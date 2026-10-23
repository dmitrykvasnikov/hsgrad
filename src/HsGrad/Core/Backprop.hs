-- | Numeric reverse-mode automatic differentiation: micrograd's
-- @backward()@.
--
-- Given a graph, its forward values ("HsGrad.Core.Eval") and one scalar
-- output @out@, backprop computes @∂out/∂n@ for every node @n@ the output
-- depends on, in one sweep over the nodes in __descending__ id order
-- (decision D2). Each node passes its gradient on to its arguments,
-- multiplied by the local derivatives of its op ('localGrads' from
-- "HsGrad.Core.Op"), and every argument __adds up__ what it receives from
-- all of its uses. Because ids are a topological order, a node's gradient
-- is complete by the time the sweep reaches it: every node that uses it has
-- a larger id and has already been processed.
--
-- Like the values, the gradients live outside the graph, in an
-- @IntMap Double@ keyed by node index (D1). Nodes the output does not
-- depend on get no entry: their gradient is zero.
--
-- A non-finite gradient (for example the derivative of @√x@ at @x = 0@) is
-- reported as an error that names the node where it appeared, just like a
-- non-finite value in the forward pass.
-- See @docs/06-core-backprop.md@.
module HsGrad.Core.Backprop
  ( -- * Gradients
    Grads
  , lookupGrad
  , gradAt

    -- * The reverse sweep
  , BackpropError (..)
  , backward

    -- * Forward and backward together
  , GradError (..)
  , forwardBackward
  , gradients

    -- * Leaf gradients by name
  , LeafGrads (..)
  , leafGrads
  ) where

import Control.Monad (foldM)
import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import HsGrad.Core.Eval
  ( Env
  , EvalError
  , Schedule
  , Values
  , evalSchedule
  , lookupValue
  , schedule
  , scheduleLeaves
  , scheduleNodes
  , valueAt
  )
import HsGrad.Core.Graph (Graph, Node (..), NodeId, nodeIndex)
import HsGrad.Core.Op (Leaf (..), Name, localGrads)

-- ---------------------------------------------------------------------------
-- Gradients

-- | The gradient @∂out/∂n@ of one output with respect to every node @n@ it
-- depends on, keyed by 'nodeIndex' (D1), like 'Values'.
--
-- Exactly the nodes reachable from the output have an entry, the output
-- itself included (with gradient @1@). Constants have entries too: they
-- are nodes like any other, even though nobody will ever update them.
type Grads = IntMap Double

-- | The gradient of a node, or 'Nothing' if the output does not depend on
-- it.
lookupGrad :: NodeId -> Grads -> Maybe Double
lookupGrad i = IntMap.lookup (nodeIndex i)

-- | The gradient of a node, @0@ if the output does not depend on it.
--
-- Unlike 'valueAt', this is total. A missing value is unknown, but a
-- missing gradient is known: if the output does not depend on a node,
-- changing the node does not change the output, so @∂out/∂n = 0@.
gradAt :: Grads -> NodeId -> Double
gradAt grads i = IntMap.findWithDefault 0 (nodeIndex i) grads

-- ---------------------------------------------------------------------------
-- The reverse sweep

-- | Everything that can go wrong in the reverse sweep.
--
-- The forward values are finite (the forward pass checks them), so this is
-- the only thing left: a local derivative rule that gives a non-finite
-- number where the forward function did not (@√x@ at @0@: the value is
-- @0@, the slope is infinite), or a gradient that overflows.
data BackpropError
  = -- | Passing the gradient of a node on to its arguments made an
    -- argument's gradient NaN or infinite. The fields are the node, the
    -- values of its arguments, its own gradient (finite), and its local
    -- gradients (one per argument).
    --
    -- This is the /first/ node in sweep order (descending ids) where a
    -- non-finite gradient appears, so it is where the problem starts. Its
    -- source span is @provSpan (nodeProv (nodeAt g i))@.
    NonFiniteGrad NodeId [Double] Double [Double]
  deriving (Eq, Show)

-- | The reverse sweep: the gradient of @out@ with respect to every node it
-- depends on.
--
-- The schedule and values must come from the same forward pass
-- (@'evalSchedule' s env == Right vals@), and @out@ must be one of the
-- nodes the schedule evaluated; otherwise this fails with an 'error', like
-- 'valueAt'. The schedule may have been built for several outputs: the
-- nodes @out@ does not depend on are skipped.
backward :: Schedule -> Values -> NodeId -> Either BackpropError Grads
backward s vals out = case lookupValue out vals of
  Nothing ->
    error ("HsGrad.Core.Backprop.backward: output " <> show out <> " was not evaluated")
  -- The seed: ∂out/∂out = 1.
  Just _ -> sweep vals (IntMap.singleton (nodeIndex out) 1) (reverse (scheduleNodes s))

-- | Visit the nodes in descending id order. When a node is reached, every
-- node that uses it (they all have larger ids) has already passed its share
-- on, so the gradient found in @grads@ is final. A node without a gradient
-- is one the output does not depend on (a node that uses the output, say,
-- or one only another output of the schedule needs), and is skipped.
sweep :: Values -> Grads -> [(NodeId, Node)] -> Either BackpropError Grads
sweep _ !grads [] = Right grads
sweep vals !grads ((i, n) : rest) = case lookupGrad i grads of
  Nothing -> sweep vals grads rest
  Just g -> do
    grads' <- foldM (passOn g) grads (zip (nodeArgs n) locals)
    sweep vals grads' rest
  where
    argVals = map (valueAt vals) (nodeArgs n)
    -- One local derivative per argument; none for a leaf.
    locals = localGrads (nodeOp n) argVals (valueAt vals i)
    -- The chain rule for one use of argument @a@: its gradient grows by
    -- ∂out/∂n · ∂n/∂a. It is /added/ to what is already there, from the
    -- argument's other uses (including the other position of the same
    -- node, as in @a + a@); an argument seen for the first time starts
    -- from 0.
    passOn g acc (a, local)
      | isFinite total = Right (IntMap.insert (nodeIndex a) total acc)
      | otherwise = Left (NonFiniteGrad i argVals g locals)
      where
        total = IntMap.findWithDefault 0 (nodeIndex a) acc + g * local

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

-- ---------------------------------------------------------------------------
-- Forward and backward together

-- | Something went wrong in the forward pass or in the reverse sweep.
data GradError
  = ForwardError EvalError
  | BackwardError BackpropError
  deriving (Eq, Show)

-- | The forward pass followed by the reverse sweep: the values of the
-- schedule's nodes and the gradients of @out@.
--
-- This is one training step's worth of differentiation (step 9): build the
-- schedule for the loss once, then call this with each new environment.
-- @out@ must be one of the schedule's nodes (see 'backward').
forwardBackward :: Schedule -> Env -> NodeId -> Either GradError (Values, Grads)
forwardBackward s env out = do
  vals <- either (Left . ForwardError) Right (evalSchedule s env)
  grads <- either (Left . BackwardError) Right (backward s vals out)
  pure (vals, grads)

-- | Values and gradients of one output, for a one-off computation. To
-- differentiate the same output many times, build the 'schedule' once and
-- call 'forwardBackward'.
--
-- Unlike 'backward', this cannot fail with an 'error': an id that is not a
-- node of the graph gives no values and no gradients.
gradients :: Graph -> Env -> NodeId -> Either GradError (Values, Grads)
gradients g env out
  | null (scheduleNodes s) = Right (IntMap.empty, IntMap.empty)
  | otherwise = forwardBackward s env out
  where
    s = schedule g [out]

-- ---------------------------------------------------------------------------
-- Leaf gradients by name

-- | The gradients of the input and parameter leaves, by name: the shape of
-- 'Env', for the gradient instead of the value.
--
-- Hash-consing (D3) makes every leaf a single node, so each name has
-- exactly one gradient, summed over all uses of the leaf.
data LeafGrads = LeafGrads
  { inputGrads :: !(Map Name Double)
  , paramGrads :: !(Map Name Double)
  }
  deriving (Eq, Show)

-- | The gradients of the schedule's input and parameter leaves.
--
-- Only the leaves the output depends on are included; a name that is
-- missing has gradient @0@ (use @'Map.findWithDefault' 0@). Constants are
-- not included: they have no name, and nothing updates them.
leafGrads :: Schedule -> Grads -> LeafGrads
leafGrads s grads =
  LeafGrads
    { inputGrads = Map.fromList [(n, g) | (i, Input n) <- leaves, Just g <- [lookupGrad i grads]]
    , paramGrads = Map.fromList [(n, g) | (i, Param n) <- leaves, Just g <- [lookupGrad i grads]]
    }
  where
    leaves = scheduleLeaves s
