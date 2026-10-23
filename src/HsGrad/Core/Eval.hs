-- | The forward pass: the value of every node a set of outputs depends on.
--
-- Evaluation takes three things:
--
-- * a 'Graph' (step 3), which never changes;
-- * an 'Env': the values of the input and parameter leaves, by name;
-- * the outputs whose values are wanted.
--
-- It computes the nodes the outputs depend on, and only those, in ascending
-- id order, which is a topological order (decision D2). The result is a map
-- from node index to value (D1), kept outside the graph.
--
-- Evaluation is split in two so that training (step 9) can run it thousands
-- of times on one graph (D10):
--
-- * 'schedule' does the work that only depends on the graph and the outputs:
--   finding the reachable nodes and their leaves. Done once.
-- * 'evalSchedule' does the work that depends on the leaf values. Done once
--   per step, after rebinding parameters and dataset slots in the 'Env'.
--
-- Two things can go wrong, and both are reported as values ('EvalError'):
-- a leaf without a value (the core side of error E0502), and a NaN or
-- infinite result (E0501). The second check is always on, and the error
-- names the first node that produced a non-finite value, so a diagnostic
-- can point at its source span.
-- See @docs/05-core-eval.md@.
module HsGrad.Core.Eval
  ( -- * Leaf values
    Env (..)
  , emptyEnv
  , bindInput
  , bindParam

    -- * Schedules
  , Schedule
  , schedule
  , scheduleNodes
  , scheduleLeaves

    -- * Evaluation
  , Values
  , EvalError (..)
  , evalSchedule
  , evaluate
  , lookupValue
  , valueAt
  ) where

import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.List.NonEmpty (NonEmpty, nonEmpty)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import HsGrad.Core.Graph (Graph, Node (..), NodeId, nodeAt, nodeIndex, reachable)
import HsGrad.Core.Op (Leaf (..), Name, Op (..), forwardBin, forwardUn)

-- ---------------------------------------------------------------------------
-- Leaf values

-- | The values of the leaves that do not carry their own: inputs and
-- parameters, by name. ('Const' leaves carry their value in the graph.)
--
-- Inputs and parameters are kept in separate maps because they are
-- different leaves even when their names are equal: @'Input' "w"@ and
-- @'Param' "w"@ are two nodes (@docs/03-core-op.md@ §3.4). Separate maps
-- also match how training changes them: after every step it replaces all
-- parameters at once (@env {envParams = ps}@) and rebinds the dataset slots
-- with 'bindInput'. Neither touches the graph (D10).
--
-- Names the graph does not use are allowed and ignored.
data Env = Env
  { envInputs :: !(Map Name Double)
  , envParams :: !(Map Name Double)
  }
  deriving (Eq, Show)

-- | No values at all. Enough for a graph whose only leaves are constants.
emptyEnv :: Env
emptyEnv = Env Map.empty Map.empty

-- | Give an input (or dataset slot) a value, replacing an earlier one.
bindInput :: Name -> Double -> Env -> Env
bindInput n v env = env {envInputs = Map.insert n v (envInputs env)}

-- | Give a parameter a value, replacing an earlier one.
bindParam :: Name -> Double -> Env -> Env
bindParam n v env = env {envParams = Map.insert n v (envParams env)}

-- | The value of a leaf in an environment, if it has one.
leafValue :: Env -> Leaf -> Maybe Double
leafValue env l = case l of
  Const c -> Just c
  Input n -> Map.lookup n (envInputs env)
  Param n -> Map.lookup n (envParams env)

-- ---------------------------------------------------------------------------
-- Schedules

-- | The nodes some outputs depend on, in the order they must be computed.
--
-- A schedule depends on the graph and the outputs, but not on any leaf
-- value, so it can be computed once and evaluated with many environments.
data Schedule = Schedule
  { schedNodes :: ![(NodeId, Node)]
  -- ^ Every reachable node, in ascending id order (a topological order).
  , schedLeaves :: ![(NodeId, Leaf)]
  -- ^ The reachable input and parameter leaves, in ascending id order.
  }

-- | The schedule for computing these outputs of this graph.
--
-- It contains the outputs and everything they depend on, each node once
-- however often it is used. Output ids that are not nodes of the graph are
-- ignored, as in 'reachable'.
schedule :: Graph -> [NodeId] -> Schedule
schedule g outs = Schedule steps leaves
  where
    steps = [(i, nodeAt g i) | i <- Set.toAscList (reachable g outs)]
    leaves = [(i, l) | (i, Node {nodeOp = Leaf l}) <- steps, needsValue l]
    needsValue (Const _) = False
    needsValue _ = True

-- | The nodes of a schedule, in evaluation order (ascending ids). Backprop
-- (step 5) walks them in reverse.
scheduleNodes :: Schedule -> [(NodeId, Node)]
scheduleNodes = schedNodes

-- | The input and parameter leaves of a schedule: the leaves that need a
-- value in the 'Env'.
scheduleLeaves :: Schedule -> [(NodeId, Leaf)]
scheduleLeaves = schedLeaves

-- ---------------------------------------------------------------------------
-- Evaluation

-- | The value of every evaluated node, keyed by 'nodeIndex' (D1).
type Values = IntMap Double

-- | Everything that can go wrong in the forward pass.
data EvalError
  = -- | Some leaves have no value in the environment: every reachable input
    -- and parameter leaf without one, in ascending id order. The core side
    -- of E0502.
    UnboundLeaves (NonEmpty (NodeId, Leaf))
  | -- | A node's value is NaN or infinite: the node, the values of its
    -- arguments (all finite), and its value. The core side of E0501.
    --
    -- This is the /first/ such node in evaluation order, so it is where the
    -- problem starts, not one of the nodes the NaN spread to. Its source
    -- span is @provSpan (nodeProv (nodeAt g i))@.
    NonFiniteValue NodeId [Double] Double
  deriving (Eq, Show)

-- | Evaluate a schedule with the given leaf values.
--
-- First every input and parameter leaf is looked up; if any are missing,
-- all of them are reported together. Then the nodes are computed in
-- ascending id order, stopping at the first non-finite value.
evalSchedule :: Schedule -> Env -> Either EvalError Values
evalSchedule s env = do
  leafVals <- bindLeaves env (schedLeaves s)
  sweep leafVals (schedNodes s)

-- | Build a schedule and evaluate it. For a single evaluation; to evaluate
-- the same outputs many times, build the 'schedule' once and call
-- 'evalSchedule' for each environment.
evaluate :: Graph -> Env -> [NodeId] -> Either EvalError Values
evaluate g env outs = evalSchedule (schedule g outs) env

-- | Look up every leaf of the schedule in the environment. Either all of
-- them have a value, or the result lists every one that does not.
bindLeaves :: Env -> [(NodeId, Leaf)] -> Either EvalError Values
bindLeaves env leaves =
  case nonEmpty [(i, l) | (i, l, Nothing) <- looked] of
    Just missing -> Left (UnboundLeaves missing)
    Nothing -> Right (IntMap.fromList [(nodeIndex i, v) | (i, _, Just v) <- looked])
  where
    looked = [(i, l, leafValue env l) | (i, l) <- leaves]

-- | Compute the nodes in order. @vals@ starts out with the leaf values from
-- 'bindLeaves' and gains one value per node; the bang keeps it evaluated,
-- so no chain of unevaluated inserts builds up.
sweep :: Values -> [(NodeId, Node)] -> Either EvalError Values
sweep !vals [] = Right vals
sweep !vals ((i, n) : rest)
  | isFinite v = sweep (IntMap.insert (nodeIndex i) v vals) rest
  | otherwise = Left (NonFiniteValue i argVals v)
  where
    -- Arguments have smaller ids, so they were computed earlier in the sweep.
    argVals = map (valueAt vals) (nodeArgs n)
    v = case (nodeOp n, argVals) of
      (Leaf (Const c), []) -> c
      (Leaf _, []) -> valueAt vals i -- bound by 'bindLeaves'
      (Un op, [x]) -> forwardUn op x
      (Bin op, [x, y]) -> forwardBin op x y
      _ -> error ("HsGrad.Core.Eval: wrong number of arguments at " <> show i)

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

-- | The value of a node, if it was evaluated.
lookupValue :: NodeId -> Values -> Maybe Double
lookupValue i = IntMap.lookup (nodeIndex i)

-- | The value of a node.
--
-- Partial, like 'nodeAt': it fails with an error if the node was not
-- evaluated, i.e. it is not an output or something an output depends on.
-- Use 'lookupValue' where that is a real possibility.
valueAt :: Values -> NodeId -> Double
valueAt vals i = case lookupValue i vals of
  Just v -> v
  Nothing ->
    error ("HsGrad.Core.Eval.valueAt: " <> show i <> " was not evaluated")
