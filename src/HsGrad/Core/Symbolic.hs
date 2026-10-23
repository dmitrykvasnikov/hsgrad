{-# LANGUAGE OverloadedStrings #-}

-- | Symbolic reverse-mode differentiation: the gradient as a graph.
--
-- "HsGrad.Core.Backprop" computes @∂out/∂n@ as /numbers/. This module runs
-- the very same sweep (decision D6), in the same order, with the same seed
-- and the same accumulation, but every step /emits graph nodes/ into the
-- 'Builder' instead of computing a number:
--
-- * the seed @∂out/∂out = 1@ is the constant @1@;
-- * a node's local derivatives are its op's rules ('partials' from
--   "HsGrad.Core.Op"), turned into nodes by a second interpreter of 'DExpr'
--   ('emitDExpr'): @DArg i@ is the argument's id, @DOut@ the node's own id;
-- * @g · local@ becomes a @*@ node, and summing over several uses becomes
--   a @+@ node.
--
-- The result maps every node the output depends on to the 'NodeId' of a
-- node whose /value/ is @∂out/∂n@. Gradient nodes are added to the forward
-- graph itself, so they refer to forward nodes (the rule for @tanh@ uses
-- the @tanh@ node), and hash-consing (D3) merges what the different
-- gradients have in common. The gradient graph is therefore about as large
-- as the forward graph, never exponentially larger, and building it costs
-- what one backprop sweep costs.
--
-- Running the sweep again, on a gradient node as the output, gives second
-- derivatives: a gradient node is a node like any other.
--
-- Only the simplifications that are free and exact are done here:
-- multiplying by a literal @0@, @1@ or @-1@ and adding a literal @0@ (see
-- 'Term'). Real simplification is "HsGrad.Core.Simplify" (step 8).
-- See @docs/08-core-symbolic.md@.
module HsGrad.Core.Symbolic
  ( -- * Gradient nodes
    GradNodes
  , lookupGradNode
  , gradNodeOrZero

    -- * The symbolic reverse sweep
  , backwardNodes
  , symbolicGradients

    -- * Leaf gradients by name
  , LeafGradNodes (..)
  , leafGradNodes

    -- * Names and the rule interpreter
  , gradientName
  , emitDExpr
  ) where

import Control.Monad (foldM)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Graph
  ( Builder
  , Graph
  , GraphError
  , Node (..)
  , NodeId
  , Provenance (..)
  , binary
  , constant
  , currentGraph
  , extend
  , label
  , lookupNode
  , nodeAt
  , nodeCount
  , nodeIndex
  , reachable
  , unary
  , withSpan
  )
import HsGrad.Core.Op (BinOp (..), DExpr (..), Leaf (..), Name (..), Op (..), UnOp (..), partials)

-- ---------------------------------------------------------------------------
-- Gradient nodes

-- | The gradient of one output with respect to every node it depends on,
-- as nodes: forward node @n@ ↦ the node whose value is @∂out/∂n@.
--
-- Exactly the nodes reachable from the output have an entry, the output
-- itself included (its gradient is the constant @1@), as in
-- 'HsGrad.Core.Backprop.Grads'.
--
-- Unlike 'HsGrad.Core.Backprop.Grads' this is a 'Map' keyed by 'NodeId',
-- not an 'Data.IntMap.IntMap' keyed by index: callers want to walk the
-- pairs (to label, caption or evaluate the gradients), and there is
-- deliberately no way back from an index to a 'NodeId'.
type GradNodes = Map NodeId NodeId

-- | The gradient node of a forward node, or 'Nothing' if the output does
-- not depend on it.
lookupGradNode :: NodeId -> GradNodes -> Maybe NodeId
lookupGradNode = Map.lookup

-- | The gradient node of a forward node, or the constant @0@ if the output
-- does not depend on it: the symbolic counterpart of
-- 'HsGrad.Core.Backprop.gradAt'. It runs in the 'Builder' because the
-- constant may have to be created.
gradNodeOrZero :: GradNodes -> NodeId -> Builder NodeId
gradNodeOrZero grads i = maybe (constant 0) pure (lookupGradNode i grads)

-- ---------------------------------------------------------------------------
-- Terms: literals that are not nodes yet

-- | A value during the sweep: either a literal that has not been turned
-- into a node, or a node.
--
-- Keeping literals apart until they are needed is what makes the light
-- simplification free. Rules are full of @DLit 1@ (the partials of @+@ and
-- @-@) and @DLit (-1)@ (@-x@, @x - y@), and the seed is @1@. Multiplying a
-- gradient by such a literal needs no node at all, and no @Const 1@ node is
-- created just to be multiplied away.
data Term
  = Lit Double
  | Ref NodeId

-- | The node for a term, creating a constant for a literal.
materialize :: Term -> Builder NodeId
materialize (Lit c) = constant c
materialize (Ref i) = pure i

-- | @s · t@. Exact for finite values: @0 · t = 0@, @1 · t = t@ and
-- @-1 · t = -t@ hold bit for bit in IEEE arithmetic (up to the sign of a
-- zero).
mulT :: Term -> Term -> Builder Term
mulT (Lit 0) _ = pure (Lit 0)
mulT _ (Lit 0) = pure (Lit 0)
mulT (Lit 1) t = pure t
mulT s (Lit 1) = pure s
mulT (Lit (-1)) t = negT t
mulT s (Lit (-1)) = negT s
mulT s t = binNode Mul s t

-- | @s + t@, dropping a literal @0@ on either side.
addT :: Term -> Term -> Builder Term
addT (Lit 0) t = pure t
addT s (Lit 0) = pure s
addT s t = binNode Add s t

-- | @-t@. The negation of a literal is a literal.
negT :: Term -> Builder Term
negT (Lit c) = pure (Lit (negate c))
negT t = unNode Neg t

-- | A unary node, without simplification.
unNode :: UnOp -> Term -> Builder Term
unNode op t = Ref <$> (unary op =<< materialize t)

-- | A binary node, without simplification.
binNode :: BinOp -> Term -> Term -> Builder Term
binNode op s t = do
  x <- materialize s
  y <- materialize t
  Ref <$> binary op x y

-- ---------------------------------------------------------------------------
-- The second interpreter of derivative rules

-- | Interpret a rule as nodes: the symbolic counterpart of 'evalDExpr'.
-- Given the argument ids of a node and the node's own id, the result is a
-- node whose value is the rule's value.
--
-- > DLit c      ↦  a Const c node
-- > DArg i      ↦  the id of argument i
-- > DOut        ↦  the node's own id
-- > DUn / DBin  ↦  a new (or hash-consed) node
emitDExpr :: [NodeId] -> NodeId -> DExpr -> Builder NodeId
emitDExpr args self e = materialize =<< emit args self e

-- | 'emitDExpr', returning a 'Term' so that literals stay literals.
-- @*@, @+@ and @-x@ go through the simplifying constructors; every other
-- op becomes a node as it is.
emit :: [NodeId] -> NodeId -> DExpr -> Builder Term
emit args self = go
  where
    go e = case e of
      DLit c -> pure (Lit c)
      -- Every rule only mentions arguments below its op's arity (tested in
      -- step 2), so the lookup cannot fail.
      DArg i -> pure (Ref (args !! i))
      DOut -> pure (Ref self)
      DUn Neg a -> negT =<< go a
      DUn op a -> unNode op =<< go a
      DBin op a b -> do
        x <- go a
        y <- go b
        case op of
          Mul -> mulT x y
          Add -> addT x y
          _ -> binNode op x y

-- ---------------------------------------------------------------------------
-- The symbolic reverse sweep

-- | Add the gradient nodes of @out@ to the graph being built, and return
-- them.
--
-- The sweep is 'HsGrad.Core.Backprop.backward' with nodes for numbers:
-- the nodes @out@ depends on are visited in descending id order (D2); each
-- passes its gradient on to its arguments, multiplied by its local
-- derivatives, and each argument adds up what it receives. The nodes
-- created for a node's arguments get that node's source span (D4), and
-- every new gradient node of a named node gets a label such as @∂L/∂a@
-- (see 'gradientName').
--
-- Only the nodes that exist when the sweep starts are differentiated, so
-- @out@ may itself be a gradient node: that gives second derivatives. An
-- id that is not a node of the graph gives no gradients at all.
backwardNodes :: NodeId -> Builder GradNodes
backwardNodes out = do
  g <- currentGraph
  case lookupNode out g of
    Nothing -> pure Map.empty
    Just _ -> do
      let schedule = [(i, nodeAt g i) | i <- Set.toDescList (reachable g [out])]
      -- The seed: ∂out/∂out = 1.
      terms <- sweep (Map.singleton out (Lit 1)) schedule
      grads <- traverse materialize terms
      labelGradients g out grads
      pure grads

-- | Visit the nodes in descending id order. As in numeric backprop, a
-- node's gradient is complete when the sweep reaches it, because every
-- node that uses it has a larger id. Every node in the schedule is
-- reachable from @out@ and so receives a gradient (possibly the literal
-- @0@) before it is visited; the 'Nothing' case is there for totality.
sweep :: Map NodeId Term -> [(NodeId, Node)] -> Builder (Map NodeId Term)
sweep !grads [] = pure grads
sweep !grads ((i, n) : rest) = case Map.lookup i grads of
  Nothing -> sweep grads rest
  Just g -> do
    grads' <- inSpanOf n (foldM (passOn g) grads (zip (nodeArgs n) (partials (nodeOp n))))
    sweep grads' rest
  where
    -- The chain rule for one use of argument @a@: its gradient grows by
    -- ∂out/∂n · ∂n/∂a. Numeric backprop adds to 0 for an argument seen for
    -- the first time; here the first contribution is used as it is.
    passOn g acc (a, rule) = do
      contribution <- case g of
        -- A gradient that is the literal 0 passes on 0, without emitting
        -- the local derivative only to multiply it away.
        Lit 0 -> pure (Lit 0)
        _ -> mulT g =<< emit (nodeArgs n) i rule
      total <- case Map.lookup a acc of
        Nothing -> pure contribution
        Just old -> addT old contribution
      pure (Map.insert a total acc)

-- | Run a builder with the span of a forward node, if it has one, so that
-- the gradient nodes a node creates are drawn in the same cluster (step 6).
inSpanOf :: Node -> Builder a -> Builder a
inSpanOf n = maybe id withSpan (provSpan (nodeProv n))

-- | Label every gradient node this sweep created with what it is
-- (@∂L/∂a@), if both the output and the forward node have names.
--
-- Only /new/, non-constant nodes are labelled. A gradient can be a node
-- that already existed (in @L = d · f@, @∂L/∂d@ is the node @f@ itself),
-- and labelling it would rename a forward node in every picture; a
-- constant can be shared by anything. Labels are "first wins" (step 3), so
-- when one node is the gradient of several named nodes, the one with the
-- smallest id names it.
labelGradients :: Graph -> NodeId -> GradNodes -> Builder ()
labelGradients before out grads = do
  after <- currentGraph
  let isNew gi =
        nodeIndex gi >= nodeCount before && not (isLeaf (nodeAt after gi))
  sequence_
    [ label name gi
    | (i, gi) <- Map.toList grads
    , isNew gi
    , Just name <- [gradientName before out i]
    ]
  where
    isLeaf n = case nodeOp n of
      Leaf _ -> True
      _ -> False

-- | Differentiate one output of an existing graph: the graph extended with
-- the gradient nodes, and the gradient node of every node the output
-- depends on.
--
-- It fails only if a rule's literal is not finite (a 'PowC' exponent too
-- large for a 'Double'), which the builder rejects. An id that is not a
-- node of the graph gives no gradients and an unchanged graph.
symbolicGradients :: Graph -> NodeId -> Either GraphError (GradNodes, Graph)
symbolicGradients g out = extend g (backwardNodes out)

-- ---------------------------------------------------------------------------
-- Names

-- | The name of the gradient of @out@ with respect to @n@, such as
-- @∂L/∂a@, if both have names: an input's or parameter's name, or a
-- node's label. An output name that is itself a fraction is put in
-- parentheses: @∂(∂y/∂x)/∂x@.
gradientName :: Graph -> NodeId -> NodeId -> Maybe Text
gradientName g out n = do
  o <- nameOf out
  x <- nameOf n
  let o' = if "/" `Text.isInfixOf` o then "(" <> o <> ")" else o
  pure ("∂" <> o' <> "/∂" <> x)
  where
    nameOf i = do
      nd <- lookupNode i g
      case nodeOp nd of
        Leaf (Input (Name t)) -> Just t
        Leaf (Param (Name t)) -> Just t
        _ -> provLabel (nodeProv nd)

-- ---------------------------------------------------------------------------
-- Leaf gradients by name

-- | The gradient nodes of the input and parameter leaves, by name: the
-- shape of 'HsGrad.Core.Eval.Env' and 'HsGrad.Core.Backprop.LeafGrads'.
data LeafGradNodes = LeafGradNodes
  { inputGradNodes :: !(Map Name NodeId)
  , paramGradNodes :: !(Map Name NodeId)
  }
  deriving (Eq, Show)

-- | The gradient nodes of the leaves the output depends on. A missing name
-- is a leaf the output does not depend on: its gradient is zero. Constants
-- are left out. The graph may be the forward graph or the extended one.
leafGradNodes :: Graph -> GradNodes -> LeafGradNodes
leafGradNodes g grads =
  LeafGradNodes
    { inputGradNodes = Map.fromList [(nm, gi) | (Leaf (Input nm), gi) <- leaves]
    , paramGradNodes = Map.fromList [(nm, gi) | (Leaf (Param nm), gi) <- leaves]
    }
  where
    leaves = [(nodeOp (nodeAt g i), gi) | (i, gi) <- Map.toList grads]
