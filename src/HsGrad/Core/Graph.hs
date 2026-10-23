-- | The core graph: a DAG of scalar operations, and the 'Builder' monad that
-- creates it.
--
-- A graph is pure data (decision D1). Every node is one 'Op' from
-- "HsGrad.Core.Op" applied to argument nodes, which it refers to by
-- 'NodeId'. Graphs are only ever created by running a 'Builder', and the
-- builder guarantees four invariants:
--
-- * __Ids are a topological order__ (D2). Ids are handed out as
--   @0, 1, 2, …@, and a node can only refer to nodes that already exist, so
--   every argument has a smaller id than the node that uses it. Ascending
--   id order is a valid forward order, descending a valid backward order.
--
-- * __Hash-consing__ (D3). Asking for a node whose op and arguments are
--   equal to an existing node's returns the existing id. So the graph never
--   holds two structurally identical nodes.
--
-- * __Arity__. A node has exactly @'arity' op@ arguments.
--
-- * __Finite constants__. A 'Const' node never holds NaN or an infinity,
--   which would break the ordering of the hash-consing table
--   (@docs/03-core-op.md@ §3.5).
--
-- Each node also carries its 'Provenance' (D4): where in the source it came
-- from and an optional human label. Provenance plays no part in
-- hash-consing.
--
-- Evaluation, backprop, symbolic differentiation and drawing live in later
-- modules; this one only builds and inspects graphs.
-- See @docs/04-core-graph.md@.
module HsGrad.Core.Graph
  ( -- * Node ids
    NodeId
  , nodeIndex

    -- * Provenance
  , Pos (..)
  , SrcSpan (..)
  , Provenance (..)
  , noProvenance

    -- * Nodes and graphs
  , Node (..)
  , Graph
  , emptyGraph
  , nodeCount
  , nodes
  , nodeIds
  , lookupNode
  , nodeAt
  , findNode
  , reachable

    -- * Building graphs
  , Builder
  , GraphError (..)
  , build
  , extend
  , node
  , constant
  , input
  , param
  , unary
  , binary
  , withSpan
  , label
  , currentGraph
  ) where

import Control.Applicative ((<|>))
import Control.Monad (unless, when)
import Control.Monad.Except (throwError)
import Control.Monad.Reader (ReaderT, ask, local, runReaderT)
import Control.Monad.State.Strict (StateT, get, gets, modify', put, runStateT)
import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import HsGrad.Core.Op (BinOp, Leaf (..), Name, Op (..), UnOp, arity)

-- ---------------------------------------------------------------------------
-- Node ids

-- | The identity of a node within one graph.
--
-- The constructor is not exported: the only way to get a 'NodeId' is from
-- the 'Builder', which hands them out in increasing order. 'Ord' is
-- therefore "was created earlier than", and that is a topological order of
-- the graph (D2).
newtype NodeId = NodeId Int
  deriving (Eq, Ord, Show)

-- | The id as a plain 'Int': @0@ for the first node of a graph, @1@ for the
-- second, and so on. Use it as the key of an 'IntMap' of per-node values
-- (D1), as the forward pass and backprop do.
nodeIndex :: NodeId -> Int
nodeIndex (NodeId i) = i

-- ---------------------------------------------------------------------------
-- Provenance

-- | A position in a source file: 1-based line and column.
data Pos = Pos
  { posLine :: !Int
  , posCol :: !Int
  }
  deriving (Eq, Ord, Show)

-- | A region of a source file, from 'spanStart' (inclusive) to 'spanEnd'
-- (exclusive).
--
-- This is the smallest span type that lets a diagnostic point back into a
-- @.hsg@ file. It lives in Core so that Core does not depend on the language
-- (layer rule, @00-architecture.md@ §2); "HsGrad.Lang.Span" (step 10) can
-- build its richer type on top of it.
data SrcSpan = SrcSpan
  { spanFile :: !FilePath
  , spanStart :: !Pos
  , spanEnd :: !Pos
  }
  deriving (Eq, Ord, Show)

-- | Where a node came from (D4): the source expression that created it and
-- an optional human-readable label, such as @W1[3,0]@ or @h@.
--
-- Both parts are optional. Nodes built directly from Haskell code (tests,
-- symbolic gradients) may have neither.
data Provenance = Provenance
  { provSpan :: !(Maybe SrcSpan)
  , provLabel :: !(Maybe Text)
  }
  deriving (Eq, Show)

-- | Combining provenance keeps what is already known and only fills in what
-- is missing: for each field, the left 'Just' wins.
--
-- This is the rule for merged nodes. When hash-consing returns an existing
-- node, the builder combines @existing <> new@: the provenance of the
-- __first__ occurrence wins, field by field.
instance Semigroup Provenance where
  Provenance s1 l1 <> Provenance s2 l2 = Provenance (s1 <|> s2) (l1 <|> l2)

instance Monoid Provenance where
  mempty = noProvenance

-- | No span and no label.
noProvenance :: Provenance
noProvenance = Provenance Nothing Nothing

-- ---------------------------------------------------------------------------
-- Nodes and graphs

-- | One node: an op applied to argument nodes.
--
-- The builder guarantees that @length nodeArgs == arity nodeOp@ and that
-- every argument has a smaller id than the node itself.
data Node = Node
  { nodeOp :: !Op
  , nodeArgs :: ![NodeId]
  , nodeProv :: !Provenance
  }
  deriving (Eq, Show)

-- | A graph of scalar operations.
--
-- The constructor is not exported, so every graph comes from a 'Builder'
-- and satisfies the invariants listed at the top of this module.
data Graph = Graph
  { graphNodes :: !(IntMap Node)
  -- ^ Every node, keyed by 'nodeIndex'. The keys are exactly
  -- @0 .. graphCount - 1@.
  , graphCount :: !Int
  -- ^ The number of nodes, which is also the next fresh id. Stored because
  -- 'IntMap.size' takes linear time.
  , graphTable :: !(Map (Op, [NodeId]) NodeId)
  -- ^ The hash-consing table (D3): op and arguments to the node that has
  -- them. Kept in the graph, not only in the builder, so that 'extend' can
  -- add nodes to an existing graph and still find its nodes.
  }
  deriving (Eq, Show)

-- | The graph with no nodes.
emptyGraph :: Graph
emptyGraph = Graph IntMap.empty 0 Map.empty

-- | The number of nodes.
nodeCount :: Graph -> Int
nodeCount = graphCount

-- | All nodes with their ids, in ascending id order, which is a topological
-- order (D2).
nodes :: Graph -> [(NodeId, Node)]
nodes g = [(NodeId i, n) | (i, n) <- IntMap.toAscList (graphNodes g)]

-- | All ids, in ascending order.
nodeIds :: Graph -> [NodeId]
nodeIds g = map NodeId [0 .. graphCount g - 1]

-- | The node with this id, or 'Nothing' if the id belongs to another graph.
lookupNode :: NodeId -> Graph -> Maybe Node
lookupNode (NodeId i) g = IntMap.lookup i (graphNodes g)

-- | The node with this id.
--
-- Partial: it fails with an error if the id is not a node of this graph.
-- That can only happen when an id is used with a graph it did not come
-- from, which is a bug in the caller. Use 'lookupNode' where that is a
-- real possibility.
nodeAt :: Graph -> NodeId -> Node
nodeAt g i = case lookupNode i g of
  Just n -> n
  Nothing ->
    error ("HsGrad.Core.Graph.nodeAt: " <> show i <> " is not a node of this graph")

-- | The id of the node with this op and these arguments, if there is one.
-- This is the hash-consing lookup on its own, without creating a node:
-- @findNode (Leaf (Input "x")) [] g@ is the id of input @x@, if @g@ has it.
findNode :: Op -> [NodeId] -> Graph -> Maybe NodeId
findNode op args g = Map.lookup (op, args) (graphTable g)

-- | The set of nodes the given outputs depend on: the outputs themselves,
-- their arguments, their arguments' arguments, and so on.
--
-- Later steps use it to skip dead nodes: the forward pass evaluates
-- @Set.toAscList (reachable g outs)@, backprop walks @Set.toDescList@.
-- Ids that are not nodes of the graph are ignored, so the result is always
-- a subset of @nodeIds g@.
reachable :: Graph -> [NodeId] -> Set NodeId
reachable g = go Set.empty
  where
    -- A depth-first search with an explicit stack of ids still to visit.
    -- Each node is expanded at most once, so this takes O(r log r) for r
    -- reachable nodes, however much sharing there is.
    go seen [] = seen
    go seen (i : stack)
      | i `Set.member` seen = go seen stack
      | otherwise = case lookupNode i g of
          Nothing -> go seen stack
          Just n -> go (Set.insert i seen) (nodeArgs n ++ stack)

-- ---------------------------------------------------------------------------
-- The builder

-- | Everything that can go wrong while building a graph.
--
-- All three are bugs in the code that calls the builder (the compiler, the
-- symbolic differentiator, the simplifier), not mistakes in a @.hsg@ file,
-- which the checker reports long before a graph is built.
data GraphError
  = -- | The op was given a different number of arguments than its arity.
    ArityMismatch Op [NodeId]
  | -- | An argument (or a node passed to 'label') is not a node of the graph
    -- being built: it came from a different graph.
    UnknownNode NodeId
  | -- | A constant that is NaN or infinite.
    NonFiniteConst Double
  deriving (Eq, Show)

-- | A computation that adds nodes to a graph and returns an @a@, usually a
-- 'NodeId' or a structure of them.
--
-- It combines three effects, one per layer:
--
-- * 'ReaderT' for the /current source span/, which 'withSpan' sets for a
--   scope and every new node records;
-- * 'StateT' for the graph under construction;
-- * 'Either' for 'GraphError'. The first error aborts the whole build.
--
-- The layers are hidden behind a newtype, so code outside this module can
-- only use the operations below and cannot break the graph's invariants.
newtype Builder a
  = Builder (ReaderT (Maybe SrcSpan) (StateT Graph (Either GraphError)) a)
  deriving (Functor, Applicative, Monad)

-- | Run a builder from the empty graph. Returns its result and the graph
-- it built, or the first error.
build :: Builder a -> Either GraphError (a, Graph)
build = extend emptyGraph

-- | Run a builder that adds nodes to an existing graph.
--
-- Old nodes keep their ids, new nodes get larger ones, and hash-consing
-- also finds the old nodes. Symbolic differentiation (step 7) uses this to
-- build gradient nodes that share subexpressions with the forward graph.
extend :: Graph -> Builder a -> Either GraphError (a, Graph)
extend g (Builder m) = runStateT (runReaderT m Nothing) g

-- | The graph built so far. Read-only: there is no way to put it back.
currentGraph :: Builder Graph
currentGraph = Builder get

-- | The node for an op applied to arguments: an existing node if the graph
-- already has one with this op and these arguments, otherwise a new node
-- with the next id.
--
-- A new node records the current span (see 'withSpan'). An existing node
-- keeps its provenance and only gains a span if it had none.
--
-- Fails with 'ArityMismatch' if the number of arguments is not
-- @'arity' op@, with 'UnknownNode' if an argument is not in the graph, and
-- with 'NonFiniteConst' for a NaN or infinite constant.
node :: Op -> [NodeId] -> Builder NodeId
node op args = Builder $ do
  when (length args /= arity op) $
    throwError (ArityMismatch op args)
  case op of
    Leaf (Const c)
      | isNaN c || isInfinite c -> throwError (NonFiniteConst c)
    _ -> pure ()
  g <- get
  mapM_ (\a -> unless (exists a g) (throwError (UnknownNode a))) args
  sp <- ask
  let prov = Provenance sp Nothing
  case Map.lookup (op, args) (graphTable g) of
    Just i -> do
      put (addProvenance i prov g)
      pure i
    Nothing -> do
      let i = NodeId (graphCount g)
      put
        Graph
          { graphNodes = IntMap.insert (nodeIndex i) (Node op args prov) (graphNodes g)
          , graphCount = graphCount g + 1
          , graphTable = Map.insert (op, args) i (graphTable g)
          }
      pure i

-- | Does the graph have a node with this id?
exists :: NodeId -> Graph -> Bool
exists (NodeId i) g = i >= 0 && i < graphCount g

-- | Combine more provenance into an existing node: @old <> new@, so what
-- the node already has wins.
addProvenance :: NodeId -> Provenance -> Graph -> Graph
addProvenance (NodeId i) prov g =
  g {graphNodes = IntMap.adjust update i (graphNodes g)}
  where
    update n = n {nodeProv = nodeProv n <> prov}

-- | A constant leaf. Fails with 'NonFiniteConst' for NaN or an infinity.
constant :: Double -> Builder NodeId
constant c = node (Leaf (Const c)) []

-- | An input leaf. Asking for the same name twice gives the same node.
input :: Name -> Builder NodeId
input n = node (Leaf (Input n)) []

-- | A parameter leaf. Asking for the same name twice gives the same node.
param :: Name -> Builder NodeId
param n = node (Leaf (Param n)) []

-- | A unary op applied to one node. Its arity is right by construction.
unary :: UnOp -> NodeId -> Builder NodeId
unary op x = node (Un op) [x]

-- | A binary op applied to two nodes, in argument order.
binary :: BinOp -> NodeId -> NodeId -> Builder NodeId
binary op x y = node (Bin op) [x, y]

-- | Run a builder with a current source span: every node it creates
-- records this span, unless an inner 'withSpan' sets a different one. The
-- previous span is back in force afterwards.
--
-- This mirrors the shape of the source: the compiler wraps the code for
-- each expression in that expression's span, so every node gets the span
-- of the innermost expression that created it.
withSpan :: SrcSpan -> Builder a -> Builder a
withSpan sp (Builder m) = Builder (local (const (Just sp)) m)

-- | Give a node a human-readable label, unless it already has one (the
-- first label wins, as for merged nodes). Fails with 'UnknownNode' if the
-- node is not in the graph.
label :: Text -> NodeId -> Builder ()
label l i = Builder $ do
  known <- gets (exists i)
  unless known $
    throwError (UnknownNode i)
  modify' (addProvenance i (Provenance Nothing (Just l)))
