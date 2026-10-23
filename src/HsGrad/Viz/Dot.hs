{-# LANGUAGE OverloadedStrings #-}

-- | Drawing graphs with Graphviz: a hand-written exporter to the DOT
-- language, in the style of micrograd's @draw_dot@.
--
-- Every node of a 'Graph' becomes up to two DOT nodes:
--
-- * a __box__ (a Graphviz @record@) for the node's /value/: its name, and,
--   when they are supplied, its forward value and its gradient, as
--   @{ e | data -6.0000 | grad -2.0000 }@;
-- * an __op node__ (an ellipse such as @*@) for the operation that
--   computes the value. Leaves have no op node.
--
-- Edges follow the data flow: from each argument's value into the op node,
-- and from the op node into its own box. A box is only drawn when it has
-- something to show, so an unnamed intermediate node without values (most
-- nodes of a gradient graph) is just its op node.
--
-- 'DotOptions' choose what to show: forward values (from
-- "HsGrad.Core.Eval"), gradients (from "HsGrad.Core.Backprop"), highlighted
-- output nodes with optional captions such as @∂L/∂a@, clusters that group
-- nodes by their source span (decision D4), and the layout direction.
--
-- The result is DOT source as 'Text'. Writing it to a file and running
-- Graphviz is left to the caller (the CLI, step 15).
-- See @docs/07-viz-dot.md@.
module HsGrad.Viz.Dot
  ( -- * Options
    DotOptions (..)
  , RankDir (..)
  , defaultDotOptions

    -- * Highlighted nodes
  , Highlight (..)
  , highlight
  , highlightAs

    -- * Rendering
  , renderDot
  , renderDotFrom

    -- * Escaping
  , escapeString
  , escapeRecordField
  ) where

import Data.Char (isControl)
import Data.IntMap.Strict qualified as IntMap
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Backprop (Grads, lookupGrad)
import HsGrad.Core.Eval (Values, lookupValue)
import HsGrad.Core.Graph
  ( Graph
  , Node (..)
  , NodeId
  , Pos (..)
  , Provenance (..)
  , SrcSpan (..)
  , nodeAt
  , nodeIndex
  , nodes
  , reachable
  )
import HsGrad.Core.Op (BinOp (..), Leaf (..), Name (..), Op (..), UnOp (..))
import Numeric (showEFloat, showFFloat)

-- ---------------------------------------------------------------------------
-- Options

-- | What to draw, and how.
data DotOptions = DotOptions
  { dotValues :: !(Maybe Values)
  -- ^ Forward values to show as @data …@ in each box. Nodes without a
  -- value (not evaluated) show none.
  , dotGrads :: !(Maybe Grads)
  -- ^ Gradients to show as @grad …@ in each box. Nodes without a
  -- gradient (the output does not depend on them) show none, rather than
  -- @0@ (@docs/06-core-backprop.md@ §4.3).
  , dotHighlights :: ![Highlight]
  -- ^ Nodes to draw emphasised, usually the outputs: the loss of a forward
  -- graph, or the @∂L/∂a@ nodes of a gradient graph.
  , dotClusters :: !Bool
  -- ^ Group nodes that share a source span into a Graphviz cluster (D4).
  -- Nodes without a span stay outside every cluster.
  , dotRankDir :: !RankDir
  -- ^ The direction the data flows in.
  }
  deriving (Eq, Show)

-- | The direction of the layout: where the leaves are, and where the
-- outputs.
data RankDir
  = -- | Leaves on the left, outputs on the right, as in micrograd.
    LeftToRight
  | -- | Leaves at the top, outputs at the bottom.
    TopToBottom
  deriving (Eq, Show)

-- | Names only: no values, no gradients, no highlights, no clusters, left
-- to right. Change fields with record update syntax:
--
-- > defaultDotOptions {dotValues = Just vals, dotGrads = Just grads}
defaultDotOptions :: DotOptions
defaultDotOptions =
  DotOptions
    { dotValues = Nothing
    , dotGrads = Nothing
    , dotHighlights = []
    , dotClusters = False
    , dotRankDir = LeftToRight
    }

-- ---------------------------------------------------------------------------
-- Highlighted nodes

-- | A node to emphasise, with an optional caption shown in its box.
--
-- A gradient graph (step 7) has one designated node per gradient, and
-- those nodes have no label of their own: the caption is how the picture
-- says which node is @∂L/∂a@. Several highlights may name the same node
-- (two gradients that hash-consing made one node); their captions are all
-- shown, in the order given.
data Highlight = Highlight
  { hlNode :: !NodeId
  , hlCaption :: !(Maybe Text)
  }
  deriving (Eq, Show)

-- | Emphasise a node, without a caption.
highlight :: NodeId -> Highlight
highlight i = Highlight i Nothing

-- | Emphasise a node and caption it: @highlightAs "∂L/∂a" i@.
highlightAs :: Text -> NodeId -> Highlight
highlightAs caption i = Highlight i (Just caption)

-- ---------------------------------------------------------------------------
-- Rendering

-- | The whole graph as DOT source.
renderDot :: DotOptions -> Graph -> Text
renderDot opts g = render opts (nodes g)

-- | The part of the graph the given outputs depend on, as DOT source. Ids
-- that are not nodes of the graph are ignored, as in 'reachable'.
--
-- This is what @hsgrad plot --of L@ draws: the loss and everything it is
-- computed from, without the nodes of other outputs.
renderDotFrom :: DotOptions -> Graph -> [NodeId] -> Text
renderDotFrom opts g outs =
  render opts [(i, nodeAt g i) | i <- Set.toAscList (reachable g outs)]

-- | Render a set of nodes, given in ascending id order. The set must be
-- closed under arguments (every argument of a node is in it too), which is
-- true of a whole graph and of a reachable set.
--
-- The output has four parts: a header, the node statements of unclustered
-- nodes, one subgraph per cluster with the node statements of its nodes,
-- and finally all edges. Edges may refer to nodes in any cluster, so
-- listing them last keeps each cluster a plain list of nodes.
render :: DotOptions -> [(NodeId, Node)] -> Text
render opts ns =
  Text.unlines $
    ["digraph hsgrad {", "  rankdir=" <> rankDir <> ";"]
      ++ concatMap (indent . nodeStatements opts) loose
      ++ concat (zipWith (cluster opts) [0 ..] (Map.toList clustered))
      ++ concatMap (indent . edgeStatements opts valueIdOf) ns
      ++ ["}"]
  where
    -- Where the edges out of each node start. Every argument of a node is
    -- in the set, so the lookup cannot fail.
    valueIds = IntMap.fromList [(nodeIndex i, valueId opts i n) | (i, n) <- ns]
    valueIdOf a = valueIds IntMap.! nodeIndex a
    rankDir = case dotRankDir opts of
      LeftToRight -> "LR"
      TopToBottom -> "TB"
    (loose, clustered)
      | dotClusters opts = clusters ns
      | otherwise = (ns, Map.empty)

-- | Indent statements by one level (two spaces).
indent :: [Text] -> [Text]
indent = map ("  " <>)

-- ---------------------------------------------------------------------------
-- Clusters

-- | Split nodes into those outside any cluster and the clusters, keyed by
-- span. Both keep ascending id order.
--
-- A cluster is only made for a span that at least two nodes share. A box
-- around a single node would add a frame and a caption without grouping
-- anything.
clusters :: [(NodeId, Node)] -> ([(NodeId, Node)], Map SrcSpan [(NodeId, Node)])
clusters ns = (filter (not . inCluster) ns, groups)
  where
    bySpan =
      Map.fromListWith
        (flip (++))
        [(sp, [(i, n)]) | (i, n) <- ns, Just sp <- [provSpan (nodeProv n)]]
    groups = Map.filter ((>= 2) . length) bySpan
    inCluster (_, n) = maybe False (`Map.member` groups) (provSpan (nodeProv n))

-- | One cluster as a DOT subgraph. Graphviz draws a subgraph as a framed
-- group only if its name starts with @cluster@.
cluster :: DotOptions -> Int -> (SrcSpan, [(NodeId, Node)]) -> [Text]
cluster opts k (sp, ns) =
  indent $
    ["subgraph cluster_" <> showT k <> " {"]
      ++ indent
        [ "label=" <> quoted (escapeString (showSpan sp)) <> ";"
        , "style=rounded;"
        , "color=gray;"
        ]
      ++ concatMap (indent . nodeStatements opts) ns
      ++ ["}"]

-- | A span as @file:line:col-line:col@, or @file:line:col-col@ when it
-- starts and ends on the same line. The end column is exclusive, as in
-- 'SrcSpan'.
showSpan :: SrcSpan -> Text
showSpan (SrcSpan file (Pos l1 c1) (Pos l2 c2))
  | l1 == l2 = Text.pack file <> ":" <> showT l1 <> ":" <> showT c1 <> "-" <> showT c2
  | otherwise =
      Text.pack file <> ":" <> showT l1 <> ":" <> showT c1 <> "-" <> showT l2 <> ":" <> showT c2

-- ---------------------------------------------------------------------------
-- Nodes

-- | The DOT id of a node's box: @n7@.
boxId :: NodeId -> Text
boxId i = "n" <> showT (nodeIndex i)

-- | The DOT id of a node's op node: @o7@.
opId :: NodeId -> Text
opId i = "o" <> showT (nodeIndex i)

-- | The fields of a node's box, from left to right: its captions, its
-- name, its value and its gradient, each only if it has one. No fields
-- means no box.
boxFields :: DotOptions -> NodeId -> Node -> [Text]
boxFields opts i n =
  concat
    [ [Text.intercalate ", " cs | let cs = captions opts i, not (null cs)]
    , maybe [] pure (nodeName n)
    , ["data " <> showNumber v | Just vals <- [dotValues opts], Just v <- [lookupValue i vals]]
    , ["grad " <> showNumber d | Just grads <- [dotGrads opts], Just d <- [lookupGrad i grads]]
    ]

-- | Does the node have a box?
hasBox :: DotOptions -> NodeId -> Node -> Bool
hasBox opts i n = not (null (boxFields opts i n))

-- | The DOT id that stands for a node's value, where edges to its users
-- start: the box if there is one, otherwise the op node. A leaf always has
-- a name, so it always has a box.
valueId :: DotOptions -> NodeId -> Node -> Text
valueId opts i n
  | hasBox opts i n = boxId i
  | otherwise = opId i

-- | The captions of a node's highlights, in the order given.
captions :: DotOptions -> NodeId -> [Text]
captions opts i = [c | Highlight j (Just c) <- dotHighlights opts, j == i]

-- | Is the node highlighted?
isHighlighted :: DotOptions -> NodeId -> Bool
isHighlighted opts i = any ((== i) . hlNode) (dotHighlights opts)

-- | The name shown in a node's box:
--
-- * an input or parameter: its name, which is what @--set@ and the
--   parameter file refer to;
-- * a constant: its value, prefixed with its label if it has one
--   (@f = -2@), so the number is visible even without 'dotValues';
-- * any other node: its label, if it has one (@e@, @h1[3]@).
nodeName :: Node -> Maybe Text
nodeName n = case nodeOp n of
  Leaf (Input (Name x)) -> Just x
  Leaf (Param (Name x)) -> Just x
  Leaf (Const c) -> Just (maybe "" (<> " = ") lbl <> showConst c)
  _ -> lbl
  where
    lbl = provLabel (nodeProv n)

-- | The DOT statements declaring a node: its box and its op node, as far
-- as it has them.
--
-- A highlighted node is drawn filled and with a thicker border. The
-- emphasis goes on its box, or on its op node if it has no box.
nodeStatements :: DotOptions -> (NodeId, Node) -> [Text]
nodeStatements opts (i, n) = box ++ op
  where
    fields = boxFields opts i n
    boxed = not (null fields)
    emphasis
      | isHighlighted opts i = ", style=filled, fillcolor=lightblue, penwidth=2"
      | otherwise = ""
    box =
      [ boxId i <> " [shape=record, label=" <> quoted (recordLabel fields) <> emphasis <> "];"
      | boxed
      ]
    op = case opLabel (nodeOp n) of
      Nothing -> []
      Just l ->
        [opId i <> " [label=" <> quoted (escapeString l) <> (if boxed then "" else emphasis) <> "];"]
    -- Graphviz lays out the top-level fields of a record across the
    -- rank direction: stacked vertically when the graph flows left to
    -- right. Braces flip that, so in both directions the fields end up
    -- side by side, as in micrograd.
    recordLabel fs = case dotRankDir opts of
      LeftToRight -> "{ " <> body <> " }"
      TopToBottom -> body
      where
        body = Text.intercalate " | " (map escapeRecordField fs)

-- | The text of an op node, or 'Nothing' for a leaf, which has none.
opLabel :: Op -> Maybe Text
opLabel op = case op of
  Leaf _ -> Nothing
  Un u -> Just $ case u of
    Neg -> "neg"
    PowC c -> "^" <> showExponent c
    Exp -> "exp"
    Log -> "log"
    Tanh -> "tanh"
    Relu -> "relu"
    Step -> "step"
  Bin b -> Just $ case b of
    Add -> "+"
    Sub -> "-"
    Mul -> "*"
    Div -> "/"

-- ---------------------------------------------------------------------------
-- Edges

-- | The edges into a node: one from each argument's value into the op
-- node, in argument order, and one from the op node into the box. A leaf
-- has none. The function argument gives the DOT id where an argument's
-- value is drawn ('valueId').
--
-- For @-@ and @/@ the order of the arguments matters, so their edges are
-- labelled @x@ and @y@ (the names @docs/03-core-op.md@ uses for the first
-- and second argument). For the other binary ops, @+@ and @*@, it does
-- not, and labels would only add clutter.
--
-- An argument used twice, as in @a + a@, gives two edges.
edgeStatements :: DotOptions -> (NodeId -> Text) -> (NodeId, Node) -> [Text]
edgeStatements opts valueIdOf (i, n) = case opLabel (nodeOp n) of
  Nothing -> []
  Just _ -> zipWith argEdge argNames (nodeArgs n) ++ outEdge
  where
    argEdge name a =
      valueIdOf a <> " -> " <> opId i <> maybe "" (\l -> " [headlabel=" <> quoted l <> "]") name <> ";"
    outEdge = [opId i <> " -> " <> boxId i <> ";" | hasBox opts i n]
    argNames = case nodeOp n of
      Bin Sub -> [Just "x", Just "y"]
      Bin Div -> [Just "x", Just "y"]
      _ -> repeat Nothing

-- ---------------------------------------------------------------------------
-- Text

-- | Escape text for a double-quoted DOT string, such as the label of an op
-- node or a cluster.
--
-- Two layers read the string. The DOT parser only treats @\\"@ specially.
-- Graphviz then interprets backslash escapes in labels (@\\n@ is a line
-- break, @\\N@ the node's name, …), so a backslash meant literally must be
-- doubled. Line breaks become @\\n@; other control characters become
-- spaces. Everything else, including non-ASCII text such as @∂L/∂a@, is
-- passed through: DOT files are UTF-8.
escapeString :: Text -> Text
escapeString = Text.concatMap escapeChar

-- | Escape text for one field of a @record@ label. On top of
-- 'escapeString', the characters that structure a record (@{@, @}@, @|@,
-- @<@, @>@) must be backslash-escaped to stand for themselves.
escapeRecordField :: Text -> Text
escapeRecordField = Text.concatMap esc
  where
    esc c
      | c `elem` ("{}|<>" :: String) = Text.pack ['\\', c]
      | otherwise = escapeChar c

-- | 'escapeString' for one character.
escapeChar :: Char -> Text
escapeChar c = case c of
  '"' -> "\\\""
  '\\' -> "\\\\"
  '\n' -> "\\n"
  _
    | isControl c -> " "
    | otherwise -> Text.singleton c

-- | Put text in double quotes. The text must already be escaped.
quoted :: Text -> Text
quoted t = "\"" <> t <> "\""

-- | A value or gradient, like micrograd's @%.4f@: four decimals. Very small
-- and very large magnitudes use scientific notation instead, so that
-- @1e-7@ does not show as @0.0000@ and @1e200@ not as 201 digits.
showNumber :: Double -> Text
showNumber x
  | x == 0 = "0.0000" -- also for -0.0
  | a >= 1e-3 && a < 1e6 = Text.pack (showFFloat (Just 4) x "")
  | otherwise = Text.pack (showEFloat (Just 4) x "")
  where
    a = abs x

-- | A constant as written in a source file: @-2@ rather than @-2.0@ for
-- whole numbers, Haskell's 'show' otherwise (@0.5@, @1.0e-4@).
showConst :: Double -> Text
showConst c
  | c == fromInteger r && abs c < 1e15 = showT r
  | otherwise = showT c
  where
    r = round c :: Integer

-- | The exponent of a 'PowC' node: @2@, @-1@, @(1/2)@.
showExponent :: Rational -> Text
showExponent c
  | denominator c == 1 = showT (numerator c)
  | otherwise = "(" <> showT (numerator c) <> "/" <> showT (denominator c) <> ")"

-- | 'show' as 'Text'.
showT :: (Show a) => a -> Text
showT = Text.pack . show
