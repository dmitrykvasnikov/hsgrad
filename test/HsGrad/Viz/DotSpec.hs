{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Viz.Dot".
--
-- * Golden tests: the exact DOT text for @intro.hsg@ (micrograd's picture)
--   and for @neuron.hsg@ with clusters.
-- * Boxes, op nodes and edges on small graphs: what gets a box, op labels,
--   argument order for @-@ and @/@, values and gradients.
-- * Highlights and captions, as a gradient graph (step 7) will use them.
-- * Escaping of awkward names and file names.
-- * Clusters, with and without provenance.
-- * Properties over random graphs: every DOT node is declared once, every
--   edge connects declared nodes, and the counts match the graph.
-- * If Graphviz is installed, it accepts every example without warnings
--   and shows awkward labels literally. Without Graphviz these tests are
--   reported as pending, not failed.
module HsGrad.Viz.DotSpec
  ( spec
  ) where

import Control.Monad (forM_)
import Data.Char (isDigit)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Backprop
import HsGrad.Core.Eval
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Test.Examples
import HsGrad.Test.Graphs (Recipe (..), built, builtRecipe, genEnv, withGraph)
import HsGrad.Test.Graphviz (toSvg, withDot)
import HsGrad.Viz.Dot
import Test.Hspec
  ( Spec
  , describe
  , it
  , shouldBe
  , shouldContain
  , shouldNotContain
  )
import Test.Hspec.QuickCheck (modifyMaxSize, prop)
import Test.QuickCheck
  ( Arbitrary (..)
  , Property
  , conjoin
  , counterexample
  , discard
  , forAll
  , generate
  , resize
  , vectorOf
  , (===)
  )

spec :: Spec
spec = do
  describe "golden" goldenSpec
  describe "boxes, op nodes and edges" drawingSpec
  describe "highlights" highlightSpec
  describe "escaping" escapingSpec
  describe "clusters" clusterSpec
  describe "random graphs" propertySpec
  describe "Graphviz" graphvizSpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Forward and backward, expecting success.
differentiated :: Graph -> Env -> NodeId -> (Values, Grads)
differentiated g env out = case gradients g env out of
  Right r -> r
  Left err -> error ("unexpected error: " <> show err)

-- | The lines of DOT text, without their indentation.
stmts :: Text -> [Text]
stmts = map Text.strip . Text.lines

-- | @intro.hsg@ with values, gradients and the loss highlighted: the
-- picture micrograd's @draw_dot(L)@ draws.
introFull :: Text
introFull = renderDotFrom opts g [introL ns]
  where
    (ns, g) = intro
    (vals, grads) = differentiated g introEnv (introL ns)
    opts =
      defaultDotOptions
        { dotValues = Just vals
        , dotGrads = Just grads
        , dotHighlights = [highlight (introL ns)]
        }

-- | @neuron.hsg@, names only, clustered, with @y@ highlighted.
neuronClustered :: Text
neuronClustered = renderDotFrom opts g [neuronY ns]
  where
    (ns, g) = neuron
    opts = defaultDotOptions {dotClusters = True, dotHighlights = [highlight (neuronY ns)]}

-- | A sketch of the gradient graph step 7 will build for @intro.hsg@:
-- gradient nodes added to the forward graph, sharing its leaves. The
-- outputs are the gradient nodes, captioned with what they are. @∂L/∂c@
-- and @∂L/∂e@ are the same node.
introGradient :: Text
introGradient = renderDotFrom opts g outs
  where
    (ns, fwd) = intro
    ((ga, gb, ge), g) = case extend fwd gradientNodes of
      Right r -> r
      Left err -> error (show err)
    gradientNodes = do
      one <- constant 1
      gd <- binary Mul one (introF ns) -- ∂L/∂d = 1 · f
      ge' <- binary Mul gd one -- ∂L/∂e = ∂L/∂d · 1 = ∂L/∂c
      ga' <- binary Mul ge' (introB ns) -- ∂L/∂a = ∂L/∂e · b
      gb' <- binary Mul ge' (introA ns) -- ∂L/∂b = ∂L/∂e · a
      pure (ga', gb', ge')
    outs = [ga, gb, ge]
    opts =
      defaultDotOptions
        { dotHighlights =
            [ highlightAs "∂L/∂a" ga
            , highlightAs "∂L/∂b" gb
            , highlightAs "∂L/∂c" ge
            , highlightAs "∂L/∂e" ge
            ]
        }

-- | A leaf whose name uses every character that needs escaping somewhere.
awkwardName :: Name
awkwardName = Name "say \"hi\" {x|y} <z> a\\b"

-- | A graph with the awkward leaf, a two-line label, and a cluster whose
-- file name needs escaping.
awkward :: Text
awkward = renderDot defaultDotOptions {dotClusters = True} g
  where
    (_, g) = built $ withSpan (SrcSpan "my \"dir\"\\x.hsg" (Pos 1 1) (Pos 2 5)) $ do
      x <- input awkwardName
      y <- unary Exp x
      label "two\nlines" y

-- ---------------------------------------------------------------------------
-- Golden tests

goldenSpec :: Spec
goldenSpec = do
  it "draws intro.hsg with values and gradients, like micrograd's draw_dot" $
    introFull
      `shouldBe` Text.unlines
        [ "digraph hsgrad {"
        , "  rankdir=LR;"
        , "  n0 [shape=record, label=\"{ a | data 2.0000 | grad 6.0000 }\"];"
        , "  n1 [shape=record, label=\"{ b | data -3.0000 | grad -4.0000 }\"];"
        , "  n2 [shape=record, label=\"{ c | data 10.0000 | grad -2.0000 }\"];"
        , "  n3 [shape=record, label=\"{ e | data -6.0000 | grad -2.0000 }\"];"
        , "  o3 [label=\"*\"];"
        , "  n4 [shape=record, label=\"{ d | data 4.0000 | grad -2.0000 }\"];"
        , "  o4 [label=\"+\"];"
        , "  n5 [shape=record, label=\"{ -2 | data -2.0000 | grad 4.0000 }\"];"
        , "  n6 [shape=record, label=\"{ L | data -8.0000 | grad 1.0000 }\", style=filled, fillcolor=lightblue, penwidth=2];"
        , "  o6 [label=\"*\"];"
        , "  n0 -> o3;"
        , "  n1 -> o3;"
        , "  o3 -> n3;"
        , "  n3 -> o4;"
        , "  n2 -> o4;"
        , "  o4 -> n4;"
        , "  n4 -> o6;"
        , "  n5 -> o6;"
        , "  o6 -> n6;"
        , "}"
        ]

  it "draws intro.hsg with names only" $ do
    let (ns, g) = intro
    renderDotFrom defaultDotOptions g [introL ns]
      `shouldBe` Text.unlines
        [ "digraph hsgrad {"
        , "  rankdir=LR;"
        , "  n0 [shape=record, label=\"{ a }\"];"
        , "  n1 [shape=record, label=\"{ b }\"];"
        , "  n2 [shape=record, label=\"{ c }\"];"
        , "  n3 [shape=record, label=\"{ e }\"];"
        , "  o3 [label=\"*\"];"
        , "  n4 [shape=record, label=\"{ d }\"];"
        , "  o4 [label=\"+\"];"
        , "  n5 [shape=record, label=\"{ -2 }\"];"
        , "  n6 [shape=record, label=\"{ L }\"];"
        , "  o6 [label=\"*\"];"
        , "  n0 -> o3;"
        , "  n1 -> o3;"
        , "  o3 -> n3;"
        , "  n3 -> o4;"
        , "  n2 -> o4;"
        , "  o4 -> n4;"
        , "  n4 -> o6;"
        , "  n5 -> o6;"
        , "  o6 -> n6;"
        , "}"
        ]

  it "draws neuron.hsg with a cluster per unrolled vector and per dot product" $
    neuronClustered
      `shouldBe` Text.unlines
        [ "digraph hsgrad {"
        , "  rankdir=LR;"
        , "  n6 [shape=record, label=\"{ b }\"];"
        , "  o12 [label=\"+\"];"
        , "  n13 [shape=record, label=\"{ y }\", style=filled, fillcolor=lightblue, penwidth=2];"
        , "  o13 [label=\"tanh\"];"
        , "  subgraph cluster_0 {"
        , "    label=\"neuron.hsg:1:1-35\";"
        , "    style=rounded;"
        , "    color=gray;"
        , "    n0 [shape=record, label=\"{ x[0] }\"];"
        , "    n1 [shape=record, label=\"{ x[1] }\"];"
        , "    n2 [shape=record, label=\"{ x[2] }\"];"
        , "  }"
        , "  subgraph cluster_1 {"
        , "    label=\"neuron.hsg:2:1-33\";"
        , "    style=rounded;"
        , "    color=gray;"
        , "    n3 [shape=record, label=\"{ w[0] }\"];"
        , "    n4 [shape=record, label=\"{ w[1] }\"];"
        , "    n5 [shape=record, label=\"{ w[2] }\"];"
        , "  }"
        , "  subgraph cluster_2 {"
        , "    label=\"neuron.hsg:6:8-17\";"
        , "    style=rounded;"
        , "    color=gray;"
        , "    o7 [label=\"*\"];"
        , "    o8 [label=\"*\"];"
        , "    o9 [label=\"*\"];"
        , "    o10 [label=\"+\"];"
        , "    o11 [label=\"+\"];"
        , "  }"
        , "  n3 -> o7;"
        , "  n0 -> o7;"
        , "  n4 -> o8;"
        , "  n1 -> o8;"
        , "  n5 -> o9;"
        , "  n2 -> o9;"
        , "  o7 -> o10;"
        , "  o8 -> o10;"
        , "  o10 -> o11;"
        , "  o9 -> o11;"
        , "  o11 -> o12;"
        , "  n6 -> o12;"
        , "  o12 -> o13;"
        , "  o13 -> n13;"
        , "}"
        ]

-- ---------------------------------------------------------------------------
-- Boxes, op nodes and edges

drawingSpec :: Spec
drawingSpec = do
  it "draws an unnamed node as just its op node, unless it has a value" $ do
    let (t, g) = built $ do
          x <- input "x"
          y <- input "y"
          s <- binary Add x y
          unary Tanh s
        plain = stmts (renderDotFrom defaultDotOptions g [t])
        vals = case evaluate g (bindInput "x" 1 (bindInput "y" 2 emptyEnv)) [t] of
          Right v -> v
          Left err -> error (show err)
        withValues = stmts (renderDotFrom defaultDotOptions {dotValues = Just vals} g [t])
    -- Without values: leaves have boxes, the two ops do not.
    filter ("n" `Text.isPrefixOf`) plain
      `shouldBe` [ "n0 [shape=record, label=\"{ x }\"];"
                 , "n1 [shape=record, label=\"{ y }\"];"
                 , "n0 -> o2;"
                 , "n1 -> o2;"
                 ]
    plain `shouldContain` ["o2 -> o3;"]
    -- With values, every node has something to show.
    withValues
      `shouldContain` [ "n2 [shape=record, label=\"{ data 3.0000 }\"];"
                      , "o2 [label=\"+\"];"
                      ]
    withValues `shouldContain` ["o2 -> n2;", "n2 -> o3;", "o3 -> n3;"]

  it "names constants by their value, labelled ones as label = value" $ do
    let (_, g) = built $ do
          h <- constant 0.5
          f <- constant (-2)
          label "f" f
          s <- constant 1.0e-4
          binary Add h =<< binary Add f s
        out = stmts (renderDot defaultDotOptions g)
    out
      `shouldContain` [ "n0 [shape=record, label=\"{ 0.5 }\"];"
                      , "n1 [shape=record, label=\"{ f = -2 }\"];"
                      , "n2 [shape=record, label=\"{ 1.0e-4 }\"];"
                      ]

  it "labels every op" $ do
    let unOps = [Neg, PowC 2, PowC (-1), PowC (1 / 2), PowC (-1 / 2), Exp, Log, Tanh, Relu, Step]
        (_, g) = built $ do
          x <- input "x"
          mapM_ (`unary` x) unOps
          mapM_ (\op -> binary op x x) [minBound .. maxBound]
        out = stmts (renderDot defaultDotOptions g)
        expected =
          ["neg", "^2", "^-1", "^(1/2)", "^(-1/2)", "exp", "log", "tanh", "relu", "step", "+", "-", "*", "/"]
    out
      `shouldContain` [ "o" <> Text.pack (show i) <> " [label=\"" <> l <> "\"];"
                      | (i, l) <- zip [1 :: Int ..] expected
                      ]

  it "labels the arguments of - and / as x and y" $ do
    let ((_, _), g) = built $ do
          a <- input "a"
          b <- input "b"
          d <- binary Sub a b
          q <- binary Div a a
          pure (d, q)
        out = stmts (renderDot defaultDotOptions g)
    out `shouldContain` ["n0 -> o2 [headlabel=\"x\"];", "n1 -> o2 [headlabel=\"y\"];"]
    out `shouldContain` ["n0 -> o3 [headlabel=\"x\"];", "n0 -> o3 [headlabel=\"y\"];"]

  it "draws a + a with two edges" $ do
    let (_, g) = built $ do
          a <- input "a"
          binary Add a a
    filter ("->" `Text.isInfixOf`) (stmts (renderDot defaultDotOptions g))
      `shouldBe` ["n0 -> o1;", "n0 -> o1;"]

  it "draws only what the outputs depend on with renderDotFrom" $ do
    let ((l, m), g) = built $ do
          a <- input "a"
          b <- input "b"
          l' <- unary Exp a
          m' <- unary Log b
          pure (l', m')
        from = renderDotFrom defaultDotOptions g [l]
        whole = renderDot defaultDotOptions g
    stmts from `shouldNotContain` ["n1 [shape=record, label=\"{ b }\"];"]
    stmts whole `shouldContain` ["n1 [shape=record, label=\"{ b }\"];"]
    renderDotFrom defaultDotOptions g [l, m] `shouldBe` whole

  it "shows no gradient for a node the output does not depend on" $ do
    let ((l, m), g) = built $ do
          a <- input "a"
          l' <- unary Exp a
          label "l" l'
          m' <- unary Neg l'
          label "m" m'
          pure (l', m')
        s = schedule g [m]
        (vals, grads) = case forwardBackward s (bindInput "a" 0 emptyEnv) l of
          Right r -> r
          Left err -> error (show err)
        out = stmts (renderDotFrom defaultDotOptions {dotValues = Just vals, dotGrads = Just grads} g [m])
    out
      `shouldContain` [ "n1 [shape=record, label=\"{ l | data 1.0000 | grad 1.0000 }\"];"
                      ]
    out `shouldContain` ["n2 [shape=record, label=\"{ m | data -1.0000 }\"];"]

  it "formats numbers like micrograd, tiny and huge ones in scientific notation" $ do
    let (x, g) = built (input "x")
        dataField v =
          stmts (renderDot defaultDotOptions {dotValues = Just (single x v)} g)
        single i v = case evaluate g (bindInput "x" v emptyEnv) [i] of
          Right vals -> vals
          Left err -> error (show err)
    dataField 0 `shouldContain` ["n0 [shape=record, label=\"{ x | data 0.0000 }\"];"]
    dataField (-0.0) `shouldContain` ["n0 [shape=record, label=\"{ x | data 0.0000 }\"];"]
    dataField 0.70710678 `shouldContain` ["n0 [shape=record, label=\"{ x | data 0.7071 }\"];"]
    dataField 1.0e-7 `shouldContain` ["n0 [shape=record, label=\"{ x | data 1.0000e-7 }\"];"]
    dataField (-1.5e200) `shouldContain` ["n0 [shape=record, label=\"{ x | data -1.5000e200 }\"];"]

  it "lays out top to bottom without the braces that flip a record" $ do
    let (ns, g) = intro
        out = stmts (renderDotFrom defaultDotOptions {dotRankDir = TopToBottom} g [introL ns])
    out `shouldContain` ["rankdir=TB;", "n0 [shape=record, label=\"a\"];"]

-- ---------------------------------------------------------------------------
-- Highlights

highlightSpec :: Spec
highlightSpec = do
  it "captions gradient nodes, joining the captions of a shared node" $ do
    let out = stmts introGradient
        emphasis = ", style=filled, fillcolor=lightblue, penwidth=2];"
    out
      `shouldContain` [ "n10 [shape=record, label=\"{ ∂L/∂a }\"" <> emphasis
                      , "o10 [label=\"*\"];"
                      ]
    out `shouldContain` ["n9 [shape=record, label=\"{ ∂L/∂c, ∂L/∂e }\"" <> emphasis]
    -- The gradient nodes use the forward graph's leaves, and nothing else
    -- of it: L, d and e are not drawn.
    out `shouldContain` ["n0 [shape=record, label=\"{ a }\"];"]
    out `shouldContain` ["n5 [shape=record, label=\"{ -2 }\"];"]
    filter (\s -> any (`Text.isPrefixOf` s) ["n3 ", "n4 ", "n6 "]) out `shouldBe` []

  it "keeps the label of a captioned node, after the captions" $ do
    let (ns, g) = intro
        out = stmts (renderDotFrom defaultDotOptions {dotHighlights = [highlightAs "∂L/∂L" (introL ns)]} g [introL ns])
    out
      `shouldContain` [ "n6 [shape=record, label=\"{ ∂L/∂L | L }\", style=filled, fillcolor=lightblue, penwidth=2];"
                      ]

  it "emphasises the op node of a highlighted node that has no box" $ do
    let (t, g) = built $ do
          x <- input "x"
          unary Tanh x
        out = stmts (renderDot defaultDotOptions {dotHighlights = [highlight t]} g)
    out `shouldContain` ["o1 [label=\"tanh\", style=filled, fillcolor=lightblue, penwidth=2];"]
    filter ("n1" `Text.isPrefixOf`) out `shouldBe` []

  it "ignores highlights of nodes that are not drawn" $ do
    let ((l, m), g) = built $ do
          a <- input "a"
          l' <- unary Exp a
          m' <- unary Log a
          pure (l', m')
    renderDotFrom defaultDotOptions {dotHighlights = [highlightAs "other" m]} g [l]
      `shouldBe` renderDotFrom defaultDotOptions g [l]

-- ---------------------------------------------------------------------------
-- Escaping

escapingSpec :: Spec
escapingSpec = do
  it "escapes quotes and backslashes in strings" $ do
    escapeString "say \"hi\"" `shouldBe` "say \\\"hi\\\""
    escapeString "a\\b" `shouldBe` "a\\\\b"
    escapeString "\\N" `shouldBe` "\\\\N"

  it "turns line breaks into \\n and other control characters into spaces" $ do
    escapeString "two\nlines" `shouldBe` "two\\nlines"
    escapeString "tab\there\r" `shouldBe` "tab here "

  it "passes everything else through, including non-ASCII text" $ do
    escapeString "∂L/∂a {x|y} <z> W1[3,0] moons#3.x[1]" `shouldBe` "∂L/∂a {x|y} <z> W1[3,0] moons#3.x[1]"

  it "escapes the structure characters of records in record fields" $ do
    escapeRecordField "{x|y} <z>" `shouldBe` "\\{x\\|y\\} \\<z\\>"
    escapeRecordField "say \"hi\" a\\b" `shouldBe` "say \\\"hi\\\" a\\\\b"

  it "escapes awkward names, labels and file names in the output" $ do
    let out = stmts awkward
    out
      `shouldContain` [ "label=\"my \\\"dir\\\"\\\\x.hsg:1:1-2:5\";"
                      ]
    out
      `shouldContain` [ "n0 [shape=record, label=\"{ say \\\"hi\\\" \\{x\\|y\\} \\<z\\> a\\\\b }\"];"
                      ]
    out `shouldContain` ["n1 [shape=record, label=\"{ two\\nlines }\"];"]

-- ---------------------------------------------------------------------------
-- Clusters

clusterSpec :: Spec
clusterSpec = do
  it "makes no cluster for a span that only one node has" $ do
    -- In intro.hsg every expression is a single scalar node.
    let (ns, g) = intro
    renderDotFrom defaultDotOptions {dotClusters = True} g [introL ns]
      `shouldBe` renderDotFrom defaultDotOptions g [introL ns]

  prop "changes nothing for a graph without provenance" $
    withGraph $ \g outs ->
      renderDotFrom defaultDotOptions {dotClusters = True} g outs
        === renderDotFrom defaultDotOptions g outs

  it "keeps nodes without a span outside every cluster" $ do
    let (ns, fwd) = neuron
        (z, g) = case extend fwd (binary Mul (neuronY ns) (neuronY ns)) of
          Right r -> r
          Left err -> error (show err)
        out = Text.lines (renderDotFrom defaultDotOptions {dotClusters = True} g [z])
    -- Top-level statements are indented by two spaces, statements inside
    -- a cluster by four.
    out `shouldContain` ["  o14 [label=\"*\"];"]
    length (filter ("subgraph cluster_" `Text.isInfixOf`) out) `shouldBe` 3

  it "puts every node of one span into one cluster, also across outputs" $ do
    -- Two let-bound expressions with the same span (as two calls of one
    -- function body would have) share a cluster.
    let body = SrcSpan "f.hsg" (Pos 3 5) (Pos 3 10)
        ((p, q), g) = built $ do
          a <- input "a"
          b <- input "b"
          p' <- withSpan body (binary Mul a a)
          q' <- withSpan body (binary Mul b b)
          pure (p', q')
        out = Text.lines (renderDotFrom defaultDotOptions {dotClusters = True} g [p, q])
    out
      `shouldContain` [ "  subgraph cluster_0 {"
                      , "    label=\"f.hsg:3:5-10\";"
                      , "    style=rounded;"
                      , "    color=gray;"
                      , "    o2 [label=\"*\"];"
                      , "    o3 [label=\"*\"];"
                      , "  }"
                      ]

-- ---------------------------------------------------------------------------
-- Properties

-- | One line of DOT output, as far as the properties care.
data Stmt
  = -- | A node statement, with the node's DOT id.
    Decl Text
  | -- | An edge, from one DOT id to another.
    Edge Text Text
  | Other
  deriving (Eq, Show)

parseStmt :: Text -> Stmt
parseStmt line = case Text.words line of
  a : "->" : b : _ -> Edge a (Text.takeWhile (/= ';') b)
  a : b : _ | isDotId a && "[" `Text.isPrefixOf` b -> Decl a
  _ -> Other
  where
    isDotId t = case Text.uncons t of
      Just (c, rest) -> c `elem` ("no" :: String) && not (Text.null rest) && Text.all isDigit rest
      Nothing -> False

-- | Check the structure of a drawing of the nodes reachable from @outs@:
-- DOT nodes are declared once, edges connect declared nodes, and there are
-- @boxes@ boxes, one op node per interior node, and @edges@ edges.
structure :: Graph -> [NodeId] -> Text -> Int -> Int -> Property
structure g outs dot boxes edges =
  counterexample (Text.unpack dot) $
    conjoin
      [ counterexample "a DOT node is declared twice" $
          Set.size (Set.fromList decls) === length decls
      , counterexample "an edge refers to an undeclared node" $
          all (\(a, b) -> Set.member a declared && Set.member b declared) es === True
      , counterexample "box count" $ length (filter ("n" `Text.isPrefixOf`) decls) === boxes
      , counterexample "op node count" $ length (filter ("o" `Text.isPrefixOf`) decls) === interior
      , counterexample "edge count" $ length es === edges
      ]
  where
    parsed = map parseStmt (stmts dot)
    decls = [d | Decl d <- parsed]
    declared = Set.fromList decls
    es = [(a, b) | Edge a b <- parsed]
    scope = [nodeAt g i | i <- Set.toList (reachable g outs)]
    interior = length [n | n <- scope, not (null (nodeArgs n))]

propertySpec :: Spec
propertySpec = modifyMaxSize (const 20) $ do
  prop "names only: a box per leaf, an edge per argument" $
    withGraph $ \g outs ->
      let scope = [nodeAt g i | i <- Set.toList (reachable g outs)]
          leaves = length [n | n <- scope, null (nodeArgs n)]
          args = sum [length (nodeArgs n) | n <- scope]
       in structure g outs (renderDotFrom defaultDotOptions g outs) leaves args

  prop "with values: a box per node, and an edge from each op node to its box" $
    withGraph $ \g outs -> forAll genEnv $ \env ->
      case evaluate g env outs of
        Left _ -> discard
        Right vals ->
          let scope = [nodeAt g i | i <- Set.toList (reachable g outs)]
              interior = length [n | n <- scope, not (null (nodeArgs n))]
              args = sum [length (nodeArgs n) | n <- scope]
           in structure
                g
                outs
                (renderDotFrom defaultDotOptions {dotValues = Just vals, dotClusters = True} g outs)
                (length scope)
                (args + interior)

-- ---------------------------------------------------------------------------
-- Graphviz

-- 'withDot' and 'toSvg' are in "HsGrad.Test.Graphviz".

graphvizSpec :: Spec
graphvizSpec = do
  it "accepts every example without warnings" $
    withDot $ \exe -> do
      let (ns, g) = neuron
          (vals, grads) = differentiated g neuronEnv (neuronY ns)
          neuronFull =
            renderDotFrom
              defaultDotOptions
                { dotValues = Just vals
                , dotGrads = Just grads
                , dotClusters = True
                , dotRankDir = TopToBottom
                }
              g
              [neuronY ns]
      forM_ [introFull, neuronClustered, neuronFull, introGradient, awkward] (toSvg exe)

  it "shows awkward labels literally" $
    withDot $ \exe -> do
      svg <- toSvg exe awkward
      -- SVG is XML, so Graphviz escapes &, < > and " once more.
      svg `shouldContain` "say &quot;hi&quot; {x|y} &lt;z&gt; a\\b"
      svg `shouldContain` "my &quot;dir&quot;\\x.hsg:1:1&#45;2:5"
      svg `shouldContain` ">two</text>"
      svg `shouldContain` ">lines</text>"

  it "accepts random graphs" $
    withDot $ \exe -> do
      recipes <- generate (vectorOf 30 (resize 15 arbitrary))
      forM_ recipes $ \(Recipe steps) -> do
        let (ids, g) = builtRecipe steps
        _ <- toSvg exe (renderDotFrom defaultDotOptions {dotHighlights = map highlight ids} g ids)
        pure ()
