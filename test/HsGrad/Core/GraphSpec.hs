{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Graph".
--
-- * Building: sharing, hash-consing, ids in creation order, extending an
--   existing graph.
-- * Errors: arity, unknown nodes, non-finite constants.
-- * Provenance: the current span, nested spans, which provenance wins when
--   nodes merge, labels.
-- * Reachability on small hand-built graphs.
-- * Properties over randomly built graphs: the builder's invariants hold,
--   and 'reachable' agrees with a simple reference definition.
module HsGrad.Core.GraphSpec
  ( spec
  ) where

import Control.Monad (foldM, replicateM_)
import Data.Set qualified as Set
import HsGrad.Core.Graph
import HsGrad.Core.Op
import Test.Hspec (Expectation, Spec, describe, expectationFailure, it, shouldBe)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck
  ( Arbitrary (..)
  , Gen
  , choose
  , conjoin
  , counterexample
  , elements
  , forAll
  , frequency
  , listOf
  , shrinkList
  , sized
  , vectorOf
  , (.&&.)
  , (===)
  )

spec :: Spec
spec = do
  describe "building" buildingSpec
  describe "errors" errorSpec
  describe "provenance" provenanceSpec
  describe "reachable" reachableSpec
  describe "random graphs" propertySpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Run a builder that is expected to succeed.
built :: Builder a -> (a, Graph)
built b = case build b of
  Right r -> r
  Left err -> error ("unexpected build error: " <> show err)

-- | Check that a builder fails, and pass the error to a check.
failsWith :: Builder a -> (GraphError -> Expectation) -> Expectation
failsWith b check = case build b of
  Left err -> check err
  Right _ -> expectationFailure "expected the build to fail, but it succeeded"

-- | The op and arguments of a node, i.e. its hash-consing key.
keyOf :: Graph -> NodeId -> (Op, [NodeId])
keyOf g i = let n = nodeAt g i in (nodeOp n, nodeArgs n)

-- | A span on line 1 of @m.hsg@, from column @a@ to column @b@.
span1 :: Int -> Int -> SrcSpan
span1 a b = SrcSpan "m.hsg" (Pos 1 a) (Pos 1 b)

-- ---------------------------------------------------------------------------
-- Building

buildingSpec :: Spec
buildingSpec = do
  it "starts from an empty graph" $ do
    nodeCount emptyGraph `shouldBe` 0
    nodes emptyGraph `shouldBe` []

  it "refers to a shared node by the same id (a DAG, not a tree)" $ do
    let ((x, sq), g) = built $ do
          x' <- input "x"
          sq' <- binary Mul x' x'
          pure (x', sq')
    nodeArgs (nodeAt g sq) `shouldBe` [x, x]
    nodeCount g `shouldBe` 2

  it "merges structurally identical nodes (hash-consing)" $ do
    let ((a, b), g) = built $ do
          x <- input "x"
          y <- param "w"
          a' <- binary Add x y
          b' <- binary Add x y
          pure (a', b')
    a `shouldBe` b
    nodeCount g `shouldBe` 3

  it "merges equal leaves and equal constants" $ do
    let ((xs, cs), g) = built $ do
          xs' <- sequence [input "x", input "x"]
          cs' <- sequence [constant 2, constant 2]
          pure (xs', cs')
    Set.size (Set.fromList xs) `shouldBe` 1
    Set.size (Set.fromList cs) `shouldBe` 1
    nodeCount g `shouldBe` 2

  it "merges whole expressions bottom-up" $ do
    -- tanh(x * y) built twice from scratch: every level merges.
    let expr = do
          x <- input "x"
          y <- input "y"
          unary Tanh =<< binary Mul x y
        ((t1, t2), g) = built ((,) <$> expr <*> expr)
    t1 `shouldBe` t2
    nodeCount g `shouldBe` 4

  it "keeps nodes apart that differ in op, arguments or argument order" $ do
    let (ids, g) = built $ do
          x <- input "x"
          y <- input "y"
          sequence
            [ binary Add x y
            , binary Add y x -- no commutativity: that is the simplifier's job
            , binary Mul x y
            , binary Add x x
            , unary (PowC 2) x
            , unary (PowC 3) x
            , input "w"
            , param "w" -- same name, different kind of leaf
            , constant 1
            , constant 2
            ]
    Set.size (Set.fromList ids) `shouldBe` length ids
    nodeCount g `shouldBe` 2 + length ids

  it "merges 0.0 and -0.0, as Double's (==) does" $ do
    let ((a, b), _) = built ((,) <$> constant 0 <*> constant (-0))
    a `shouldBe` b

  it "hands out ids 0, 1, 2, ... in creation order" $ do
    let (ids, g) = built $ do
          x <- input "x"
          y <- constant 3
          z <- binary Mul x y
          w <- unary Neg z
          pure [x, y, z, w]
    map nodeIndex ids `shouldBe` [0, 1, 2, 3]
    nodeIds g `shouldBe` ids
    map fst (nodes g) `shouldBe` ids

  it "does not use up an id when a node is merged" $ do
    let (ids, _) = built $ do
          x <- input "x"
          _ <- input "x"
          y <- input "y"
          pure [x, y]
    map nodeIndex ids `shouldBe` [0, 1]

  it "stores op, arguments and provenance of each node" $ do
    let (z, g) = built $ do
          x <- input "x"
          y <- input "y"
          binary Sub x y
    nodeAt g z `shouldBe` Node (Bin Sub) (map fst (take 2 (nodes g))) noProvenance
    lookupNode z g `shouldBe` Just (nodeAt g z)

  it "finds nodes by op and arguments without building them" $ do
    let ((x, y), g) = built $ do
          x' <- input "x"
          y' <- unary Exp x'
          pure (x', y')
    findNode (Leaf (Input "x")) [] g `shouldBe` Just x
    findNode (Un Exp) [x] g `shouldBe` Just y
    findNode (Un Log) [x] g `shouldBe` Nothing

  it "extends an existing graph, keeping its nodes and ids" $ do
    let ((x, e), g1) = built $ do
          x' <- input "x"
          e' <- unary Exp x'
          pure (x', e')
        more = do
          x2' <- input "x" -- found in the old graph
          sq' <- binary Mul e e
          pure (x2', sq')
    case extend g1 more of
      Left err -> expectationFailure (show err)
      Right ((x2, sq), g2) -> do
        x2 `shouldBe` x
        nodeIndex sq `shouldBe` 2
        take 2 (nodes g2) `shouldBe` nodes g1

  it "gives read-only access to the graph built so far" $ do
    let (n, _) = built $ do
          replicateM_ 3 (input "x")
          _ <- input "y"
          nodeCount <$> currentGraph
    n `shouldBe` 2

-- ---------------------------------------------------------------------------
-- Errors

errorSpec :: Spec
errorSpec = do
  it "rejects an op with too few or too many arguments" $ do
    node (Bin Add) [] `failsWith` (`shouldBe` ArityMismatch (Bin Add) [])
    let oneArg = do
          x <- input "x"
          node (Bin Add) [x]
    oneArg `failsWith` \err -> case err of
      ArityMismatch (Bin Add) [_] -> pure ()
      other -> expectationFailure ("wrong error: " <> show other)
    let leafWithArg = do
          x <- input "x"
          node (Leaf (Const 1)) [x]
    leafWithArg `failsWith` isArityMismatch
    let unWithTwo = do
          x <- input "x"
          node (Un Tanh) [x, x]
    unWithTwo `failsWith` isArityMismatch

  it "accepts every op with exactly its arity" $ do
    let (_, g) = built $ do
          x <- input "x"
          _ <- node (Leaf (Param "w")) []
          _ <- node (Un Relu) [x]
          node (Bin Div) [x, x]
    nodeCount g `shouldBe` 4

  it "rejects an argument that is not a node of the graph" $ do
    -- An id from another graph: input "y" is node 1 there, and this graph
    -- has only node 0.
    let ((_, foreignId), _) = built ((,) <$> input "x" <*> input "y")
        useForeign = do
          x <- input "x"
          binary Add x foreignId
    useForeign `failsWith` (`shouldBe` UnknownNode foreignId)
    label "y" foreignId `failsWith` (`shouldBe` UnknownNode foreignId)

  it "rejects non-finite constants" $ do
    let isNonFinite p err = case err of
          NonFiniteConst c | p c -> pure ()
          other -> expectationFailure ("wrong error: " <> show other)
    constant (1 / 0) `failsWith` isNonFinite (> 0)
    constant (-1 / 0) `failsWith` isNonFinite (< 0)
    constant (0 / 0) `failsWith` isNonFinite isNaN
    node (Leaf (Const (0 / 0))) [] `failsWith` isNonFinite isNaN

  it "aborts the whole build at the first error" $ do
    let b = do
          _ <- input "x"
          _ <- constant (1 / 0)
          input "never built"
    b `failsWith` \err -> show err `shouldBe` "NonFiniteConst Infinity"
  where
    isArityMismatch err = case err of
      ArityMismatch _ _ -> pure ()
      other -> expectationFailure ("wrong error: " <> show other)

-- ---------------------------------------------------------------------------
-- Provenance

provenanceSpec :: Spec
provenanceSpec = do
  it "records no provenance outside withSpan" $ do
    let (x, g) = built (input "x")
    nodeProv (nodeAt g x) `shouldBe` noProvenance

  it "records the current span in every new node" $ do
    let ((x, y), g) = built $ withSpan (span1 1 6) $ do
          x' <- input "x"
          y' <- unary Exp x'
          pure (x', y')
    provSpan (nodeProv (nodeAt g x)) `shouldBe` Just (span1 1 6)
    provSpan (nodeProv (nodeAt g y)) `shouldBe` Just (span1 1 6)

  it "uses the innermost span, and restores the outer one afterwards" $ do
    let ((x, y, z, w), g) = built $ do
          x' <- input "x"
          (y', z') <- withSpan (span1 1 10) $ do
            y'' <- withSpan (span1 5 8) (unary Exp x')
            z'' <- unary Neg y''
            pure (y'', z'')
          w' <- unary Tanh z'
          pure (x', y', z', w')
        spanOf i = provSpan (nodeProv (nodeAt g i))
    spanOf x `shouldBe` Nothing
    spanOf y `shouldBe` Just (span1 5 8)
    spanOf z `shouldBe` Just (span1 1 10)
    spanOf w `shouldBe` Nothing

  it "keeps the first span when nodes merge" $ do
    let ((a, b), g) = built $ do
          x <- input "x"
          a' <- withSpan (span1 1 4) (unary Exp x)
          b' <- withSpan (span1 9 12) (unary Exp x)
          pure (a', b')
    a `shouldBe` b
    provSpan (nodeProv (nodeAt g a)) `shouldBe` Just (span1 1 4)

  it "fills in a missing span when nodes merge" $ do
    let (a, g) = built $ do
          x <- input "x"
          _ <- unary Exp x -- no span
          withSpan (span1 9 12) (unary Exp x)
    provSpan (nodeProv (nodeAt g a)) `shouldBe` Just (span1 9 12)

  it "labels a node, and keeps the first label" $ do
    let (x, g) = built $ do
          x' <- param "W1[3,0]"
          label "W1[3,0]" x'
          label "other" x'
          pure x'
    provLabel (nodeProv (nodeAt g x)) `shouldBe` Just "W1[3,0]"

  it "does not let provenance affect hash-consing" $ do
    let ((a, b), g) = built $ do
          a' <- withSpan (span1 1 2) (input "x")
          label "a" a'
          b' <- withSpan (span1 3 4) (input "x")
          pure (a', b')
    a `shouldBe` b
    nodeProv (nodeAt g a) `shouldBe` Provenance (Just (span1 1 2)) (Just "a")

  it "combines provenance field by field, the left side winning" $ do
    let s = Provenance (Just (span1 1 2)) Nothing
        l = Provenance Nothing (Just "h")
        both = Provenance (Just (span1 3 4)) (Just "z")
    s <> l `shouldBe` Provenance (Just (span1 1 2)) (Just "h")
    s <> both `shouldBe` Provenance (Just (span1 1 2)) (Just "z")
    both <> s `shouldBe` both
    mempty <> s `shouldBe` s

-- ---------------------------------------------------------------------------
-- Reachability

reachableSpec :: Spec
reachableSpec = do
  -- x, y; a = x * y; b = x + y; out = a - b; dead = exp y
  let ((x, y, a, b, out, dead), g) = built $ do
        x' <- input "x"
        y' <- input "y"
        a' <- binary Mul x' y'
        b' <- binary Add x' y'
        out' <- binary Sub a' b'
        dead' <- unary Exp y'
        pure (x', y', a', b', out', dead')

  it "contains the outputs and everything they depend on" $
    reachable g [out] `shouldBe` Set.fromList [x, y, a, b, out]

  it "leaves out nodes the outputs do not depend on" $ do
    Set.member dead (reachable g [out]) `shouldBe` False
    reachable g [a] `shouldBe` Set.fromList [x, y, a]

  it "takes the union over several outputs" $
    reachable g [a, dead] `shouldBe` Set.fromList [x, y, a, dead]

  it "is empty for no outputs" $
    reachable g [] `shouldBe` Set.empty

  it "is just the leaf for a leaf" $
    reachable g [y] `shouldBe` Set.singleton y

  it "ignores ids that are not nodes of the graph" $ do
    -- Node 6 of a bigger graph; g only has nodes 0 .. 5.
    let (bigIds, _) = built (mapM input ["p", "q", "r", "s", "t", "u", "v"])
    reachable g [last bigIds] `shouldBe` Set.empty

-- ---------------------------------------------------------------------------
-- Random graphs

-- | A recipe for a random graph: a list of steps, each creating (or
-- finding) one node. Arguments are indices into the ids produced by the
-- steps before, taken modulo their number, so every list of steps is a
-- valid recipe, and so is every sublist (which keeps shrinking simple).
newtype Recipe = Recipe [Step]
  deriving (Show)

data Step
  = SLeaf Leaf
  | SUn UnOp Int
  | SBin BinOp Int Int
  deriving (Show)

instance Arbitrary Recipe where
  arbitrary = sized $ \n -> do
    len <- choose (1, max 1 (2 * n))
    Recipe <$> vectorOf len genStep
  shrink (Recipe steps) = map Recipe (shrinkList (const []) steps)

-- | Few distinct leaves and ops, so that random graphs contain many
-- structurally identical nodes for hash-consing to merge.
genStep :: Gen Step
genStep =
  frequency
    [ (2, SLeaf <$> elements leaves)
    , (3, SUn <$> elements unOps <*> genIndex)
    , (5, SBin <$> elements [minBound .. maxBound] <*> genIndex <*> genIndex)
    ]
  where
    leaves = [Input "x", Input "y", Param "w", Const 0, Const 1, Const (-2.5)]
    unOps = [Neg, Exp, Tanh, Relu, PowC 2, PowC (1 / 2)]
    genIndex = choose (0, 1000)

-- | Run a recipe. Returns the id produced by every step, in step order.
runRecipe :: [Step] -> Builder [NodeId]
runRecipe = fmap reverse . foldM step []
  where
    -- @done@ holds the ids produced so far, most recent first. A step that
    -- needs an argument when there is none yet becomes a leaf.
    step done s = (: done) <$> case s of
      SLeaf l -> node (Leaf l) []
      SUn op i
        | null done -> input "x"
        | otherwise -> unary op (pick done i)
      SBin op i j
        | null done -> input "y"
        | otherwise -> binary op (pick done i) (pick done j)
    pick done i = done !! (i `mod` length done)

-- | Build a recipe; random recipes never fail.
builtRecipe :: [Step] -> ([NodeId], Graph)
builtRecipe = built . runRecipe

-- | The reference definition of reachability: start from the outputs and
-- keep adding the arguments of everything in the set until nothing
-- changes. Obviously correct, and slow, which is fine for small graphs.
reachableRef :: Graph -> [NodeId] -> Set.Set NodeId
reachableRef g outs = fixpoint (Set.fromList outs)
  where
    fixpoint s =
      let s' = Set.union s (Set.fromList (concatMap (nodeArgs . nodeAt g) (Set.toList s)))
       in if s' == s then s else fixpoint s'

propertySpec :: Spec
propertySpec = do
  prop "ids are 0 .. n-1 in ascending order" $ \(Recipe steps) ->
    let (_, g) = builtRecipe steps
     in map nodeIndex (nodeIds g) === [0 .. nodeCount g - 1]

  prop "every argument has a smaller id than its node (topological order)" $
    \(Recipe steps) ->
      let (_, g) = builtRecipe steps
       in conjoin
            [ counterexample (show (i, n)) (all (< i) (nodeArgs n))
            | (i, n) <- nodes g
            ]

  prop "every node has as many arguments as its op's arity" $
    \(Recipe steps) ->
      let (_, g) = builtRecipe steps
       in conjoin [length (nodeArgs n) === arity (nodeOp n) | (_, n) <- nodes g]

  prop "no two nodes have the same op and arguments" $ \(Recipe steps) ->
    let (_, g) = builtRecipe steps
     in Set.size (Set.fromList (map (keyOf g) (nodeIds g))) === nodeCount g

  prop "findNode finds every node under its own key" $ \(Recipe steps) ->
    let (_, g) = builtRecipe steps
     in conjoin
          [ uncurry findNode (keyOf g i) g === Just i
          | i <- nodeIds g
          ]

  prop "every id a step returns is a node of the graph" $
    \(Recipe steps) ->
      let (ids, g) = builtRecipe steps
       in conjoin [counterexample (show i) (lookupNode i g /= Nothing) | i <- ids]

  prop "running a recipe twice creates no new nodes and returns the same ids" $
    \(Recipe steps) ->
      let ((first, second), g) = built ((,) <$> runRecipe steps <*> runRecipe steps)
          (_, g1) = builtRecipe steps
       in first === second .&&. nodeCount g === nodeCount g1

  prop "building is deterministic" $ \(Recipe steps) ->
    builtRecipe steps === builtRecipe steps

  prop "extend keeps the old graph as a prefix" $
    \(Recipe steps1) (Recipe steps2) ->
      let (_, g1) = builtRecipe steps1
       in case extend g1 (runRecipe steps2) of
            Left err -> counterexample (show err) False
            Right (_, g2) ->
              take (nodeCount g1) (nodes g2) === nodes g1

  prop "reachable agrees with the reference definition" $
    \(Recipe steps) -> forAll (listOf (choose (0, 1000))) $ \picks ->
      let (ids, g) = builtRecipe steps
          outs = [ids !! (p `mod` length ids) | p <- picks]
       in reachable g outs === reachableRef g outs

  prop "reachable is closed under arguments" $
    \(Recipe steps) (k :: Int) ->
      let (ids, g) = builtRecipe steps
          r = reachable g [ids !! (k `mod` length ids)]
       in conjoin
            [ counterexample (show i) (all (`Set.member` r) (nodeArgs (nodeAt g i)))
            | i <- Set.toList r
            ]
