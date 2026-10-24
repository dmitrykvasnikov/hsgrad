{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Simplify".
--
-- * Canonical forms: like terms, powers, coefficients, the 0/1 identities,
--   no distribution over sums, the layout of the result (hand cases,
--   checked through "HsGrad.Core.Pretty").
-- * Constant folding, and the folds that are refused because the result
--   would not be finite.
-- * The domain caveats: identities that make an expression defined where
--   it was not.
-- * The milestone: the symbolic gradients of @intro.hsg@ print as @-2·b@,
--   @-2·a@, @-2@; and other simplified gradients.
-- * Provenance and the shape of the result graph.
-- * Sharing: deep shared chains stay small.
-- * Properties over random graphs: simplification preserves values (away
--   from singular points), is idempotent, gives commuted expressions the
--   same result, and leaves nothing to fold.
module HsGrad.Core.SimplifySpec
  ( spec
  ) where

import Data.Map.Strict qualified as Map
import Data.Ratio (denominator)
import Data.Set qualified as Set
import Data.Text (Text)
import HsGrad.Core.Backprop (gradAt, gradients)
import HsGrad.Core.Eval
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Core.Pretty
import HsGrad.Core.Simplify
import HsGrad.Core.Symbolic
import HsGrad.Test.Examples
import HsGrad.Test.Graphs (Recipe (..), SmoothRecipe (..), built, builtRecipe, copyWith, genEnv, sameExpressions, withGraph)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (modifyMaxSuccess, prop)
import Test.QuickCheck
  ( Property
  , conjoin
  , counterexample
  , discard
  , forAll
  , property
  , (===)
  )

spec :: Spec
spec = do
  describe "canonical forms" canonicalSpec
  describe "constant folding" foldingSpec
  describe "domain" domainSpec
  describe "gradients" gradientSpec
  describe "provenance and result graph" provenanceSpec
  describe "sharing" sharingSpec
  describe "random graphs" propertySpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Simplify, expecting success. Returns the simplified node of each given
-- node, in order.
simplified :: Graph -> [NodeId] -> ([NodeId], Graph)
simplified g outs = case simplify g outs of
  Right (m, g') -> (map (m Map.!) outs, g')
  Left err -> error ("unexpected build error: " <> show err)

-- | Simplify one node.
simplified1 :: Graph -> NodeId -> (NodeId, Graph)
simplified1 g o = case simplify g [o] of
  Right (m, g') -> (m Map.! o, g')
  Left err -> error ("unexpected build error: " <> show err)

-- | The simplified expression a builder returns, printed.
simp :: Builder NodeId -> Text
simp b = prettyNode g' o'
  where
    (o, g) = built b
    (o', g') = simplified1 g o

-- | Two inputs.
xy :: Builder (NodeId, NodeId)
xy = (,) <$> input "x" <*> input "y"

-- | Symbolic differentiation, expecting success.
symbolic :: Graph -> NodeId -> (GradNodes, Graph)
symbolic g out = case symbolicGradients g out of
  Right r -> r
  Left err -> error ("unexpected build error: " <> show err)

-- | The gradient node of a forward node, which must exist.
gradNode :: GradNodes -> NodeId -> NodeId
gradNode grads i = case lookupGradNode i grads of
  Just gi -> gi
  Nothing -> error ("no gradient node for " <> show i)

-- | The simplified gradients of some nodes, rendered together.
simplifiedGradients :: (Graph, NodeId) -> [NodeId] -> Rendered
simplifiedGradients (g, out) wrt = renderNodes defaultPrettyOptions g'' roots'
  where
    (gns, g') = symbolic g out
    (roots', g'') = simplified g' (map (gradNode gns) wrt)

-- | Forward values, expecting success.
valuesOf :: Graph -> Env -> [NodeId] -> Values
valuesOf g env outs = case evaluate g env outs of
  Right vals -> vals
  Left err -> error ("unexpected evaluation error: " <> show err)

-- | An environment with only inputs.
inputs :: [(Name, Double)] -> Env
inputs bs = emptyEnv {envInputs = Map.fromList bs}

-- | Is the value of every node the outputs depend on away from the places
-- where rounding errors are amplified without bound (division by almost
-- zero, @log@ and negative or fractional powers near zero, the kink of
-- @relu@ and @step@) and of moderate size?
wellConditioned :: Graph -> Values -> [NodeId] -> Bool
wellConditioned g vals outs = all ok (Set.toList (reachable g outs))
  where
    v = valueAt vals
    ok i =
      let n = nodeAt g i
       in abs (v i) <= 1e6 && case (nodeOp n, map v (nodeArgs n)) of
            (Bin Div, [_, y]) -> abs y >= 1e-2
            (Un (PowC c), [x]) | c < 0 || denominator c /= 1 -> abs x >= 1e-2
            (Un Log, [x]) -> x >= 1e-2
            (Un Relu, [x]) -> abs x >= 1e-6
            (Un Step, [x]) -> abs x >= 1e-6
            _ -> True

-- | The largest magnitude of a value the outputs depend on, at least 1:
-- the scale of the rounding errors that reordering the arithmetic can
-- introduce.
scaleOf :: Graph -> Values -> [NodeId] -> Double
scaleOf g vals outs = maximum (1 : [abs (valueAt vals i) | i <- Set.toList (reachable g outs)])

-- | Simplifying preserves values: wherever the original outputs are
-- defined (and well conditioned), the simplified ones are defined too and
-- equal up to rounding, relative to the scale of the computation.
preservesValues :: Graph -> [NodeId] -> Env -> Property
preservesValues g outs env = case evaluate g env outs of
  -- Outside the domain of the original: nothing to compare.
  Left _ -> discard
  Right vals
    | not (wellConditioned g vals outs) -> discard
    | otherwise ->
        let (outs', g') = simplified g outs
            scale = scaleOf g vals outs
         in counterexample (show (nodes g')) $ case evaluate g' env outs' of
              Left err -> counterexample ("simplified graph fails: " <> show err) False
              Right vals' ->
                conjoin
                  [ counterexample (show (o, a, b)) (abs (a - b) <= 1e-10 * scale)
                  | (o, o') <- zip outs outs'
                  , let a = valueAt vals o
                        b = valueAt vals' o'
                  ]

-- ---------------------------------------------------------------------------
-- Canonical forms

canonicalSpec :: Spec
canonicalSpec = do
  it "gathers like terms: x + x = 2·x" $
    simp (do x <- input "x"; binary Add x x) `shouldBe` "2·x"
  it "merges powers: x·x·x = x³" $
    simp (do x <- input "x"; xx <- binary Mul x x; binary Mul xx x) `shouldBe` "x³"
  it "folds coefficients: 2·x·3 = 6·x" $
    simp (do x <- input "x"; two <- constant 2; three <- constant 3; t <- binary Mul two x; binary Mul t three)
      `shouldBe` "6·x"
  it "cancels: x - x = 0, x·y - y·x = 0" $ do
    simp (do x <- input "x"; binary Sub x x) `shouldBe` "0"
    simp (do (x, y) <- xy; a <- binary Mul x y; binary Sub a =<< binary Mul y x) `shouldBe` "0"
  it "cancels exponents: x/x = 1, x·y/x = y, x²·x⁻¹ = x" $ do
    simp (do x <- input "x"; binary Div x x) `shouldBe` "1"
    simp (do (x, y) <- xy; a <- binary Mul x y; binary Div a x) `shouldBe` "y"
    simp (do x <- input "x"; a <- unary (PowC 2) x; binary Mul a =<< unary (PowC (-1)) x) `shouldBe` "x"
  it "applies the identities of 0 and 1" $ do
    simp (do x <- input "x"; binary Add x =<< constant 0) `shouldBe` "x"
    simp (do x <- input "x"; one <- constant 1; binary Mul one x) `shouldBe` "x"
    simp (do x <- input "x"; binary Div x =<< constant 1) `shouldBe` "x"
    simp (do x <- input "x"; unary (PowC 1) x) `shouldBe` "x"
    simp (do x <- input "x"; unary (PowC 0) x) `shouldBe` "1"
    simp (do x <- input "x"; m <- constant (-1); binary Mul x m) `shouldBe` "-x"
    simp (do x <- input "x"; unary Neg =<< unary Neg x) `shouldBe` "x"
  it "orders terms and factors by structure, not by creation order" $ do
    simp (do (x, y) <- xy; binary Add y x) `shouldBe` "x + y"
    simp (do y <- input "y"; x <- input "x"; binary Add y x) `shouldBe` "x + y"
    simp (do (x, y) <- xy; binary Mul y x) `shouldBe` "x·y"
    -- inputs before parameters, leaves before applications
    simp (do x <- input "x"; w <- param "w"; e <- unary Exp x; a <- binary Add e w; binary Add a x)
      `shouldBe` "x + w + exp(x)"
  it "does not distribute products over sums" $ do
    simp (do (x, y) <- xy; z <- input "z"; binary Mul x =<< binary Add y z) `shouldBe` "x·(y + z)"
    simp (do (x, y) <- xy; s <- binary Add x y; binary Mul s s) `shouldBe` "(x + y)²"
    simp (do (x, y) <- xy; s <- binary Add x y; binary Mul s =<< binary Add y x) `shouldBe` "(x + y)²"
    simp (do (x, y) <- xy; s <- binary Add x y; binary Div s s) `shouldBe` "1"
  it "distributes constant factors, which adds no terms" $ do
    simp (do (x, y) <- xy; two <- constant 2; binary Mul two =<< binary Add y x) `shouldBe` "2·x + 2·y"
    simp (do (x, y) <- xy; unary Neg =<< binary Sub x y) `shouldBe` "-x + y"
    simp (do (x, y) <- xy; two <- constant 2; binary Div (x) =<< binary Mul two y) `shouldBe` "0.5·x/y"
  it "lays out results: constant first, subtraction, quotients" $ do
    simp (do x <- input "x"; one <- constant 1; binary Sub one =<< unary (PowC 2) =<< unary Tanh x)
      `shouldBe` "1 - tanh(x)²"
    simp (do (x, y) <- xy; two <- constant 2; binary Sub x =<< binary Mul two y) `shouldBe` "x - 2·y"
    simp (do x <- input "x"; unary (PowC (-1)) x) `shouldBe` "1/x"
    simp (do x <- input "x"; two <- constant 2; unary (PowC (-1)) =<< binary Mul two x) `shouldBe` "0.5/x"
    simp (do (x, y) <- xy; m <- constant (-1); a <- binary Mul m x; binary Mul a y) `shouldBe` "-x·y"
    simp (do (x, y) <- xy; z <- input "z"; a <- binary Div x y; binary Div a z) `shouldBe` "x/(y·z)"
  it "puts a sum left on its own by a product back as a sum" $ do
    simp (do (x, y) <- xy; z <- input "z"; s <- binary Add x y; p <- binary Mul s z; binary Div p z)
      `shouldBe` "x + y"
    simp (do (x, y) <- xy; three <- constant 3; t <- binary Mul three x; s <- binary Add x y; binary Mul t =<< binary Div s x)
      `shouldBe` "3·x + 3·y"
  it "multiplies a product of sums by its coefficient as a whole" $ do
    simp (do y <- input "y"; one <- constant 1; s <- binary Add one y; t <- binary Add s one; p <- binary Mul s t; binary Add p p)
      `shouldBe` "2·((1 + y)·(2 + y))"
    simp (do (x, y) <- xy; z <- input "z"; s <- binary Add x y; d <- binary Div s z; unary Neg d)
      `shouldBe` "-((x + y)/z)"
  it "multiplies integer powers through products: (2·x·y)² = 4·x²·y²" $
    simp (do (x, y) <- xy; two <- constant 2; a <- binary Mul two x; unary (PowC 2) =<< binary Mul a y)
      `shouldBe` "4·x²·y²"
  it "keeps non-polynomial ops as atoms, with simplified arguments" $ do
    simp (do x <- input "x"; unary Relu =<< binary Add x x) `shouldBe` "relu(2·x)"
    simp (do x <- input "x"; z <- constant 0; a <- unary Tanh x; b <- unary Tanh =<< binary Add x z; binary Sub a b)
      `shouldBe` "0"
    -- fractional powers are atoms, so they only merge with themselves
    simp (do x <- input "x"; r <- unary (PowC (1 / 2)) x; binary Mul r r) `shouldBe` "(x^0.5)²"
    simp (do x <- input "x"; r <- unary (PowC (1 / 2)) x; binary Mul r x) `shouldBe` "x·x^0.5"

-- ---------------------------------------------------------------------------
-- Constant folding

foldingSpec :: Spec
foldingSpec = do
  it "folds constants with the forward pass's arithmetic" $ do
    simp (constant 0 >>= unary Exp) `shouldBe` "1"
    simp (constant 2 >>= unary (PowC 3)) `shouldBe` "8"
    simp (constant 4 >>= unary (PowC (1 / 2))) `shouldBe` "2"
    simp (do a <- constant 1; b <- constant 1; binary Add a b) `shouldBe` "2"
    simp (constant (-3) >>= unary Relu) `shouldBe` "0"
    simp (do x <- input "x"; a <- constant 3; b <- constant 0.5; s <- binary Mul a b; binary Add x s)
      `shouldBe` "1.5 + x"
  it "refuses folds whose result is not finite, keeping the node" $ do
    simp (do x <- input "x"; binary Div x =<< constant 0) `shouldBe` "x/0"
    simp (constant 0 >>= unary Log) `shouldBe` "log(0)"
    simp (constant (-4) >>= unary (PowC (1 / 2))) `shouldBe` "(-4)^0.5"
    simp (constant 1000 >>= unary Exp) `shouldBe` "exp(1000)"
    simp (do x <- input "x"; e <- unary Exp =<< constant 1000; binary Mul x e) `shouldBe` "x·exp(1000)"
  it "keeps the errors of refused folds: evaluating still fails" $ do
    let (o, g) = built (do x <- input "x"; binary Div x =<< constant 0)
        (o', g') = simplified1 g o
    evaluate g' (inputs [("x", 1)]) [o'] `shouldSatisfy` either (const True) (const False)

-- ---------------------------------------------------------------------------
-- Domain

domainSpec :: Spec
domainSpec = do
  it "x/x = 1 is defined at x = 0, where x/x is not" $ do
    let (o, g) = built (do x <- input "x"; binary Div x x)
        (o', g') = simplified1 g o
        env = inputs [("x", 0)]
    evaluate g env [o] `shouldSatisfy` either (const True) (const False)
    fmap (`valueAt` o') (evaluate g' env [o']) `shouldBe` Right 1
  it "0·log(x) = 0 is defined at x < 0, where log x is not" $ do
    let (o, g) = built (do x <- input "x"; z <- constant 0; binary Mul z =<< unary Log x)
        (o', g') = simplified1 g o
        env = inputs [("x", -1)]
    evaluate g env [o] `shouldSatisfy` either (const True) (const False)
    fmap (`valueAt` o') (evaluate g' env [o']) `shouldBe` Right 0

-- ---------------------------------------------------------------------------
-- Gradients

gradientSpec :: Spec
gradientSpec = do
  it "prints intro.hsg's gradients as -2·b, -2·a, -2 (the milestone)" $ do
    let (ex, g) = intro
    simplifiedGradients (g, introL ex) [introA ex, introB ex, introC ex]
      `shouldBe` Rendered [] ["-2·b", "-2·a", "-2"]
  it "gives the same for (a*b + c) * -2 built without names" $ do
    let ((a, b, c, l), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          c' <- input "c"
          ab <- binary Mul a' b'
          s <- binary Add ab c'
          l' <- binary Mul s =<< constant (-2)
          pure (a', b', c', l')
    simplifiedGradients (g, l) [a, b, c] `shouldBe` Rendered [] ["-2·b", "-2·a", "-2"]
    simplifiedGradients (g, l) [l] `shouldBe` Rendered [] ["1"]
  it "simplifies intro.hsg's L and its other gradients" $ do
    let (ex, g) = intro
        (gns, g') = symbolic g (introL ex)
        (outs, g'') = simplified g' [introL ex, gradNode gns (introD ex), gradNode gns (introF ex)]
    map (prettyNode g'') outs `shouldBe` ["-2·a·b - 2·c", "-2", "a·b + c"]
  it "folds the 1 + 1 of a + a into 2" $ do
    let ((a, out), g) = built (do a' <- input "a"; o <- binary Add a' a'; pure (a', o))
    simplifiedGradients (g, out) [a] `shouldBe` Rendered [] ["2"]
  it "shares 1 - tanh(…)² between the neuron's gradients" $ do
    let (ex, g) = neuron
    simplifiedGradients (g, neuronY ex) (neuronB ex : neuronW ex ++ neuronX ex)
      `shouldBe` Rendered
        [("t1", "1 - tanh(x[0]·w[0] + x[1]·w[1] + x[2]·w[2] + b)²")]
        ["t1", "x[0]·t1", "x[1]·t1", "x[2]·t1", "w[0]·t1", "w[1]·t1", "w[2]·t1"]
  it "gives clean higher derivatives of x³: 3·x², 6·x, 6" $ do
    let ((x, y), g) = built (do x' <- input "x"; y' <- unary (PowC 3) x'; pure (x', y'))
        (d1, g1) = symbolic g y
        (d2, g2) = symbolic g1 (gradNode d1 x)
        (d3, g3) = symbolic g2 (gradNode d2 x)
        (outs, g4) = simplified g3 (map (`gradNode` x) [d1, d2, d3])
    map (prettyNode g4) outs `shouldBe` ["3·x²", "6·x", "6"]
  it "simplifies tanh'' into one product" $ do
    let ((x, y), g) = built (do x' <- input "x"; y' <- unary Tanh x'; pure (x', y'))
        (d1, g1) = symbolic g y
        (d2, g2) = symbolic g1 (gradNode d1 x)
        (o, g3) = simplified1 g2 (gradNode d2 x)
    prettyNode g3 o `shouldBe` "-2·tanh(x)·(1 - tanh(x)²)"

-- ---------------------------------------------------------------------------
-- Provenance and the result graph

provenanceSpec :: Spec
provenanceSpec = do
  it "keeps labels of outputs and gradient nodes, but never labels a leaf" $ do
    let (ex, g) = intro
        (gns, g') = symbolic g (introL ex)
        (outs, g'') = simplified g' [introL ex, gradNode gns (introA ex), gradNode gns (introC ex)]
        labelOf i = provLabel (nodeProv (nodeAt g'' i))
    map labelOf outs `shouldBe` [Just "L", Just "∂L/∂a", Nothing]
  it "keeps the span of the source node on its simplified node" $ do
    let (ex, g) = intro
        (gns, g') = symbolic g (introL ex)
        (ga, g'') = simplified1 g' (gradNode gns (introA ex))
    provSpan (nodeProv (nodeAt g'' ga)) `shouldBe` provSpan (nodeProv (nodeAt g' (gradNode gns (introA ex))))
  it "returns a graph of exactly the simplified expressions" $ do
    let ((x, o), g) = built (do x' <- input "x"; e <- unary Exp x'; s <- binary Sub e e; o' <- binary Add s x'; pure (x', o'))
        (outs, g') = simplified g [o, x]
    Set.size (reachable g' outs) `shouldBe` nodeCount g'
    nodeCount g' `shouldBe` 1
  it "leaves out ids that are not nodes of the graph" $ do
    let (foreignId, _) = built (input "a" >> input "b")
        (x, g) = built (input "x")
    fmap (Map.keys . fst) (simplify g [x, foreignId]) `shouldBe` Right [x]

-- ---------------------------------------------------------------------------
-- Sharing

sharingSpec :: Spec
sharingSpec = do
  it "turns a chain of 30 squarings into one power" $ do
    let ((x, out), g) = chain 30
        (o, g') = simplified1 g out
        (gns, gg) = symbolic g out
        (d, gg') = simplified1 gg (gradNode gns x)
    prettyNode g' o `shouldBe` "x¹⁰⁷³⁷⁴¹⁸²⁴"
    nodeCount g' `shouldBe` 2
    prettyNode gg' d `shouldBe` "1073741824·x¹⁰⁷³⁷⁴¹⁸²³"
    nodeCount gg' `shouldBe` 4
  it "keeps a deep chain that reuses every level small, and its value" $ do
    -- xₖ₊₁ = tanh(xₖ·xₖ - 1): written out as a tree, 2²⁰⁰ copies of x.
    let levels = 200 :: Int
        ((x, out), g) = built $ do
          x' <- input "x"
          one <- constant 1
          let level v = unary Tanh =<< (\s -> binary Sub s one) =<< binary Mul v v
          out' <- foldr (=<<) (pure x') (replicate levels level)
          pure (x', out')
        (o, g') = simplified1 g out
        env = inputs [("x", 0.7)]
    nodeCount g' `shouldSatisfy` (<= 3 * levels + 2)
    valueAt (valuesOf g' env [o]) o `shouldBe` valueAt (valuesOf g env [out]) out
    -- Its gradient: a product of 2·xₖ·(1 - xₖ₊₁²) over all levels.
    let (gns, gg) = symbolic g out
        (d, gg') = simplified1 gg (gradNode gns x)
        grads = case gradients g env out of
          Right (_, gs) -> gs
          Left err -> error (show err)
        expected = gradAt grads x
    Set.size (reachable gg' [d]) `shouldSatisfy` (<= 7 * levels)
    valueAt (valuesOf gg' env [d]) d `shouldSatisfy` (\v -> abs (v - expected) <= 1e-12 * abs expected)

-- | @x₀ = x@, @xₖ₊₁ = xₖ · xₖ@, @n@ times.
chain :: Int -> ((NodeId, NodeId), Graph)
chain n = built $ do
  x <- input "x"
  out <- foldr (=<<) (pure x) (replicate n (\v -> binary Mul v v))
  pure (x, out)

-- ---------------------------------------------------------------------------
-- Properties over random graphs

propertySpec :: Spec
propertySpec = modifyMaxSuccess (const 1000) $ do
  prop "preserves values (graphs with constants 0 and divisions by zero)" $
    onGraphs (\g outs -> forAll genEnv (preservesValues g outs))
  prop "preserves values (every op)" $ \(SmoothRecipe steps) ->
    onGraphs (\g outs -> forAll genEnv (preservesValues g outs)) (Recipe steps)
  prop "preserves the values of symbolic gradients" $ \(SmoothRecipe steps) ->
    let (ids, g) = builtRecipe steps
        (gns, g') = symbolic g (last ids)
     in not (null ids) ==> forAll genEnv (preservesValues g' (Map.elems gns))
  prop "is idempotent: simplifying twice is simplifying once" $
    onGraphs $ \g outs ->
      let (outs1, g1) = simplified g outs
          (outs2, g2) = simplified g1 outs1
       in conjoin
            [ nodeCount g2 === nodeCount g1
            , property (sameExpressions (outs1, g1) (outs2, g2))
            ]
  prop "gives commuted expressions the same result" $
    onGraphs $ \g outs ->
      let (mouts, mg) = built (copyWith swap g outs)
          swap op args = case (op, args) of
            (Bin Add, [a, b]) -> [b, a]
            (Bin Mul, [a, b]) -> [b, a]
            _ -> args
       in property (sameExpressions (simplified g outs) (simplified mg mouts))
  prop "leaves nothing to fold" $
    onGraphs $ \g outs ->
      let (_, g') = simplified g outs
       in conjoin [counterexample (show (i, n)) (normal g' n) | (i, n) <- nodes g']
  where
    p ==> q = if p then q else discard
    -- 'withGraph', but recipes can shrink to no steps at all, where it
    -- would have no node to pick outputs from.
    onGraphs check (Recipe steps) = not (null steps) ==> withGraph check (Recipe steps)
    -- No multiplication by 0 or 1 and no addition of 0 is left, and every
    -- node whose arguments are all constants has a value that could not be
    -- folded (it is not finite).
    normal g' n =
      let consts = [c | a <- nodeArgs n, Leaf (Const c) <- [nodeOp (nodeAt g' a)]]
          allConst = not (null (nodeArgs n)) && length consts == length (nodeArgs n)
          value = case evaluate g' emptyEnv [i | (i, m) <- nodes g', m == n] of
            Right vals -> [valueAt vals i | (i, m) <- nodes g', m == n]
            Left _ -> []
       in case nodeOp n of
            Bin Mul | any (`elem` [0, 1]) consts -> False
            Bin Add | 0 `elem` consts -> False
            Bin Sub | 0 `elem` consts -> False
            _ -> not allConst || null value
