{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Symbolic".
--
-- * The rule interpreter: emitting a rule as nodes and evaluating them
--   gives the number 'localGrads' gives.
-- * micrograd's examples: the gradient graphs of @intro.hsg@ and the tanh
--   neuron have the expected shape, and evaluate to backprop's numbers.
-- * Light simplification: no node is created to multiply by a literal
--   @1@, @0@ or @-1@ or to add a literal @0@.
-- * Sharing: the gradient graph of a chain of squarings grows linearly,
--   although written out as a tree it would be exponentially large.
-- * Reachability: exactly the nodes the output depends on get a gradient.
-- * Second derivatives: differentiating a gradient node again.
-- * Provenance: labels such as @∂L/∂a@ and source spans.
-- * Properties over random graphs: evaluating the gradient graph gives
--   exactly backprop's numbers, also for second derivatives, and the
--   gradient graph is at most linearly larger than the forward graph.
module HsGrad.Core.SymbolicSpec
  ( spec
  ) where

import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import HsGrad.Core.Backprop
import HsGrad.Core.Eval
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Core.Symbolic
import HsGrad.Test.Examples
import HsGrad.Test.Graphs (SmoothRecipe (..), built, builtRecipe, genEnv, withGraph)
import Test.Hspec
  ( Expectation
  , Spec
  , describe
  , expectationFailure
  , it
  , shouldBe
  , shouldSatisfy
  )
import Test.Hspec.QuickCheck (modifyMaxSize, prop)
import Test.QuickCheck
  ( Property
  , classify
  , conjoin
  , counterexample
  , discard
  , elements
  , forAll
  , property
  , (===)
  )

spec :: Spec
spec = do
  describe "rule interpreter" interpreterSpec
  describe "micrograd examples" examplesSpec
  describe "light simplification" simplificationSpec
  describe "sharing" sharingSpec
  describe "reachability" reachabilitySpec
  describe "second derivatives" secondSpec
  describe "provenance" provenanceSpec
  describe "random graphs" propertySpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Symbolic differentiation, expecting success.
symbolic :: Graph -> NodeId -> (GradNodes, Graph)
symbolic g out = case symbolicGradients g out of
  Right r -> r
  Left err -> error ("unexpected build error: " <> show err)

-- | Forward values, expecting success.
valuesOf :: Graph -> Env -> [NodeId] -> Values
valuesOf g env outs = case evaluate g env outs of
  Right vals -> vals
  Left err -> error ("unexpected evaluation error: " <> show err)

-- | Numeric gradients, expecting success.
numericGrads :: Graph -> Env -> NodeId -> Grads
numericGrads g env out = case gradients g env out of
  Right (_, grads) -> grads
  Left err -> error ("unexpected error: " <> show err)

-- | The gradient node of a forward node, which must exist.
gradNode :: GradNodes -> NodeId -> NodeId
gradNode grads i = case lookupGradNode i grads of
  Just gi -> gi
  Nothing -> error ("no gradient node for " <> show i)

-- | An environment with only inputs.
inputs :: [(Name, Double)] -> Env
inputs bs = emptyEnv {envInputs = Map.fromList bs}

-- | The value of the symbolic gradient of every node, evaluated, next to
-- backprop's number for it.
compared :: Graph -> Env -> NodeId -> [(NodeId, Double, Double)]
compared g env out = [(n, valueAt vals gn, gradAt grads n) | (n, gn) <- Map.toList gns]
  where
    (gns, g') = symbolic g out
    vals = valuesOf g' env (Map.elems gns)
    grads = numericGrads g env out

-- | Approximate equality, for results of transcendental functions.
shouldBeNear :: Double -> Double -> Expectation
shouldBeNear x y =
  x `shouldSatisfy` \v -> abs (v - y) <= 1e-12 * max 1 (abs y)

-- | The op and arguments of a node.
shape :: Graph -> NodeId -> (Op, [NodeId])
shape g i = (nodeOp n, nodeArgs n)
  where
    n = nodeAt g i

-- ---------------------------------------------------------------------------
-- The rule interpreter

interpreterSpec :: Spec
interpreterSpec =
  prop "emitDExpr builds nodes that evaluate to localGrads" $
    forAll (elements ops) $ \op -> forAll genEnv $ \env ->
      let ((args, self, emitted), g) = built $ do
            x <- input "x"
            y <- input "y"
            let as = take (arity op) [x, y]
            n <- node op as
            es <- mapM (emitDExpr as n) (partials op)
            pure (as, n, es)
       in case evaluate g env emitted of
            -- Outside the op's domain (log of a negative number, …): the
            -- numeric rule is not finite either, nothing to compare.
            Left _ -> discard
            Right vals ->
              map (valueAt vals) emitted
                === localGrads op (map (valueAt vals) args) (valueAt vals self)
  where
    ops =
      map Un [Neg, PowC 0, PowC 1, PowC 2, PowC 3, PowC (-1), PowC (1 / 2), PowC (-1 / 2), Exp, Log, Tanh, Relu, Step]
        ++ map Bin [minBound .. maxBound]

-- ---------------------------------------------------------------------------
-- micrograd examples

examplesSpec :: Spec
examplesSpec = do
  it "differentiates intro.hsg into three new nodes" $ do
    -- L = d · f, d = e + c, e = a · b. With the seed 1 and the partials 1
    -- of +, the gradients of d, e and c are all the forward node f itself,
    -- ∂L/∂f is d, and only ∂L/∂a = f · b and ∂L/∂b = f · a are new, plus
    -- the constant 1 for ∂L/∂L.
    let (ns, g) = intro
        (gns, g') = symbolic g (introL ns)
        grad = gradNode gns
    nodeCount g' - nodeCount g `shouldBe` 3
    map grad [introD ns, introE ns, introC ns] `shouldBe` replicate 3 (introF ns)
    grad (introF ns) `shouldBe` introD ns
    shape g' (grad (introA ns)) `shouldBe` (Bin Mul, [introF ns, introB ns])
    shape g' (grad (introB ns)) `shouldBe` (Bin Mul, [introF ns, introA ns])
    shape g' (grad (introL ns)) `shouldBe` (Leaf (Const 1), [])

  it "evaluates intro.hsg's gradient nodes to ∂L/∂a = 6, ∂L/∂b = -4, ∂L/∂c = -2" $ do
    let (ns, g) = intro
        (gns, g') = symbolic g (introL ns)
        vals = valuesOf g' introEnv (Map.elems gns)
        LeafGradNodes ins ps = leafGradNodes g' gns
    map (valueAt vals) (Map.elems ins) `shouldBe` [6, -4, -2]
    Map.keys ins `shouldBe` ["a", "b", "c"]
    ps `shouldBe` Map.empty
    let results = compared g introEnv (introL ns)
    [s | (_, s, _) <- results] `shouldBe` [b | (_, _, b) <- results]

  it "shares 1 - y² between all gradients of the tanh neuron" $ do
    -- y = tanh s. Every gradient below y is ∂y/∂s = 1 - y² (passed through
    -- the + nodes unchanged), times one leaf: ∂y/∂w[i] = t · x[i],
    -- ∂y/∂x[i] = t · w[i], ∂y/∂b = t.
    let (ns, g) = neuron
        (gns, g') = symbolic g (neuronY ns)
        t = gradNode gns (neuronB ns)
    case shape g' t of
      (Bin Sub, [one, sq]) -> do
        shape g' one `shouldBe` (Leaf (Const 1), [])
        shape g' sq `shouldBe` (Un (PowC 2), [neuronY ns])
        -- ∂y/∂y is the constant 1: the very node the tanh rule created.
        gradNode gns (neuronY ns) `shouldBe` one
      other -> expectationFailure ("1 - y² expected, got " <> show other)
    [shape g' (gradNode gns w) | w <- neuronW ns] `shouldBe` [(Bin Mul, [t, x]) | x <- neuronX ns]
    [shape g' (gradNode gns x) | x <- neuronX ns] `shouldBe` [(Bin Mul, [t, w]) | w <- neuronW ns]

  it "evaluates the tanh neuron's gradients to backprop's numbers" $ do
    let (ns, g) = neuron
        results = compared g neuronEnv (neuronY ns)
    length results `shouldBe` nodeCount g
    [s | (_, s, _) <- results] `shouldBe` [b | (_, _, b) <- results]

-- ---------------------------------------------------------------------------
-- Light simplification

simplificationSpec :: Spec
simplificationSpec = do
  it "x - y: the gradients are the constants 1 and -1" $ do
    let ((x, y, out), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          out' <- binary Sub x' y'
          pure (x', y', out')
        (gns, g') = symbolic g out
    shape g' (gradNode gns x) `shouldBe` (Leaf (Const 1), [])
    shape g' (gradNode gns y) `shouldBe` (Leaf (Const (-1)), [])

  it "multiplying by -1 is a negation: ∂tanh(x - y)/∂y = -(1 - tanh²)" $ do
    let ((x, y, out), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          out' <- unary Tanh =<< binary Sub x' y'
          pure (x', y', out')
        (gns, g') = symbolic g out
    shape g' (gradNode gns y) `shouldBe` (Un Neg, [gradNode gns x])
    shape g' (gradNode gns x) `shouldSatisfy` ((== Bin Sub) . fst)

  it "a gradient of literal 0 passes on 0 without emitting anything" $ do
    -- step has the partial 0, so ∂L/∂(x·y) = 0, and x and y get 0 too:
    -- the sweep never emits the partials of · just to multiply them by 0.
    -- The only new nodes are the constants 1 (for ∂L/∂L) and 0.
    let ((x, y, xy, out), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          xy' <- binary Mul x' y'
          out' <- unary Step xy'
          pure (x', y', xy', out')
        (gns, g') = symbolic g out
    nodeCount g' - nodeCount g `shouldBe` 2
    [shape g' (gradNode gns i) | i <- [x, y, xy]] `shouldBe` replicate 3 (Leaf (Const 0), [])

  it "√(step x) at x < 0: ∂/∂x is 0, where backprop reports ∞ · 0" $ do
    -- The slope of step is the literal 0, so ∂/∂x is the constant 0 (the
    -- function is constant near x). Numerically, √'s slope at step x = 0
    -- is infinite: backprop fails, and so does evaluating ∂/∂(step x).
    let ((x, st, out), g) = built $ do
          x' <- input "x"
          st' <- unary Step x'
          out' <- unary (PowC (1 / 2)) st'
          pure (x', st', out')
        (gns, g') = symbolic g out
        env = inputs [("x", -1)]
    shape g' (gradNode gns x) `shouldBe` (Leaf (Const 0), [])
    gradients g env out `shouldSatisfy` either (const True) (const False)
    evaluate g' env [gradNode gns st] `shouldSatisfy` either (const True) (const False)

  it "a + a: two literal 1s are added as nodes, ∂/∂a = 2" $ do
    let ((a, out), g) = built $ do
          a' <- input "a"
          out' <- binary Add a' a'
          pure (a', out')
        (gns, g') = symbolic g out
    valueAt (valuesOf g' (inputs [("a", 5)]) [gradNode gns a]) (gradNode gns a) `shouldBe` 2

  it "never multiplies by a constant the sweep created" $ do
    -- In the README expression of micrograd (plenty of +, -, unary minus),
    -- no new * node has a new constant 0, 1 or -1 as an argument.
    let (out, g) = readme
        (_, g') = symbolic g out
        new = drop (nodeCount g) (nodes g')
        -- A constant among cs that the sweep created (the README's own
        -- constant 1 may legitimately be multiplied with).
        newConst cs i = case nodeOp (nodeAt g' i) of
          Leaf (Const c) -> c `elem` cs && nodeIndex i >= nodeCount g
          _ -> False
    [n | (_, n@Node {nodeOp = Bin Mul}) <- new, any (newConst [0, 1, -1]) (nodeArgs n)] `shouldBe` []
    [n | (_, n@Node {nodeOp = Bin Add}) <- new, any (newConst [0]) (nodeArgs n)] `shouldBe` []

-- | The expression from micrograd's README (as in the Backprop tests),
-- with lots of sharing, +, -, unary minus, powers and divisions.
readme :: (NodeId, Graph)
readme = built $ do
  a <- input "a"
  b <- input "b"
  one <- constant 1
  two <- constant 2
  three <- constant 3
  ten <- constant 10
  c0 <- binary Add a b
  d0 <- do
    ab <- binary Mul a b
    b3 <- unary (PowC 3) b
    binary Add ab b3
  c1 <- binary Add c0 =<< binary Add c0 one
  c2 <- do
    t <- binary Add c1 one
    na <- unary Neg a
    binary Add c1 =<< binary Add t na
  d1 <- do
    t <- binary Mul d0 two
    r <- unary Relu =<< binary Add b a
    binary Add d0 =<< binary Add t r
  d2 <- do
    t <- binary Mul d1 three
    r <- unary Relu =<< binary Sub b a
    binary Add d1 =<< binary Add t r
  e <- binary Sub c2 d2
  f <- unary (PowC 2) e
  g0 <- binary Div f two
  binary Add g0 =<< binary Div ten f

-- ---------------------------------------------------------------------------
-- Sharing

sharingSpec :: Spec
sharingSpec = do
  it "grows by 2 nodes per level on a chain of 30 squarings" $ do
    -- x₀ = x, xₖ₊₁ = xₖ · xₖ, so x₃₀ = x^(2³⁰). The gradient of xₖ is
    -- m + m with m = ∂/∂xₖ₊₁ · xₖ: one * and one + per level (the top
    -- level needs only the +, and the seed adds the constant 1).
    let n = 30 :: Int
        ((x, out), g) = chain n
        (gns, g') = symbolic g out
        gx = gradNode gns x
    nodeCount g `shouldBe` n + 1
    nodeCount g' - nodeCount g `shouldBe` 2 * n
    -- d/dx x^(2^30) = 2^30 · x^(2^30 - 1), which is 2^30 at x = 1.
    valueAt (valuesOf g' (inputs [("x", 1)]) [gx]) gx `shouldBe` 2 ^ n

  it "would be exponentially large written out as a tree" $ do
    -- Textbook symbolic differentiation works on expression trees, where
    -- every use of xₖ is a copy. Unfolding the gradient node into a tree
    -- gives about 1.5 · 10^18 nodes; the graph has 31 + 60.
    let ((x, out), g) = chain 30
        (gns, g') = symbolic g out
    treeSize g' (gradNode gns x) `shouldSatisfy` (> 2 ^ (30 :: Int))
    treeSize g' out `shouldBe` 2 ^ (31 :: Int) - 1

-- | @x₀ = x@, @xₖ₊₁ = xₖ · xₖ@, @n@ times.
chain :: Int -> ((NodeId, NodeId), Graph)
chain n = built $ do
  x <- input "x"
  let square v = binary Mul v v
  out <- foldr (=<<) (pure x) (replicate n square)
  pure (x, out)

-- | The number of nodes of a node's expression written out as a tree,
-- where every use of a shared node counts as a copy. Computed bottom-up
-- over the whole graph, so it takes linear time however large the number.
treeSize :: Graph -> NodeId -> Integer
treeSize g i = sizes Map.! i
  where
    sizes = foldl add Map.empty (nodes g)
    add acc (j, n) = Map.insert j (1 + sum [acc Map.! a | a <- nodeArgs n]) acc

-- ---------------------------------------------------------------------------
-- Reachability

reachabilitySpec :: Spec
reachabilitySpec = do
  it "gives no gradient node to nodes the output does not depend on" $ do
    -- l1 = x² · w, l2 = l1 + exp y. Differentiating l1 leaves out y,
    -- exp y and l2.
    let ((x, y, w, sq, l1, l2), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          w' <- param "w"
          sq' <- unary (PowC 2) x'
          l1' <- binary Mul sq' w'
          l2' <- binary Add l1' =<< unary Exp y'
          pure (x', y', w', sq', l1', l2')
        (gns, g') = symbolic g l1
    Map.keys gns `shouldBe` [x, w, sq, l1]
    lookupGradNode y gns `shouldBe` Nothing
    lookupGradNode l2 gns `shouldBe` Nothing
    leafGradNodes g' gns
      `shouldBe` LeafGradNodes (Map.fromList [("x", gradNode gns x)]) (Map.fromList [("w", gradNode gns w)])

  it "gives the constant 0 on request for an unrelated leaf" $ do
    let ((y, out), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          out' <- unary Exp x'
          pure (y', out')
        (z, g') = case extend g (backwardNodes out >>= \gns -> gradNodeOrZero gns y) of
          Right r -> r
          Left err -> error (show err)
    shape g' z `shouldBe` (Leaf (Const 0), [])

  it "gives nothing, and adds nothing, for an id of another graph" $ do
    let (other, _) = built (input "x" >> input "y" >> input "z")
        (_, g) = built (input "x")
    symbolicGradients g other `shouldBe` Right (Map.empty, g)

-- ---------------------------------------------------------------------------
-- Second derivatives

secondSpec :: Spec
secondSpec = do
  it "d²/dx² x³ = 6x, d³/dx³ x³ = 6, and x³ has no fourth derivative node" $ do
    let ((x, y), g) = built $ do
          x' <- input "x"
          y' <- unary (PowC 3) x'
          pure (x', y')
        (d1, g1) = symbolic g y
        dy = gradNode d1 x -- 3 · x²
        (d2, g2) = symbolic g1 dy
        ddy = gradNode d2 x -- 3 · (2 · x)
        (d3, g3) = symbolic g2 ddy
        dddy = gradNode d3 x -- 3 · 2
        (d4, _) = symbolic g3 dddy
        at v i = valueAt (valuesOf g3 (inputs [("x", v)]) [i]) i
    [at v dy | v <- [2, -1.5]] `shouldBe` [12, 6.75]
    [at v ddy | v <- [2, -1.5]] `shouldBe` [12, -9]
    [at v dddy | v <- [2, -1.5]] `shouldBe` [6, 6]
    -- The third derivative is the product of two constants: x is not
    -- reachable from it, so the fourth derivative has no entry (it is 0).
    lookupGradNode x d4 `shouldBe` Nothing

  it "d²/dx² tanh x = -2 tanh x (1 - tanh² x)" $ do
    let ((x, y), g) = built $ do
          x' <- input "x"
          y' <- unary Tanh x'
          pure (x', y')
        (d1, g1) = symbolic g y
        (d2, g2) = symbolic g1 (gradNode d1 x)
        ddy = gradNode d2 x
        t = tanh 0.7
    valueAt (valuesOf g2 (inputs [("x", 0.7)]) [ddy]) ddy `shouldBeNear` (-2 * t * (1 - t * t))

  it "mixed partials of x · y: ∂/∂y (∂(x·y)/∂x) = 1, ∂/∂x (∂(x·y)/∂x) = 0" $ do
    -- ∂(x·y)/∂x is the node y itself. Its gradient with respect to y is
    -- the constant 1; x is not reachable from it.
    let ((x, y, out), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          out' <- binary Mul x' y'
          pure (x', y', out')
        (d1, g1) = symbolic g out
        (d2, g2) = symbolic g1 (gradNode d1 x)
    gradNode d1 x `shouldBe` y
    shape g2 (gradNode d2 y) `shouldBe` (Leaf (Const 1), [])
    lookupGradNode x d2 `shouldBe` Nothing

-- ---------------------------------------------------------------------------
-- Provenance

provenanceSpec :: Spec
provenanceSpec = do
  it "labels new gradient nodes ∂L/∂a, and leaves forward nodes alone" $ do
    let (ns, g) = intro
        (gns, g') = symbolic g (introL ns)
        labelOf i = provLabel (nodeProv (nodeAt g' i))
    labelOf (gradNode gns (introA ns)) `shouldBe` Just "∂L/∂a"
    labelOf (gradNode gns (introB ns)) `shouldBe` Just "∂L/∂b"
    -- ∂L/∂c is the forward constant f, which keeps having no label; ∂L/∂L
    -- is a constant, which may be shared by anything, so it gets none.
    labelOf (introF ns) `shouldBe` Nothing
    labelOf (gradNode gns (introL ns)) `shouldBe` Nothing

  it "does not label a forward node that is a gradient" $ do
    -- In L = x · exp y, ∂L/∂x is the forward node exp y. Labelling it
    -- ∂L/∂x would rename it in every picture of the forward graph.
    let ((x, ey, l), g) = built $ do
          x' <- input "x"
          ey' <- unary Exp =<< input "y"
          l' <- binary Mul x' ey'
          label "L" l'
          pure (x', ey', l')
        (gns, g') = symbolic g l
    gradNode gns x `shouldBe` ey
    provLabel (nodeProv (nodeAt g' ey)) `shouldBe` Nothing

  it "gives gradient nodes the span of the node that passes the gradient on" $ do
    -- ∂L/∂a = f · b is created while e = a · b (line 6, columns 9-13)
    -- passes its gradient on.
    let (ns, g) = intro
        (gns, g') = symbolic g (introL ns)
    provSpan (nodeProv (nodeAt g' (gradNode gns (introA ns))))
      `shouldBe` Just (SrcSpan "intro.hsg" (Pos 6 9) (Pos 6 14))

  it "names gradients, with parentheses around a fraction" $ do
    let ((x, y), g) = built $ do
          x' <- input "x"
          y' <- unary (PowC 3) x'
          label "y" y'
          pure (x', y')
        (d1, g1) = symbolic g y
        (d2, g2) = symbolic g1 (gradNode d1 x)
        labelOf gr i = provLabel (nodeProv (nodeAt gr i))
    gradientName g y x `shouldBe` Just "∂y/∂x"
    labelOf g1 (gradNode d1 x) `shouldBe` Just "∂y/∂x"
    labelOf g2 (gradNode d2 x) `shouldBe` Just "∂(∂y/∂x)/∂x"

  it "does not label gradients of an unnamed output" $ do
    let ((x, out), g) = built $ do
          x' <- input "x"
          out' <- binary Mul x' =<< unary Exp x'
          pure (x', out')
        (gns, g') = symbolic g out
    gradientName g out x `shouldBe` Nothing
    provLabel (nodeProv (nodeAt g' (gradNode gns x))) `shouldBe` Nothing

-- ---------------------------------------------------------------------------
-- Properties over random graphs

propertySpec :: Spec
propertySpec = do
  modifyMaxSize (const 15) $ do
    prop "evaluates to exactly backprop's gradients (every op)" $
      \(SmoothRecipe steps) -> forAll genEnv $ \env ->
        let (ids, g) = builtRecipe steps
         in if null ids then discard else agreesWithBackprop g env (last ids)

    prop "second derivatives are backprop on the gradient graph" $
      \(SmoothRecipe steps) -> forAll genEnv $ \env ->
        let (ids, g) = builtRecipe steps
            out = last ids
            (gns, g1) = symbolic g out
            -- The gradient with respect to the first leaf the output
            -- depends on, differentiated again.
            firstLeafGrad =
              take 1
                [ gl
                | l <- [Input "x", Input "y", Param "w"]
                , Just i <- [findNode (Leaf l) [] g]
                , Just gl <- [lookupGradNode i gns]
                ]
         in case firstLeafGrad of
              _ | null ids -> discard
              [] -> discard -- the output depends on no input or parameter
              gl : _ -> agreesWithBackprop g1 env gl

  prop "evaluates to exactly backprop's gradients (any output)" $
    withGraph $ \g outs -> forAll genEnv $ \env -> agreesWithBackprop g env (firstOutput outs)

  prop "gives a gradient node to exactly the nodes the output depends on" $
    withGraph $ \g outs ->
      let out = firstOutput outs
       in Map.keysSet (fst (symbolic g out)) === reachable g [out]

  prop "adds at most 6 nodes per edge of the forward graph, plus 3" $
    withGraph $ \g outs ->
      let out = firstOutput outs
          edges = sum [length (nodeArgs (nodeAt g i)) | i <- Set.toList (reachable g [out])]
          added = nodeCount (snd (symbolic g out)) - nodeCount g
       in counterexample (show added <> " nodes added for " <> show edges <> " edges") $
            property (added <= 6 * edges + 3)

-- | 'withGraph' gives at least one output; differentiate the first.
firstOutput :: [NodeId] -> NodeId
firstOutput (o : _) = o
firstOutput [] = error "withGraph gave no outputs"

-- | Evaluating the symbolic gradient of every node gives exactly the
-- number backprop computes for it. Exactly: the gradient graph performs
-- the same floating-point operations in the same order (the simplified
-- cases, @1 · g = g@ and so on, are exact). Samples where backprop fails
-- are discarded; where it succeeds, evaluating the gradient graph must
-- succeed too.
agreesWithBackprop :: Graph -> Env -> NodeId -> Property
agreesWithBackprop g env out = case gradients g env out of
  Left _ -> discard
  Right (_, grads) ->
    let (gns, g') = symbolic g out
     in counterexample (show (nodes g')) $
          counterexample ("output " <> show out <> ", " <> show env) $
            classify (Map.size gns > 5) "more than 5 gradients" $
              case evaluate g' env (Map.elems gns) of
                Left err -> counterexample ("gradient graph: " <> show err) False
                Right vals ->
                  conjoin
                    [ counterexample (show n <> ": symbolic " <> show s <> ", backprop " <> show b) (s == b)
                    | (n, gn) <- Map.toList gns
                    , let s = valueAt vals gn
                          b = gradAt grads n
                    ]
