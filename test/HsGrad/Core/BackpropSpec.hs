{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Backprop".
--
-- * micrograd's examples, built by hand, give micrograd's gradients.
-- * Sharing: a node used several times (also twice by the same node, as in
--   @a + a@) receives the sum of the gradients of all its uses.
-- * Reachability: nodes the output does not depend on get no gradient.
-- * Errors: non-finite gradients are reported at the node where they
--   appear; forward errors pass through 'forwardBackward'.
-- * Properties over random graphs: gradients agree with central finite
--   differences, and exist for exactly the reachable nodes.
module HsGrad.Core.BackpropSpec
  ( spec
  ) where

import Data.IntMap.Strict qualified as IntMap
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import HsGrad.Core.Backprop
import HsGrad.Core.Eval
import HsGrad.Core.Graph
import HsGrad.Core.Op
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
import Test.Hspec.QuickCheck (modifyMaxSize, modifyMaxSuccess, prop)
import Test.QuickCheck
  ( Property
  , classify
  , conjoin
  , counterexample
  , discard
  , forAll
  , (===)
  )

spec :: Spec
spec = do
  describe "micrograd examples" examplesSpec
  describe "sharing" sharingSpec
  describe "reachability" reachabilitySpec
  describe "errors" errorSpec
  describe "random graphs" propertySpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Forward and backward, expecting success.
differentiated :: Graph -> Env -> NodeId -> (Values, Grads)
differentiated g env out = case gradients g env out of
  Right r -> r
  Left err -> error ("unexpected error: " <> show err)

-- | The gradients alone.
gradsOf :: Graph -> Env -> NodeId -> Grads
gradsOf g env out = snd (differentiated g env out)

-- | Check that differentiation fails, and pass the error to a check.
failsWith :: Either GradError (Values, Grads) -> (GradError -> Expectation) -> Expectation
failsWith r check = case r of
  Left err -> check err
  Right (_, grads) -> expectationFailure ("expected an error, got " <> show grads)

-- | An environment with only inputs.
inputs :: [(Name, Double)] -> Env
inputs bs = emptyEnv {envInputs = Map.fromList bs}

-- | Approximate equality, for results of transcendental functions.
shouldBeNear :: Double -> Double -> Expectation
shouldBeNear x y =
  x `shouldSatisfy` \v -> abs (v - y) <= 1e-12 * max 1 (abs y)

-- | Equality to the four decimals micrograd's README prints.
shouldBeAbout :: Double -> Double -> Expectation
shouldBeAbout x y = x `shouldSatisfy` \v -> abs (v - y) < 5e-5

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

-- ---------------------------------------------------------------------------
-- micrograd examples

examplesSpec :: Spec
examplesSpec = do
  it "differentiates intro.hsg: ∂L/∂a = 6, ∂L/∂b = -4, ∂L/∂c = -2" $ do
    -- docs/01-language.md §1.1. L = (a·b + c) · (-2).
    let ((a, b, c, e, d, f, l), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          c' <- input "c"
          e' <- binary Mul a' b'
          d' <- binary Add e' c'
          f' <- constant (-2)
          l' <- binary Mul d' f'
          pure (a', b', c', e', d', f', l')
        grads = gradsOf g (inputs [("a", 2), ("b", -3), ("c", 10)]) l
    -- ∂L/∂L = 1, ∂L/∂d = f = -2, ∂L/∂f = d = 4, ∂L/∂e = ∂L/∂c = ∂L/∂d,
    -- ∂L/∂a = ∂L/∂e · b, ∂L/∂b = ∂L/∂e · a.
    map (gradAt grads) [l, d, f, e, c, a, b] `shouldBe` [1, -2, 4, -2, -2, 6, -4]

  it "gives intro.hsg's leaf gradients by name" $ do
    let (l, g) = built $ do
          a <- input "a"
          b <- input "b"
          c <- input "c"
          e <- binary Mul a b
          d <- binary Add e c
          binary Mul d =<< constant (-2)
        s = schedule g [l]
    case forwardBackward s (inputs [("a", 2), ("b", -3), ("c", 10)]) l of
      Right (vals, grads) -> do
        valueAt vals l `shouldBe` -8
        leafGrads s grads
          `shouldBe` LeafGrads
            { inputGrads = Map.fromList [("a", 6), ("b", -4), ("c", -2)]
            , paramGrads = Map.empty
            }
      Left err -> expectationFailure (show err)

  it "differentiates micrograd's neuron: tanh(x1*w1 + x2*w2 + b)" $ do
    -- The bias is chosen so that o = 1/√2, hence ∂o/∂n = 1 - o² = 1/2.
    let (o, g) = built $ do
          x1 <- input "x1"
          x2 <- input "x2"
          w1 <- param "w1"
          w2 <- param "w2"
          bias <- param "b"
          x1w1 <- binary Mul x1 w1
          x2w2 <- binary Mul x2 w2
          sm <- binary Add x1w1 x2w2
          n <- binary Add sm bias
          unary Tanh n
        env =
          Env
            { envInputs = Map.fromList [("x1", 2), ("x2", 0)]
            , envParams = Map.fromList [("w1", -3), ("w2", 1), ("b", 6.881373587019543)]
            }
        s = schedule g [o]
    case forwardBackward s env o of
      Right (_, grads) -> do
        let LeafGrads ins ps = leafGrads s grads
        -- micrograd: x1.grad = -1.5, w1.grad = 1.0, x2.grad = 0.5, w2.grad = 0.0
        Map.keys ins `shouldBe` ["x1", "x2"]
        Map.keys ps `shouldBe` ["b", "w1", "w2"]
        ins Map.! "x1" `shouldBeNear` (-1.5)
        ins Map.! "x2" `shouldBeNear` 0.5
        ps Map.! "w1" `shouldBeNear` 1
        ps Map.! "w2" `shouldBeNear` 0
        ps Map.! "b" `shouldBeNear` 0.5
      Left err -> expectationFailure (show err)

  it "differentiates the expression from micrograd's README" $ do
    -- micrograd's README, line by line. Python's `c += e` is `c = c + e`, a
    -- new node; `1 + c` is `c + 1`, `3 * d` is `d * 3`. The README prints
    -- g = 24.7041, ∂g/∂a = 138.8338, ∂g/∂b = 645.5773.
    let ((a, b, gOut), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          one <- constant 1
          two <- constant 2
          three <- constant 3
          ten <- constant 10
          c0 <- binary Add a' b'
          d0 <- do
            ab <- binary Mul a' b'
            b3 <- unary (PowC 3) b'
            binary Add ab b3
          c1 <- binary Add c0 =<< binary Add c0 one
          c2 <- do
            t <- binary Add c1 one
            na <- unary Neg a'
            binary Add c1 =<< binary Add t na
          d1 <- do
            t <- binary Mul d0 two
            r <- unary Relu =<< binary Add b' a'
            binary Add d0 =<< binary Add t r
          d2 <- do
            t <- binary Mul d1 three
            r <- unary Relu =<< binary Sub b' a'
            binary Add d1 =<< binary Add t r
          e <- binary Sub c2 d2
          f <- unary (PowC 2) e
          g0 <- binary Div f two
          g1 <- binary Add g0 =<< binary Div ten f
          pure (a', b', g1)
        (vals, grads) = differentiated g (inputs [("a", -4), ("b", 2)]) gOut
    valueAt vals gOut `shouldBeAbout` 24.7041
    gradAt grads a `shouldBeAbout` 138.8338
    gradAt grads b `shouldBeAbout` 645.5773

-- ---------------------------------------------------------------------------
-- Sharing

sharingSpec :: Spec
sharingSpec = do
  it "a + a: both uses count, ∂/∂a = 2" $ do
    -- Hash-consing makes this one node with arguments [a, a]. micrograd's
    -- lecture shows the bug this guards against: with `=` instead of `+=`
    -- the second use overwrites the first, and the gradient comes out as 1.
    let ((a, out), g) = built $ do
          a' <- input "a"
          out' <- binary Add a' a'
          pure (a', out')
    nodeArgs (nodeAt g out) `shouldBe` [a, a]
    gradAt (gradsOf g (inputs [("a", 3)]) out) a `shouldBe` 2

  it "a * a: ∂/∂a = a + a = 2a" $ do
    let ((a, out), g) = built $ do
          a' <- input "a"
          out' <- binary Mul a' a'
          pure (a', out')
    gradAt (gradsOf g (inputs [("a", 3)]) out) a `shouldBe` 6

  it "a / a: the two partials cancel, ∂/∂a = 0" $ do
    let ((a, out), g) = built $ do
          a' <- input "a"
          out' <- binary Div a' a'
          pure (a', out')
    gradAt (gradsOf g (inputs [("a", 3)]) out) a `shouldBe` 0

  it "sums over different users: L = a·b + exp(a)" $ do
    let ((a, b, out), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          ab <- binary Mul a' b'
          ea <- unary Exp a'
          out' <- binary Add ab ea
          pure (a', b', out')
        grads = gradsOf g (inputs [("a", 0.5), ("b", -2)]) out
    gradAt grads a `shouldBeNear` (-2 + exp 0.5)
    gradAt grads b `shouldBe` 0.5

  it "passes a shared node's total gradient on: L = d * d, d = a + b" $ do
    -- ∂L/∂d = 2d = 6 is complete before d is visited, then a and b get it.
    let ((a, b, d, l), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          d' <- binary Add a' b'
          l' <- binary Mul d' d'
          pure (a', b', d', l')
        grads = gradsOf g (inputs [("a", 1), ("b", 2)]) l
    map (gradAt grads) [d, a, b] `shouldBe` [6, 6, 6]

  it "stays linear on a chain of doublings: ∂/∂x = 2^40 over 2^40 paths" $ do
    -- d₀ = x, dₖ₊₁ = dₖ + dₖ. Each path from the top to x contributes 1, and
    -- there are 2^40 of them; the sweep visits each of the 41 nodes once.
    let ((x, out), g) = built $ do
          x' <- input "x"
          let double d = binary Add d d
          out' <- foldr (=<<) (pure x') (replicate 40 double)
          pure (x', out')
        grads = gradsOf g (inputs [("x", 1)]) out
    gradAt grads x `shouldBe` 2 ^ (40 :: Int)
    IntMap.size grads `shouldBe` 41

-- ---------------------------------------------------------------------------
-- Reachability

reachabilitySpec :: Spec
reachabilitySpec = do
  it "the output's gradient is 1" $ do
    let (out, g) = built (unary Tanh =<< input "x")
    gradAt (gradsOf g (inputs [("x", 0.3)]) out) out `shouldBe` 1

  it "gives no gradient to nodes the output does not depend on" $ do
    -- Two outputs of one schedule. Differentiating l1 skips everything that
    -- only l2 uses, and l2 itself.
    let ((x, y, w, sq, l1, l2), g) = built $ do
          x' <- input "x"
          y' <- input "y"
          w' <- param "w"
          sq' <- unary (PowC 2) x'
          l1' <- binary Mul sq' w'
          l2' <- binary Add l1' =<< unary Exp y'
          pure (x', y', w', sq', l1', l2')
        s = schedule g [l1, l2]
        env = bindParam "w" 3 (inputs [("x", 2), ("y", 0)])
    case evalSchedule s env of
      Left err -> expectationFailure (show err)
      Right vals -> case backward s vals l1 of
        Left err -> expectationFailure (show err)
        Right grads -> do
          IntMap.keys grads `shouldBe` map nodeIndex [x, w, sq, l1]
          lookupGrad y grads `shouldBe` Nothing
          lookupGrad l2 grads `shouldBe` Nothing
          gradAt grads y `shouldBe` 0
          leafGrads s grads
            `shouldBe` LeafGrads (Map.fromList [("x", 12)]) (Map.fromList [("w", 4)])

  it "keeps the gradients of Input \"w\" and Param \"w\" apart" $ do
    let ((wi, wp, out), g) = built $ do
          wi' <- input "w"
          wp' <- param "w"
          out' <- binary Mul wi' =<< binary Mul wp' wp'
          pure (wi', wp', out')
        env = bindParam "w" 3 (inputs [("w", 2)])
        s = schedule g [out]
    case forwardBackward s env out of
      Left err -> expectationFailure (show err)
      Right (_, grads) -> do
        -- out = wi · wp²: ∂/∂wi = wp² = 9, ∂/∂wp = 2·wi·wp = 12
        map (gradAt grads) [wi, wp] `shouldBe` [9, 12]
        leafGrads s grads
          `shouldBe` LeafGrads (Map.fromList [("w", 9)]) (Map.fromList [("w", 12)])

  it "gives nothing for an id of another graph" $ do
    let (other, _) = built (input "x" >> input "y")
        (_, g) = built (input "x")
    gradients g (inputs [("x", 1)]) other `shouldBe` Right (IntMap.empty, IntMap.empty)

-- ---------------------------------------------------------------------------
-- Errors

errorSpec :: Spec
errorSpec = do
  it "reports an infinite slope: √x at x = 0" $ do
    -- The value √0 = 0 is fine; its derivative ½·0^(-½) is not.
    let ((x, sq), g) = built $ do
          x' <- input "x"
          sq' <- unary (PowC (1 / 2)) x'
          pure (x', sq')
    nodeArgs (nodeAt g sq) `shouldBe` [x]
    gradients g (inputs [("x", 0)]) sq
      `shouldBe` Left (BackwardError (NonFiniteGrad sq [0] 1 [1 / 0]))

  it "reports 0 · ∞ at the node with the infinite slope: √(relu x), x < 0" $ do
    -- ∂out/∂√ = 1 and relu passes on 0, so the √ node's gradient is
    -- finite (1), but √'s slope at relu(x) = 0 is infinite: ∞ · 1 at the
    -- relu node, then 0 · ∞ = NaN would follow at x. The error names √.
    let ((r, sq), g) = built $ do
          r' <- unary Relu =<< input "x"
          sq' <- unary (PowC (1 / 2)) r'
          pure (r', sq')
    gradients g (inputs [("x", -1)]) sq `failsWith` \err ->
      err `shouldBe` BackwardError (NonFiniteGrad sq [0] 1 [1 / 0])
    nodeArgs (nodeAt g sq) `shouldBe` [r]

  it "reports an overflowing product at the node that passes it on" $ do
    -- h = x · 10^200 = 1, L = h · 10^200 = 10^200. All values are finite,
    -- and so is ∂L/∂h = 10^200, but ∂L/∂x = 10^200 · 10^200 overflows.
    let ((h, l), g) = built $ do
          x <- input "x"
          big <- constant 1e200
          h' <- binary Mul x big
          l' <- binary Mul h' big
          pure (h', l')
    gradients g (inputs [("x", 1e-200)]) l `failsWith` \err ->
      err `shouldBe` BackwardError (NonFiniteGrad h [1e-200, 1e200] 1e200 [1e200, 1e-200])

  it "reports an overflowing sum at the node whose share tips it over" $ do
    -- Two different users of x each pass on 10^308, which is finite; the
    -- sum is not. The users are visited in descending order, m2 first, so
    -- it is m1's share that overflows.
    let ((m1, l), g) = built $ do
          x <- input "x"
          big <- constant 1e308
          m1' <- binary Mul x big
          m2 <- binary Mul big x
          l' <- binary Add m1' m2
          pure (m1', l')
    gradients g (inputs [("x", 1e-308)]) l `failsWith` \case
      BackwardError (NonFiniteGrad i _ gi locals) -> do
        i `shouldBe` m1
        gi `shouldBe` 1
        locals `shouldSatisfy` all isFinite
      other -> expectationFailure (show other)

  it "passes forward errors through" $ do
    let (out, g) = built (unary Log =<< input "x")
    gradients g emptyEnv out `failsWith` \case
      ForwardError (UnboundLeaves _) -> pure ()
      other -> expectationFailure (show other)
    gradients g (inputs [("x", -1)]) out `failsWith` \case
      ForwardError (NonFiniteValue i _ _) -> i `shouldBe` out
      other -> expectationFailure (show other)

  it "takes relu's gradient at the kink to be 0, as micrograd does" $ do
    let ((x, out), g) = built $ do
          x' <- input "x"
          out' <- unary Relu x'
          pure (x', out')
    gradAt (gradsOf g (inputs [("x", 0)]) out) x `shouldBe` 0

-- ---------------------------------------------------------------------------
-- Properties over random graphs

propertySpec :: Spec
propertySpec = do
  -- Recipes of at most 30 steps: deeper random graphs mostly overflow or
  -- become too ill-conditioned for a finite difference.
  modifyMaxSize (const 15) . modifyMaxSuccess (const 500) $
    prop "agrees with central finite differences" gradientCheck

  prop "gives a gradient to exactly the nodes the output depends on" $
    withGraph $ \g outs -> forAll genEnv $ \env ->
      let out = firstOutput outs
       in case gradients g env out of
            Right (_, grads) ->
              IntMap.keys grads === map nodeIndex (Set.toAscList (reachable g [out]))
            Left _ -> discard

  prop "does not depend on the other outputs of the schedule" $
    withGraph $ \g outs -> forAll genEnv $ \env ->
      let out = firstOutput outs
          s = schedule g outs
       in case (evalSchedule s env, gradients g env out) of
            (Right vals, Right (_, grads)) -> backward s vals out === Right grads
            _ -> discard

-- | 'withGraph' gives at least one output; differentiate the first.
firstOutput :: [NodeId] -> NodeId
firstOutput (o : _) = o
firstOutput [] = error "withGraph gave no outputs"

-- | The gradient check (docs/06-core-backprop.md §5.1). For a random
-- graph and point, and for every input and parameter leaf @v@ the output
-- depends on:
--
-- > ∂out/∂v ≈ (out(v + h) - out(v - h)) / 2h,     h = 10⁻⁶ · max(1, |v|)
--
-- with a relative tolerance of @10⁻⁵@ (absolute below 1). Samples where
-- the finite difference cannot be trusted are discarded: see
-- 'trustworthy'.
gradientCheck :: SmoothRecipe -> Property
gradientCheck (SmoothRecipe steps) =
  forAll genEnv $ \env ->
    let (ids, g) = builtRecipe steps
        -- The node of the last step: it tends to depend on more of the
        -- graph than a node picked at random.
        out = last ids
        s = schedule g [out]
     in case forwardBackward s env out of
          _ | null ids -> discard -- only when shrinking: an empty recipe
          Left _ -> discard
          Right (vals, grads) ->
            let LeafGrads ins ps = leafGrads s grads
                checks =
                  [check ("input " <> show n) v (\x -> bindInput n x env) (ins Map.! n) | (n, v) <- Map.toList (envInputs env), Map.member n ins]
                    ++ [check ("param " <> show n) v (\x -> bindParam n x env) (ps Map.! n) | (n, v) <- Map.toList (envParams env), Map.member n ps]
                check what v rebind grad = do
                  let h = 1e-6 * max 1 (abs v)
                  plus <- either (const Nothing) Just (evalSchedule s (rebind (v + h)))
                  minus <- either (const Nothing) Just (evalSchedule s (rebind (v - h)))
                  let fd = (valueAt plus out - valueAt minus out) / (2 * h)
                  pure (what, grad, fd, [plus, minus])
             in case sequence checks of
                  Nothing -> discard -- a perturbed point is outside the domain
                  Just results
                    | null results -> discard -- nothing to check: no leaves
                    | not (trustworthy s (vals : concat [vs | (_, _, _, vs) <- results]) results) -> discard
                    | otherwise ->
                        classify (shared s) "with a shared node" $
                          classify (any isKink (scheduleNodes s)) "with relu or step" $
                          counterexample (show (nodes g)) $
                            counterexample ("output " <> show out <> ", " <> show env) $
                              conjoin
                                [ counterexample (what <> ": backprop " <> show grad <> ", finite difference " <> show fd) $
                                  abs (grad - fd) <= 1e-5 * max 1 (abs fd)
                                | (what, grad, fd, _) <- results
                                ]

-- | Whether a finite difference is a fair reference at this point:
--
-- * every value, at the point and at the perturbed points, is at most
--   @10³@ in magnitude, and so is every finite difference: this keeps
--   rounding error small (it grows with the size of the values divided by
--   @h@) and stays away from poles such as @1/x@ near @0@;
-- * every @relu@ and @step@ argument has the same sign, never 0, at the
--   point and at the perturbed points: the difference does not straddle a
--   kink or a jump.
trustworthy :: Schedule -> [Values] -> [(String, Double, Double, [Values])] -> Bool
trustworthy s valss results =
  all (all ((<= 1e3) . abs) . IntMap.elems) valss
    && all (\(_, _, fd, _) -> abs fd <= 1e3) results
    && all sameSide kinks
  where
    kinks = [a | entry@(_, Node {nodeArgs = [a]}) <- scheduleNodes s, isKink entry]
    sameSide a = case map (signum . (`valueAt` a)) valss of
      sgn : rest -> sgn /= 0 && all (== sgn) rest
      [] -> True

-- | Whether some node of the schedule is used more than once (by different
-- nodes, or twice by the same one), so that gradients have to be summed.
shared :: Schedule -> Bool
shared s = any (> 1) (Map.elems uses)
  where
    uses = Map.fromListWith (+) [(a, 1 :: Int) | (_, n) <- scheduleNodes s, a <- nodeArgs n]

isKink :: (NodeId, Node) -> Bool
isKink (_, n) = nodeOp n `elem` [Un Relu, Un Step]
