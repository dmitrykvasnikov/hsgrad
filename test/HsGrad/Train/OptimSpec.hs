{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Train.Optim".
--
-- * The three schedules give the documented rates; @linear@ reproduces
--   micrograd's demo.
-- * 'sgdStep' moves every parameter by exactly @−η·g@, leaves parameters
--   without a gradient alone, and on a real graph (with gradients from
--   backprop) a small step lowers the loss.
module HsGrad.Train.OptimSpec
  ( spec
  ) where

import Data.Map.Strict qualified as Map
import HsGrad.Core.Backprop (LeafGrads (..), gradients, leafGrads)
import HsGrad.Core.Eval (Env (..), evaluate, schedule, valueAt)
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Test.Graphs (built)
import HsGrad.Train.Optim
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Positive (..), choose, counterexample, forAll, (==>))

-- | @loss = (w·x − y)² + (v − 1)²@ with inputs @x = 2@, @y = 1@.
smallLoss :: (NodeId, Graph)
smallLoss = built $ do
  w <- param "w"
  v <- param "v"
  x <- input "x"
  y <- input "y"
  one <- constant 1
  a <- binary Mul w x >>= \wx -> binary Sub wx y >>= unary (PowC 2)
  b <- binary Sub v one >>= unary (PowC 2)
  binary Add a b

lossAt :: Map.Map Name Double -> Double
lossAt ps = case evaluate g env [l] of
  Right vals -> valueAt vals l
  Left err -> error (show err)
  where
    (l, g) = smallLoss
    env = inputs ps

-- | The inputs @x = 2@, @y = 1@ and these parameters.
inputs :: Map.Map Name Double -> Env
inputs = Env (Map.fromList [("x", 2), ("y", 1)])

-- | The parameter gradients of 'smallLoss', by backprop.
gradsAt :: Map.Map Name Double -> Map.Map Name Double
gradsAt ps = case gradients g (inputs ps) l of
  Right (_, gs) -> paramGrads (leafGrads (schedule g [l]) gs)
  Left err -> error (show err)
  where
    (l, g) = smallLoss

spec :: Spec
spec = do
  describe "learningRate" $ do
    it "const: the base rate at every step" $
      map (learningRate (Sgd 0.5 ConstantRate) 10) [0, 5, 9] `shouldBe` [0.5, 0.5, 0.5]
    it "linear: the base rate at step 0, END at step n" $
      map (learningRate (Sgd 1 (LinearTo 0.1)) 100) [0, 100] `shouldBe` [1, 0.1]
    it "linear: micrograd's 1.0 - 0.9*k/100" $
      [learningRate (Sgd 1 (LinearTo 0.1)) 100 k - (1 - 0.9 * fromIntegral k / 100) | k <- [0 .. 99]]
        `shouldSatisfy` all (\d -> abs d < 1e-12)
    it "linear: halfway is halfway" $
      learningRate (Sgd 2 (LinearTo 0)) 10 5 `shouldBe` 1
    it "linear with no steps is the base rate" $
      learningRate (Sgd 2 (LinearTo 0)) 0 0 `shouldBe` 2
    it "step: multiplies by F every K steps" $
      map (learningRate (Sgd 1 (StepDecay 10 0.5)) 100) [0, 9, 10, 19, 20, 35]
        `shouldBe` [1, 1, 0.5, 0.5, 0.25, 0.125]
    it "step with K < 1 counts as K = 1" $
      map (learningRate (Sgd 1 (StepDecay 0 0.5)) 10) [0, 1, 2] `shouldBe` [1, 0.5, 0.25]

  describe "sgdStep" $ do
    it "moves each parameter by −η·gradient" $
      sgdStep 0.1 (Map.fromList [("a", 2), ("b", -4)]) (Map.fromList [("a", 1), ("b", 1)])
        `shouldBe` Map.fromList [("a", 1 - 0.1 * 2), ("b", 1 + 0.1 * 4)]
    it "leaves a parameter without a gradient alone" $
      sgdStep 0.1 (Map.fromList [("a", 2)]) (Map.fromList [("a", 1), ("c", 5)])
        `shouldBe` Map.fromList [("a", 0.8), ("c", 5)]
    it "ignores gradients of names that are not parameters" $
      sgdStep 0.1 (Map.fromList [("x", 2)]) (Map.fromList [("a", 1)])
        `shouldBe` Map.fromList [("a", 1)]
    prop "moves against the gradient" $ \(Positive eta) p g ->
      let p' = sgdStep eta (Map.singleton "a" g) (Map.singleton "a" p) Map.! "a"
       in counterexample (show p') ((p' - p) * g <= 0)
    prop "a small step lowers the loss" $
      forAll ((,) <$> choose (-3, 3) <*> choose (-3, 3)) $ \(w, v) ->
        let ps = Map.fromList [("w", w), ("v", v)]
            grads = gradsAt ps
            norm = sum (map abs (Map.elems grads))
         in norm > 1e-6 ==> lossAt (sgdStep 1e-3 grads ps) < lossAt ps
    it "one step on (w·x − y)² + (v − 1)² by hand" $ do
      -- At w = 0, v = 0: ∂/∂w = 2(w·x − y)·x = −4, ∂/∂v = 2(v − 1) = −2.
      let ps = Map.fromList [("w", 0), ("v", 0)]
          grads = gradsAt ps
          ps' = sgdStep 0.25 grads ps
      grads `shouldBe` Map.fromList [("v", -2), ("w", -4)]
      ps' `shouldBe` Map.fromList [("v", 0.5), ("w", 1)]
      (lossAt ps, lossAt ps') `shouldBe` (2, 1.25)
