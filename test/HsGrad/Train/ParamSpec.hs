{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Train.Param".
--
-- * Element names follow the lowering of @docs/01-language.md@ §8.
-- * Initialisation is deterministic per seed, respects the ranges, draws
--   in a documented order (so that a constant parameter does not shift
--   the random ones), and 'Normal' has the right mean and spread.
-- * The two random streams of a seed are different.
module HsGrad.Train.ParamSpec
  ( spec
  ) where

import Control.Monad (replicateM)
import Control.Monad.State.Strict (evalState)
import Data.Map.Strict qualified as Map
import HsGrad.Train.Loop (sampleBatch)
import HsGrad.Train.Param
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Gen, choose, forAll, oneof, (.&&.), (===))

-- | Shapes small enough to enumerate.
genShape :: Gen Shape
genShape =
  oneof
    [ pure Scalar
    , Vector <$> choose (1, 6)
    , Matrix <$> choose (1, 4) <*> choose (1, 4)
    ]

-- | The mean and (population) standard deviation of some numbers.
meanStd :: [Double] -> (Double, Double)
meanStd xs = (m, sqrt (sum [(x - m) ^ (2 :: Int) | x <- xs] / n))
  where
    n = fromIntegral (length xs)
    m = sum xs / n

spec :: Spec
spec = do
  describe "elementNames" $ do
    it "a Real is one leaf with the parameter's name" $
      elementNames (ParamSpec "b2" Scalar (Constant 0)) `shouldBe` ["b2"]
    it "a vector has one indexed leaf per element" $
      elementNames (ParamSpec "w" (Vector 3) (Constant 0)) `shouldBe` ["w[0]", "w[1]", "w[2]"]
    it "a matrix is row-major, with [row,column] indices" $
      elementNames (ParamSpec "W1" (Matrix 2 3) (Constant 0))
        `shouldBe` ["W1[0,0]", "W1[0,1]", "W1[0,2]", "W1[1,0]", "W1[1,1]", "W1[1,2]"]
    prop "has shapeSize distinct names" $ forAll genShape $ \sh ->
      let names = elementNames (ParamSpec "p" sh (Constant 0))
       in (length names, Map.size (Map.fromList [(n, ()) | n <- names])) === (shapeSize sh, shapeSize sh)

  describe "initParams" $ do
    let specs =
          [ ParamSpec "W" (Matrix 3 2) (Uniform (-1) 1)
          , ParamSpec "b" (Vector 3) (Constant 0)
          , ParamSpec "v" (Vector 4) (Normal 0 1)
          ]
    it "gives every element of every spec a value" $
      Map.keys (initParams 0 specs) `shouldBe` Map.keys (Map.fromList [(n, ()) | s <- specs, n <- elementNames s])
    prop "is deterministic for a seed" $ \seed ->
      initParams seed specs === initParams seed specs
    it "depends on the seed" $
      initParams 1 specs `shouldSatisfy` (/= initParams 2 specs)
    prop "uniform values lie between the bounds, in either order" $ \seed lo hi ->
      let ps = initParams seed [ParamSpec "u" (Vector 50) (Uniform lo hi)]
       in all (\x -> x >= min lo hi && x <= max lo hi) (Map.elems ps)
    prop "constants are exact" $ \seed c ->
      Map.elems (initParams seed [ParamSpec "c" (Matrix 2 2) (Constant c)]) === replicate 4 c
    prop "a constant parameter draws nothing" $ \seed ->
      let u = ParamSpec "u" (Vector 5) (Uniform 0 1)
          c = ParamSpec "c" (Vector 5) (Constant 3)
       in Map.delete "c[0]" (Map.filterWithKey (\k _ -> k `elem` elementNames u) (initParams seed [c, u]))
            === initParams seed [u]
    prop "explicit values are exact, in element order, and draw nothing" $ \seed ->
      let u = ParamSpec "u" (Vector 5) (Uniform 0 1)
          v = ParamSpec "v" (Matrix 2 2) (Values [1, 2, 3, 4])
          ps = initParams seed [v, u]
       in map (ps Map.!) (elementNames v) === [1, 2, 3, 4]
            .&&. Map.filterWithKey (\k _ -> k `elem` elementNames u) ps === initParams seed [u]
    prop "earlier parameters do not depend on later ones" $ \seed ->
      let a = ParamSpec "a" (Vector 3) (Uniform (-1) 1)
          b = ParamSpec "b" (Vector 3) (Normal 0 1)
       in Map.filterWithKey (\k _ -> k `elem` elementNames a) (initParams seed [a, b])
            === initParams seed [a]
    it "normal values have the requested mean and standard deviation" $ do
      let xs = Map.elems (initParams 42 [ParamSpec "n" (Vector 20000) (Normal 2 0.5)])
          (m, s) = meanStd xs
      -- Standard error of the mean: 0.5 / √20000 ≈ 0.0035.
      abs (m - 2) `shouldSatisfy` (< 0.02)
      abs (s - 0.5) `shouldSatisfy` (< 0.02)
    it "standard normal samples fall within one deviation about 68% of the time" $ do
      let zs = evalState (replicateM 20000 standardNormal) (streamGen 3 InitStream)
          inside = length (filter (\z -> abs z < 1) zs)
      fromIntegral inside / (20000 :: Double) `shouldSatisfy` (\f -> abs (f - 0.6827) < 0.01)
    it "a normal with standard deviation 0 is its mean" $
      Map.elems (initParams 0 [ParamSpec "n" (Vector 3) (Normal 7 0)]) `shouldBe` [7, 7, 7]

  describe "paramValues" $ do
    let w = ParamSpec "w" (Vector 3) (Constant 0)
    it "reads a parameter's elements in order" $
      paramValues w (Map.fromList [("w[2]", 3), ("w[0]", 1), ("w[1]", 2)]) `shouldBe` Just [1, 2, 3]
    it "is Nothing when an element is missing" $
      paramValues w (Map.fromList [("w[0]", 1)]) `shouldBe` Nothing

  describe "streamGen" $
    prop "initialisation and batches get different streams" $ \seed ->
      let draws s = evalState (sampleBatch 1000 5) (streamGen seed s)
       in draws InitStream /= draws BatchStream
