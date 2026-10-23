{-# LANGUAGE OverloadedStrings #-}

-- | Trainable parameters: what they are, how they start, where they live.
--
-- Core sees parameters as scalar leaves with names like @W1[3,0]@
-- ("HsGrad.Core.Op"). The language declares them as whole vectors and
-- matrices with an initialiser (@docs/01-language.md@ §3):
--
-- > param W1 : Mat 16 2 ~ uniform(-1, 1)
-- > param b1 : Vec 16   = 0
--
-- This module connects the two:
--
-- * a 'ParamSpec' describes one @param@ declaration: name, 'Shape', 'Init'.
--   It is plain data, so the driver (step 15) can build it from the
--   checked program without Train knowing anything about the language;
-- * 'elementNames' gives the scalar leaf names of a spec, the names the
--   compiler (step 14) gives the param leaves (@01-language.md@ §8);
-- * 'Params', the parameter store, maps every leaf name to its current
--   value. It is exactly what 'HsGrad.Core.Eval.envParams' wants;
-- * 'initParams' fills the store from a seed, deterministically.
--
-- Randomness comes from @random@'s 'StdGen'. A run uses one seed (@--seed@)
-- for two independent purposes, initialisation and mini-batch sampling
-- ("HsGrad.Train.Loop"); 'streamGen' derives a separate generator for each.
-- See @docs/10-train.md@.
module HsGrad.Train.Param
  ( -- * Describing parameters
    Shape (..)
  , Init (..)
  , ParamSpec (..)
  , elementNames
  , shapeSize

    -- * The parameter store
  , Params
  , initParams
  , paramValues

    -- * Seeds and random streams
  , Seed
  , Stream (..)
  , streamGen

    -- * Sampling
  , sampleInit
  , standardNormal
  ) where

import Control.Monad.State.Strict (State, evalState, state)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Op (Name (..))
import System.Random (StdGen, mkStdGen, splitGen, uniformR)

-- ---------------------------------------------------------------------------
-- Describing parameters

-- | The shape of a parameter, as declared: @Real@, @Vec n@ or @Mat r c@.
--
-- This is deliberately not the language's type ("HsGrad.Lang.Shape", step
-- 12), which also has shape variables: by the time a parameter reaches
-- training, its shape is concrete.
data Shape
  = Scalar
  | Vector !Int
  | Matrix !Int !Int
  deriving (Eq, Show)

-- | How a parameter's elements start out (@01-language.md@ §3).
data Init
  = -- | @~ uniform(lo, hi)@: every element drawn uniformly from the
    -- interval between the two bounds (both included, in either order).
    Uniform !Double !Double
  | -- | @~ normal(mean, std)@: every element drawn from a normal
    -- distribution.
    Normal !Double !Double
  | -- | @= c@: every element equal to @c@. No randomness is used.
    Constant !Double
  deriving (Eq, Show)

-- | One @param@ declaration: @param W1 : Mat 16 2 ~ uniform(-1, 1)@ is
--
-- > ParamSpec "W1" (Matrix 16 2) (Uniform (-1) 1)
data ParamSpec = ParamSpec
  { specName :: !Text
  , specShape :: !Shape
  , specInit :: !Init
  }
  deriving (Eq, Show)

-- | The names of a parameter's scalar leaves, in row-major order: the
-- lowering of @01-language.md@ §8.
--
-- > elementNames (ParamSpec "b" Scalar …)        == ["b"]
-- > elementNames (ParamSpec "w" (Vector 3) …)    == ["w[0]", "w[1]", "w[2]"]
-- > elementNames (ParamSpec "W" (Matrix 2 2) …)  == ["W[0,0]", "W[0,1]", "W[1,0]", "W[1,1]"]
--
-- The compiler must create the leaves under exactly these names, or the
-- forward pass will find parameter leaves without a value.
elementNames :: ParamSpec -> [Name]
elementNames (ParamSpec base sh _) = map Name $ case sh of
  Scalar -> [base]
  Vector n -> [indexed [i] | i <- [0 .. n - 1]]
  Matrix r c -> [indexed [i, j] | i <- [0 .. r - 1], j <- [0 .. c - 1]]
  where
    indexed is = base <> "[" <> Text.intercalate "," (map (Text.pack . show) is) <> "]"

-- | The number of scalar elements of a shape.
shapeSize :: Shape -> Int
shapeSize sh = case sh of
  Scalar -> 1
  Vector n -> n
  Matrix r c -> r * c

-- ---------------------------------------------------------------------------
-- The parameter store

-- | The current value of every parameter leaf, by leaf name.
--
-- A plain map, so that it can be dropped into the environment as it is
-- (@env {envParams = ps}@) and updated with the gradients from
-- 'HsGrad.Core.Backprop.paramGrads', which are keyed the same way.
type Params = Map Name Double

-- | A fresh parameter store: every element of every spec, initialised from
-- the seed.
--
-- The random elements are drawn from one generator ('streamGen' with
-- 'InitStream'), spec by spec in list order and element by element in the
-- order of 'elementNames'. So the same seed and the same specs always give
-- the same store; list the specs in declaration order to get what the
-- @.hsg@ file means. Specs with a 'Constant' initialiser draw nothing.
--
-- Spec names are expected to be distinct (the checker reports duplicate
-- declarations); if two specs share a leaf name, the later one wins.
initParams :: Seed -> [ParamSpec] -> Params
initParams seed specs =
  Map.fromList (evalState (concat <$> mapM initSpec specs) (streamGen seed InitStream))
  where
    initSpec spec = mapM (\n -> (,) n <$> sampleInit (specInit spec)) (elementNames spec)

-- | The values of one parameter's elements, in the order of
-- 'elementNames', if the store has all of them. For printing a trained
-- parameter in its declared shape (@--save-params@).
paramValues :: ParamSpec -> Params -> Maybe [Double]
paramValues spec ps = mapM (`Map.lookup` ps) (elementNames spec)

-- ---------------------------------------------------------------------------
-- Seeds and random streams

-- | The seed of a run (@--seed@).
type Seed = Int

-- | What a run uses randomness for.
data Stream
  = -- | Initial parameter values ('initParams').
    InitStream
  | -- | Choosing the rows of each mini-batch ("HsGrad.Train.Loop").
    BatchStream
  deriving (Eq, Show, Enum, Bounded)

-- | The generator for one purpose of a run with this seed.
--
-- The seed's generator is split in two, and each purpose gets its own half.
-- The two streams are independent: changing the batch size (and so how
-- many numbers the batches use) does not change the initial parameters,
-- and adding a parameter does not change which rows the batches pick.
streamGen :: Seed -> Stream -> StdGen
streamGen seed s = case s of
  InitStream -> forInit
  BatchStream -> forBatches
  where
    (forInit, forBatches) = splitGen (mkStdGen seed)

-- ---------------------------------------------------------------------------
-- Sampling

-- | One element's initial value, drawn with the generator in the state.
sampleInit :: Init -> State StdGen Double
sampleInit i = case i of
  Uniform lo hi -> state (uniformR (lo, hi))
  Normal mean std -> (\z -> mean + std * z) <$> standardNormal
  Constant c -> pure c

-- | A sample from the standard normal distribution (mean 0, standard
-- deviation 1), by the Box–Muller transform.
--
-- Two independent uniform numbers @u₁@ in @(0, 1]@ and @u₂@ in @[0, 1]@
-- give
--
-- > z = √(-2 ln u₁) · cos(2π u₂)
--
-- which is exactly standard normal. (The same @u₁@, @u₂@ with @sin@ give a
-- second, independent sample; we throw it away to keep the state simple.)
-- @u₁@ must not be @0@, where the logarithm is infinite: 'uniformR'
-- includes both ends of its range, so a @0@ is simply drawn again.
standardNormal :: State StdGen Double
standardNormal = do
  u1 <- positiveUnit
  u2 <- state (uniformR (0, 1))
  pure (sqrt (-2 * log u1) * cos (2 * pi * u2))
  where
    positiveUnit = do
      u <- state (uniformR (0, 1))
      if u > 0 then pure u else positiveUnit
