-- | Generate @examples/data/moons.csv@, the dataset of @examples/moons.hsg@:
-- scikit-learn's @make_moons(n_samples=100, noise=0.1)@ with labels -1 and
-- +1, as in micrograd's demo, but from a fixed seed of our own.
--
-- Two interleaving half circles of 50 points each:
--
-- * the upper moon, label -1: @(cos t, sin t)@;
-- * the lower moon, label +1: @(1 - cos t, 0.5 - sin t)@;
--
-- for 50 values of @t@ evenly spaced from 0 to π, plus Gaussian noise with
-- standard deviation 0.1 on every coordinate. Rows alternate between the
-- moons. The noise comes from "HsGrad.Train.Param"'s 'standardNormal' with
-- seed 1337, so the file only changes if the @random@ package (pinned by
-- the index-state in @cabal.project@) or that sampler changes.
--
-- Run it from the package directory:
--
-- > cabal exec -v0 -- runghc --ghc-arg=-package=hsgrad scripts/MakeMoons.hs > examples/data/moons.csv
module Main
  ( main
  ) where

import Control.Monad (replicateM)
import Control.Monad.State.Strict (evalState)
import HsGrad.Train.Param (standardNormal)
import System.Random (mkStdGen)
import Text.Printf (printf)

-- | Points per moon.
perMoon :: Int
perMoon = 50

noise :: Double
noise = 0.1

main :: IO ()
main = do
  putStrLn "x1,x2,label"
  mapM_ row (concat [[upper k, lower k] | k <- [0 .. perMoon - 1]] `withNoise` normals)
  where
    angle k = pi * fromIntegral k / fromIntegral (perMoon - 1)
    upper k = (cos (angle k), sin (angle k), -1 :: Int)
    lower k = (1 - cos (angle k), 0.5 - sin (angle k), 1)
    normals = evalState (replicateM (4 * perMoon) standardNormal) (mkStdGen 1337)
    withNoise ((a, b, label) : points) (n1 : n2 : ns) = (a + noise * n1, b + noise * n2, label) : withNoise points ns
    withNoise _ _ = []
    row (a, b, label) = printf "%.6f,%.6f,%d\n" a b label
