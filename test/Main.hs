-- | Entry point of the test suite.
--
-- Every library module @HsGrad.X.Y@ gets a spec module @HsGrad.X.YSpec@
-- exporting @spec :: Spec@. Each one is imported here and listed below.
module Main
  ( main
  ) where

import HsGradSpec qualified
import Test.Hspec (describe, hspec)

main :: IO ()
main = hspec $ do
  describe "HsGrad" HsGradSpec.spec
