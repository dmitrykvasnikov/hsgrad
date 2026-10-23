-- | Entry point of the test suite.
--
-- Every library module @HsGrad.X.Y@ gets a spec module @HsGrad.X.YSpec@
-- exporting @spec :: Spec@. Each one is imported here and listed below.
module Main
  ( main
  ) where

import HsGrad.Core.EvalSpec qualified
import HsGrad.Core.GraphSpec qualified
import HsGrad.Core.OpSpec qualified
import HsGradSpec qualified
import Test.Hspec (describe, hspec)

main :: IO ()
main = hspec $ do
  describe "HsGrad" HsGradSpec.spec
  describe "HsGrad.Core.Op" HsGrad.Core.OpSpec.spec
  describe "HsGrad.Core.Graph" HsGrad.Core.GraphSpec.spec
  describe "HsGrad.Core.Eval" HsGrad.Core.EvalSpec.spec
