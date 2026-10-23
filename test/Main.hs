-- | Entry point of the test suite.
--
-- Every library module @HsGrad.X.Y@ gets a spec module @HsGrad.X.YSpec@
-- exporting @spec :: Spec@. Each one is imported here and listed below.
module Main
  ( main
  ) where

import HsGrad.Core.BackpropSpec qualified
import HsGrad.Core.EvalSpec qualified
import HsGrad.Core.GraphSpec qualified
import HsGrad.Core.OpSpec qualified
import HsGrad.Core.PrettySpec qualified
import HsGrad.Core.SimplifySpec qualified
import HsGrad.Core.SymbolicSpec qualified
import HsGrad.Data.CsvSpec qualified
import HsGrad.Data.DatasetSpec qualified
import HsGrad.Lang.CheckSpec qualified
import HsGrad.Lang.DiagnosticSpec qualified
import HsGrad.Lang.ParserSpec qualified
import HsGrad.Lang.ShapeSpec qualified
import HsGrad.Lang.SpanSpec qualified
import HsGrad.Lang.SyntaxSpec qualified
import HsGrad.Train.LoopSpec qualified
import HsGrad.Train.OptimSpec qualified
import HsGrad.Train.ParamSpec qualified
import HsGrad.Viz.DotSpec qualified
import HsGradSpec qualified
import Test.Hspec (describe, hspec)

main :: IO ()
main = hspec $ do
  describe "HsGrad" HsGradSpec.spec
  describe "HsGrad.Core.Op" HsGrad.Core.OpSpec.spec
  describe "HsGrad.Core.Graph" HsGrad.Core.GraphSpec.spec
  describe "HsGrad.Core.Eval" HsGrad.Core.EvalSpec.spec
  describe "HsGrad.Core.Backprop" HsGrad.Core.BackpropSpec.spec
  describe "HsGrad.Core.Symbolic" HsGrad.Core.SymbolicSpec.spec
  describe "HsGrad.Viz.Dot" HsGrad.Viz.DotSpec.spec
  describe "HsGrad.Core.Simplify" HsGrad.Core.SimplifySpec.spec
  describe "HsGrad.Core.Pretty" HsGrad.Core.PrettySpec.spec
  describe "HsGrad.Train.Param" HsGrad.Train.ParamSpec.spec
  describe "HsGrad.Train.Optim" HsGrad.Train.OptimSpec.spec
  describe "HsGrad.Train.Loop" HsGrad.Train.LoopSpec.spec
  describe "HsGrad.Lang.Span" HsGrad.Lang.SpanSpec.spec
  describe "HsGrad.Lang.Diagnostic" HsGrad.Lang.DiagnosticSpec.spec
  describe "HsGrad.Lang.Syntax" HsGrad.Lang.SyntaxSpec.spec
  describe "HsGrad.Lang.Parser" HsGrad.Lang.ParserSpec.spec
  describe "HsGrad.Lang.Shape" HsGrad.Lang.ShapeSpec.spec
  describe "HsGrad.Lang.Check" HsGrad.Lang.CheckSpec.spec
  describe "HsGrad.Data.Csv" HsGrad.Data.CsvSpec.spec
  describe "HsGrad.Data.Dataset" HsGrad.Data.DatasetSpec.spec
