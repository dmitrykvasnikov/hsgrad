{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Cli": command lines as @optparse-applicative@ parses
-- them ('parseArgs' is 'execParserPure', no IO).
--
-- * Every command with its options, and the defaults.
-- * The re-assembled command line and the spans of its values, which
--   diagnostics point at.
-- * Wrong command lines fail with a message; @--help@ has examples.
-- * The readers: schedules, comma-separated names.
module HsGrad.CliSpec
  ( spec
  ) where

import Data.List (isInfixOf)
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Cli (parseArgs, readSchedule, splitNames)
import HsGrad.Driver
import HsGrad.Lang.Span (Pos (..), SrcSpan (..))
import HsGrad.Train.Optim (LrSchedule (..))
import Options.Applicative (ParserResult (..), renderFailure)
import System.Exit (ExitCode (..))
import Test.Hspec hiding (Arg)

spec :: Spec
spec = do
  describe "commands" commandSpec
  describe "the command line and its spans" spanSpec
  describe "errors and help" failureSpec
  describe "readers" readerSpec

-- | The options of a command line that must parse.
parsed :: [String] -> IO Options
parsed args = case parseArgs args of
  Success o -> pure o
  Failure f -> fail (fst (renderFailure f "hsgrad"))
  CompletionInvoked _ -> fail "completion"

-- | The text of the arguments.
texts :: [Arg] -> [Text]
texts = map argText

commandSpec :: Spec
commandSpec = do
  it "parses check" $ do
    o <- parsed ["check", "examples/xor.hsg"]
    optCommand o `shouldBe` Check
    argText (optFile o) `shouldBe` "examples/xor.hsg"
    optColour o `shouldBe` ColourAuto

  it "parses eval with --set and --show, in any order" $ do
    o <- parsed ["eval", "f.hsg", "--show", "L", "--set", "a=3", "--set", "x=[1, 2]", "--show", "e", "--seed", "7"]
    case optCommand o of
      Eval e -> do
        texts (bindSet (evalBindings e)) `shouldBe` ["a=3", "x=[1, 2]"]
        texts (evalShow e) `shouldBe` ["L", "e"]
        bindSeed (evalBindings e) `shouldBe` 7
        bindParams (evalBindings e) `shouldBe` Nothing
      c -> expectationFailure (show c)

  it "parses derive, splitting --wrt at commas outside brackets" $ do
    o <- parsed ["derive", "f.hsg", "--of", "loss", "--wrt", "W1[0,1],b2", "--simplify", "--dot", "g.dot", "--format", "dot"]
    case optCommand o of
      Derive d -> do
        argText (deriveOf d) `shouldBe` "loss"
        texts (deriveWrt d) `shouldBe` ["W1[0,1]", "b2"]
        deriveSimplify d `shouldBe` True
        deriveDot d `shouldBe` Just "g.dot"
        deriveFormat d `shouldBe` DotFormat
      c -> expectationFailure (show c)

  it "parses plot" $ do
    o <- parsed ["plot", "f.hsg", "--of", "y", "--dot", "-", "--cluster", "--color", "never"]
    case optCommand o of
      Plot p -> (argText (plotOf p), plotDot p, plotCluster p) `shouldBe` ("y", "-", True)
      c -> expectationFailure (show c)
    optColour o `shouldBe` ColourNever

  it "parses train with every option" $ do
    o <- parsed ["train", "f.hsg", "--steps", "50", "--lr", "1e-2", "--lr-schedule", "step:10:0.5", "--batch", "16", "--seed", "3", "--report-every", "5", "--save-params", "p.params", "--set", "a=1"]
    case optCommand o of
      Train t -> do
        (trainingSteps t, trainingRate t, trainingSchedule t, trainingBatch t) `shouldBe` (50, 0.01, StepDecay 10 0.5, Just 16)
        (trainingSeed t, trainingReportEvery t, trainingSaveParams t) `shouldBe` (3, 5, Just "p.params")
        texts (trainingSet t) `shouldBe` ["a=1"]
      c -> expectationFailure (show c)

  it "has defaults for train" $ do
    o <- parsed ["train", "f.hsg"]
    case optCommand o of
      Train t -> do
        (trainingSteps t, trainingRate t, trainingSchedule t, trainingBatch t) `shouldBe` (100, 0.1, ConstantRate, Nothing)
        (trainingSeed t, trainingReportEvery t, trainingSaveParams t) `shouldBe` (defaultSeed, 10, Nothing)
      c -> expectationFailure (show c)

  it "parses predict" $ do
    o <- parsed ["predict", "f.hsg", "--params", "p.params", "--set", "x=[0, 1]"]
    case optCommand o of
      Predict p -> (argText (predictParams p), texts (predictSet p)) `shouldBe` ("p.params", ["x=[0, 1]"])
      c -> expectationFailure (show c)

spanSpec :: Spec
spanSpec = do
  it "re-assembles the command line from the values, in the parsers' order" $ do
    o <- parsed ["eval", "f.hsg", "--show", "L", "--set", "x=[1, 2]"]
    optCommandLine o `shouldBe` "hsgrad eval f.hsg --set x=[1, 2] --show L"

  it "gives every value the span of its text in that line" $ do
    o <- parsed ["derive", "f.hsg", "--of", "L", "--wrt", "a,W[0,1]", "--set", "b=2"]
    let line = optCommandLine o
        slice a = case argSpan a of
          SrcSpan file (Pos 1 c1) (Pos 1 c2) | file == commandLineFile -> Text.take (c2 - c1) (Text.drop (c1 - 1) line)
          sp -> error (show sp)
    case optCommand o of
      Derive d -> do
        map slice ([optCommandWord o, optFile o, deriveOf d] <> deriveWrt d <> bindSet (deriveBindings d))
          `shouldBe` ["derive", "f.hsg", "L", "a", "W[0,1]", "b=2"]
      c -> expectationFailure (show c)

failureSpec :: Spec
failureSpec = do
  let failsWith args needle = case parseArgs args of
        Failure f -> fst (renderFailure f "hsgrad") `shouldSatisfy` (needle `isInfixOf`)
        _ -> expectationFailure ("parsed: " <> unwords args)
  it "rejects a step count below 1" $ failsWith ["train", "f.hsg", "--steps", "0"] "at least 1"
  it "rejects a learning rate that isn't a positive number" $ failsWith ["train", "f.hsg", "--lr", "NaN"] "positive number"
  it "rejects a bad schedule" $ failsWith ["train", "f.hsg", "--lr-schedule", "step:0:0.5"] "expected const"
  it "rejects --set without =" $ failsWith ["eval", "f.hsg", "--set", "x"] "NAME=VALUE"
  it "requires --of for derive" $ failsWith ["derive", "f.hsg", "--wrt", "a"] "--of"
  it "requires --params for predict" $ failsWith ["predict", "f.hsg"] "--params"
  it "rejects an unknown format" $ failsWith ["derive", "f.hsg", "--of", "L", "--wrt", "a", "--format", "svg"] "expected one of text, dot"
  it "rejects an unknown command" $ failsWith ["fit", "f.hsg"] "fit"
  it "has examples in the help of every command" $
    mapM_
      ( \cmd -> case parseArgs [cmd, "--help"] of
          Failure f -> do
            let (text, code) = renderFailure f "hsgrad"
            code `shouldBe` ExitSuccess
            text `shouldSatisfy` isInfixOf ("Examples:\n  hsgrad " <> cmd)
          _ -> expectationFailure cmd
      )
      ["check", "eval", "derive", "plot", "train", "predict"]

readerSpec :: Spec
readerSpec = do
  it "reads schedules" $ do
    readSchedule "const" `shouldBe` Right ConstantRate
    readSchedule "linear:0.1" `shouldBe` Right (LinearTo 0.1)
    readSchedule "step:25:0.5" `shouldBe` Right (StepDecay 25 0.5)
    mapM_ (\s -> readSchedule s `shouldSatisfy` either (const True) (const False)) ["linear", "linear:-1", "step:2", "step:0:1", "step:2:0", "constant"]

  it "splits names at commas outside brackets, with offsets" $ do
    splitNames "a,b,c" `shouldBe` [(0, "a"), (2, "b"), (4, "c")]
    splitNames "W1[0,1],b" `shouldBe` [(0, "W1[0,1]"), (8, "b")]
    splitNames "x" `shouldBe` [(0, "x")]
