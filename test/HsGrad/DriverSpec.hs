{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Driver" and "HsGrad.Driver.ParamsFile": whole
-- commands, from a command line to their output, without a process.
--
-- * Every command on the examples: @derive@ prints @01-language.md@ §1.1
--   (simplified and not), expands vectors and matrices in @--wrt@ and
--   shares subexpressions; @eval@ prints values and takes @--set@;
--   @plot@ and @derive --dot@ produce DOT that Graphviz accepts; @train@
--   reaches accuracy 1 on XOR with a fixed seed and saves parameters that
--   @predict@ reads back; @check@ summarises.
-- * Parameter files: saving and reading gives back exactly the same
--   numbers (a property), hand-written files work.
-- * Golden files of whole runs, @test/golden/run/*.hsg@: runtime errors
--   (E0501–E0505) and problems with command-line values, rendered as on
--   stderr. Each file's first line, @-- args: …@, is its command line.
module HsGrad.DriverSpec
  ( spec
  ) where

import Control.Monad (forM_)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Cli (parseArgs)
import HsGrad.Driver
import HsGrad.Driver.ParamsFile (readParams, renderParams)
import HsGrad.Lang.Check (checkedParams)
import HsGrad.Lang.Compile (Compiled (..), compileProgram)
import HsGrad.Lang.Diagnostic (Code (..), codeText, defaultRenderOptions, diagCode)
import HsGrad.Test.Golden (goldenFiles)
import HsGrad.Test.Graphviz (toSvg, withDot)
import HsGrad.Train.Param (elementNames, initParams)
import Options.Applicative (ParserResult (..))
import System.Directory (createDirectoryIfMissing, getTemporaryDirectory, listDirectory)
import System.FilePath ((</>))
import Test.Hspec hiding (Arg)
import Test.QuickCheck (Gen, choose, forAll, frequency, vectorOf)
import Test.Hspec.QuickCheck (prop)

spec :: Spec
spec = do
  describe "derive" deriveSpec
  describe "eval" evalSpec
  describe "plot" plotSpec
  describe "train and predict" trainSpec
  describe "check" checkSpec
  describe "parameter files" paramsSpec
  describe "golden runs" goldenSpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Parse a command line and run it.
transcriptOf :: [String] -> IO Transcript
transcriptOf args = case parseArgs args of
  Success opts -> execute opts
  _ -> fail ("the command line does not parse: " <> unwords args)

-- | The output lines of a run that must succeed without diagnostics.
outputOf :: [String] -> IO [Text]
outputOf args = do
  t <- transcriptOf args
  okRun t
  pure (transcriptLines t)

-- | Expect success without diagnostics.
okRun :: Transcript -> Expectation
okRun t = case (transcriptResult t, transcriptDiagnostics t) of
  (Ok (), []) -> pure ()
  _ -> expectationFailure (Text.unpack (renderTranscript defaultRenderOptions t))

-- | The one file a run writes.
writtenFile :: Transcript -> IO Text
writtenFile t = case transcriptFiles t of
  [(_, text)] -> pure text
  fs -> fail ("expected one file, got " <> show (map fst fs))

-- ---------------------------------------------------------------------------
-- derive

deriveSpec :: Spec
deriveSpec = do
  it "prints the simplified gradients of intro.hsg as in 01-language.md §1.1" $
    outputOf ["derive", "examples/intro.hsg", "--of", "L", "--wrt", "a,b,c", "--simplify", "--format", "text"]
      `shouldReturn` ["∂L/∂a = -2·b  = 6", "∂L/∂b = -2·a  = -4", "∂L/∂c = -2    = -2"]

  it "prints the gradient graph as it is without --simplify" $
    -- The symbolic sweep of step 7 already gives -2·b: ∂L/∂a is the node
    -- f·b, with f the constant -2 of the source.
    outputOf ["derive", "examples/intro.hsg", "--of", "L", "--wrt", "a,e"]
      `shouldReturn` ["∂L/∂a = -2·b  = 6", "∂L/∂e = -2    = -2"]

  it "shares the common factor of neuron.hsg's gradients" $ do
    ls <- outputOf ["derive", "examples/neuron.hsg", "--of", "y", "--wrt", "w,b", "--simplify"]
    map (Text.takeWhile (/= ' ')) (take 4 ls) `shouldBe` ["∂y/∂w[0]", "∂y/∂w[1]", "∂y/∂w[2]", "∂y/∂b"]
    [Text.strip (Text.takeWhile (/= '=') (Text.drop 11 l)) | l <- take 4 ls] `shouldBe` ["x[0]·t1", "x[1]·t1", "x[2]·t1", "t1"]
    drop 4 ls `shouldBe` ["where", "  t1 = 1 - tanh(x[0]·w[0] + x[1]·w[1] + x[2]·w[2] + b)²"]

  it "expands a matrix in --wrt to its elements, row-major" $ do
    ls <- outputOf ["derive", "examples/xor.hsg", "--of", "loss", "--wrt", "W1"]
    map (Text.takeWhile (/= ' ')) (takeWhile (/= "where") ls)
      `shouldBe` ["∂loss/∂W1[" <> i <> "," <> j <> "]" | i <- ["0", "1", "2", "3"], j <- ["0", "1"]]

  it "takes single elements and mixes them with whole names" $ do
    ls <- outputOf ["derive", "examples/xor.hsg", "--of", "loss", "--wrt", "W1[1,0],b2", "--simplify"]
    map (Text.takeWhile (/= ' ')) (takeWhile (/= "where") ls) `shouldBe` ["∂loss/∂W1[1,0]", "∂loss/∂b2"]

  it "gives the derivative of an unrelated leaf as 0" $
    outputOf ["derive", "examples/intro.hsg", "--of", "e", "--wrt", "c"]
      `shouldReturn` ["∂e/∂c = 0  = 0"]

  it "leaves out the values when an input has none" $ do
    ls <- outputOf ["derive", "test/golden/run/missing-input.hsg", "--of", "y", "--wrt", "x,c"]
    ls `shouldBe` ["∂y/∂x[0] = c", "∂y/∂x[1] = c", "∂y/∂c    = x[0] + x[1]"]

  it "takes the values of --set" $
    outputOf ["derive", "examples/intro.hsg", "--of", "L", "--wrt", "a", "--simplify", "--set", "b=5"]
      `shouldReturn` ["∂L/∂a = -2·b  = -10"]

  it "prints DOT with --format dot, and writes it with --dot" $ do
    t <- transcriptOf ["derive", "examples/neuron.hsg", "--of", "y", "--wrt", "w", "--simplify", "--format", "dot", "--dot", "grad.dot"]
    okRun t
    dot <- writtenFile t
    Text.unlines (transcriptLines t) `shouldBe` dot
    map fst (transcriptFiles t) `shouldBe` ["grad.dot"]
    -- Each gradient is highlighted with its name.
    forM_ ["∂y/∂w[0]", "∂y/∂w[1]", "∂y/∂w[2]"] $ \n -> dot `shouldSatisfy` Text.isInfixOf n
    withDot $ \exe -> () <$ toSvg exe dot

-- ---------------------------------------------------------------------------
-- eval

evalSpec :: Spec
evalSpec = do
  it "shows every let of intro.hsg" $
    outputOf ["eval", "examples/intro.hsg"] `shouldReturn` ["e = -6", "d = 4", "L = -8"]

  it "takes --set and --show" $
    outputOf ["eval", "examples/intro.hsg", "--set", "a=3", "--set", "c=1", "--show", "L", "--show", "a"]
      `shouldReturn` ["L = 16", "a = 3"]

  it "shows vectors, elements and the loss" $ do
    ls <- outputOf ["eval", "examples/neuron.hsg", "--set", "x=[1, 0, -1]", "--show", "x", "--show", "x[2]", "--show", "b"]
    ls `shouldBe` ["x    = [1, 0, -1]", "x[2] = -1", "b    = 0"]
    w1 <- outputOf ["eval", "examples/xor.hsg", "--show", "W1"]
    map (Text.take 7) w1 `shouldBe` ["W1 = [["]

  it "shows the loss of a program without lets, over the whole dataset" $ do
    ls <- outputOf ["eval", "examples/xor.hsg", "--seed", "1"]
    map (Text.take 7) ls `shouldBe` ["loss = "]

  it "reads parameters from a file" $
    outputOf ["eval", "test/golden/run/predict.hsg", "--params", "test/golden/run/net.params", "--show", "w", "--show", "b"]
      `shouldReturn` ["w = [0.5, -1]", "b = 0.25"]

-- ---------------------------------------------------------------------------
-- plot

plotSpec :: Spec
plotSpec = do
  it "draws neuron.hsg with values and gradients, as DOT that Graphviz accepts" $ do
    t <- transcriptOf ["plot", "examples/neuron.hsg", "--of", "y", "--dot", "neuron.dot"]
    okRun t
    transcriptLines t `shouldBe` []
    map fst (transcriptFiles t) `shouldBe` ["neuron.dot"]
    dot <- writtenFile t
    dot `shouldSatisfy` Text.isPrefixOf "digraph"
    dot `shouldSatisfy` Text.isInfixOf "data "
    dot `shouldSatisfy` Text.isInfixOf "grad "
    withDot $ \exe -> () <$ toSvg exe dot

  it "draws the loss of xor.hsg in clusters" $ do
    t <- transcriptOf ["plot", "examples/xor.hsg", "--of", "loss", "--dot", "-", "--cluster"]
    okRun t
    dot <- writtenFile t
    dot `shouldSatisfy` Text.isInfixOf "subgraph cluster"
    withDot $ \exe -> () <$ toSvg exe dot

  it "draws names only when an input has no value" $ do
    t <- transcriptOf ["plot", "test/golden/run/missing-input.hsg", "--of", "y", "--dot", "y.dot"]
    okRun t
    dot <- writtenFile t
    dot `shouldNotSatisfy` Text.isInfixOf "data "

-- ---------------------------------------------------------------------------
-- train and predict

trainSpec :: Spec
trainSpec = do
  it "trains xor.hsg to accuracy 1 with seed 1, reporting every 10 steps" $ do
    ls <- outputOf ["train", "examples/xor.hsg", "--steps", "200", "--seed", "1"]
    take 1 ls `shouldBe` ["train examples/xor.hsg: 200 steps, lr 0.1 (constant), seed 1, xor: 4 of 4 rows per step"]
    map (Text.take 8) (drop 1 ls) `shouldBe` ["step " <> Text.justifyRight 3 ' ' (Text.pack (show n)) | n <- [0 :: Int, 10 .. 190] <> [199]]
    -- The loss falls below 1e-3 and the signs are all right at the end.
    case reverse ls of
      final : _ -> do
        final `shouldSatisfy` Text.isSuffixOf "accuracy 100%"
        lossOf final `shouldSatisfy` (< 1e-3)
      [] -> expectationFailure "no output"

  it "reports every N steps and shows a changing rate" $ do
    ls <- outputOf ["train", "examples/xor.hsg", "--steps", "20", "--report-every", "8", "--lr-schedule", "linear:0", "--lr", "0.5"]
    map (Text.take 7) (drop 1 ls) `shouldBe` ["step  0", "step  8", "step 16", "step 19"]
    take 1 (drop 1 ls) `shouldSatisfy` all (Text.isSuffixOf "lr 0.5")

  it "trains moons.hsg on mini-batches" $ do
    ls <- outputOf ["train", "examples/moons.hsg", "--steps", "3", "--batch", "8", "--report-every", "1"]
    take 1 ls `shouldBe` ["train examples/moons.hsg: 3 steps, lr 0.1 (constant), seed 1337, moons: 8 of 100 rows per step"]
    length ls `shouldBe` 4
    drop 1 ls `shouldSatisfy` all (Text.isInfixOf "accuracy")

  it "saves parameters that predict reads back" $ do
    t <- transcriptOf ["train", "examples/xor.hsg", "--steps", "200", "--seed", "1", "--save-params", "xor.params"]
    okRun t
    map fst (transcriptFiles t) `shouldBe` ["xor.params"]
    saved <- writtenFile t
    dir <- (</> "hsgrad-test") <$> getTemporaryDirectory
    createDirectoryIfMissing True dir
    let path = dir </> "xor.params"
    Utf8.writeFile path saved
    -- Every row of the XOR table gets the sign of its label.
    forM_ [("[0,0]", -1), ("[0,1]", 1), ("[1,0]", 1), ("[1,1]", -1 :: Double)] $ \(x, label) -> do
      out <- outputOf ["predict", "examples/xor.hsg", "--params", path, "--set", "x=" <> x]
      case out of
        [l] -> do
          l `shouldSatisfy` Text.isPrefixOf ("net(" <> Text.pack (commaSpaced x) <> ") = ")
          signum (read (Text.unpack (Text.drop 1 (Text.dropWhile (/= '=') l))) :: Double) `shouldBe` label
        _ -> expectationFailure (show out)

  it "predicts with a hand-written parameter file" $
    outputOf ["predict", "test/golden/run/predict.hsg", "--params", "test/golden/run/net.params", "--set", "x=[2, 1]"]
      `shouldReturn` ["net([2, 1]) = 0.25"]
  where
    lossOf l = read (Text.unpack (Text.takeWhile (/= ' ') (snd (Text.breakOnEnd "loss " l)))) :: Double
    commaSpaced = concatMap (\ch -> if ch == ',' then ", " else [ch])

-- ---------------------------------------------------------------------------
-- check

checkSpec :: Spec
checkSpec = do
  it "summarises a program" $ do
    ls <- outputOf ["check", "examples/xor.hsg"]
    ls `shouldBe` ["data xor : { x : Vec 2, y : Real }", "param W1 : Mat 4 2", "param b1 : Vec 4", "param w2 : Vec 4", "param b2 : Real", "model net : Vec 2 → Real", "loss : Real", "examples/xor.hsg: ok, 154 nodes"]

  it "reports a missing file at its name on the command line" $ do
    t <- transcriptOf ["check", "examples/nothere.hsg"]
    map diagCode (transcriptDiagnostics t) `shouldBe` [E0401]
    transcriptResult t `shouldBe` Stopped

-- ---------------------------------------------------------------------------
-- Parameter files

paramsSpec :: Spec
paramsSpec = do
  src <- runIO (Utf8.readFile "examples/xor.hsg")
  let checked = case checkSource "examples/xor.hsg" src of
        Outcome _ (Ok c) -> c
        _ -> error "xor.hsg does not check"
      specs = case compileProgram Map.empty checked of
        Right c -> compiledParams c
        Left e -> error (show e)
      leaves = Map.keys (initParams 1 specs)
  prop "reads back exactly the numbers it saved" $
    forAll (vectorOf (length leaves) number) $ \xs -> do
      let store = Map.fromList (zip leaves xs)
      case renderParams ["a comment"] specs store of
        Nothing -> expectationFailure "no text"
        Just text -> readParams "p.params" text (checkedParams checked) `shouldBe` Right store

  it "writes one row of a matrix per line, names and types in columns" $ do
    let store = initParams 1 specs
    case Text.lines <$> renderParams ["c"] specs store of
      Just ls -> do
        take 2 ls `shouldBe` ["-- c", "param W1 : Mat 4 2 = ["]
        filter (Text.isPrefixOf "param") ls `shouldSatisfy` all (\l -> Text.index l 9 == ':' && Text.index l 19 == '=')
        length ls `shouldBe` 1 + 1 + 4 + 1 + 3
      Nothing -> expectationFailure "no text"

  it "fills a shape from one number" $
    readParams "p.params" "param W1 : Mat 4 2 = 0\nparam b1 : Vec 4 = 1\nparam w2 : Vec 4 = 2\nparam b2 : Real = 3\n" (checkedParams checked)
      `shouldBe` Right (Map.fromList (zip (concatMap elementNames specs) (replicate 8 0 <> replicate 4 1 <> replicate 4 2 <> [3])))
  where
    -- Numbers of every size, including ones that print in scientific
    -- notation, and simple ones.
    number :: Gen Double
    number =
      frequency
        [ (3, choose (-10, 10))
        , (1, (*) <$> choose (-1, 1) <*> ((10 **) <$> choose (-300, 300)))
        , (1, fromInteger <$> choose (-5, 5))
        ]

-- ---------------------------------------------------------------------------
-- Golden runs

goldenSpec :: Spec
goldenSpec = do
  goldenFiles "test/golden/run" ".hsg" $ \name src -> do
    let path = "test/golden/run" </> name
    args <- case Text.lines src of
      first : _ | Just rest <- Text.stripPrefix "-- args:" first -> pure (map Text.unpack (Text.words rest))
      _ -> fail (name <> " has no `-- args:` line")
    case args of
      command : rest -> renderTranscript defaultRenderOptions <$> transcriptOf (command : path : rest)
      [] -> fail (name <> ": empty `-- args:` line")

  it "covers every runtime code E05xx" $ do
    names <- filter ((== ".err") . reverse . take 4 . reverse) <$> listDirectory "test/golden/run"
    texts <- mapM (Utf8.readFile . ("test/golden/run" </>)) names
    [codeText c | c <- [E0501 .. E0505], not (any (Text.isInfixOf ("[" <> codeText c <> "]")) texts)] `shouldBe` []
