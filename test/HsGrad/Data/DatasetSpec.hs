{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Data.Dataset".
--
-- * The examples: @xor.hsg@'s inline rows and @moons.hsg@'s CSV file give
--   datasets that match their schemas.
-- * Building: path resolution, mapping columns by name in schema order,
--   unmapped columns, the stages and their codes.
-- * Training: slot names, the feed, the batch-size check, and the XOR net
--   of step 9 trained on the rows read from @xor.hsg@.
-- * Golden files in @test/golden/data/@: every E04xx code rendered with
--   the snippets of both files, the @.hsg@ and the CSV file.
module HsGrad.Data.DatasetSpec
  ( spec
  ) where

import Control.Monad (forM_)
import Data.Foldable (toList)
import Data.List (isPrefixOf, sort)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Core.Pretty (showConstant)
import HsGrad.Data.Dataset
import HsGrad.Lang.Check (Checked, DatasetInfo (..), checkProgram, checkedDatasets)
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Parser (parseProgram)
import HsGrad.Lang.Shape (Shape (..))
import HsGrad.Test.Examples (xorNet, xorRows, xorSlots, xorSpecs)
import HsGrad.Test.Golden (goldenFiles)
import HsGrad.Train.Loop
import HsGrad.Train.Optim (LrSchedule (..), Sgd (..))
import HsGrad.Train.Param (initParams)
import System.Directory (listDirectory)
import System.FilePath (takeExtension, (</>))
import Test.Hspec (Spec, describe, expectationFailure, it, runIO, shouldBe, shouldSatisfy)

-- | Parse and check a program that must check.
checked :: FilePath -> Text -> Checked
checked file src = case parseProgram file src of
  Left ds -> error ("parse error in test program: " <> show (toList ds))
  Right p -> case checkProgram p of
    (_, Just c) -> c
    (ds, Nothing) -> error ("test program doesn't check: " <> show (map diagCode ds))

-- | The datasets of a program that must check.
infos :: Text -> [DatasetInfo]
infos = checkedDatasets . checked "t.hsg"

-- | The one dataset of a program, built with the given files.
built :: Files -> Text -> Either [Diagnostic] Dataset
built files src = case infos src of
  [info] -> buildDataset "t.hsg" files info
  is -> error ("expected one dataset, got " <> show (length is))

-- | The codes of a build that must fail.
codesOf :: Either [Diagnostic] a -> [Code]
codesOf = either (map diagCode) (const [])

-- | A one-file 'Files' map.
oneFile :: FilePath -> Text -> Files
oneFile path text = Map.singleton path (Right text)

spec :: Spec
spec = do
  exampleSpec
  buildSpec
  trainSpec
  goldenSpec

-- ---------------------------------------------------------------------------
-- Examples

exampleSpec :: Spec
exampleSpec = describe "examples" $ do
  it "reads xor.hsg's inline rows" $ do
    src <- Utf8.readFile "examples/xor.hsg"
    let c = checked "examples/xor.hsg" src
    (_, result) <- loadDatasets "examples/xor.hsg" (checkedDatasets c)
    case result of
      Right [d] -> do
        dataName d `shouldBe` "xor"
        dataFields d `shouldBe` [("x", Vec 2), ("y", Real)]
        datasetColumns d `shouldBe` ["x[0]", "x[1]", "y"]
        dataRows d `shouldBe` xorRows
      other -> expectationFailure (show other)

  it "reads moons.hsg's CSV file, relative to the .hsg file" $ do
    src <- Utf8.readFile "examples/moons.hsg"
    let c = checked "examples/moons.hsg" src
    (sources, result) <- loadDatasets "examples/moons.hsg" (checkedDatasets c)
    Map.keys sources `shouldBe` ["examples/data/moons.csv"]
    case result of
      Right [d] -> do
        dataFields d `shouldBe` [("x", Vec 2), ("y", Real)]
        datasetSize d `shouldBe` 100
        dataRows d `shouldSatisfy` all ((== 3) . length)
        take 2 (dataRows d) `shouldBe` [[0.909158, -0.135722, -1], [0.082757, 0.700558, 1]]
        -- 50 points per moon, labels -1 and +1, the points near the moons.
        let labels = map (!! 2) (dataRows d)
        (length (filter (== -1) labels), length (filter (== 1) labels)) `shouldBe` (50, 50)
        dataRows d `shouldSatisfy` all (\r -> all (\v -> abs v < 2.5) r)
      other -> expectationFailure (show other)

-- ---------------------------------------------------------------------------
-- Building

buildSpec :: Spec
buildSpec = describe "buildDataset" $ do
  it "resolves CSV paths relative to the .hsg file" $ do
    let paths hsg csv = map (csvPath hsg) (infos ("data d : { y : Real } = csv \"" <> csv <> "\" { y = c }"))
    paths "examples/moons.hsg" "data/moons.csv" `shouldBe` [Just "examples/data/moons.csv"]
    paths "moons.hsg" "data/moons.csv" `shouldBe` [Just "data/moons.csv"]
    paths "/home/u/m.hsg" "../d.csv" `shouldBe` [Just "/home/u/../d.csv"]
    paths "examples/moons.hsg" "/srv/d.csv" `shouldBe` [Just "/srv/d.csv"]
    map (csvPath "t.hsg") (infos "data d : { y : Real } = rows [ 1 ]") `shouldBe` [Nothing]

  it "maps columns by name, in schema order, whatever the header's order" $
    fmap dataRows (built (oneFile "d.csv" "a,label,b\n1,2,3\n4,5,6\n") "data d : { x : Vec 2, y : Real } = csv \"d.csv\" { y = label, x = [b, a] }")
      `shouldBe` Right [[3, 1, 2], [6, 4, 5]]

  it "may read one column for several fields" $
    fmap dataRows (built (oneFile "d.csv" "a\n7\n") "data d : { x : Vec 2, y : Real } = csv \"d.csv\" { x = [a, a], y = a }")
      `shouldBe` Right [[7, 7, 7]]

  it "ignores unmapped columns, whatever is in them" $
    fmap dataRows (built (oneFile "d.csv" "id,x\nbob,1\n\"x, y\",2\n") "data d : { x : Real } = csv \"d.csv\" { x = x }")
      `shouldBe` Right [[1], [2]]

  it "reads a one-field dataset's bare literals" $
    fmap dataRows (built Map.empty "data d : { v : Vec 2 } = rows [ [1, 2], [3, -4] ]") `shouldBe` Right [[1, 2], [3, -4]]

  it "reports a file missing from the Files map as unreadable" $
    codesOf (built Map.empty "data d : { y : Real } = csv \"d.csv\" { y = c }") `shouldBe` [E0401]

  it "reports why a file could not be read" $
    forM_ [FileNotFound, PermissionDenied, NotUtf8, OtherReadError "is a directory"] $ \p ->
      codesOf (built (Map.singleton "d.csv" (Left p)) "data d : { y : Real } = csv \"d.csv\" { y = c }") `shouldBe` [E0401]

  it "checks the mapping before reading the file" $
    codesOf (built Map.empty "data d : { y : Real } = csv \"d.csv\" { y = [c] }") `shouldBe` [E0404]

  it "reports the problems of one stage together" $
    codesOf (built (oneFile "d.csv" "a,b\n1,x\ny,2\n") "data d : { x : Real, y : Real } = csv \"d.csv\" { x = a, y = b }")
      `shouldBe` [E0403, E0403]

  it "reports a column missing from the header before any bad cell" $
    codesOf (built (oneFile "d.csv" "a\nx\n") "data d : { x : Real, y : Real } = csv \"d.csv\" { x = a, y = b }") `shouldBe` [E0402]

  it "reports every bad inline row" $
    codesOf (built Map.empty "data d : { x : Vec 2, y : Real } = rows [ ([1, 2], 3), (1, 2), ([1], 2), ([1, 2], [3]), ([1, 2]) ]")
      `shouldBe` [E0404, E0404, E0404, E0404]

  it "rejects a mapping with the wrong number of columns for a vector" $ do
    let files = oneFile "d.csv" "a,b,c\n1,2,3\n"
    codesOf (built files "data d : { x : Vec 2, y : Real } = csv \"d.csv\" { x = [a], y = b }") `shouldBe` [E0404]
    codesOf (built files "data d : { x : Vec 2, y : Real } = csv \"d.csv\" { x = [a, b, c], y = b }") `shouldBe` [E0404]

  it "builds every dataset, or reports the problems of all" $ do
    let two = infos "data a : { y : Real } = rows [ 1 ]\ndata b : { y : Real } = csv \"b.csv\" { y = c }\ndata e : { y : Real } = rows [ [1] ]"
    fmap (map dataName) (buildDatasets "t.hsg" (oneFile "b.csv" "c\n2\n") (take 2 two)) `shouldBe` Right ["a", "b"]
    codesOf (buildDatasets "t.hsg" Map.empty two) `shouldBe` [E0401, E0404]

-- ---------------------------------------------------------------------------
-- Training

xorData :: Dataset
xorData = case built Map.empty "data xor : { x : Vec 2, y : Real } = rows [ ([0, 0], -1), ([0, 1], 1), ([1, 0], 1), ([1, 1], -1) ]" of
  Right d -> d
  Left ds -> error (show ds)

trainSpec :: Spec
trainSpec = describe "training" $ do
  it "names slot leaves like 01-language §8" $ do
    slotNames "moons" [("x", Vec 2), ("y", Real)] 2
      `shouldBe` [["moons#0.x[0]", "moons#0.x[1]", "moons#0.y"], ["moons#1.x[0]", "moons#1.x[1]", "moons#1.y"]]
    slotNames "d" [("v", Real)] 0 `shouldBe` []

  it "gives the feed of the hand-built XOR net of step 9" $
    toFeed xorData (slotNames "xor" (dataFields xorData) 4) `shouldBe` Feed "xor" xorRows (xorSlots 4)

  it "trains the XOR net on the rows read from the declaration" $ do
    let ((loss, pairs), g) = xorNet 4
        setup = Setup g loss Map.empty [toFeed xorData (slotNames "xor" (dataFields xorData) 4)] pairs
        cfg = TrainConfig 200 (Sgd 0.1 ConstantRate) 1337
        reports = runReports (train cfg setup (initParams 1337 xorSpecs))
    map reportAccuracy (drop 199 reports) `shouldBe` [Just 1]

  it "accepts batches up to the dataset size, and reports larger ones" $ do
    checkBatch 1 xorData `shouldBe` []
    checkBatch 4 xorData `shouldBe` []
    map diagCode (checkBatch 5 xorData) `shouldBe` [E0405]

-- ---------------------------------------------------------------------------
-- Golden files

goldenDir :: FilePath
goldenDir = "test/golden/data"

-- | What the driver will show for a file's datasets: the diagnostics,
-- with the snippets of the @.hsg@ file and of every CSV file read, or a
-- description of the datasets.
--
-- Spans use the path below the package directory
-- (@test/golden/data/x.hsg@), so that the CSV file is found relative to
-- the @.hsg@ file just as in the driver. Warnings of the checker are left
-- out (most of these programs have unused datasets). A line
-- @-- golden: --batch N@ in the program also checks that batch size.
rendered :: FilePath -> Text -> IO Text
rendered name src = case parseProgram path src of
  Left ds -> pure (render (Map.singleton path src) (toList ds))
  Right p -> case checkProgram p of
    (ds, Nothing) -> pure (render (Map.singleton path src) ds)
    (_, Just c) -> do
      (sources, result) <- loadDatasets path (checkedDatasets c)
      let allSources = Map.insert path src sources
      pure $ case result of
        Left ds -> render allSources ds
        Right datasets -> case concatMap (checkBatch batch) datasets of
          [] -> Text.concat (map describeDataset datasets)
          ds -> render allSources ds
  where
    path = goldenDir </> name
    render = renderDiagnostics defaultRenderOptions
    batch = case [Text.strip rest | l <- Text.lines src, Just rest <- [Text.stripPrefix "-- golden: --batch " l]] of
      n : _ -> read (Text.unpack n)
      [] -> 1
    describeDataset d =
      "dataset `" <> dataName d <> "`: " <> Text.pack (show (datasetSize d)) <> " rows, columns "
        <> Text.intercalate ", " (datasetColumns d)
        <> "\n"
        <> Text.concat ["  " <> Text.intercalate ", " (map showConstant r) <> "\n" | r <- dataRows d]

goldenSpec :: Spec
goldenSpec = describe "golden files" $ do
  goldenFiles goldenDir ".hsg" rendered

  errs <- runIO (sort . filter ((== ".err") . takeExtension) <$> listDirectory goldenDir)
  texts <- runIO (mapM (Utf8.readFile . (goldenDir </>)) errs)
  it "show every E04xx code" $
    forM_ [c | c <- [minBound .. maxBound :: Code], "E04" `isPrefixOf` show c] $ \c ->
      (show c, any (Text.isInfixOf ("[" <> codeText c <> "]")) texts) `shouldBe` (show c, True)
