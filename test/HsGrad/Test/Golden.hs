-- | Golden-file tests: compare output with a file checked into the
-- repository.
--
-- A golden directory holds input files (for example @*.hsg@) and, next to
-- each, the expected output with the extension @.err@. 'goldenFiles' makes
-- one test per input file. When an output changes on purpose, regenerate
-- the expected files instead of editing them by hand:
--
-- > HSGRAD_ACCEPT=1 cabal test
--
-- With @HSGRAD_ACCEPT@ set (to anything but @0@) every golden test writes
-- its actual output to the expected file and passes. Review the change with
-- @git diff test/golden@ before committing it.
--
-- Paths are relative to the package directory, where @cabal test@ runs the
-- suite. Files are read and written as UTF-8 whatever the locale.
module HsGrad.Test.Golden
  ( goldenFiles
  , goldenText
  , expectedPath
  ) where

import Control.Monad (forM_, when)
import Data.List (sort)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import System.Directory (doesFileExist, listDirectory)
import System.Environment (lookupEnv)
import System.FilePath (replaceExtension, takeExtension, (</>))
import Test.Hspec (Expectation, Spec, expectationFailure, it, runIO)

-- | One test per file with the given extension (such as @".hsg"@) in the
-- directory, in name order. The function gets the file's name (without the
-- directory) and its contents, and produces the output to compare with
-- the file's 'expectedPath'.
--
-- The output function may do IO (reading data files next to the input,
-- say); for a pure one, use @\\name src -> pure (f name src)@.
goldenFiles :: FilePath -> String -> (FilePath -> Text -> IO Text) -> Spec
goldenFiles dir ext run = do
  inputs <- runIO (sort . filter ((== ext) . takeExtension) <$> listDirectory dir)
  when (null inputs) $
    it ("finds " <> ext <> " files in " <> dir) $
      expectationFailure ("no " <> ext <> " files in " <> dir)
  forM_ inputs $ \name ->
    it name $ do
      source <- Utf8.readFile (dir </> name)
      actual <- run name source
      goldenText (expectedPath (dir </> name)) actual

-- | The expected output for an input file: same name, extension @.err@.
expectedPath :: FilePath -> FilePath
expectedPath input = replaceExtension input "err"

-- | Compare text with a golden file, or overwrite the file when accepting.
-- A missing file is a failure (not silently created), so a forgotten
-- @git add@ cannot make a test pass.
goldenText :: FilePath -> Text -> Expectation
goldenText path actual = do
  accepting <- maybe False (`notElem` ["", "0"]) <$> lookupEnv "HSGRAD_ACCEPT"
  exists <- doesFileExist path
  if accepting
    then Utf8.writeFile path actual
    else
      if not exists
        then expectationFailure (path <> " does not exist; create it with HSGRAD_ACCEPT=1 cabal test")
        else do
          expected <- Utf8.readFile path
          when (expected /= actual) $
            expectationFailure (difference path expected actual)

-- | Where two texts first differ, and the whole actual output.
difference :: FilePath -> Text -> Text -> String
difference path expected actual =
  unlines
    [ path <> " differs from the output, first at line " <> show n <> ":"
    , "  expected: " <> show e
    , "  actual:   " <> show a
    , "whole output:"
    , Text.unpack actual
    , "(if the new output is right, accept it with HSGRAD_ACCEPT=1 cabal test)"
    ]
  where
    le = Text.lines expected
    la = Text.lines actual
    pad ls = map Just ls <> repeat Nothing
    pairs = take (max (length le) (length la)) (zip3 [1 :: Int ..] (pad le) (pad la))
    (n, e, a) = case [p | p@(_, x, y) <- pairs, x /= y] of
      p : _ -> p
      -- Same lines, so they differ only in the final newline.
      [] -> (length (Text.lines actual) + 1, Nothing, Nothing)
