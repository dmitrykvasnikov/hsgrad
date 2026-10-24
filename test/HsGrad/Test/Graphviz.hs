-- | Running Graphviz in tests: does @dot@ accept a DOT text?
--
-- Moved here from @DotSpec@ (step 6) so that the driver's tests (step 15)
-- check the pictures of @plot@ and @derive --dot@ the same way.
module HsGrad.Test.Graphviz
  ( withDot
  , toSvg
  ) where

import Data.Text (Text)
import Data.Text qualified as Text
import GHC.IO.Encoding (setLocaleEncoding, utf8)
import System.Directory (findExecutable)
import System.Exit (ExitCode (..))
import System.Process (readProcessWithExitCode)
import Test.Hspec (Expectation, expectationFailure, pendingWith)

-- | Run a test with the path of Graphviz's @dot@, or mark it pending if
-- Graphviz is not installed: the test suite must not require it.
withDot :: (FilePath -> Expectation) -> Expectation
withDot k = do
  found <- findExecutable "dot"
  case found of
    Nothing -> pendingWith "Graphviz (dot) is not on PATH"
    Just exe -> do
      -- The DOT text is UTF-8 (∂L/∂a), whatever the system locale says.
      setLocaleEncoding utf8
      k exe

-- | Render DOT text to SVG with Graphviz. Fails the test if @dot@ exits
-- with an error or prints anything (a warning) on stderr.
toSvg :: FilePath -> Text -> IO String
toSvg exe dot = do
  (code, out, err) <- readProcessWithExitCode exe ["-Tsvg"] (Text.unpack dot)
  case (code, err) of
    (ExitSuccess, "") -> pure out
    _ ->
      expectationFailure ("dot failed: " <> show code <> "\n" <> err <> "\n" <> Text.unpack dot)
        >> pure out
