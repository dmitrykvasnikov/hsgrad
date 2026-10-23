-- | The @hsgrad@ command-line program.
--
-- Placeholder until roadmap step 15, which replaces it with an
-- @optparse-applicative@ CLI (see @docs/00-architecture.md@ §5).
module Main
  ( main
  ) where

import HsGrad qualified
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hPutStr, hPutStrLn, stderr)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [] -> putStr usage
    ["--help"] -> putStr usage
    ["--version"] -> putStrLn ("hsgrad " <> HsGrad.version)
    _ -> do
      hPutStrLn stderr "hsgrad: commands are not implemented yet"
      hPutStrLn stderr ""
      hPutStr stderr usage
      exitFailure

usage :: String
usage =
  unlines
    [ "hsgrad " <> HsGrad.version <> ": scalar autodiff with a small model language"
    , ""
    , "Usage:"
    , "  hsgrad --help       show this message"
    , "  hsgrad --version    show the version"
    , ""
    , "Planned commands: check, eval, derive, plot, train, predict"
    , "(see docs/00-architecture.md, section 5)."
    ]
