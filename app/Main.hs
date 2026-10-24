{-# LANGUAGE OverloadedStrings #-}

-- | The @hsgrad@ program: the IO around "HsGrad.Driver".
--
-- Parse the command line ("HsGrad.Cli"), load the file and run the
-- command (the driver), then do what the result says: print lines on
-- stdout as they come (training reports while training runs), write
-- files, print diagnostics on stderr, and exit with
--
-- * 0 when everything worked (warnings are fine),
-- * 1 when there were errors (or the command line was wrong),
-- * 2 on an internal error, a bug in hsgrad.
--
-- See @docs/16-cli.md@.
module Main
  ( main
  ) where

import Control.Exception (IOException, displayException, try)
import Data.Maybe (isNothing)
import Data.Text (Text)
import Data.Text.IO qualified as TextIO
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Cli (cliInfo, cliPrefs)
import HsGrad.Driver
import HsGrad.Lang.Diagnostic
import Options.Applicative (customExecParser)
import System.Environment (lookupEnv)
import System.Exit (ExitCode (..), exitWith)
import System.IO (BufferMode (..), hIsTerminalDevice, hPutStrLn, hSetBuffering, hSetEncoding, stderr, stdout, utf8)

main :: IO ()
main = do
  -- Output has ∂ and · in it: UTF-8, whatever the locale says.
  mapM_ (`hSetEncoding` utf8) [stdout, stderr]
  hSetBuffering stdout LineBuffering
  opts <- customExecParser cliPrefs cliInfo
  colour <- case optColour opts of
    ColourAlways -> pure True
    ColourNever -> pure False
    ColourAuto -> (&&) <$> hIsTerminalDevice stderr <*> (isNothing <$> lookupEnv "NO_COLOR")
  let render = defaultRenderOptions {renderColour = colour}
  (sources, loaded) <- load opts
  case loaded >>= run opts of
    Outcome ds (Ok stream) -> do
      printDiagnostics render sources ds
      Outcome ds' result <- printLines stream
      printDiagnostics render sources ds'
      writeProblems <- case result of
        Ok files -> concat <$> mapM writeOutput files
        _ -> pure []
      mapM_ (hPutStrLn stderr) writeProblems
      finish (ds <> ds') (if null writeProblems then () <$ result else Stopped)
    Outcome ds result -> do
      printDiagnostics render sources ds
      finish ds (() <$ result)

-- | Print the lines of a stream as they are produced; then its end.
printLines :: Stream -> IO (Outcome [(FilePath, Text)])
printLines (Line l rest) = TextIO.putStrLn l >> printLines rest
printLines (End end) = pure end

-- | Diagnostics on stderr, each followed by a blank line. (The summary
-- line comes once, at the end.)
printDiagnostics :: RenderOptions -> Sources -> [Diagnostic] -> IO ()
printDiagnostics render sources ds =
  mapM_ (\d -> TextIO.hPutStr stderr (renderDiagnostic render sources d <> "\n")) (sortDiagnostics ds)

-- | Write an output file (UTF-8; @-@ is stdout). A problem as a message.
writeOutput :: (FilePath, Text) -> IO [String]
writeOutput ("-", text) = [] <$ TextIO.putStr text
writeOutput (path, text) = do
  written <- try (Utf8.writeFile path text)
  case written of
    Right () -> [] <$ hPutStrLn stderr ("wrote " <> path)
    Left e -> pure ["hsgrad: cannot write " <> path <> ": " <> displayException (e :: IOException)]

-- | The summary line, and the exit code.
finish :: [Diagnostic] -> Result () -> IO ()
finish ds result = do
  if null ds
    then pure ()
    else TextIO.hPutStrLn stderr (summary ds)
  case result of
    Ok () -> exitWith ExitSuccess
    Stopped -> exitWith (ExitFailure 1)
    Internal msg -> do
      TextIO.hPutStrLn stderr ("hsgrad: internal error (a bug in hsgrad): " <> msg)
      exitWith (ExitFailure 2)
