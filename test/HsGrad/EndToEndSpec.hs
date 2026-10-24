{-# LANGUAGE OverloadedStrings #-}

-- | End-to-end tests: the built @hsgrad@ program, run as a process the
-- way a user runs it, on the files in @examples/@.
--
-- "HsGrad.DriverSpec" already runs whole commands in-process, with the
-- real command-line parser and 'HsGrad.Driver.execute'. What only a
-- process shows is the shell around them, @app/Main.hs@: exit codes,
-- results on stdout and diagnostics on stderr, files written to disk,
-- UTF-8 output in a C locale. The test suite declares
-- @build-tool-depends: hsgrad:hsgrad@, so cabal builds the program before
-- the tests and puts it on their @PATH@.
--
-- * The examples: all but @errors.hsg@ check cleanly, @errors.hsg@ gives
--   its errors; @moons.hsg@ trains to at least 95% accuracy and @xor.hsg@
--   to 100%; saved parameters work with @eval@ and @predict@; @derive@
--   prints formulas and writes DOT that Graphviz accepts.
-- * The tutorial: every @$ hsgrad …@ transcript in @docs/17-examples.md@
--   is run and must print exactly what the page shows. A block marked
--   @\<!-- slow --\>@ (micrograd's full moons run, about 10 s) runs only
--   with @HSGRAD_SLOW=1@ set.
module HsGrad.EndToEndSpec
  ( spec
  ) where

import Control.Monad (forM_, unless)
import Data.List (isPrefixOf, sort)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import GHC.IO.Encoding (setLocaleEncoding, utf8)
import HsGrad.Test.Graphviz (toSvg, withDot)
import System.Directory
  ( createDirectoryIfMissing
  , createDirectoryLink
  , doesFileExist
  , findExecutable
  , getTemporaryDirectory
  , listDirectory
  , makeAbsolute
  , removePathForcibly
  )
import System.Environment (getEnvironment, lookupEnv)
import System.Exit (ExitCode (..))
import System.FilePath (takeExtension, (</>))
import System.IO (hGetContents', hSetEncoding)
import System.Process
  ( CreateProcess (..)
  , StdStream (..)
  , createPipe
  , createProcess
  , proc
  , readCreateProcessWithExitCode
  , waitForProcess
  )
import Test.Hspec

spec :: Spec
spec = do
  -- The output of hsgrad is UTF-8 (∂, ·); read it as such.
  runIO (setLocaleEncoding utf8)
  describe "the examples" exampleSpec
  describe "docs/17-examples.md" tutorialSpec

-- ---------------------------------------------------------------------------
-- Running the program

-- | What a run printed, and how it ended.
data Run = Run
  { runExit :: ExitCode
  , runStdout :: Text
  , runStderr :: Text
  }
  deriving (Show)

-- | The @hsgrad@ program that cabal built for the tests.
program :: IO FilePath
program =
  findExecutable "hsgrad"
    >>= maybe (fail "hsgrad is not on the PATH (cabal test puts it there: build-tool-depends)") pure

-- | The environment of a run: ours, in the C locale. hsgrad must print
-- UTF-8 whatever the locale says (docs/16-cli.md §4.8).
cLocale :: IO [(String, String)]
cLocale = do
  env <- getEnvironment
  pure (("LC_ALL", "C") : [(k, v) | (k, v) <- env, k /= "LANG", not ("LC_" `isPrefixOf` k)])

-- | Run hsgrad in a directory, with stdout and stderr apart.
hsgrad :: FilePath -> [String] -> IO Run
hsgrad dir args = do
  exe <- program
  env <- cLocale
  (code, out, err) <- readCreateProcessWithExitCode (proc exe args) {cwd = Just dir, env = Just env} ""
  pure (Run code (Text.pack out) (Text.pack err))

-- | Run hsgrad in a directory with stdout and stderr in one pipe, as a
-- terminal shows them: training reports, then "wrote FILE".
terminal :: FilePath -> [String] -> IO (ExitCode, Text)
terminal dir args = do
  exe <- program
  env <- cLocale
  (r, w) <- createPipe
  hSetEncoding r utf8
  -- createProcess closes w here; the child keeps its copies.
  (_, _, _, ph) <- createProcess (proc exe args) {cwd = Just dir, env = Just env, std_out = UseHandle w, std_err = UseHandle w}
  out <- hGetContents' r
  code <- waitForProcess ph
  pure (code, Text.pack out)

-- | A fresh, empty directory for the files a run writes, with a link
-- @examples@ to the examples, so that paths read as in the docs.
scratch :: String -> IO FilePath
scratch name = do
  dir <- (</> ("hsgrad-e2e" </> name)) <$> getTemporaryDirectory
  removePathForcibly dir
  createDirectoryIfMissing True dir
  examples <- makeAbsolute "examples"
  createDirectoryLink examples (dir </> "examples")
  pure dir

-- | Expect a run that worked and said nothing on stderr.
quiet :: Run -> Expectation
quiet r = (runExit r, runStderr r) `shouldBe` (ExitSuccess, "")

-- | The number at the end of a line @… = 0.5@.
valueOf :: Text -> Double
valueOf l = read (Text.unpack (Text.strip (snd (Text.breakOnEnd "= " l))))

-- | The values of a run's output lines.
valuesOf :: Run -> [Double]
valuesOf = map valueOf . Text.lines . runStdout

-- | The accuracy of a training report line, in percent.
accuracyOf :: Text -> Maybe Int
accuracyOf l = case Text.breakOnEnd "accuracy " l of
  ("", _) -> Nothing
  (_, rest) -> Just (read (Text.unpack (Text.takeWhile (/= '%') rest)))

-- | The accuracy of the last report of a training run.
finalAccuracy :: Run -> Maybe Int
finalAccuracy r = case reverse (Text.lines (runStdout r)) of
  l : _ -> accuracyOf l
  [] -> Nothing

-- ---------------------------------------------------------------------------
-- The examples

exampleSpec :: Spec
exampleSpec = do
  files <- runIO (sort . filter ((== ".hsg") . takeExtension) <$> listDirectory "examples")

  describe "check" $ do
    forM_ (filter (/= "errors.hsg") files) $ \file ->
      it (file <> " checks without a diagnostic") $ do
        let path = "examples/" <> Text.pack file
        r <- hsgrad "." ["check", Text.unpack path]
        quiet r
        take 1 (reverse (Text.lines (runStdout r))) `shouldSatisfy` all ((path <> ": ok, ") `Text.isPrefixOf`)

    it "errors.hsg gives all its errors, sorted, and exit code 1" $ do
      r <- hsgrad "." ["check", "examples/errors.hsg"]
      runExit r `shouldBe` ExitFailure 1
      runStdout r `shouldBe` ""
      let ls = Text.lines (runStderr r)
          codes = [Text.takeWhile (/= ']') (Text.drop 1 rest) | l <- ls, any (`Text.isPrefixOf` l) ["error[", "warning["], let rest = Text.dropWhile (/= '[') l]
      codes `shouldBe` ["W0202", "E0201", "E0303", "E0302", "E0308"]
      take 1 (reverse ls) `shouldBe` ["4 errors, 1 warning"]

  describe "train" $ do
    it "trains moons.hsg to at least 95% accuracy in 30 steps of micrograd's schedule" $ do
      -- micrograd's own run (100 steps, docs/17-examples.md §5) takes
      -- about 10 s; a third of it, at the full batch, already separates
      -- the moons to 97%.
      r <- hsgrad "." ["train", "examples/moons.hsg", "--steps", "30", "--lr", "1", "--lr-schedule", "linear:0.1", "--seed", "1337"]
      quiet r
      finalAccuracy r `shouldSatisfy` maybe False (>= 95)

    it "trains xor.hsg to 100%, and predict gives every row the sign of its label" $ do
      dir <- scratch "xor"
      r <- hsgrad dir ["train", "examples/xor.hsg", "--steps", "200", "--seed", "1", "--save-params", "xor.params"]
      (runExit r, runStderr r) `shouldBe` (ExitSuccess, "wrote xor.params\n")
      finalAccuracy r `shouldBe` Just 100
      doesFileExist (dir </> "xor.params") `shouldReturn` True
      forM_ [("[0,0]", -1), ("[0,1]", 1), ("[1,0]", 1), ("[1,1]", -1)] $ \(x, label) -> do
        p <- hsgrad dir ["predict", "examples/xor.hsg", "--params", "xor.params", "--set", "x=" <> x]
        quiet p
        map signum (valuesOf p) `shouldBe` [label]

    it "fits linreg.hsg's least-squares line w = 2, b = 1" $ do
      dir <- scratch "linreg"
      r <- hsgrad dir ["train", "examples/linreg.hsg", "--save-params", "linreg.params"]
      runExit r `shouldBe` ExitSuccess
      e <- hsgrad dir ["eval", "examples/linreg.hsg", "--params", "linreg.params", "--show", "w", "--show", "b"]
      quiet e
      valuesOf e `shouldSatisfy` close [2, 1]
      p <- hsgrad dir ["predict", "examples/linreg.hsg", "--params", "linreg.params", "--set", "x=10"]
      quiet p
      valuesOf p `shouldSatisfy` close [21]

  describe "derive and plot" $ do
    it "prints the gradients of intro.hsg (01-language.md §1.1), in UTF-8 in a C locale" $ do
      r <- hsgrad "." ["derive", "examples/intro.hsg", "--of", "L", "--wrt", "a,b,c", "--simplify", "--format", "text"]
      quiet r
      runStdout r `shouldBe` "∂L/∂a = -2·b  = 6\n∂L/∂b = -2·a  = -4\n∂L/∂c = -2    = -2\n"

    forM_ [[], ["--simplify"]] $ \flags ->
      it ("writes neuron.hsg's gradient graph as DOT that Graphviz accepts " <> show flags) $ do
        dir <- scratch "neuron"
        r <- hsgrad dir (["derive", "examples/neuron.hsg", "--of", "y", "--wrt", "w", "--dot", "grad.dot"] <> flags)
        (runExit r, runStderr r) `shouldBe` (ExitSuccess, "wrote grad.dot\n")
        map (Text.takeWhile (/= ' ')) (take 3 (Text.lines (runStdout r))) `shouldBe` ["∂y/∂w[0]", "∂y/∂w[1]", "∂y/∂w[2]"]
        dot <- Utf8.readFile (dir </> "grad.dot")
        -- --format dot prints the same graph on stdout.
        d <- hsgrad dir (["derive", "examples/neuron.hsg", "--of", "y", "--wrt", "w", "--format", "dot"] <> flags)
        quiet d
        runStdout d `shouldBe` dot
        withDot $ \exe -> () <$ toSvg exe dot

    it "plots to stdout with --dot -" $ do
      r <- hsgrad "." ["plot", "examples/neuron.hsg", "--of", "y", "--dot", "-", "--cluster"]
      quiet r
      runStdout r `shouldSatisfy` Text.isPrefixOf "digraph"
      withDot $ \exe -> () <$ toSvg exe (runStdout r)

    it "agrees with the derivatives written down in calculus.hsg" $
      -- derive of cube / s against eval of the hand-written dcube / ds.
      forM_ [("cube", "dcube"), ("s", "ds")] $ \(f, df) -> do
        d <- hsgrad "." ["derive", "examples/calculus.hsg", "--of", f, "--wrt", "x"]
        e <- hsgrad "." ["eval", "examples/calculus.hsg", "--show", df]
        quiet d >> quiet e
        -- The first line is ∂f/∂x; a `where` with shared terms may follow.
        take 1 (valuesOf d) `shouldSatisfy` close (valuesOf e)

  describe "exit codes" $ do
    it "is 1 for a command line that doesn't parse" $ do
      r <- hsgrad "." ["frobnicate", "examples/intro.hsg"]
      runExit r `shouldBe` ExitFailure 1
      runStderr r `shouldNotBe` ""

    it "is 1 for an error at runtime, after the output so far" $ do
      r <- hsgrad "." ["eval", "examples/neuron.hsg", "--set", "x=[1,2]"]
      runExit r `shouldBe` ExitFailure 1
      runStderr r `shouldSatisfy` Text.isPrefixOf "error[E0503]"
  where
    close expected actual =
      length expected == length actual
        && and (zipWith (\a b -> abs (a - b) <= 1e-6 * max 1 (abs a)) expected actual)

-- ---------------------------------------------------------------------------
-- The tutorial

-- | A code block of the tutorial that is a transcript: commands, each
-- followed by the output it prints.
data Block = Block
  { blockLine :: Int
  , blockSlow :: Bool
  , blockRuns :: [(Text, [Text])]
  }

-- | The transcript blocks of a Markdown text: fenced blocks whose first
-- line starts with @$ hsgrad@. Within one, every line @$ COMMAND@ starts a
-- command, and the lines up to the next one are its output. The line
-- @\<!-- slow --\>@ right before a block marks it slow.
transcripts :: Text -> [Block]
transcripts = go . zip [1 ..] . Text.lines
  where
    go ls = case break (("```" `Text.isPrefixOf`) . snd) ls of
      (_, []) -> []
      (preceding, (n, _) : rest) ->
        let (body, following) = break ((== "```") . snd) rest
            slow = case reverse preceding of
              (_, l) : _ -> Text.strip l == "<!-- slow -->"
              [] -> False
            block = Block (n + 1) slow (runs (map snd body))
         in [block | take 1 (map snd body) `startsWith` "$ hsgrad "] <> go (drop 1 following)
    startsWith ls p = any (p `Text.isPrefixOf`) ls
    runs (l : ls) | Just command <- Text.stripPrefix "$ " l =
      let (out, rest) = break ("$ " `Text.isPrefixOf`) ls
       in (command, out) : runs rest
    runs _ = []

-- | Split a command line into words, like a shell: at spaces, except
-- inside double quotes. (Enough for the tutorial: no escapes.)
shellWords :: String -> [String]
shellWords s = case dropWhile (== ' ') s of
  "" -> []
  s' -> let (w, rest) = word s' in w : shellWords rest
  where
    word ('"' : t) = let (q, rest) = break (== '"') t; (w, rest') = word (drop 1 rest) in (q <> w, rest')
    word (c : t) | c /= ' ' = let (w, rest) = word t in (c : w, rest)
    word t = ("", t)

tutorialSpec :: Spec
tutorialSpec = do
  page <- runIO (Utf8.readFile "docs/17-examples.md")
  slowOn <- runIO (maybe False (`notElem` ["", "0"]) <$> lookupEnv "HSGRAD_SLOW")
  let blocks = transcripts page

  it "has transcripts of every example" $ do
    files <- filter ((== ".hsg") . takeExtension) <$> listDirectory "examples"
    let commands = [c | b <- blocks, (c, _) <- blockRuns b]
    [f | f <- files, not (any (("examples/" <> Text.pack f) `Text.isInfixOf`) commands)] `shouldBe` []

  forM_ blocks $ \b ->
    it ("prints what the transcript at line " <> show (blockLine b) <> " shows") $
      if blockSlow b && not slowOn
        then pendingWith "slow: run with HSGRAD_SLOW=1 cabal test"
        else do
          dir <- scratch ("tutorial-" <> show (blockLine b))
          forM_ (blockRuns b) $ \(command, expected) -> do
            words' <- case shellWords (Text.unpack command) of
              "hsgrad" : args -> pure args
              _ -> fail ("only hsgrad commands can be checked: " <> Text.unpack command)
            (_, out) <- terminal dir words'
            let actual = Text.lines out
            unless (actual == expected) $
              expectationFailure
                ( unlines
                    [ "$ " <> Text.unpack command
                    , "printed:"
                    , Text.unpack out
                    , "but docs/17-examples.md shows:"
                    , Text.unpack (Text.unlines expected)
                    ]
                )
