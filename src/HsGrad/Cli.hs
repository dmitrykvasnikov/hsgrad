{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The command line of @hsgrad@, described with @optparse-applicative@
-- (@00-architecture.md@ §5): what each command and flag means, their
-- help texts, and how they become the driver's 'Options'.
--
-- Values that a diagnostic may point at (@--set x=[1, 2]@, @--of L@, the
-- file name) are 'Arg's with a span. The spans point into a one-line
-- text that the parser re-assembles from those values, in the order the
-- parsers are written, for example
--
-- @
-- hsgrad eval examples/neuron.hsg --set x=[1, 2]
-- @
--
-- so an error in a @--set@ value can be shown in a snippet of the command
-- line, like an error in a file. Each parser produces, besides its value,
-- a 'Layout' action that places its words on that line: the parsers are
-- 'Compose'd with 'Layout', which is again an applicative, so placing
-- comes for free with combining parsers.
-- See @docs/16-cli.md@ §2.3.
module HsGrad.Cli
  ( -- * The parser
    cliInfo
  , cliPrefs
  , parseArgs

    -- * Readers (for tests)
  , readSchedule
  , splitNames
  ) where

import Control.Monad.State.Strict (State, get, put, runState)
import Data.Functor.Compose (Compose (..))
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad qualified
import HsGrad.Data.Csv (readNumber)
import HsGrad.Driver
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..))
import HsGrad.Train.Optim (LrSchedule (..))
import Options.Applicative
import Options.Applicative.Help.Pretty (Doc, pretty, vsep)

-- ---------------------------------------------------------------------------
-- Placing words on the command line

-- | Building the one-line command line: the next free column and the
-- words so far (last first).
type Layout = State (Int, [Text])

-- | A parser of the command line that also places what it read.
type CliParser = Compose Parser Layout

-- | Place a word; its span.
word :: Text -> Layout Span
word w = do
  (col, ws) <- get
  put (col + Text.length w + 1, w : ws)
  pure (SrcSpan commandLineFile (Pos 1 col) (Pos 1 (col + Text.length w)))

-- | Place a flag and its value; the value as an 'Arg'.
flagged :: Text -> Text -> Layout Arg
flagged name v = word name *> (Arg v <$> word v)

-- | A parser that places nothing.
plain :: Parser a -> CliParser a
plain p = Compose (pure <$> p)

-- | A parser that places a fixed word (the program, the command).
fixed :: Text -> CliParser Arg
fixed w = Compose (pure (Arg w <$> word w))

-- | 'many' and 'optional' for placing parsers: 'Compose' has its own,
-- but they would recurse forever through optparse's 'Parser'.
manyArgs :: CliParser a -> CliParser [a]
manyArgs (Compose p) = Compose (sequenceA <$> many p)

optionalArg :: CliParser a -> CliParser (Maybe a)
optionalArg (Compose p) = Compose (sequenceA <$> optional p)

-- | Run the layout: the command line text, and the options that point
-- into it.
finish :: Layout (Text -> Options) -> Options
finish layout = make (Text.unwords (reverse ws))
  where
    (make, (_, ws)) = runState (word "hsgrad" *> layout) (1, [])

-- ---------------------------------------------------------------------------
-- The parser

-- | Preferences: show the help when a command is given no arguments.
cliPrefs :: ParserPrefs
cliPrefs = prefs (showHelpOnEmpty <> showHelpOnError)

-- | Parse arguments without IO (for tests): 'Success', or 'Failure' with
-- the message optparse would print.
parseArgs :: [String] -> ParserResult Options
parseArgs = execParserPure cliPrefs cliInfo

-- | The whole command line.
cliInfo :: ParserInfo Options
cliInfo =
  info
    (commands <**> helper <**> simpleVersioner HsGrad.version)
    ( fullDesc
        <> header ("hsgrad " <> HsGrad.version <> ": scalar autodiff with a small model language")
        <> progDesc
          "Read a .hsg file (inputs, parameters, data, model, loss) and check it, \
          \evaluate it, differentiate it, draw it, train it or predict with it. \
          \Run `hsgrad COMMAND --help` for the options of a command."
        <> footerDoc
          ( examples
              [ "hsgrad check examples/moons.hsg"
              , "hsgrad derive examples/intro.hsg --of L --wrt a,b,c --simplify"
              , "hsgrad train examples/xor.hsg --steps 200 --seed 1 --save-params xor.params"
              , "hsgrad predict examples/xor.hsg --params xor.params --set \"x=[0, 1]\""
              ]
          )
    )
  where
    commands =
      fmap finish . hsubparser $
        mconcat
          [ subcommand "check" "Parse and check a file, load its data, and compile it; report every problem." checkExamples (pure Check)
          , subcommand "eval" "Evaluate named values (by default every `let` and the loss)." evalExamples (Eval <$> evalOptions)
          , subcommand "derive" "Differentiate a value with respect to inputs, parameters or other values, as formulas or a graph." deriveExamples (Derive <$> deriveOptions)
          , subcommand "plot" "Draw the graph of a value as Graphviz DOT, with values and gradients." plotExamples (Plot <$> plotOptions)
          , subcommand "train" "Train the parameters on the loss with gradient descent." trainExamples (Train <$> trainOptions)
          , subcommand "predict" "Evaluate the model on arguments given with --set, with saved parameters." predictExamples (Predict <$> predictOptions)
          ]
    checkExamples = ["hsgrad check examples/moons.hsg"]
    evalExamples =
      [ "hsgrad eval examples/intro.hsg"
      , "hsgrad eval examples/intro.hsg --set a=3 --show L"
      , "hsgrad eval examples/neuron.hsg --set \"x=[1, 0, -1]\" --show y --seed 7"
      ]
    deriveExamples =
      [ "hsgrad derive examples/intro.hsg --of L --wrt a,b,c --simplify"
      , "hsgrad derive examples/neuron.hsg --of y --wrt w,b --simplify --dot neuron-grad.dot"
      , "hsgrad derive examples/xor.hsg --of loss --wrt \"W1[0,1]\" --format dot"
      ]
    plotExamples =
      [ "hsgrad plot examples/neuron.hsg --of y --dot neuron.dot"
      , "dot -Tsvg neuron.dot -o neuron.svg"
      , "hsgrad plot examples/xor.hsg --of loss --dot - --cluster | dot -Tsvg > xor.svg"
      ]
    trainExamples =
      [ "hsgrad train examples/xor.hsg --steps 200 --seed 1 --save-params xor.params"
      , "hsgrad train examples/moons.hsg --steps 100 --lr 1 --lr-schedule linear:0.1 --batch 32"
      ]
    predictExamples =
      [ "hsgrad predict examples/xor.hsg --params xor.params --set \"x=[0, 1]\""
      ]

-- | One command: its name, description, examples and options. Every
-- command takes the file and @--color@.
subcommand :: String -> String -> [String] -> CliParser Command -> Mod CommandFields (Layout (Text -> Options))
subcommand name desc exs body =
  command name $
    info
      -- No 'helper' here: 'hsubparser' adds @--help@ to every command.
      (getCompose parser)
      (fullDesc <> progDesc desc <> footerDoc (examples exs))
  where
    parser = make <$> fixed (Text.pack name) <*> file <*> body <*> plain colour
    make cmd f b c line = Options line c cmd f b
    file = Compose ((\v -> Arg v <$> word v) . Text.pack <$> strArgument (metavar "FILE" <> help "The .hsg file"))

-- | An "Examples:" footer.
examples :: [String] -> Maybe Doc
examples exs = Just (vsep (pretty ("Examples:" :: String) : map (pretty . ("  " <>)) exs))

-- ---------------------------------------------------------------------------
-- Options of the commands

colour :: Parser Colour
colour =
  option
    (enumeration [("auto", ColourAuto), ("always", ColourAlways), ("never", ColourNever)])
    ( long "color"
        <> metavar "auto|always|never"
        <> value ColourAuto
        <> help "Colour in error messages; auto means when stderr is a terminal and NO_COLOR is not set (default: auto)"
    )

-- | @--set NAME=VALUE@, any number of times.
sets :: String -> CliParser [Arg]
sets what =
  manyArgs . Compose $
    flagged "--set" . Text.pack
      <$> option
        (eitherReader setting)
        (long "set" <> metavar "NAME=VALUE" <> help what)
  where
    setting s = case break (== '=') s of
      (name, '=' : _) | not (null name) -> Right s
      _ -> Left ("expected NAME=VALUE, like x=[1, 2] or a=3, not " <> show s)

-- | How leaves get values: @--set@, @--params@, @--seed@.
bindings :: CliParser Bindings
bindings =
  Bindings
    <$> sets "Give an input a value: a number, a vector [1, 2] or a matrix [[1, 2], [3, 4]] (repeatable)"
    <*> optionalArg (valueArg "--params" (long "params" <> metavar "PARAMS" <> help "Read the parameters from a file saved by `train --save-params`"))
    <*> plain seed

seed :: Parser Int
seed = option auto (long "seed" <> metavar "N" <> value defaultSeed <> showDefault <> help "Seed for the random initial parameters")

-- | A flag whose value is an 'Arg'.
valueArg :: Text -> Mod OptionFields String -> CliParser Arg
valueArg name mods = Compose (flagged name . Text.pack <$> strOption mods)

evalOptions :: CliParser EvalOptions
evalOptions =
  EvalOptions
    <$> bindings
    <*> manyArgs (valueArg "--show" (long "show" <> metavar "NAME" <> help "Show this value: an input, parameter, `let`, `loss`, or an element like W1[0,1] (repeatable; default: every `let` and the loss)"))

deriveOptions :: CliParser DeriveOptions
deriveOptions =
  DeriveOptions
    <$> bindings
    <*> valueArg "--of" (long "of" <> metavar "NAME" <> help "The value to differentiate (`loss`, a `let`, …; a vector gives one gradient per element)")
    <*> wrt
    <*> plain (switch (long "simplify" <> help "Simplify the gradients (gather like terms, fold constants)"))
    <*> plain (optional (strOption (long "dot" <> metavar "OUT" <> help "Also write the gradient graph as Graphviz DOT to OUT (- for stdout)")))
    <*> plain (option (enumeration [("text", TextFormat), ("dot", DotFormat)]) (long "format" <> metavar "dot|text" <> value TextFormat <> help "What to print: formulas with values, or DOT (default: text)"))
  where
    wrt = Compose (placeNames . Text.pack <$> strOption (long "wrt" <> metavar "NAME[,NAME...]" <> help "Differentiate with respect to these (whole vectors and matrices are expanded to their elements)"))
    placeNames v = do
      _ <- word "--wrt"
      sp <- word v
      pure [Arg n (shifted sp from (Text.length n)) | (from, n) <- splitNames v]
    shifted (SrcSpan f (Pos l c) _) from len = SrcSpan f (Pos l (c + from)) (Pos l (c + from + len))

-- | Split @a,b,W1[0,1]@ at the commas that are not inside brackets, with
-- each name's offset in the text.
splitNames :: Text -> [(Int, Text)]
splitNames t = go 0 0 0 (Text.unpack t)
  where
    go start i _ [] = [(start, slice start i)]
    go start i depth (ch : rest) = case ch of
      ',' | depth == (0 :: Int) -> (start, slice start i) : go (i + 1) (i + 1) depth rest
      '[' -> go start (i + 1) (depth + 1) rest
      ']' -> go start (i + 1) (max 0 (depth - 1)) rest
      _ -> go start (i + 1) depth rest
    slice from to = Text.take (to - from) (Text.drop from t)

plotOptions :: CliParser PlotOptions
plotOptions =
  PlotOptions
    <$> bindings
    <*> valueArg "--of" (long "of" <> metavar "NAME" <> help "The value whose graph to draw")
    <*> plain (strOption (long "dot" <> metavar "OUT" <> help "Write the graph as Graphviz DOT to OUT (- for stdout)"))
    <*> plain (switch (long "cluster" <> help "Group the nodes of each source expression in a box"))

trainOptions :: CliParser TrainOptions
trainOptions =
  TrainOptions
    <$> sets "Give an input a value (repeatable)"
    <*> plain (option positive (long "steps" <> metavar "N" <> value 100 <> showDefault <> help "Number of gradient steps"))
    <*> plain (option rate (long "lr" <> metavar "X" <> value 0.1 <> showDefault <> help "Learning rate"))
    <*> plain (option (eitherReader readSchedule) (long "lr-schedule" <> metavar "const|linear:END|step:K:F" <> value ConstantRate <> help "Learning-rate schedule: constant, linear to END at the last step, or times F every K steps (default: const)"))
    <*> plain (optional (option positive (long "batch" <> metavar "N" <> help "Rows per step, sampled at random (default: every row)")))
    <*> plain seed
    <*> plain (option positive (long "report-every" <> metavar "N" <> value 10 <> showDefault <> help "Print a report every N steps (and after the last)"))
    <*> plain (optional (strOption (long "save-params" <> metavar "OUT" <> help "Save the trained parameters to OUT, for `predict --params`")))

predictOptions :: CliParser PredictOptions
predictOptions =
  PredictOptions
    <$> valueArg "--params" (long "params" <> metavar "PARAMS" <> help "The parameters, saved by `train --save-params`")
    <*> sets "Give a model argument (or an input) a value (repeatable)"

-- ---------------------------------------------------------------------------
-- Readers

-- | One of a few words.
enumeration :: [(String, a)] -> ReadM a
enumeration table = eitherReader $ \s -> case lookup s table of
  Just a -> Right a
  Nothing -> Left ("expected one of " <> commaList (map fst table) <> ", not " <> show s)
  where
    commaList = foldr1 (\a b -> a <> ", " <> b)

-- | A whole number, at least 1.
positive :: ReadM Int
positive = eitherReader $ \s -> case reads s of
  [(n, "")] | n >= 1 -> Right n
  _ -> Left ("expected a whole number of at least 1, not " <> show s)

-- | A positive finite number, read exactly like a CSV cell.
rate :: ReadM Double
rate = eitherReader $ \s -> case readNumber (Text.pack s) of
  Right x | x > 0 -> Right x
  _ -> Left ("expected a positive number, like 0.1, not " <> show s)

-- | @const@, @linear:END@ or @step:K:F@.
readSchedule :: String -> Either String LrSchedule
readSchedule s = case Text.splitOn ":" (Text.pack s) of
  ["const"] -> Right ConstantRate
  ["linear", end] | Right e <- readNumber end, e >= 0 -> Right (LinearTo e)
  ["step", k, f]
    | [(n, "")] <- reads (Text.unpack k)
    , n >= 1
    , Right x <- readNumber f
    , x > 0 ->
        Right (StepDecay n x)
  _ ->
    Left
      ( "expected const, linear:END (END ≥ 0) or step:K:F (K ≥ 1 steps, F > 0), not "
          <> show s
      )
