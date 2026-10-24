{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The driver: everything between the command line and the libraries.
--
-- A run of @hsgrad@ is a pipeline (@00-architecture.md@ §1):
--
-- @
-- FILE ──► parse ──► check ──► load datasets ──► compile ──► run the command
-- @
--
-- Each stage either produces a value or stops with diagnostics; warnings
-- are collected along the way and never stop anything. The pipeline is
-- written with 'Outcome', a small monad for exactly that. Only two things
-- are 'IO': reading files ('load') and, in @app/Main.hs@, printing and
-- writing the results. Everything in between is pure, so the tests run
-- whole commands without a process or a terminal.
--
-- The driver also turns the core's runtime errors into diagnostics that
-- point into the source (E0501–E0504, through the provenance of graph
-- nodes, D4), and problems with values given on the command line into
-- diagnostics that point into the command line itself.
-- See @docs/16-cli.md@.
module HsGrad.Driver
  ( -- * What to run
    Options (..)
  , Colour (..)
  , Arg (..)
  , Command (..)
  , Bindings (..)
  , EvalOptions (..)
  , DeriveOptions (..)
  , Format (..)
  , PlotOptions (..)
  , TrainOptions (..)
  , PredictOptions (..)
  , commandLineFile
  , defaultSeed
  , paramsArg

    -- * Outcomes
  , Outcome (..)
  , Result (..)

    -- * Loading a program
  , Session (..)
  , load
  , checkSource

    -- * Running a command
  , Stream (..)
  , run

    -- * Everything at once
  , Transcript (..)
  , execute
  , renderTranscript
  ) where

import Control.Monad (ap, forM)
import Data.Char (isSpace)
import Data.Foldable (toList)
import Data.Function ((&))
import Data.List (find)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, isJust, mapMaybe)
import Numeric (showEFloat)
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Backprop (BackpropError (..), GradError (..), gradients)
import HsGrad.Core.Eval (Env (..), EvalError (..), Values, evaluate, valueAt)
import HsGrad.Core.Graph (Graph, Node (..), NodeId, Provenance (..), extend, nodeAt, nodeCount, nodeIndex)
import HsGrad.Core.Op (BinOp (..), Leaf (..), Name (..), Op (..), UnOp (..))
import HsGrad.Core.Pretty (Rendered (..), defaultPrettyOptions, renderNodes, showConstant, showExponent)
import HsGrad.Core.Simplify (simplify)
import HsGrad.Core.Symbolic (backwardNodes, gradNodeOrZero)
import HsGrad.Data.Dataset (Dataset (..), ReadProblem (..), checkBatch, datasetSize, loadDatasets, readCsvFile, toFeed)
import HsGrad.Driver.ParamsFile (readParams, renderParams, trainShape)
import HsGrad.Lang.Check
  ( CheckedDecl (..)
  , Checked (..)
  , DatasetInfo (..)
  , FunctionInfo (..)
  , InputInfo (..)
  , ModelInfo (..)
  , checkProgram
  , checkedDatasets
  , checkedInputs
  , checkedModel
  , checkedParams
  , describeChecked
  , suggestName
  )
import HsGrad.Lang.Compile
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Parser (parseLiteral, parseProgram)
import HsGrad.Lang.Shape (Shape, prettyShape)
import HsGrad.Lang.Shape qualified as Shape
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..), pointSpan, startPos)
import HsGrad.Lang.Syntax (Literal (..), Number (..))
import HsGrad.Train.Loop qualified as Loop
import HsGrad.Train.Optim (LrSchedule (..), Sgd (..))
import HsGrad.Train.Param (ParamSpec (..), Params, Seed, initParams, leafNames)
import HsGrad.Viz.Dot (DotOptions (..), defaultDotOptions, highlight, highlightAs, renderDotFrom)

-- ---------------------------------------------------------------------------
-- Options

-- | What the user asked for: the parsed command line ("HsGrad.Cli" makes
-- it). The @.hsg@ file says /what/ is computed, these options say /how/
-- (D11).
data Options = Options
  { optCommandLine :: !Text
  -- ^ The command line as the driver understood it, on one line: the
  -- text that the spans of every 'Arg' point into (file
  -- 'commandLineFile'). Diagnostics about a @--set@ value show it.
  , optColour :: !Colour
  , optCommandWord :: !Arg
  -- ^ The command (@train@), for "this command needs a @loss@".
  , optFile :: !Arg
  -- ^ The @.hsg@ file.
  , optCommand :: !Command
  }
  deriving (Eq, Show)

-- | @--color auto|always|never@: colour in diagnostics. @auto@ means
-- "when stderr is a terminal" (decided in @app/Main.hs@).
data Colour = ColourAuto | ColourAlways | ColourNever
  deriving (Eq, Show)

-- | A value from the command line, with its place in 'optCommandLine'.
data Arg = Arg
  { argText :: !Text
  , argSpan :: !Span
  }
  deriving (Eq, Show)

-- | The commands of @00-architecture.md@ §5.
data Command
  = Check
  | Eval !EvalOptions
  | Derive !DeriveOptions
  | Plot !PlotOptions
  | Train !TrainOptions
  | Predict !PredictOptions
  deriving (Eq, Show)

-- | Where the leaves get their values, for the commands that evaluate
-- the graph (@eval@, @derive@, @plot@): inputs from their defaults and
-- @--set NAME=VALUE@, parameters from a file (@--params@) or from their
-- initialisers with @--seed@.
data Bindings = Bindings
  { bindSet :: ![Arg]
  -- ^ @--set NAME=VALUE@, each as written (@x=[1, 2]@).
  , bindParams :: !(Maybe Arg)
  , bindSeed :: !Seed
  }
  deriving (Eq, Show)

-- | @eval FILE [--set NAME=VALUE ...] [--show NAME ...]@.
data EvalOptions = EvalOptions
  { evalBindings :: !Bindings
  , evalShow :: ![Arg]
  -- ^ What to show; nothing given means every @let@, then the loss.
  }
  deriving (Eq, Show)

-- | What @derive@ prints.
data Format = TextFormat | DotFormat
  deriving (Eq, Show)

-- | @derive FILE --of NAME --wrt NAME[,NAME...] [--simplify] [--dot OUT]
-- [--format dot|text]@.
data DeriveOptions = DeriveOptions
  { deriveBindings :: !Bindings
  , deriveOf :: !Arg
  , deriveWrt :: ![Arg]
  -- ^ One per name of the comma-separated list.
  , deriveSimplify :: !Bool
  , deriveDot :: !(Maybe FilePath)
  -- ^ Also write the gradient graph as DOT to this file (@-@: stdout).
  , deriveFormat :: !Format
  -- ^ What to print on stdout: the gradients as text, or as DOT.
  }
  deriving (Eq, Show)

-- | @plot FILE --of NAME --dot OUT [--cluster]@.
data PlotOptions = PlotOptions
  { plotBindings :: !Bindings
  , plotOf :: !Arg
  , plotDot :: !FilePath
  , plotCluster :: !Bool
  }
  deriving (Eq, Show)

-- | @train FILE [--steps N] [--lr X] [--lr-schedule …] [--batch N]
-- [--seed N] [--report-every N] [--save-params OUT]@.
data TrainOptions = TrainOptions
  { trainingSet :: ![Arg]
  , trainingSteps :: !Int
  , trainingRate :: !Double
  , trainingSchedule :: !LrSchedule
  , trainingBatch :: !(Maybe Int)
  -- ^ Rows per step; 'Nothing' means every row (D10).
  , trainingSeed :: !Seed
  , trainingReportEvery :: !Int
  , trainingSaveParams :: !(Maybe FilePath)
  }
  deriving (Eq, Show)

-- | @predict FILE --params PARAMS [--set NAME=VALUE ...]@.
data PredictOptions = PredictOptions
  { predictParams :: !Arg
  , predictSet :: ![Arg]
  }
  deriving (Eq, Show)

-- | The file name of the command line in spans and 'Sources'.
commandLineFile :: FilePath
commandLineFile = "<command line>"

-- | The seed when @--seed@ is not given (micrograd's).
defaultSeed :: Seed
defaultSeed = 1337

-- | The parameter file a command reads, if any.
paramsArg :: Command -> Maybe Arg
paramsArg = \case
  Eval o -> bindParams (evalBindings o)
  Derive o -> bindParams (deriveBindings o)
  Plot o -> bindParams (plotBindings o)
  Predict o -> Just (predictParams o)
  Check -> Nothing
  Train _ -> Nothing

-- ---------------------------------------------------------------------------
-- Outcomes

-- | The outcome of a stage: the diagnostics it reported (warnings, and
-- errors if it stopped), and its result.
--
-- As a monad, @a >>= f@ runs @f@ only when @a@ produced a value, and
-- keeps the diagnostics of both: a warning of the checker is still there
-- when the command fails later. It is @Writer [Diagnostic]@ combined with
-- @Either@, written out by hand.
data Outcome a = Outcome
  { outcomeDiagnostics :: ![Diagnostic]
  , outcomeResult :: !(Result a)
  }
  deriving (Show)

-- | How a stage ended.
data Result a
  = Ok a
  | -- | Stopped by errors; they are in the diagnostics.
    Stopped
  | -- | A bug: a promise of an earlier stage was broken. Not the user's
    -- fault, so not a diagnostic.
    Internal Text
  deriving (Eq, Show, Functor)

instance Functor Outcome where
  fmap f (Outcome ds r) = Outcome ds (fmap f r)

instance Applicative Outcome where
  pure = Outcome [] . Ok
  (<*>) = ap

instance Monad Outcome where
  Outcome ds r >>= f = case r of
    Ok a -> let Outcome ds' r' = f a in Outcome (ds <> ds') r'
    Stopped -> Outcome ds Stopped
    Internal t -> Outcome ds (Internal t)

-- | Report diagnostics; stop if one of them is an error.
report :: [Diagnostic] -> Outcome ()
report ds = Outcome ds (if hasErrors ds then Stopped else Ok ())

-- | Stop with errors.
failWith :: [Diagnostic] -> Outcome a
failWith ds = Outcome ds Stopped

-- | Stop with a bug.
internal :: Text -> Outcome a
internal = Outcome [] . Internal

-- | Two independent stages: both results, or the problems of both (not
-- only of the first, as with '>>=').
alongside :: Outcome a -> Outcome b -> Outcome (a, b)
alongside (Outcome d1 r1) (Outcome d2 r2) = Outcome (d1 <> d2) $ case (r1, r2) of
  (Ok a, Ok b) -> Ok (a, b)
  (Internal t, _) -> Internal t
  (_, Internal t) -> Internal t
  _ -> Stopped

-- | Every problem of several independent checks, or all their results.
-- (Like 'traverse', but it does not stop at the first 'Left'.)
collect :: [Either [Diagnostic] a] -> Outcome [a]
collect results = case concat [ds | Left ds <- results] of
  [] -> pure [a | Right a <- results]
  ds -> failWith ds

-- ---------------------------------------------------------------------------
-- Loading

-- | A program that parsed, checked and has its data: what every command
-- starts from.
data Session = Session
  { sessionPath :: !FilePath
  , sessionChecked :: !Checked
  , sessionDatasets :: ![Dataset]
  -- ^ Every dataset of the program, with its rows.
  , sessionParamsFile :: !(Maybe (FilePath, Text))
  -- ^ The text of the @--params@ file, if the command has one.
  }

-- | Read the @.hsg@ file, parse and check it, read its datasets, and the
-- @--params@ file if the command has one.
--
-- Also returns the text of every file that was read, and of the command
-- line: what diagnostics point into ('renderDiagnostics' needs them).
load :: Options -> IO (Sources, Outcome Session)
load opts = do
  let path = Text.unpack (argText (optFile opts))
      sources0 = Map.singleton commandLineFile (optCommandLine opts)
  readText (optFile opts) >>= \case
    Left d -> pure (sources0, failWith [d])
    Right src -> do
      let sources1 = Map.insert path src sources0
      case checkSource path src of
        Outcome ds (Ok checked) -> do
          (csvSources, built) <- loadDatasets path (checkedDatasets checked)
          paramsText <- traverse readArgFile (paramsArg (optCommand opts))
          let sources2 = Map.unions [sources1, csvSources, Map.fromList [p | Just (Right p) <- [paramsText]]]
              session = do
                datasets <- either failWith pure built
                params <- traverse (either (failWith . pure) pure) paramsText
                pure (Session path checked datasets params)
          pure (sources2, Outcome ds (Ok ()) >> session)
        Outcome ds r -> pure (sources1, Outcome ds (Stopped <$ r) >> failWith [])
  where
    readArgFile a = fmap (\t -> (Text.unpack (argText a), t)) <$> readText a

-- | Read a file named on the command line, as UTF-8 text. A problem is an
-- E0401 pointing at the name on the command line.
readText :: Arg -> IO (Either Diagnostic Text)
readText a = either (Left . unreadable) Right <$> readCsvFile (Text.unpack (argText a))
  where
    unreadable problem =
      diagnostic E0401 ("cannot read `" <> argText a <> "`") (argSpan a) $ case problem of
        FileNotFound -> "no such file"
        PermissionDenied -> "permission denied"
        NotUtf8 -> "not UTF-8 text"
        OtherReadError t -> t

-- | Parse and check a program: its diagnostics, and the checked program
-- unless there was an error. Parse errors stop before the checker
-- (step 11).
checkSource :: FilePath -> Text -> Outcome Checked
checkSource path src = case parseProgram path src of
  Left ds -> failWith (toList ds)
  Right prog -> do
    let (ds, checked) = checkProgram prog
    report ds
    maybe (internal "the checker reported no error but gave no program") pure checked

-- | Compile the program, with a batch size for training (@--batch@) or
-- every row of every dataset.
compileSession :: Maybe Int -> Session -> Outcome Compiled
compileSession batch s =
  case compileProgram (batchSizes batch (sessionDatasets s)) (sessionChecked s) of
    Left e -> internal ("the compiler failed: " <> Text.pack (show e))
    Right c -> do
      -- E0405 only for the datasets the graph actually iterates over.
      report
        [ d'
        | Just b <- [batch]
        , d <- sessionDatasets s
        , dataName d `Map.member` compiledSlots c
        , d' <- checkBatch b d
        ]
      pure c

-- ---------------------------------------------------------------------------
-- Leaf values

-- | Something the user can give a value with @--set@: an input of the
-- program, or an argument of the model (for @predict@).
data Settable = Settable
  { settableSpec :: !InputSpec
  , settableWhat :: !Text
  -- ^ For messages: "input `x`", "argument `x` of `net`".
  , settableSpan :: !Span
  -- ^ Its declaration.
  }

-- | The inputs of the program.
inputSettables :: Session -> Compiled -> [Settable]
inputSettables s c =
  [ Settable spec ("input `" <> inputName i <> "`") (inputNameSpan i)
  | (spec, i) <- zip (compiledInputs c) (checkedInputs (sessionChecked s))
  ]

-- | The arguments of the model.
argumentSettables :: Session -> Predictor -> [Settable]
argumentSettables s p =
  [ Settable spec ("argument `" <> inputSpecName spec <> "` of `" <> predictorName p <> "`") sp
  | spec <- predictorArguments p
  , let sp = maybe (pointSpan (sessionPath s) startPos) modelNameSpan (checkedModel (sessionChecked s))
  ]

-- | The values of @--set NAME=VALUE@ arguments, by leaf name. Each value
-- is a literal, parsed by the language's parser and checked against the
-- shape of what it sets (E0503). The first settable with the name wins
-- (for @predict@: a model argument before an input of the same name).
settings :: [Settable] -> [Text] -> [Arg] -> Outcome (Map Name Double)
settings targets params args = Map.fromList . concat <$> collect (map setting args)
  where
    setting a = do
      let (name, rest) = Text.breakOn "=" (argText a)
          value = Text.drop 1 rest
          nameSpan = argPart a 0 (Text.length name)
          offset = Text.length name + 1
          valueSpan = argPart a offset (Text.length value)
      settable <- case find ((== name) . inputSpecName . settableSpec) targets of
        Just t -> Right t
        Nothing -> Left [unknownSetting name nameSpan]
      lit <- either (Left . map (shiftDiagnostic (argColumn a + offset - 1)) . toList) (Right . fmap (shiftSpan (argColumn a + offset - 1))) (parseLiteral commandLineFile value)
      (shape, values) <- either (Left . pure) Right (literalValues lit)
      let spec = settableSpec settable
          want = inputSpecShape spec
      if shape == want
        then Right (zip (inputSpecLeaves spec) values)
        else
          Left
            [ diagnostic E0503 ("the value for " <> settableWhat settable <> " has the wrong shape") valueSpan ("a `" <> prettyShape shape <> "`")
                & withSecondary (settableSpan settable) ("declared as a `" <> prettyShape want <> "` here")
                & maybe id (\e -> withHint ("write a `" <> prettyShape want <> "`, like `" <> name <> "=" <> e <> "`")) (exampleLiteral want)
            ]
    known = map (inputSpecName . settableSpec) targets
    unknownSetting name sp
      | name `elem` params =
          diagnostic E0201 ("`" <> name <> "` is a parameter, not an input") sp "can't be set with `--set`"
            & withHint "parameters come from their initialiser and `--seed`, or from a `--params` file"
      | otherwise =
          diagnostic E0201 ("there is no input named `" <> name <> "`") sp "unknown input"
            & suggestion name known
            & withNote (if null known then "the program has no inputs" else "the inputs are " <> commas known)

-- | The shape and the row-major numbers of a literal (E0312 for a
-- malformed one: rows of different lengths, numbers mixed with rows, a
-- matrix inside a matrix).
literalValues :: Literal Span -> Either Diagnostic (Shape Int, [Double])
literalValues = \case
  LitNumber n -> Right (Shape.Real, [fromRational (numberValue n)])
  LitVector sp items
    | Just xs <- mapM number items -> Right (Shape.Vec (length xs), xs)
    | Just rows@(r : _) <- mapM row items ->
        if all ((== length r) . length) rows
          then Right (Shape.Mat (length rows) (length r), concat rows)
          else Left (malformed sp "rows of different lengths")
    | otherwise -> Left (malformed sp "mixes numbers and rows, or nests too deeply")
  where
    number = \case
      LitNumber n -> Just (fromRational (numberValue n))
      LitVector _ _ -> Nothing
    row = \case
      LitVector _ xs -> mapM number xs
      LitNumber _ -> Nothing
    malformed sp what = diagnostic E0312 "malformed vector or matrix literal" sp what

-- | An example literal of a shape, for hints: @[1, 2, 3]@. Only for small
-- shapes.
exampleLiteral :: Shape Int -> Maybe Text
exampleLiteral sh
  | Shape.shapeSize sh > 6 = Nothing
  | otherwise = Just $ case sh of
      Shape.Real -> "1.5"
      Shape.Vec n -> list (take n numbers)
      Shape.Mat r c -> list [list (take c (drop (i * c) numbers)) | i <- [0 .. r - 1]]
  where
    numbers = map (Text.pack . show) [1 :: Int ..]
    list xs = "[" <> Text.intercalate ", " xs <> "]"

-- | The parameters: from the @--params@ file if there is one, else from
-- their initialisers.
parameterStore :: Session -> Compiled -> Seed -> Outcome Params
parameterStore s c seed = case sessionParamsFile s of
  Just (path, text) -> either failWith pure (readParams path text (checkedParams (sessionChecked s)))
  Nothing -> pure (initParams seed (compiledParams c))

-- | Every row of every dataset bound to its slot (the full batch: slot
-- @i@ holds row @i@), for evaluating a graph outside training.
slotInputs :: Session -> Map Text [[Name]] -> Map Name Double
slotInputs s slots =
  Map.fromList
    [ binding
    | d <- sessionDatasets s
    , Just named <- [Map.lookup (dataName d) slots]
    , (leaves, row) <- zip named (dataRows d)
    , binding <- zip leaves row
    ]

-- | The environment of @eval@, @derive@ and @plot@.
environment :: Session -> Compiled -> Bindings -> Outcome Env
environment s c b = do
  set <- settings (inputSettables s c) (map specName (compiledParams c)) (bindSet b)
  params <- parameterStore s c (bindSeed b)
  let inputs = Map.unions [set, inputDefaults c, slotInputs s (compiledSlots c)]
  pure (Env inputs params)

-- ---------------------------------------------------------------------------
-- Names on the command line

-- | A value named on the command line (@--of L@, @--show h@, @--wrt W1@):
-- the name as written (spaces removed) and its value.
data Target = Target
  { targetName :: !Text
  , targetValue :: !Value
  }

-- | Look a name up: an input, a parameter, a top-level @let@, @loss@, or
-- one element of a vector or matrix (@x[1]@, @W1[0,1]@).
target :: Session -> Compiled -> Arg -> Either [Diagnostic] Target
target s c a
  | name == "loss", Just l <- compiledLoss c = Right (Target name (Real l))
  | Just v <- lookupName name c = Right (Target name v)
  | base /= name, Just v <- lookupName base c = case lookup name (elementsOf base v) of
      Just i -> Right (Target name (Real i))
      Nothing ->
        Left
          [ diagnostic E0305 ("`" <> base <> "` has no element `" <> name <> "`") (argSpan a) ("`" <> base <> "` is a `" <> prettyShape (valueShape v) <> "`")
              & withNote ("its elements are " <> range (map fst (elementsOf base v)))
          ]
  | name == "loss" =
      Left [diagnostic E0201 ("`" <> Text.pack (sessionPath s) <> "` has no `loss`") (argSpan a) "no `loss` declaration"]
  | Just kind <- lookup name kinds =
      Left
        [ diagnostic E0204 ("`" <> name <> "` is " <> kind <> ", not a value") (argSpan a) "needs a value"
            & withNote "values are inputs, parameters, top-level `let`s and `loss`"
        ]
  | otherwise =
      Left
        [ diagnostic E0201 ("no value named `" <> name <> "` in `" <> Text.pack (sessionPath s) <> "`") (argSpan a) "unknown name"
            & suggestion name (Map.keys (compiledNames c) <> ["loss" | isJust (compiledLoss c)])
            & withNote "values are inputs, parameters, top-level `let`s and `loss`"
        ]
  where
    name = Text.filter (not . isSpace) (argText a)
    base = Text.takeWhile (/= '[') name
    kinds = flip mapMaybe (checkedDecls (sessionChecked s)) $ \case
      CFunction f -> Just (functionName f, "a function")
      CModel m -> Just (modelName m, "the model")
      CData d -> Just (datasetName d, "a dataset")
      _ -> Nothing
    range ns = case ns of
      [n] -> "just `" <> n <> "`"
      _ -> "`" <> head' ns <> "` to `" <> last' ns <> "`"
    head' = maybe "" fst . uncons'
    last' = head' . reverse
    uncons' xs = case xs of
      x : rest -> Just (x, rest)
      [] -> Nothing

-- | The scalar outputs of several names ('elements' of each 'target').
names :: Session -> Compiled -> [Arg] -> Outcome [(Text, NodeId)]
names s c args = concatMap elements <$> collect (map (target s c) args)

-- | The elements of a value with their names: @x[0]@, @x[1]@ (row-major).
elementsOf :: Text -> Value -> [(Text, NodeId)]
elementsOf base v = zip [n | Name n <- leafNames base (trainShape (valueShape v))] (valueNodes v)

-- | The scalar outputs a target stands for: itself if it is a 'Real',
-- else its elements.
elements :: Target -> [(Text, NodeId)]
elements (Target n v) = case v of
  Real i -> [(n, i)]
  _ -> elementsOf n v

-- ---------------------------------------------------------------------------
-- Runtime errors as diagnostics

-- | What a runtime diagnostic needs to know about the graph being run.
data Runtime = Runtime
  { rtGraph :: !Graph
  , rtSettables :: ![Settable]
  -- ^ What an unbound input leaf belongs to (E0502).
  , rtIsGradient :: !(NodeId -> Bool)
  -- ^ Does a node compute a gradient (E0504) rather than a value (E0501)?
  , rtFallback :: !Span
  -- ^ For a node without a span (never, for compiled graphs).
  }

-- | The runtime of a program's own graph.
runtime :: Session -> Compiled -> Graph -> Runtime
runtime s c g = Runtime g (inputSettables s c) (const False) (pointSpan (sessionPath s) startPos)

-- | Evaluate, or stop with the diagnostics of 'evalFailure'.
evaluated :: Runtime -> Env -> [NodeId] -> Outcome Values
evaluated rt env outs = either (evalFailure rt) pure (evaluate (rtGraph rt) env outs)

-- | Evaluate if every input has a value; 'Nothing' if one is missing
-- (values are optional in @derive@ and @plot@). A non-finite value is
-- still an error.
optionalValues :: Runtime -> Env -> [NodeId] -> Outcome (Maybe Values)
optionalValues rt env outs = case evaluate (rtGraph rt) env outs of
  Right vals -> pure (Just vals)
  Left (UnboundLeaves _) -> pure Nothing
  Left e -> evalFailure rt e

-- | A failed forward pass: inputs without a value (E0502), or a value
-- that is not finite (E0501; E0504 for a node of a gradient graph).
evalFailure :: Runtime -> EvalError -> Outcome a
evalFailure rt = \case
  UnboundLeaves leaves -> case missingInputs (rtSettables rt) [n | (_, Input n) <- toList leaves] of
    [] -> internal ("unbound leaves: " <> Text.pack (show (toList leaves)))
    ds -> failWith ds
  NonFiniteValue i args v ->
    failWith [nonFinite (if rtIsGradient rt i then E0504 else E0501) rt i args v]

-- | E0502, once per input (not per element) with a leaf in the list.
missingInputs :: [Settable] -> [Name] -> [Diagnostic]
missingInputs targets unbound =
  [ diagnostic E0502 (settableWhat t <> " has no value") (settableSpan t) "needs a value"
      & withHint ("give it one on the command line: `--set " <> inputSpecName spec <> "=" <> example spec <> "`")
  | t <- targets
  , let spec = settableSpec t
  , any (`elem` unbound) (inputSpecLeaves spec)
  ]
  where
    example spec = fromMaybe "…" (exampleLiteral (inputSpecShape spec))

-- | A node whose value is NaN or ±∞, at the span of the expression it
-- came from (D4).
nonFinite :: Code -> Runtime -> NodeId -> [Double] -> Double -> Diagnostic
nonFinite code rt i args v =
  diagnostic code (headline <> ": " <> problem) (nodeSpan rt i) ("`" <> computation op args <> "` is " <> showValue v)
    & maybe id withHint hint
  where
    op = nodeOp (nodeAt (rtGraph rt) i)
    (problem, hint) = case (code, op, args) of
      -- In a gradient graph, x ^ -0.5 at 0 is the slope of sqrt (step 7).
      (E0504, Un (PowC k), [0]) | k < 0 -> infiniteSlope
      _ -> opProblem op args
    headline = if code == E0504 then "non-finite gradient" else "non-finite value"

-- | The problem of a power below 1 at 0: an infinite derivative.
infiniteSlope :: (Text, Maybe Text)
infiniteSlope = ("infinite slope at 0", Just "the slope of `sqrt(x)` (`x ^ 0.5`), and of any power below 1, is infinite at x = 0")

-- | A failed backward pass (E0504): the gradient passed back through a
-- node is not finite.
gradFailure :: Runtime -> BackpropError -> Outcome a
gradFailure rt (NonFiniteGrad i args g locals) =
  failWith
    [ diagnostic E0504 ("non-finite gradient: " <> problem) (nodeSpan rt i) ("the gradient passed back through `" <> computation op args <> "` is not finite")
        & withNote ("the gradient arriving there is " <> showValue g <> ", the local derivatives are " <> Text.intercalate ", " (map showValue locals))
        & maybe id withHint hint
    ]
  where
    op = nodeOp (nodeAt (rtGraph rt) i)
    (problem, hint) = case (op, args) of
      (Un (PowC k), [0]) | k < 1 -> infiniteSlope
      _ -> ("overflow", Just "a gradient is too large for a Double (about 1.8e308)")

-- | Where a node came from.
nodeSpan :: Runtime -> NodeId -> Span
nodeSpan rt i = fromMaybe (rtFallback rt) (provSpan (nodeProv (nodeAt (rtGraph rt) i)))

-- | Why an operation gave a non-finite result, and a hint.
opProblem :: Op -> [Double] -> (Text, Maybe Text)
opProblem op args = case (op, args) of
  (Un Log, [x]) | x <= 0 -> (if x == 0 then "`log` of 0" else "`log` of a negative number", Just "`log` is defined only for positive numbers")
  (Un Exp, _) -> ("`exp` overflows", Just "`exp(x)` is too large for a Double when x is above about 709.78")
  (Un (PowC k), [x])
    | x < 0 && fractional k -> ("fractional power of a negative number", Just "a power like `^ 0.5` (or `sqrt`) needs a number that is not negative")
    | x == 0 && k < 0 -> ("negative power of 0", Just "`x ^ -1` is `1 / x`: it needs x ≠ 0")
  (Bin Div, [_, 0]) -> ("division by zero", Nothing)
  (Leaf (Param _), _) -> ("a parameter is not finite", Nothing)
  _ -> ("the result is too large", Just "every value must fit in a Double (up to about 1.8e308)")
  where
    fractional k = k /= fromInteger (round k)

-- | An operation applied to numbers, as text: @log(-1)@, @1 / 0@,
-- @(-4) ^ 0.5@.
computation :: Op -> [Double] -> Text
computation op args = case (op, map showValue args) of
  (Leaf (Input (Name n)), _) -> n
  (Leaf (Param (Name n)), _) -> n
  (Leaf (Const k), _) -> showValue k
  (Un Neg, [x]) -> "-" <> paren x
  (Un (PowC k), [x]) -> paren x <> " ^ " <> showExponent k
  (Un f, [x]) -> function f <> "(" <> x <> ")"
  (Bin b, [x, y]) -> x <> " " <> symbol b <> " " <> paren y
  _ -> Text.pack (show op)
  where
    paren t = if "-" `Text.isPrefixOf` t then "(" <> t <> ")" else t
    function = \case
      Exp -> "exp"
      Log -> "log"
      Tanh -> "tanh"
      Relu -> "relu"
      Step -> "step"
      Neg -> "-"
      PowC _ -> "pow"
    symbol = \case
      Add -> "+"
      Sub -> "-"
      Mul -> "*"
      Div -> "/"

-- ---------------------------------------------------------------------------
-- Running a command

-- | What a command prints, as it is computed: lines for stdout, then the
-- end: files to write (path @-@ means stdout), or diagnostics.
--
-- Training prints a report line every few steps while it runs; the stream
-- is lazy, so @app/Main.hs@ prints each line as soon as the step that
-- makes it is done.
data Stream
  = Line !Text Stream
  | End !(Outcome [(FilePath, Text)])

-- | Lines, then nothing more to do.
linesOnly :: [Text] -> Stream
linesOnly = foldr Line (End (pure []))

-- | Run the command on a loaded program. Problems found before any
-- output (a bad @--set@, an unknown name) are in the outer 'Outcome';
-- problems found later (a non-finite value at step 50) end the 'Stream'.
run :: Options -> Session -> Outcome Stream
run opts s = case optCommand opts of
  Check -> runCheck s
  Eval o -> runEval s o
  Derive o -> runDerive s o
  Plot o -> runPlot s o
  Train o -> runTrain opts s o
  Predict o -> runPredict opts s o

-- | @check@: everything up to and including the compiler, then a summary
-- of the declarations.
runCheck :: Session -> Outcome Stream
runCheck s = do
  c <- compileSession Nothing s
  pure . linesOnly $
    describeChecked (sessionChecked s)
      <> [Text.pack (sessionPath s) <> ": ok, " <> showInt (nodeCount (compiledGraph c)) <> " nodes"]

-- | @eval@: the values of named things, one line each.
runEval :: Session -> EvalOptions -> Outcome Stream
runEval s o = do
  c <- compileSession Nothing s
  (targets, env) <-
    alongside
      ( if null (evalShow o)
          then pure ([Target n v | (n, v) <- compiledLets c] <> [Target "loss" (Real l) | Just l <- [compiledLoss c]])
          else collect (map (target s c) (evalShow o))
      )
      (environment s c (evalBindings o))
  vals <- evaluated (runtime s c (compiledGraph c)) env (concatMap (valueNodes . targetValue) targets)
  pure (linesOnly (aligned [(targetName t, valueText vals (targetValue t)) | t <- targets]))

-- | @derive@: gradients of an output with respect to leaves (or any named
-- values), as text and/or DOT.
runDerive :: Session -> DeriveOptions -> Outcome Stream
runDerive s o = do
  c <- compileSession Nothing s
  ((outs, wrts), env) <-
    alongside
      (alongside (names s c [deriveOf o]) (names s c (deriveWrt o)))
      (environment s c (deriveBindings o))
  (g, named) <- gradientGraph c outs wrts
  let ids = map snd named
      forward = nodeCount (compiledGraph c)
      rt = (runtime s c g) {rtIsGradient = \i -> nodeIndex i >= forward}
  values <- optionalValues rt env ids
  (shown, shownIds) <- if deriveSimplify o then simplified g ids else pure (g, ids)
  let Rendered bindings roots = renderNodes defaultPrettyOptions shown shownIds
      text =
        equations [(n, r, showValue . (`valueAt` i) <$> values) | ((n, i), r) <- zip named roots]
          <> ["where" | not (null bindings)]
          <> ["  " <> b <> " = " <> e | (b, e) <- bindings]
      -- Values of the simplified graph's own nodes, for the picture.
      shownValues
        | deriveSimplify o = either (const Nothing) Just (evaluate shown env shownIds)
        | otherwise = values
      dot =
        renderDotFrom
          defaultDotOptions
            { dotValues = shownValues
            , dotHighlights =
                [ if displayName shown i == Just n then highlight i else highlightAs n i
                | (n, i) <- zip (map fst named) shownIds
                ]
            }
          shown
          shownIds
      files = [(path, dot) | Just path <- [deriveDot o]]
      out = case deriveFormat o of
        TextFormat -> text
        DotFormat -> Text.lines dot
  pure (foldr Line (End (pure files)) out)

-- | The gradient of every output with respect to every leaf, named
-- @∂out/∂leaf@, in a graph that extends the program's (step 7).
gradientGraph :: Compiled -> [(Text, NodeId)] -> [(Text, NodeId)] -> Outcome (Graph, [(Text, NodeId)])
gradientGraph c outs wrts = case extend (compiledGraph c) derivatives of
  Left e -> internal ("symbolic differentiation failed: " <> Text.pack (show e))
  Right (named, g) -> pure (g, named)
  where
    derivatives = fmap concat . forM outs $ \(o, out) -> do
      grads <- backwardNodes out
      forM wrts $ \(w, leaf) -> (,) ("∂" <> o <> "/∂" <> w) <$> gradNodeOrZero grads leaf

-- | Simplify (step 8), keeping the order of the outputs.
simplified :: Graph -> [NodeId] -> Outcome (Graph, [NodeId])
simplified g ids = case simplify g ids of
  Left e -> internal ("simplification failed: " <> Text.pack (show e))
  Right (m, g') -> maybe (internal "simplification lost an output") (pure . (g',)) (mapM (`Map.lookup` m) ids)

-- | The name a node shows in pictures: its label or leaf name.
displayName :: Graph -> NodeId -> Maybe Text
displayName g i = case nodeAt g i of
  Node (Leaf (Input (Name n))) _ _ -> Just n
  Node (Leaf (Param (Name n))) _ _ -> Just n
  n -> provLabel (nodeProv n)

-- | @plot@: the forward graph of an output as DOT, with values and (for
-- a 'Real' output) gradients when every input has a value.
runPlot :: Session -> PlotOptions -> Outcome Stream
runPlot s o = do
  c <- compileSession Nothing s
  (outs, env) <- alongside (names s c [plotOf o]) (environment s c (plotBindings o))
  let g = compiledGraph c
      rt = runtime s c g
      ids = map snd outs
  values <- optionalValues rt env ids
  grads <- case (values, ids) of
    (Just _, [out]) -> case gradients g env out of
      Right (_, gs) -> pure (Just gs)
      Left (BackwardError e) -> gradFailure rt e
      Left (ForwardError e) -> evalFailure rt e
    _ -> pure Nothing
  let dot =
        renderDotFrom
          defaultDotOptions
            { dotValues = values
            , dotGrads = grads
            , dotClusters = plotCluster o
            , dotHighlights = [highlightAs n i | (n, i) <- outs, displayName g i /= Just n]
            }
          g
          ids
  pure (End (pure [(plotDot o, dot)]))

-- | @train@: SGD on the loss (step 9), a report line every few steps, and
-- the trained parameters at the end.
runTrain :: Options -> Session -> TrainOptions -> Outcome Stream
runTrain opts s o = do
  c <- compileSession (trainingBatch o) s
  loss <- maybe (failWith [missingDeclaration opts "loss" "`train` minimises the `loss`"]) pure (compiledLoss c)
  set <- settings (inputSettables s c) (map specName (compiledParams c)) (trainingSet o)
  let feeds = [toFeed d slots | d <- sessionDatasets s, Just slots <- [Map.lookup (dataName d) (compiledSlots c)]]
      setup = Loop.Setup (compiledGraph c) loss (Map.union set (inputDefaults c)) feeds (compiledAccuracy c)
      sgd = Sgd (trainingRate o) (trainingSchedule o)
      config = Loop.TrainConfig (trainingSteps o) sgd (trainingSeed o)
      start = initParams (trainingSeed o) (compiledParams c)
      rt = runtime s c (compiledGraph c)
      width = Text.length (showInt (max 0 (trainingSteps o - 1)))
      every = max 1 (trainingReportEvery o)
      header =
        "train " <> Text.pack (sessionPath s) <> ": " <> showInt (trainingSteps o) <> " steps, lr " <> showConstant (trainingRate o)
          <> " (" <> scheduleText (trainingSchedule o) <> "), seed " <> showInt (trainingSeed o)
          <> Text.concat [", " <> dataName d <> ": " <> showInt (length slots) <> " of " <> showInt (datasetSize d) <> " rows per step" | d <- sessionDatasets s, Just slots <- [Map.lookup (dataName d) (compiledSlots c)]]
      go lastReport = \case
        Loop.Step r rest
          | Loop.reportStep r `mod` every == 0 || isLast rest -> Line (reportLine width (trainingSchedule o) r) (go (Just r) rest)
          | otherwise -> go (Just r) rest
        Loop.Done ps -> End (save c ps lastReport)
        Loop.Failed e -> End (trainFailure rt e)
      isLast = \case
        Loop.Step _ _ -> False
        _ -> True
      save c' ps lastReport = case trainingSaveParams o of
        Nothing -> pure []
        Just path ->
          maybe (internal "a trained parameter has no value") (\t -> pure [(path, t)]) $
            renderParams
              ( [ "Parameters of " <> Text.pack (sessionPath s) <> ", saved by `hsgrad train`."
                , header
                ]
                  <> ["last " <> Text.unwords (Text.words (reportLine width (trainingSchedule o) r)) | Just r <- [lastReport]]
              )
              (compiledParams c')
              ps
  pure (Line header (go Nothing (Loop.train config setup start)))

-- | A failed training run: the runtime error of the step, with the step.
trainFailure :: Runtime -> Loop.TrainError -> Outcome a
trainFailure rt = \case
  Loop.StepFailed t (ForwardError e) -> during t (evalFailure rt e)
  Loop.StepFailed t (BackwardError e) -> during t (gradFailure rt e)
  e -> internal ("training could not start: " <> Text.pack (show e))
  where
    during t (Outcome ds r) = Outcome (map (annotate t) ds) r
    annotate t d =
      d
        & withNote ("in training step " <> showInt t)
        -- Non-finite numbers after some steps usually mean the steps were
        -- too large; at step 0 the program itself is the problem.
        & if t > 0 && diagCode d `elem` [E0501, E0504] then withHint "training may have diverged: try a smaller `--lr`" else id

-- | One report line: @step  10  loss 0.512345  accuracy 75%@ (and the
-- rate, if it changes).
reportLine :: Int -> LrSchedule -> Loop.Report -> Text
reportLine width sched r =
  Text.stripEnd $
    "step " <> Text.justifyRight width ' ' (showInt (Loop.reportStep r))
      <> "  loss " <> Text.justifyLeft 11 ' ' (significant 6 (Loop.reportLoss r))
      <> maybe "" (\a -> "  accuracy " <> Text.justifyLeft 6 ' ' (percent a)) (Loop.reportAccuracy r)
      <> (if sched == ConstantRate then "" else "  lr " <> significant 4 (Loop.reportRate r))

-- | A schedule in words.
scheduleText :: LrSchedule -> Text
scheduleText = \case
  ConstantRate -> "constant"
  LinearTo end -> "linear to " <> showConstant end
  StepDecay k f -> "times " <> showConstant f <> " every " <> showInt k <> " steps"

-- | @predict@: the model on the arguments given with @--set@, with the
-- parameters of a file.
runPredict :: Options -> Session -> PredictOptions -> Outcome Stream
runPredict opts s o = do
  c <- compileSession Nothing s
  p <- maybe (failWith [missingDeclaration opts "model" "`predict` evaluates the `model`"]) pure (compiledPredictor c)
  let targets = argumentSettables s p <> inputSettables s c
  set <- settings targets (map specName (compiledParams c)) (predictSet o)
  params <- parameterStore s c defaultSeed
  let inputs = Map.unions [set, inputDefaults c, slotInputs s (predictorSlots p)]
      rt = Runtime (predictorGraph p) targets (const False) (pointSpan (sessionPath s) startPos)
  vals <- evaluated rt (Env inputs params) (valueNodes (predictorOutput p))
  let argument spec = shapedText (inputSpecShape spec) [Map.findWithDefault 0 n inputs | n <- inputSpecLeaves spec]
  pure . linesOnly $
    [ predictorName p <> "(" <> Text.intercalate ", " (map argument (predictorArguments p)) <> ") = "
        <> valueText vals (predictorOutput p)
    ]

-- | E0201 for a command that needs a declaration the file lacks.
missingDeclaration :: Options -> Text -> Text -> Diagnostic
missingDeclaration opts what why =
  diagnostic E0201 ("`" <> argText (optFile opts) <> "` has no `" <> what <> "`") (argSpan (optCommandWord opts)) ("needs a `" <> what <> "` declaration")
    & withNote why

-- ---------------------------------------------------------------------------
-- Everything at once

-- | Everything a run printed and wrote, collected: for tests, and for
-- golden files of whole runs.
data Transcript = Transcript
  { transcriptSources :: !Sources
  , transcriptDiagnostics :: ![Diagnostic]
  , transcriptLines :: ![Text]
  , transcriptFiles :: ![(FilePath, Text)]
  -- ^ What would be written (nothing is).
  , transcriptResult :: !(Result ())
  }

-- | Load and run, collecting the whole output. The same pipeline as the
-- program's, without printing anything or writing files.
execute :: Options -> IO Transcript
execute opts = do
  (sources, loaded) <- load opts
  pure $ case loaded >>= run opts of
    Outcome ds (Ok stream) ->
      let (ls, Outcome ds' r) = drain stream
       in Transcript sources (ds <> ds') ls (case r of Ok fs -> fs; _ -> []) (() <$ r)
    Outcome ds r -> Transcript sources ds [] [] (() <$ r)
  where
    drain = \case
      Line l rest -> let (ls, end) = drain rest in (l : ls, end)
      End end -> ([], end)

-- | A transcript as text: the output lines, the files that would be
-- written, then the diagnostics as rendered on stderr.
renderTranscript :: RenderOptions -> Transcript -> Text
renderTranscript ro t =
  Text.unlines (transcriptLines t)
    <> Text.concat ["(writes " <> Text.pack p <> ")\n" | (p, _) <- transcriptFiles t]
    <> renderDiagnostics ro (transcriptSources t) (transcriptDiagnostics t)
    <> case transcriptResult t of
      Internal msg -> "internal error: " <> msg <> "\n"
      _ -> ""

-- ---------------------------------------------------------------------------
-- Text helpers

-- | @name = value@ lines, the names padded to one width.
aligned :: [(Text, Text)] -> [Text]
aligned rows = [Text.justifyLeft width ' ' n <> " = " <> v | (n, v) <- rows]
  where
    width = maximum (0 : map (Text.length . fst) rows)

-- | @name = expression  = value@ lines in three columns (@01-language.md@
-- §1.1); without values, two. Expressions longer than 40 characters
-- don't widen the value column: their value follows them.
equations :: [(Text, Text, Maybe Text)] -> [Text]
equations rows =
  [ Text.stripEnd (Text.justifyLeft nameWidth ' ' n <> " = " <> maybe e (\v -> Text.justifyLeft exprWidth ' ' e <> "  = " <> v) value)
  | (n, e, value) <- rows
  ]
  where
    nameWidth = maximum (0 : [Text.length n | (n, _, _) <- rows])
    exprWidth = maximum (0 : [Text.length e | (_, e, _) <- rows, Text.length e <= 40])

-- | A value's numbers as a literal of its shape: @-8@, @[1, 2]@,
-- @[[1, 2], [3, 4]]@.
valueText :: Values -> Value -> Text
valueText vals v = shapedText (valueShape v) (map (valueAt vals) (valueNodes v))

-- | Row-major numbers as a literal of a shape.
shapedText :: Shape Int -> [Double] -> Text
shapedText sh xs = case sh of
  Shape.Real -> Text.concat (map showValue xs)
  Shape.Vec _ -> list (map showValue xs)
  Shape.Mat _ cols -> list [list (map showValue row) | row <- chunks cols xs]
  where
    list ys = "[" <> Text.intercalate ", " ys <> "]"
    chunks n ys = case splitAt (max 1 n) ys of
      (row, []) -> [row]
      (row, rest) -> row : chunks n rest

-- | A number as the user reads it: the shortest exact decimal ('showConstant'),
-- or @NaN@, @∞@, @-∞@.
showValue :: Double -> Text
showValue x
  | isNaN x = "NaN"
  | isInfinite x = if x > 0 then "∞" else "-∞"
  | otherwise = showConstant x

-- | A number rounded to some significant digits: @1.20341@, @3.04512e-5@.
-- 'showEFloat' rounds the decimal digits; reading them back and printing
-- with 'showConstant' drops trailing zeros and picks the notation.
significant :: Int -> Double -> Text
significant n x
  | isNaN x || isInfinite x = showValue x
  | otherwise = showConstant (read (showEFloat (Just (n - 1)) x ""))

-- | A fraction as a percentage with at most one decimal: @75%@, @96.9%@.
percent :: Double -> Text
percent a = showConstant (fromInteger (round (a * 1000)) / 10) <> "%"

showInt :: Int -> Text
showInt = Text.pack . show

-- | "did you mean `L`?", if a name is close.
suggestion :: Text -> [Text] -> Diagnostic -> Diagnostic
suggestion name candidates = maybe id (\s -> withHint ("did you mean `" <> s <> "`?")) (suggestName name candidates)

-- | @`a`, `b`, `c`@.
commas :: [Text] -> Text
commas = Text.intercalate ", " . map (\n -> "`" <> n <> "`")

-- ---------------------------------------------------------------------------
-- Spans on the command line

-- | The column where an argument starts.
argColumn :: Arg -> Int
argColumn = posCol . spanStart . argSpan

-- | The span of @len@ characters of an argument, from character @from@.
argPart :: Arg -> Int -> Int -> Span
argPart a from len = SrcSpan f (Pos l (c + from)) (Pos l (c + from + len))
  where
    SrcSpan f (Pos l c) _ = argSpan a

-- | Move a span @n@ columns to the right (a literal parsed on its own,
-- placed where it is on the command line).
shiftSpan :: Int -> Span -> Span
shiftSpan n (SrcSpan f (Pos l1 c1) (Pos l2 c2)) = SrcSpan f (Pos l1 (c1 + n)) (Pos l2 (c2 + n))

shiftDiagnostic :: Int -> Diagnostic -> Diagnostic
shiftDiagnostic n d = d {diagPrimary = shift (diagPrimary d), diagSecondary = map shift (diagSecondary d)}
  where
    shift (Label sp t) = Label (shiftSpan n sp) t
