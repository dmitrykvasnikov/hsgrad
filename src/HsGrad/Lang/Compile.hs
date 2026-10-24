{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The compiler: from a 'Checked' program to a scalar graph.
--
-- The checker (step 12) has resolved every name and given every
-- expression a concrete shape. What is left is to /lower/ the program
-- into the scalar graph of "HsGrad.Core.Graph" (decision D5,
-- @01-language.md@ §8):
--
-- * every expression becomes a compile-time 'Value': one node id for a
--   @Real@, a list of them for a @Vec n@, a list of rows for a @Mat r c@.
--   Vector and matrix code is /unrolled/: @a + b@ on two @Vec 3@ is three
--   scalar @+@ nodes;
-- * @input@ and @param@ declarations become leaves named by their
--   elements (@x[0]@, @W[1,0]@), comprehensions over a dataset one copy of
--   their body per /slot/ of input leaves (@moons#3.x[1]@, D10), and
--   comprehensions over @params@ one copy per parameter;
-- * @let@s are compiled once and referred to by node id, function calls
--   are inlined with their parameters bound to the arguments' node ids;
-- * the built-in functions become core ops or the small formulas of
--   @01-language.md@ §6 (@sigmoid(e) = 1 / (1 + exp(-e))@, …);
-- * every node records the span of the expression that created it, and
--   the results of top-level @let@s are labelled with their names (D4).
--
-- The result, 'Compiled', is the graph plus everything the commands of
-- the driver (step 15) need to use it: the nodes of every named value,
-- the loss, the parameters in the form "HsGrad.Train.Param" wants, the
-- inputs and their defaults, the dataset slots, and the model compiled
-- into a graph of its own for @predict@.
--
-- Nothing is checked again: a checked program always compiles (except
-- for a missing slot count, 'MissingBatchSize'). See @docs/15-compile.md@.
module HsGrad.Lang.Compile
  ( -- * Compiling
    compileProgram
  , batchSizes
  , CompileError (..)

    -- * The compiled program
  , Compiled (..)
  , InputSpec (..)
  , Predictor (..)
  , lookupName
  , inputDefaults

    -- * Compile-time values
  , Value (..)
  , valueNodes
  , valueShape
  ) where

import Control.Monad (foldM, forM, unless, when, zipWithM, zipWithM_)
import Control.Monad.Except (ExceptT, mapExceptT, runExceptT, throwError)
import Control.Monad.Reader (ReaderT, asks, mapReaderT, runReaderT)
import Control.Monad.State.Strict (StateT, gets, mapStateT, modify', runStateT)
import Control.Monad.Trans (lift)
import Data.List (transpose)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes, fromMaybe, listToMaybe, mapMaybe)
import Data.Set qualified as Set
import Data.Text (Text)
import HsGrad.Core.Graph
  ( Builder
  , Graph
  , GraphError
  , NodeId
  , binary
  , build
  , constant
  , input
  , label
  , param
  , reachable
  , unary
  , withSpan
  )
import HsGrad.Core.Op (BinOp (..), Name (..), UnOp (..))
import HsGrad.Data.Dataset (Dataset (..), datasetSize, slotNames)
import HsGrad.Lang.Check
import HsGrad.Lang.Shape (Builtin (..), Shape)
import HsGrad.Lang.Shape qualified as Shape
import HsGrad.Lang.Span (Span)
import HsGrad.Lang.Syntax (DataSource (..), Operator (..))
import HsGrad.Train.Param qualified as Train

-- ---------------------------------------------------------------------------
-- Compile-time values

-- | What an expression compiles to: the graph nodes of its elements
-- (@01-language.md@ §8). Matrices are row-major, a list of rows.
--
-- The constructors are named after the shapes on purpose: a @Vec 3@
-- compiles to a 'Vec' of three node ids. (Import "HsGrad.Lang.Shape"
-- qualified, or without its constructors, next to this module.)
data Value
  = Real NodeId
  | Vec [NodeId]
  | Mat [[NodeId]]
  deriving (Eq, Show)

-- | The nodes of a value's elements, row-major: the order of
-- 'Train.leafNames'.
valueNodes :: Value -> [NodeId]
valueNodes = \case
  Real x -> [x]
  Vec xs -> xs
  Mat rows -> concat rows

-- | The shape of a value.
valueShape :: Value -> Shape Int
valueShape = \case
  Real _ -> Shape.Real
  Vec xs -> Shape.Vec (length xs)
  Mat rows -> Shape.Mat (length rows) (case rows of r : _ -> length r; [] -> 0)

-- ---------------------------------------------------------------------------
-- The compiled program

-- | A compiled program: the graph, and handles into it.
data Compiled = Compiled
  { compiledGraph :: !Graph
  -- ^ The program's graph. It holds the leaves of every input and every
  -- parameter (whether used or not), the nodes of every top-level @let@
  -- and of the loss. The model is in 'compiledPredictor'.
  , compiledInputs :: ![InputSpec]
  -- ^ The @input@ declarations, in declaration order.
  , compiledParams :: ![Train.ParamSpec]
  -- ^ The @param@ declarations, in declaration order, ready for
  -- 'Train.initParams'. Their leaves are named by 'Train.elementNames'.
  , compiledNames :: !(Map Text Value)
  -- ^ Every input, parameter and top-level @let@, by name: what
  -- @eval --show@, @derive --of@ and @--wrt@ look up.
  , compiledLets :: ![(Text, Value)]
  -- ^ The top-level @let@s, in declaration order.
  , compiledLoss :: !(Maybe NodeId)
  -- ^ The @loss@ node, labelled @loss@ (unless it already had a label).
  , compiledSlots :: !(Map Text [[Name]])
  -- ^ For every dataset that a comprehension in the graph runs over: its
  -- slots, one list of leaf names per slot in column order
  -- ('slotNames'). With the rows, that is a 'HsGrad.Train.Loop.Feed'.
  , compiledAccuracy :: ![(NodeId, NodeId)]
  -- ^ For sign accuracy during training: per slot of a comprehension in
  -- the loss, the result of the model call in that slot and the leaf of
  -- the dataset's last field (when both are 'Real'). Empty if there is
  -- no such pair; the form of 'HsGrad.Train.Loop.setupAccuracy'.
  , compiledPredictor :: !(Maybe Predictor)
  -- ^ The @model@, compiled on its own, if the file has one.
  }
  deriving (Eq, Show)

-- | A value fed in from outside: a declared @input@, or an argument of
-- the model. The counterpart of 'Train.ParamSpec' for input leaves.
data InputSpec = InputSpec
  { inputSpecName :: !Text
  -- ^ The name the user writes (@--set x=[1, 2]@).
  , inputSpecShape :: !(Shape Int)
  , inputSpecLeaves :: ![Name]
  -- ^ The input leaves, row-major.
  , inputSpecDefault :: !(Maybe [Double])
  -- ^ The declared default, one number per leaf.
  }
  deriving (Eq, Show)

-- | The @model@ as a graph of its own, for @predict@: its arguments are
-- input leaves, and its result is 'predictorOutput'.
--
-- The graph also holds the leaves of every input and parameter of the
-- program, under the same names as in 'compiledGraph', so the same
-- parameter store and input values bind it.
data Predictor = Predictor
  { predictorName :: !Text
  , predictorGraph :: !Graph
  , predictorArguments :: ![InputSpec]
  -- ^ One per model parameter, in order. Their leaves are named after the
  -- model (@net.x[0]@), so they never meet an @input@ of the same name.
  , predictorOutput :: !Value
  -- ^ The result, labelled with the model's name.
  , predictorSlots :: !(Map Text [[Name]])
  -- ^ Slots of datasets the model body iterates over (normally none).
  }
  deriving (Eq, Show)

-- | The value of an input, a parameter or a top-level @let@.
lookupName :: Text -> Compiled -> Maybe Value
lookupName x c = Map.lookup x (compiledNames c)

-- | The default values of the inputs that have one, by leaf name: the
-- fixed inputs of 'HsGrad.Train.Loop.setupInputs' before any @--set@.
inputDefaults :: Compiled -> Map Name Double
inputDefaults c =
  Map.fromList
    [ (n, v)
    | spec <- compiledInputs c
    , Just vs <- [inputSpecDefault spec]
    , (n, v) <- zip (inputSpecLeaves spec) vs
    ]

-- | What can go wrong. Only 'MissingBatchSize' and 'BadBatchSize' are
-- the caller's mistakes; the others are bugs (or, for 'GraphFailure', a
-- number literal too large for a 'Double').
data CompileError
  = -- | A comprehension runs over this dataset, and no slot count was
    -- given for it. The default is only known for inline @rows@; for a
    -- CSV dataset the caller passes its size (see 'batchSizes').
    MissingBatchSize Text
  | -- | A slot count below 1 for this dataset.
    BadBatchSize Text Int
  | -- | The graph builder failed: a non-finite constant (a literal like
    -- @1e999@), or a bug.
    GraphFailure GraphError
  | -- | The checked program broke one of the checker's promises.
    InternalError Text
  deriving (Eq, Show)

-- | The slot counts for 'compileProgram': @--batch b@ for every dataset,
-- or, without it, each dataset's size.
batchSizes :: Maybe Int -> [Dataset] -> Map Text Int
batchSizes batch ds = Map.fromList [(dataName d, fromMaybe (datasetSize d) batch) | d <- ds]

-- ---------------------------------------------------------------------------
-- The compiler's monad

-- | What every part of the compiler can look at.
data Context = Context
  { ctxChecked :: Checked
  , ctxBatch :: Map Text Int
  -- ^ Slots per dataset (B, D10).
  , ctxDatasets :: Map Text DatasetInfo
  , ctxLets :: Map Text LetInfo
  , ctxModel :: Maybe InstanceKey
  -- ^ The model's instance: its calls are the predictions for accuracy.
  }

-- | What the compiler accumulates.
data St = St
  { stGlobals :: Map Text Value
  -- ^ The inputs, parameters and top-level @let@s compiled so far.
  , stSlots :: Map Text [[Name]]
  , stPredictions :: Map (Text, Int) (NodeId, NodeId)
  -- ^ Per (dataset, slot): the first model call there and the label leaf.
  }

-- | The compiler: building a graph (the innermost layer), failing with a
-- 'CompileError', keeping 'St', reading the 'Context'.
type Compile = ReaderT Context (StateT St (ExceptT CompileError Builder))

-- | Run a compilation on an empty graph.
runCompile :: Context -> Compile a -> Either CompileError (a, St, Graph)
runCompile ctx m = case build (runExceptT (runStateT (runReaderT m ctx) emptySt)) of
  Left err -> Left (GraphFailure err)
  Right (Left err, _) -> Left err
  Right (Right (a, st), g) -> Right (a, st, g)
  where
    emptySt = St Map.empty Map.empty Map.empty

-- | A builder action as a compiler step.
liftB :: Builder a -> Compile a
liftB = lift . lift . lift

-- | 'withSpan' for compiler steps: every node created inside records
-- this span, unless an inner one says otherwise.
spanned :: Span -> Compile a -> Compile a
spanned sp = mapReaderT (mapStateT (mapExceptT (withSpan sp)))

internal :: Text -> Compile a
internal = throwError . InternalError

-- ---------------------------------------------------------------------------
-- Programs

-- | Compile a checked program. The map gives the number of slots (the
-- batch size B) of each dataset a comprehension runs over; a dataset
-- that is not in it gets one slot per row if its rows are inline, and is
-- a 'MissingBatchSize' error otherwise.
compileProgram :: Map Text Int -> Checked -> Either CompileError Compiled
compileProgram batch c = do
  (losses, st, g) <- runCompile ctx (mapM declare decls)
  predictor <- traverse (compilePredictor ctx) (checkedModel c)
  let globals = stGlobals st
      loss = listToMaybe (catMaybes losses)
      live = maybe Set.empty (\l -> reachable g [l]) loss
  pure
    Compiled
      { compiledGraph = g
      , compiledInputs = map inputSpec (checkedInputs c)
      , compiledParams = map paramSpec (checkedParams c)
      , compiledNames = globals
      , compiledLets = mapMaybe (\l -> (,) (letName l) <$> Map.lookup (letName l) globals) (checkedLets c)
      , compiledLoss = loss
      , compiledSlots = stSlots st
      , compiledAccuracy =
          [ (p, y)
          | (p, y) <- Map.elems (stPredictions st)
          , p `Set.member` live
          , y `Set.member` live
          ]
      , compiledPredictor = predictor
      }
  where
    decls = checkedDecls c
    ctx =
      Context
        { ctxChecked = c
        , ctxBatch = Map.union batch (Map.fromList (mapMaybe inlineSize (checkedDatasets c)))
        , ctxDatasets = Map.fromList [(datasetName d, d) | d <- checkedDatasets c]
        , ctxLets = Map.fromList [(letName l, l) | l <- checkedLets c]
        , ctxModel = modelInstance <$> checkedModel c
        }
    inlineSize d = case datasetSource d of
      Rows _ rows -> Just (datasetName d, length rows)
      Csv {} -> Nothing

-- | One declaration, in order; the loss node for the loss. Leaves are
-- made here, at their declaration, so they record its span. Functions and
-- the model are only compiled where they are called, datasets where a
-- comprehension runs over them.
declare :: CheckedDecl -> Compile (Maybe NodeId)
declare = \case
  CInput i -> Nothing <$ declareInput i
  CParam p -> Nothing <$ declareParam p
  CLet l -> Nothing <$ compileLet l
  CLoss l -> Just <$> compileLoss l
  CData _ -> pure Nothing
  CFunction _ -> pure Nothing
  CModel _ -> pure Nothing

declareInput :: InputInfo -> Compile ()
declareInput i = do
  v <- leaves input (inputDeclSpan i) (inputName i) (inputShape i)
  setGlobal (inputName i) v

declareParam :: ParamInfo -> Compile ()
declareParam p = do
  v <- leaves param (paramDeclSpan p) (paramName p) (paramShape p)
  setGlobal (paramName p) v

-- | A top-level @let@: compiled once, its nodes labelled with its name
-- (@L@, or @h[0]@ … for a vector).
compileLet :: LetInfo -> Compile Value
compileLet l = do
  v <- compileExpr topScope (letExpr l)
  labelValue (letName l) v
  setGlobal (letName l) v
  pure v

-- | The loss: a 'Real', labelled @loss@.
compileLoss :: LossInfo -> Compile NodeId
compileLoss l =
  compileExpr topScope (lossExpr l) >>= \case
    Real x -> x <$ liftB (label "loss" x)
    _ -> internal "the loss is not a Real"

-- | The model in a graph of its own: the leaves of every input and
-- parameter (in declaration order, as in the main graph), then the
-- model's arguments as input leaves, then its body. Top-level @let@s are
-- compiled when the body first refers to them.
compilePredictor :: Context -> ModelInfo -> Either CompileError Predictor
compilePredictor ctx m = do
  ((args, out), st, g) <- runCompile ctx $ do
    mapM_ declareLeaf (checkedDecls (ctxChecked ctx))
    args <- forM (modelParams m) $ \(x, s) ->
      leaves input (modelNameSpan m) (modelName m <> "." <> x) s
    out <- inline topScope (modelInstance m) args
    labelValue (modelName m) out
    pure (args, out)
  pure
    Predictor
      { predictorName = modelName m
      , predictorGraph = g
      , predictorArguments =
          [ InputSpec x s (leafNamesOf (modelName m <> "." <> x) s) Nothing
          | ((x, s), _) <- zip (modelParams m) args
          ]
      , predictorOutput = out
      , predictorSlots = stSlots st
      }
  where
    declareLeaf = \case
      CInput i -> declareInput i
      CParam p -> declareParam p
      _ -> pure ()

-- | An input declaration, for the driver.
inputSpec :: InputInfo -> InputSpec
inputSpec i =
  InputSpec
    { inputSpecName = inputName i
    , inputSpecShape = inputShape i
    , inputSpecLeaves = leafNamesOf (inputName i) (inputShape i)
    , inputSpecDefault = map fromRational <$> inputDefault i
    }

-- | A parameter declaration, as "HsGrad.Train.Param" describes it.
paramSpec :: ParamInfo -> Train.ParamSpec
paramSpec p = Train.ParamSpec (paramName p) (trainShape (paramShape p)) initialiser
  where
    initialiser = case paramInit p of
      InitFill c -> Train.Constant (fromRational c)
      InitValues cs -> Train.Values (map fromRational cs)
      InitUniform lo hi -> Train.Uniform (fromRational lo) (fromRational hi)
      InitNormal mean std -> Train.Normal (fromRational mean) (fromRational std)

-- ---------------------------------------------------------------------------
-- Leaves, names and labels

-- | Train's description of a shape.
trainShape :: Shape Int -> Train.Shape
trainShape = \case
  Shape.Real -> Train.Scalar
  Shape.Vec n -> Train.Vector n
  Shape.Mat r c -> Train.Matrix r c

-- | The element names of a value of this shape: @x@, @x[0]@, @W[1,0]@.
leafNamesOf :: Text -> Shape Int -> [Name]
leafNamesOf x s = Train.leafNames x (trainShape s)

-- | The leaves of a declared name, with the declaration's span.
leaves :: (Name -> Builder NodeId) -> Span -> Text -> Shape Int -> Compile Value
leaves make sp x s = do
  ids <- spanned sp (liftB (mapM make (leafNamesOf x s)))
  shaped s ids

-- | Element nodes, row-major, as a value of this shape.
shaped :: Shape Int -> [NodeId] -> Compile Value
shaped s ids = case (s, ids) of
  (Shape.Real, [x]) -> pure (Real x)
  (Shape.Vec n, _) | length ids == n -> pure (Vec ids)
  (Shape.Mat r c, _) | c >= 1, length ids == r * c -> pure (Mat (chunks ids))
    where
      chunks [] = []
      chunks xs = let (row, rest) = splitAt c xs in row : chunks rest
  _ -> internal ("wrong number of elements for " <> Shape.prettyShape s)

-- | Label a value's nodes with a name: @L@ for a 'Real', @h[0]@ … for a
-- vector (first label wins, so a node that already has one keeps it).
labelValue :: Text -> Value -> Compile ()
labelValue x v =
  liftB (zipWithM_ label [t | Name t <- leafNamesOf x (valueShape v)] (valueNodes v))

setGlobal :: Text -> Value -> Compile ()
setGlobal x v = modify' (\st -> st {stGlobals = Map.insert x v (stGlobals st)})

-- ---------------------------------------------------------------------------
-- Expressions

-- | What an expression can see besides the globals.
data Scope = Scope
  { scopeLocals :: Map Text Value
  -- ^ Function parameters, @let … in@ and comprehension variables.
  , scopeSlot :: Maybe (Text, Int, Maybe NodeId)
  -- ^ Inside a comprehension over a dataset: the dataset, the slot, and
  -- the leaf of its last field if that is a 'Real' (the label).
  }

topScope :: Scope
topScope = Scope Map.empty Nothing

bindLocal :: Text -> Value -> Scope -> Scope
bindLocal x v scope = scope {scopeLocals = Map.insert x v (scopeLocals scope)}

-- | Compile an expression. Its nodes are made under its span, so each node
-- records the innermost expression that created it.
compileExpr :: Scope -> TExpr -> Compile Value
compileExpr scope (TExpr sp _ n) = spanned sp $ case n of
  NNum r -> Real <$> number r
  NRef r -> reference scope r
  NVector es -> mapM (compileExpr scope) es >>= vector
  NIndex e i j -> compileExpr scope e >>= index i j
  -- A negative literal (@-2.0@) is one constant, as in the source.
  NNegate (TExpr _ _ (NNum r)) -> Real <$> number (negate r)
  NNegate e -> compileExpr scope e >>= elementwise (unary Neg)
  NPower e k -> compileExpr scope e >>= elementwise (unary (PowC k))
  NBinary op l r -> do
    a <- compileExpr scope l
    b <- compileExpr scope r
    operator op a b
  NBuiltin b args -> mapM (compileExpr scope) args >>= builtin b
  NReduce b it -> reduce scope b it
  NCall key args -> mapM (compileExpr scope) args >>= inline scope key
  NLet x bound body -> do
    v <- compileExpr scope bound
    compileExpr (bindLocal x v scope) body

-- | A constant node for a number literal.
number :: Rational -> Compile NodeId
number r = liftB (constant (fromRational r))

-- | A name. A top-level @let@ that has not been compiled yet (the model's
-- graph compiles only what the model uses) is compiled now.
reference :: Scope -> Ref -> Compile Value
reference scope = \case
  RLocal x -> maybe (internal ("unknown local " <> x)) pure (Map.lookup x (scopeLocals scope))
  RInput x -> global x
  RParam x -> global x
  RLet x ->
    gets (Map.lookup x . stGlobals) >>= \case
      Just v -> pure v
      Nothing -> asks (Map.lookup x . ctxLets) >>= maybe (internal ("unknown let " <> x)) compileLet

global :: Text -> Compile Value
global x = gets (Map.lookup x . stGlobals) >>= maybe (internal ("unknown name " <> x)) pure

-- | A vector literal of 'Real's, or a matrix literal of rows.
vector :: [Value] -> Compile Value
vector vs = case traverse asReal vs of
  Just xs -> pure (Vec xs)
  Nothing -> maybe (internal "malformed vector literal") (pure . Mat) (traverse asRow vs)
  where
    asReal = \case Real x -> Just x; _ -> Nothing
    asRow = \case Vec xs -> Just xs; _ -> Nothing

-- | @v[i]@, @M[i]@ (a row), @M[i, j]@. No new nodes.
index :: Int -> Maybe Int -> Value -> Compile Value
index i j v = case (v, j) of
  (Vec xs, Nothing) -> Real <$> at i xs
  (Mat rows, Nothing) -> Vec <$> at i rows
  (Mat rows, Just k) -> at i rows >>= fmap Real . at k
  _ -> internal "malformed index"
  where
    at k xs = case drop k xs of
      x : _ | k >= 0 -> pure x
      _ -> internal "index out of bounds"

-- ---------------------------------------------------------------------------
-- Elementwise operations, broadcasting, products

-- | A scalar operation applied to every element.
elementwise :: (NodeId -> Builder NodeId) -> Value -> Compile Value
elementwise f = \case
  Real x -> Real <$> liftB (f x)
  Vec xs -> Vec <$> liftB (mapM f xs)
  Mat rows -> Mat <$> liftB (mapM (mapM f) rows)

-- | A binary scalar operation applied elementwise (@01-language.md@
-- §4.1): two values of the same shape element by element, or a 'Real'
-- with every element of the other operand (in operand order).
broadcasting :: (NodeId -> NodeId -> Builder NodeId) -> Value -> Value -> Compile Value
broadcasting f a b = case (a, b) of
  (Real x, _) -> elementwise (f x) b
  (_, Real y) -> elementwise (`f` y) a
  (Vec xs, Vec ys) -> Vec <$> pairwise xs ys
  (Mat xs, Mat ys) -> do
    sameLength xs ys
    Mat <$> zipWithM pairwise xs ys
  _ -> internal "elementwise operands of different shapes"
  where
    pairwise xs ys = sameLength xs ys >> liftB (zipWithM f xs ys)
    sameLength xs ys = unless (length xs == length ys) (internal "elementwise operands of different lengths")

-- | A binary operator.
operator :: Operator -> Value -> Value -> Compile Value
operator = \case
  MatMul -> matMul
  Times -> broadcasting (binary Mul)
  Divide -> broadcasting (binary Div)
  Plus -> broadcasting (binary Add)
  Minus -> broadcasting (binary Sub)

-- | @W \@ x@ and @A \@ B@: every result element is the 'dotNodes' of a row
-- and a column.
matMul :: Value -> Value -> Compile Value
matMul a b = case (a, b) of
  (Mat rows, Vec xs) -> Vec <$> mapM (`dotNodes` xs) rows
  (Mat rows, Mat others) -> do
    let columns = transpose others
    Mat <$> mapM (\r -> mapM (dotNodes r) columns) rows
  _ -> internal "malformed matrix product"

-- | @Σ u[i]·v[i]@: all the products first, then their sum from the left.
dotNodes :: [NodeId] -> [NodeId] -> Compile NodeId
dotNodes us vs = do
  unless (length us == length vs) (internal "dot product of different lengths")
  liftB (zipWithM (binary Mul) us vs) >>= sumNodes

-- | A chain of @+@ from the left: @(x0 + x1) + x2@. One node is itself.
sumNodes :: [NodeId] -> Compile NodeId
sumNodes = \case
  x : xs -> liftB (foldM (binary Add) x xs)
  [] -> internal "sum of nothing"

-- | @x / n@, for a mean over @n@ things.
divideBy :: Int -> NodeId -> Builder NodeId
divideBy n x = constant (fromIntegral n) >>= binary Div x

-- ---------------------------------------------------------------------------
-- Built-in functions (@01-language.md@ §6)

builtin :: Builtin -> [Value] -> Compile Value
builtin b args = case (b, args) of
  (BExp, [v]) -> elementwise (unary Exp) v
  (BLog, [v]) -> elementwise (unary Log) v
  (BTanh, [v]) -> elementwise (unary Tanh) v
  (BRelu, [v]) -> elementwise (unary Relu) v
  (BStep, [v]) -> elementwise (unary Step) v
  (BSigmoid, [v]) -> elementwise sigmoid v
  (BSqrt, [v]) -> elementwise (unary (PowC (1 / 2))) v
  (BAbs, [v]) -> elementwise absolute v
  (BMax, [u, v]) -> broadcasting maximum2 u v
  (BMin, [u, v]) -> broadcasting minimum2 u v
  (BSum, [v]) -> Real <$> sumNodes (valueNodes v)
  -- The mean of a 'Real' is the value itself, not a division by 1.
  (BMean, [Real x]) -> pure (Real x)
  (BMean, [v]) -> do
    let xs = valueNodes v
    total <- sumNodes xs
    Real <$> liftB (divideBy (length xs) total)
  (BDot, [Vec us, Vec vs]) -> Real <$> dotNodes us vs
  (BTranspose, [Mat rows]) -> pure (Mat (transpose rows))
  _ -> internal ("malformed call of " <> Shape.builtinName b)
  where
    -- 1 / (1 + exp(-e))
    sigmoid x = do
      e <- unary Neg x >>= unary Exp
      one <- constant 1
      binary Add one e >>= binary Div one
    -- relu(e) + relu(-e)
    absolute x = do
      r <- unary Relu x
      r' <- unary Neg x >>= unary Relu
      binary Add r r'
    -- a + relu(b - a)
    maximum2 x y = binary Sub y x >>= unary Relu >>= binary Add x
    -- a - relu(a - b)
    minimum2 x y = binary Sub x y >>= unary Relu >>= binary Sub x

-- ---------------------------------------------------------------------------
-- Calls and comprehensions

-- | A call of a function or the model: its instance's body, compiled with
-- the parameters bound to the arguments. The body sees only its
-- parameters (and the globals), as the checker did.
--
-- A call of the model inside a comprehension over a dataset is that
-- slot's prediction, for accuracy.
inline :: Scope -> InstanceKey -> [Value] -> Compile Value
inline scope key args = do
  instances <- asks (checkedInstances . ctxChecked)
  inst <- maybe (internal ("unknown instance of " <> instanceFunction key)) pure (Map.lookup key instances)
  unless (length args == length (instanceParams inst)) $
    internal ("wrong number of arguments for " <> instanceFunction key)
  let inner = Scope (Map.fromList (zip (instanceParams inst) args)) (scopeSlot scope)
  v <- compileExpr inner (instanceBody inst)
  model <- asks ctxModel
  case (v, scopeSlot scope) of
    (Real p, Just (d, i, Just y)) | model == Just key -> notePrediction d i p y
    _ -> pure ()
  pure v

-- | Remember the first prediction of a slot.
notePrediction :: Text -> Int -> NodeId -> NodeId -> Compile ()
notePrediction d i p y =
  modify' (\st -> st {stPredictions = Map.insertWith (\_ old -> old) (d, i) (p, y) (stPredictions st)})

-- | @sum [ … ]@ or @mean [ … ]@: one copy of the body per slot or per
-- parameter, added up elementwise (and divided by the count for @mean@).
reduce :: Scope -> Builtin -> Iteration -> Compile Value
reduce scope b it = do
  copies <- case it of
    OverData d xs body -> overData scope d xs body
    OverParams x bodies -> forM bodies $ \(p, body) -> do
      v <- global p
      compileExpr (bindLocal x v scope) body
  total <- case copies of
    v : vs -> foldM (broadcasting (binary Add)) v vs
    [] -> internal "comprehension without iterations"
  case b of
    BSum -> pure total
    BMean -> elementwise (divideBy (length copies)) total
    _ -> internal ("reduction with " <> Shape.builtinName b)

-- | The copies of a comprehension body over a dataset (D10): slot @i@
-- binds the pattern's variables to the input leaves @d#i.field[j]@,
-- made just before the copy that uses them, with the span of the
-- dataset's declaration. Another comprehension over the same dataset
-- gets the same leaves (they are found by name), so all of them see the
-- same rows in a training step.
overData :: Scope -> Text -> [Text] -> TExpr -> Compile [Value]
overData scope d xs body = do
  info <- asks (Map.lookup d . ctxDatasets) >>= maybe (internal ("unknown dataset " <> d)) pure
  b <- asks (Map.lookup d . ctxBatch) >>= maybe (throwError (MissingBatchSize d)) pure
  when (b < 1) (throwError (BadBatchSize d b))
  let fields = datasetFields info
      slots = slotNames d fields b
  modify' (\st -> st {stSlots = Map.insert d slots (stSlots st)})
  forM (zip [0 ..] slots) $ \(i, names) -> do
    ids <- spanned (datasetDeclSpan info) (liftB (mapM input names))
    values <- splitFields fields ids
    let labelLeaf = case (reverse fields, reverse ids) of
          ((_, Shape.Real) : _, y : _) -> Just y
          _ -> Nothing
        inner = Scope (Map.union (Map.fromList (zip xs values)) (scopeLocals scope)) (Just (d, i, labelLeaf))
    compileExpr inner body
  where
    splitFields [] [] = pure []
    splitFields ((_, s) : fs) ids = do
      let (now, rest) = splitAt (Shape.shapeSize s) ids
      (:) <$> shaped s now <*> splitFields fs rest
    splitFields [] _ = internal "more slot leaves than fields"
