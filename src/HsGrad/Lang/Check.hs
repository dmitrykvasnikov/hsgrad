{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The checker: from a parsed 'Program' to either diagnostics or a
-- 'Checked' program that the compiler (step 14) can lower to a graph
-- without checking anything again.
--
-- It does three jobs in one walk over the declarations, in order:
--
-- * __Names__ (@01-language.md@ §3, §7): one namespace, define before use
--   (D9), no duplicates, kinds (values, functions, datasets), arities,
--   comprehension patterns. Every name in the output is resolved: an
--   input, a parameter, a top-level @let@ or a local.
-- * __Shapes__ (§4, §5): every expression gets a concrete 'Shape', using
--   the rules of "HsGrad.Lang.Shape". Functions are checked "template
--   style" (D8): a function whose signature is concrete is checked once,
--   at its declaration; a generic one is checked for names at its
--   declaration and for shapes once per distinct list of argument shapes,
--   at the first call with those shapes (an 'Instance').
-- * __Warnings__: shadowing (W0201), unused names (W0202) and parameters
--   that cannot affect the loss (W0203).
--
-- All independent errors are reported. A name whose declaration failed is
-- /poisoned/: its kind is still known, but its shape is not, and an
-- expression with an unknown shape produces no further errors (§7.6).
-- See @docs/13-checker.md@.
module HsGrad.Lang.Check
  ( -- * Checking
    checkProgram

    -- * The checked program
  , Checked (..)
  , CheckedDecl (..)
  , InputInfo (..)
  , ParamInfo (..)
  , Init (..)
  , DatasetInfo (..)
  , LetInfo (..)
  , FunctionInfo (..)
  , ModelInfo (..)
  , LossInfo (..)
  , checkedInputs
  , checkedParams
  , checkedDatasets
  , checkedLets
  , checkedFunctions
  , checkedModel
  , checkedLoss
  , describeChecked

    -- * Typed expressions
  , TExpr (..)
  , Node (..)
  , Ref (..)
  , refName
  , Iteration (..)
  , InstanceKey (..)
  , Instance (..)

    -- * Suggestions
  , editDistance
  , suggestName
  ) where

import Control.Monad (forM, forM_, guard, join, unless, when)
import Control.Monad.State.Strict (State, gets, modify', runState)
import Data.Foldable (toList)
import Data.Function ((&))
import Data.List (nub, sortOn)
import Data.List.NonEmpty (NonEmpty (..), nonEmpty)
import Data.Map qualified as LazyMap
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, isJust, isNothing, listToMaybe, mapMaybe)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Shape
import HsGrad.Lang.Span (Span, SrcSpan (..), showLocation, showPos)
import HsGrad.Lang.Syntax

-- ---------------------------------------------------------------------------
-- The checked program

-- | A program that passed the checker: every declaration with concrete
-- shapes, every expression typed and every name resolved, plus the
-- function instances that calls refer to.
data Checked = Checked
  { checkedDecls :: [CheckedDecl]
  -- ^ In source order, so that a @let@ only refers to earlier ones.
  , checkedInstances :: Map InstanceKey Instance
  -- ^ Every function body at every list of argument shapes it is called
  -- with ('NCall' refers to these).
  }
  deriving (Eq, Show)

-- | One checked declaration.
data CheckedDecl
  = CInput InputInfo
  | CParam ParamInfo
  | CData DatasetInfo
  | CLet LetInfo
  | CFunction FunctionInfo
  | CModel ModelInfo
  | CLoss LossInfo
  deriving (Eq, Show)

-- | @input x : Vec 3 = [1, 2, 3]@.
data InputInfo = InputInfo
  { inputName :: Text
  , inputNameSpan :: Span
  , inputDeclSpan :: Span
  , inputShape :: Shape Int
  , inputDefault :: Maybe [Rational]
  -- ^ The default value, row-major, exactly 'shapeSize' numbers.
  }
  deriving (Eq, Show)

-- | @param W : Mat 16 2 ~ uniform(-1, 1)@.
data ParamInfo = ParamInfo
  { paramName :: Text
  , paramNameSpan :: Span
  , paramDeclSpan :: Span
  , paramShape :: Shape Int
  , paramInit :: Init
  }
  deriving (Eq, Show)

-- | How a parameter gets its initial value.
data Init
  = -- | @= 0@: every element the same number.
    InitFill Rational
  | -- | @= [1, 2, 3]@: one number per element, row-major.
    InitValues [Rational]
  | -- | @~ uniform(lo, hi)@
    InitUniform Rational Rational
  | -- | @~ normal(mean, std)@
    InitNormal Rational Rational
  deriving (Eq, Show)

-- | @data moons : { x : Vec 2, y : Real } = csv …@. Only the schema is
-- checked here; the rows are read by step 13.
data DatasetInfo = DatasetInfo
  { datasetName :: Text
  , datasetNameSpan :: Span
  , datasetDeclSpan :: Span
  , datasetFields :: [(Text, Shape Int)]
  , datasetSource :: DataSource Span
  }
  deriving (Eq, Show)

-- | A top-level @let@.
data LetInfo = LetInfo
  { letName :: Text
  , letNameSpan :: Span
  , letDeclSpan :: Span
  , letShape :: Shape Int
  , letExpr :: TExpr
  }
  deriving (Eq, Show)

-- | A @fn@: its signature and the argument shapes it is called with.
data FunctionInfo = FunctionInfo
  { functionName :: Text
  , functionNameSpan :: Span
  , functionSignature :: Signature
  , functionInstances :: [InstanceKey]
  }
  deriving (Eq, Show)

-- | The @model@: a function with a concrete signature.
data ModelInfo = ModelInfo
  { modelName :: Text
  , modelNameSpan :: Span
  , modelParams :: [(Text, Shape Int)]
  , modelResult :: Shape Int
  , modelInstance :: InstanceKey
  -- ^ Its body, in 'checkedInstances'.
  }
  deriving (Eq, Show)

-- | The @loss@. Its expression is always a 'Real'.
data LossInfo = LossInfo
  { lossKeywordSpan :: Span
  , lossDeclSpan :: Span
  , lossExpr :: TExpr
  }
  deriving (Eq, Show)

-- | A function at concrete argument shapes: which body a call inlines.
data InstanceKey = InstanceKey
  { instanceFunction :: Text
  , instanceArguments :: [Shape Int]
  }
  deriving (Eq, Ord, Show)

-- | A function body checked at the shapes of its key.
data Instance = Instance
  { instanceParams :: [Text]
  -- ^ Parameter names, in order; the body refers to them as 'RLocal'.
  , instanceBody :: TExpr
  , instanceResult :: Shape Int
  }
  deriving (Eq, Show)

-- | Every declaration of one sort, in source order.
checkedInputs :: Checked -> [InputInfo]
checkedInputs c = [i | CInput i <- checkedDecls c]

checkedParams :: Checked -> [ParamInfo]
checkedParams c = [p | CParam p <- checkedDecls c]

checkedDatasets :: Checked -> [DatasetInfo]
checkedDatasets c = [d | CData d <- checkedDecls c]

checkedLets :: Checked -> [LetInfo]
checkedLets c = [l | CLet l <- checkedDecls c]

checkedFunctions :: Checked -> [FunctionInfo]
checkedFunctions c = [f | CFunction f <- checkedDecls c]

-- | The model and the loss, if the file has them.
checkedModel :: Checked -> Maybe ModelInfo
checkedModel c = listToMaybe [m | CModel m <- checkedDecls c]

checkedLoss :: Checked -> Maybe LossInfo
checkedLoss c = listToMaybe [l | CLoss l <- checkedDecls c]

-- ---------------------------------------------------------------------------
-- Typed expressions

-- | An expression with its span and concrete shape. Every node of every
-- checked expression has one.
data TExpr = TExpr
  { texprSpan :: Span
  , texprShape :: Shape Int
  , texprNode :: Node
  }
  deriving (Eq, Show)

-- | The forms of a typed expression. They follow the syntax, with the
-- questions the compiler would otherwise have to ask again answered:
-- names are resolved, built-in calls are told apart from user calls (which
-- name their instance), and @sum [ … ]@ over a comprehension is its own
-- node.
data Node
  = NNum Rational
  | NRef Ref
  | -- | A vector of 'Real's (shape @Vec n@), or of rows (shape @Mat r c@).
    NVector [TExpr]
  | NIndex TExpr Int (Maybe Int)
  | NNegate TExpr
  | NPower TExpr Rational
  | -- | @\@@ (the shapes say matrix-vector or matrix-matrix) or an
    -- elementwise operator (a 'Real' operand is applied to every element
    -- of the other).
    NBinary Operator TExpr TExpr
  | -- | A built-in function applied to values (not to a comprehension).
    NBuiltin Builtin [TExpr]
  | -- | 'BSum' or 'BMean' of a comprehension.
    NReduce Builtin Iteration
  | -- | A call of a @fn@ or the @model@: inline the instance's body with
    -- its parameters bound to the arguments.
    NCall InstanceKey [TExpr]
  | -- | @let x = bound in body@: @bound@ is computed once.
    NLet Text TExpr TExpr
  deriving (Eq, Show)

-- | What a name refers to.
data Ref
  = RInput Text
  | RParam Text
  | -- | A top-level @let@.
    RLet Text
  | -- | A function parameter, a @let … in@ or a comprehension variable.
    -- Locals follow lexical scope: the innermost binding of the name.
    RLocal Text
  deriving (Eq, Ord, Show)

-- | The name in a reference.
refName :: Ref -> Text
refName = \case
  RInput x -> x
  RParam x -> x
  RLet x -> x
  RLocal x -> x

-- | What a comprehension runs over, with its body.
data Iteration
  = -- | Over a dataset: its name, one local per field (schema order), and
    -- the body.
    OverData Text [Text] TExpr
  | -- | Over @params@: the local, and the body checked once per parameter
    -- (all of them, in declaration order).
    OverParams Text [(Text, TExpr)]
  deriving (Eq, Show)

-- | One line per declaration, with its type: @model mlp : Vec 2 → Real@,
-- @loss : Real@; a function with one line per instance. For tests and
-- for a future @hsgrad check --verbose@.
describeChecked :: Checked -> [Text]
describeChecked c = concatMap describe (checkedDecls c)
  where
    describe = \case
      CInput i -> ["input " <> inputName i <> " : " <> prettyShape (inputShape i)]
      CParam p -> ["param " <> paramName p <> " : " <> prettyShape (paramShape p)]
      CData d ->
        ["data " <> datasetName d <> " : { " <> Text.intercalate ", " [x <> " : " <> prettyShape s | (x, s) <- datasetFields d] <> " }"]
      CLet l -> ["let " <> letName l <> " : " <> prettyShape (letShape l)]
      CFunction f ->
        ("fn " <> prettySignature (functionSignature f))
          : ["  at " <> arrow k | k <- functionInstances f]
      CModel m -> ["model " <> modelName m <> " : " <> prettyArrow (map snd (modelParams m)) (modelResult m)]
      CLoss _ -> ["loss : Real"]
    arrow k = maybe "?" (prettyArrow (instanceArguments k) . instanceResult) (Map.lookup k (checkedInstances c))

-- ---------------------------------------------------------------------------
-- The checker's state

-- | Where a dimension's size was first written down: a declaration whose
-- type states it. Used for notes like "h1 : Vec 8 because W1 : Mat 8 2
-- (declared at 6:7)".
data Origin = Origin
  { originName :: Text
  , originShape :: Shape Int
  , originSpan :: Span
  }

-- | A top-level name, as later declarations see it.
data Global = Global
  { globalSpan :: Span
  -- ^ The name in its declaration.
  , globalEntry :: Entry
  }

-- | What a top-level name is. A shape of 'Nothing' means the declaration
-- failed: the name is poisoned.
data Entry
  = EInput (Maybe (Shape Int))
  | EParam (Maybe (Shape Int))
  | -- | A @let@, with the origins of its dimensions (computed lazily).
    ELet (Maybe (Shape Int)) [Maybe Origin]
  | EFunction Function
  | EDataset (Maybe [(Text, Span, Shape Int)])

-- | A @fn@ or @model@ declaration, kept for checking calls.
data Function = Function
  { fnName :: Text
  , fnSpan :: Span
  , fnParams :: [FnParam]
  , fnResult :: Maybe (Template, Span)
  , fnBody :: Expr Span
  , fnGlobals :: Map Text Global
  -- ^ The top-level names its body can see: those declared before it.
  , fnBroken :: Bool
  -- ^ The declaration had errors: calls are checked against the
  -- signature, but the body is never instantiated.
  , fnFallback :: Maybe (Shape Int)
  -- ^ The result shape of a call that can't be instantiated, if known
  -- anyway (a concrete result annotation).
  , fnIsModel :: Bool
  }

-- | A parameter of a function: its name, its template (if annotated and
-- valid) and the span of the annotation.
data FnParam = FnParam (Ident Span) (Maybe Template) (Maybe Span)

-- | A local name in scope.
data Local = Local
  { localSpan :: Span
  , localSort :: LocalSort
  , localShape :: Maybe (Shape Int)
  , localOrigins :: [Maybe Origin]
  -- ^ Lazy: only computed when a note needs it.
  }

-- | How a local was bound.
data LocalSort = FnParameter | LetLocal | PatternVariable
  deriving (Eq)

-- | What is known while checking an expression.
data Scope = Scope
  { scopeGlobals :: Map Text Global
  , scopeLocals :: Map Text Local
  , scopeVariables :: Maybe (Map Text Int)
  -- ^ Inside a function: the values of its shape variables (as far as
  -- known). Outside: 'Nothing', and shape variables are not allowed.
  , scopeContext :: [Text]
  -- ^ Notes for shape errors, innermost first: "while checking the call
  -- at …".
  , scopeQuiet :: Bool
  -- ^ Re-checking something whose names were already checked (a function
  -- instance, the second parameter of @params@): report only shape errors.
  , scopeDecl :: Maybe Text
  -- ^ The top-level declaration being checked (for the recursion note).
  , scopeFile :: FileInfo
  }

-- | Facts about the whole file, collected before checking.
data FileInfo = FileInfo
  { fileNames :: Map Text Span
  -- ^ Every top-level name and where it is (first) declared.
  , fileParams :: [(Text, Span, Maybe (Shape Int))]
  -- ^ Every parameter, for @params@, which means all of them.
  }

-- | The result of checking a function body at some argument shapes.
data Outcome
  = Succeeded Instance
  | -- | Errors were reported; the result shape, if known anyway.
    Failed (Maybe (Shape Int))

-- | What the checker has collected so far.
data St = St
  { stDiagnostics :: [Diagnostic]
  -- ^ Newest first.
  , stErrors :: !Int
  , stUsedGlobals :: Set Text
  , stUsedLocals :: Set Span
  -- ^ Locals are identified by the span of their binding.
  , stInstances :: Map InstanceKey Outcome
  }

type Check = State St

-- | What the checker knows about an expression: nothing (an error was
-- reported inside it, or it depends on a poisoned name), only its shape
-- (an error inside, but an annotation still tells the shape), or
-- everything.
data Typed
  = Unknown
  | ShapeOnly (Shape Int)
  | Full TExpr

typedShape :: Typed -> Maybe (Shape Int)
typedShape = \case
  Unknown -> Nothing
  ShapeOnly s -> Just s
  Full e -> Just (texprShape e)

typedExpr :: Typed -> Maybe TExpr
typedExpr = \case
  Full e -> Just e
  _ -> Nothing

-- | A node whose shape is known; the tree only if all its parts are.
node :: Span -> Shape Int -> Maybe Node -> Typed
node sp s = maybe (ShapeOnly s) (Full . TExpr sp s)

-- | Report a diagnostic. In a quiet scope only shape errors are reported
-- (the rest were reported the first time round); shape errors get the
-- scope's context notes.
report :: Scope -> Diagnostic -> Check ()
report scope d
  | scopeQuiet scope && not shapeError = pure ()
  | otherwise = modify' $ \st ->
      st
        { stDiagnostics = d' : stDiagnostics st
        , stErrors = stErrors st + (if severity d == Error then 1 else 0)
        }
  where
    shapeError = dependsOnShapes (diagCode d)
    d'
      | shapeError = foldl (flip withNote) d (scopeContext scope)
      | otherwise = d

-- | The errors that can differ between two instances of the same function
-- body: shape errors, and E0314, which is found on typed bodies. All
-- others (names, kinds, arities, patterns, annotations) depend on the
-- text alone.
dependsOnShapes :: Code -> Bool
dependsOnShapes c = c `elem` [E0301, E0302, E0303, E0304, E0305, E0306, E0307, E0308, E0310, E0312, E0314]

errorCount :: Check Int
errorCount = gets stErrors

useGlobal :: Text -> Check ()
useGlobal x = modify' $ \st -> st {stUsedGlobals = Set.insert x (stUsedGlobals st)}

useLocal :: Local -> Check ()
useLocal l = modify' $ \st -> st {stUsedLocals = Set.insert (localSpan l) (stUsedLocals st)}

-- ---------------------------------------------------------------------------
-- The program

-- | Check a parsed program. The diagnostics are sorted by position; the
-- 'Checked' program is there exactly when none of them is an error.
checkProgram :: Program Span -> ([Diagnostic], Maybe Checked)
checkProgram (Program decls) = (sortDiagnostics (diagnostics <> influence), checked)
  where
    file = fileInfo decls
    ((outputs, globals), st) = runState (checkDeclarations file decls) emptyState
    emptyState = St [] 0 Set.empty Set.empty Map.empty
    diagnostics = reverse (stDiagnostics st) <> unusedGlobals globals (stUsedGlobals st)
    instances = Map.fromList [(k, i) | (k, Succeeded i) <- Map.toList (stInstances st)]
    checked
      | hasErrors diagnostics = Nothing
      | otherwise = case sequence outputs of
          Just ds -> Just (Checked (map (withInstances instances) ds) instances)
          Nothing -> error "checkProgram: a declaration failed without an error"
    influence = maybe [] (lossInfluence (stUsedGlobals st)) checked

-- | Fill in a function's instances, known only at the end.
withInstances :: Map InstanceKey Instance -> CheckedDecl -> CheckedDecl
withInstances instances = \case
  CFunction f -> CFunction f {functionInstances = [k | k <- Map.keys instances, instanceFunction k == functionName f]}
  d -> d

-- | Every top-level name with its first declaration, and every parameter.
fileInfo :: [Decl Span] -> FileInfo
fileInfo decls = FileInfo names params
  where
    names = Map.fromListWith (\_ old -> old) [(identName x, identAnn x) | Just x <- map declName decls]
    params =
      [ (identName x, identAnn x, toShape (template ty) >>= positive)
      | ParamDecl _ x ty _ <- decls
      , Map.lookup (identName x) names == Just (identAnn x)
      , isNothing (lookupBuiltin (identName x))
      ]
    positive s = if all (>= 1) (shapeDims s) then Just s else Nothing

-- | The declared name, if the declaration has one.
declName :: Decl a -> Maybe (Ident a)
declName = \case
  InputDecl _ x _ _ -> Just x
  ParamDecl _ x _ _ -> Just x
  DataDecl _ x _ _ -> Just x
  FnDecl _ x _ _ _ -> Just x
  ModelDecl _ x _ _ _ -> Just x
  LetDecl _ x _ _ -> Just x
  LossDecl {} -> Nothing

-- | What is threaded through the declarations: the names declared so far,
-- and the first @model@ and @loss@ (E0206).
data Loop = Loop
  { loopGlobals :: Map Text Global
  , loopModel :: Maybe Span
  , loopLoss :: Maybe Span
  }

checkDeclarations :: FileInfo -> [Decl Span] -> Check ([Maybe CheckedDecl], Map Text Global)
checkDeclarations file = go (Loop Map.empty Nothing Nothing)
  where
    go loop [] = pure ([], loopGlobals loop)
    go loop (d : ds) = do
      (out, loop') <- checkDecl file loop d
      (outs, globals) <- go loop' ds
      pure (out : outs, globals)

-- | The scope of a top-level declaration.
topScope :: FileInfo -> Map Text Global -> Maybe Text -> Scope
topScope file globals decl = Scope globals Map.empty Nothing [] False decl file

-- | Check one declaration: its output (if it succeeded) and the loop state
-- with its name added.
checkDecl :: FileInfo -> Loop -> Decl Span -> Check (Maybe CheckedDecl, Loop)
checkDecl file loop = \case
  InputDecl sp x ty def -> do
    let scope = scopeFor x
    fresh <- declare scope x
    ms <- concreteType scope "an `input`" ty
    mv <- forM def (checkDefault scope ty ms)
    let out = CInput <$> (InputInfo (identName x) (identAnn x) sp <$> ms <*> sequence mv)
    pure (out, add fresh x (EInput ms))
  ParamDecl sp x ty ini -> do
    let scope = scopeFor x
    fresh <- declare scope x
    ms <- concreteType scope "a `param`" ty
    mi <- checkInit scope ty ms ini
    let out = CParam <$> (ParamInfo (identName x) (identAnn x) sp <$> ms <*> mi)
    pure (out, add fresh x (EParam ms))
  DataDecl sp x fields src -> do
    let scope = scopeFor x
    fresh <- declare scope x
    duplicates scope "field" [f | Field f _ <- fields]
    checkedFields <- forM fields $ \(Field f ty) -> fmap (\s -> (identName f, identAnn f, s)) <$> fieldType scope ty
    let schema = sequence checkedFields
        out = CData . (\fs -> DatasetInfo (identName x) (identAnn x) sp [(n, s) | (n, _, s) <- fs] src) <$> schema
    pure (out, add fresh x (EDataset schema))
  FnDecl _ x binders res body -> checkFunctionDecl file loop False x binders res body
  ModelDecl _ x binders res body -> checkFunctionDecl file loop True x binders (Just res) body
  LetDecl sp x ann e -> do
    let scope = scopeFor x
    fresh <- declare scope x
    mt <- forM ann (concreteType scope "a top-level `let`")
    t <- checkExpr scope e
    shape <- case (ann, join mt, typedShape t) of
      (Just ty, Just s, Just s')
        | s /= s' -> do
            report scope (annotationMismatch e s' (typeAnn ty) ("`" <> prettyShape s <> "`") "value doesn't match the `let` annotation")
            pure (Just s)
      (Just _, ms, _) -> pure ms
      (Nothing, _, ms) -> pure ms
    instances <- gets stInstances
    let origins = case join mt of
          Just s -> selfOrigins (identName x) s (identAnn x)
          Nothing -> maybe [] (dimOrigins scope instances) (typedExpr t)
        out = CLet <$> (LetInfo (identName x) (identAnn x) sp <$> shape <*> typedExpr t)
    pure (out, add fresh x (ELet shape origins))
  LossDecl sp kw e -> do
    let scope = topScope file (loopGlobals loop) Nothing
    forM_ (loopLoss loop) $ \first ->
      report scope $
        diagnostic E0206 "more than one `loss`" kw "a second `loss`"
          & withSecondary first "the first `loss` is here"
          & withNote "a file has at most one `loss`, the objective that `hsgrad train` minimises"
    t <- checkExpr scope e
    real <- case typedShape t of
      Just s | s /= Real -> do
        report scope $
          diagnostic E0307 "`loss` must be a `Real`" (exprAnn e) ("this is `" <> prettyShape s <> "`")
            & withSecondary kw "the loss is minimised, so it must be one number"
            & withHint "reduce it to a number with `sum(…)` or `mean(…)`"
        pure False
      _ -> pure True
    let out
          | isJust (loopLoss loop) || not real = Nothing
          | otherwise = CLoss . LossInfo kw sp <$> typedExpr t
    pure (out, loop {loopLoss = loopLoss loop <|> Just kw})
  where
    scopeFor x = topScope file (loopGlobals loop) (Just (identName x))
    add fresh x entry
      | fresh = loop {loopGlobals = Map.insert (identName x) (Global (identAnn x) entry) (loopGlobals loop)}
      | otherwise = loop
    a <|> b = maybe b Just a

-- | Is the name new? E0202 if it is already declared, or a built-in
-- function. Only the first declaration of a name counts.
declare :: Scope -> Ident Span -> Check Bool
declare scope (Ident sp x)
  | Just g <- Map.lookup x (scopeGlobals scope) = do
      report scope $
        diagnostic E0202 ("duplicate declaration of `" <> x <> "`") sp "declared again here"
          & withSecondary (globalSpan g) "first declared here"
      pure False
  | Just _ <- lookupBuiltin x = do
      report scope $
        diagnostic E0202 ("`" <> x <> "` is a built-in function") sp "declared again here"
          & withNote "built-in functions can't be redefined; choose another name"
      pure False
  | otherwise = pure True

-- | E0202 for every name that repeats an earlier one in the list (of
-- parameters, fields or pattern variables).
duplicates :: Scope -> Text -> [Ident Span] -> Check ()
duplicates scope what = go Map.empty
  where
    go _ [] = pure ()
    go seen (Ident sp x : rest) = case Map.lookup x seen of
      Just first -> do
        report scope $
          diagnostic E0202 ("duplicate " <> what <> " `" <> x <> "`") sp "declared again here"
            & withSecondary first "first declared here"
        go seen rest
      Nothing -> go (Map.insert x sp seen) rest

-- | Unused top-level names (W0202): inputs, parameters, datasets and
-- functions that nothing refers to. A @let@ or the @model@ may be an
-- output of the program (@hsgrad eval --show@, @predict@), so they are
-- never unused.
unusedGlobals :: Map Text Global -> Set Text -> [Diagnostic]
unusedGlobals globals used =
  [ diagnostic W0202 ("`" <> x <> "` is never used") (globalSpan g) ""
  | (x, g) <- Map.toList globals
  , x `Set.notMember` used
  , not ("_" `Text.isPrefixOf` x)
  , case globalEntry g of
      ELet {} -> False
      EFunction fn -> not (fnIsModel fn)
      _ -> True
  ]

-- ---------------------------------------------------------------------------
-- Types in declarations

-- | A syntactic type as a template.
template :: Type a -> Template
template = \case
  TReal _ -> Real
  TVec _ d -> Vec (dimension d)
  TMat _ r c -> Mat (dimension r) (dimension c)
  where
    dimension = \case
      DimNat n -> Fixed (natValue n)
      DimVar v -> Variable (identName v)

-- | Check an annotation and return its template: E0309 for a dimension 0,
-- and for shape variables unless they are allowed here.
annotation :: Scope -> Maybe Text -> Type Span -> Check (Maybe Template)
annotation scope concreteFor ty = do
  let dims = case ty of
        TReal _ -> []
        TVec _ d -> [d]
        TMat _ r c -> [r, c]
  oks <- forM dims $ \case
    DimNat (Nat sp 0) -> do
      report scope $
        diagnostic E0309 "a dimension must be at least 1" sp "this dimension is 0"
      pure False
    DimNat _ -> pure True
    DimVar (Ident sp v) -> case concreteFor of
      Just what -> do
        report scope $
          diagnostic E0309 ("shape variable `" <> v <> "` in " <> what) sp "a shape variable"
            & withNote (what <> " needs a concrete type: shape variables belong in `fn` signatures")
        pure False
      Nothing -> pure True
  pure (if and oks then Just (template ty) else Nothing)

-- | An annotation that must be concrete: of an input, a parameter, a
-- top-level @let@, a @model@.
concreteType :: Scope -> Text -> Type Span -> Check (Maybe (Shape Int))
concreteType scope what ty = (>>= toShape) <$> annotation scope (Just what) ty

-- | A dataset field: @Real@ or @Vec n@ (@01-language.md@ §3).
fieldType :: Scope -> Type Span -> Check (Maybe (Shape Int))
fieldType scope ty = case ty of
  TMat sp _ _ -> do
    report scope $
      diagnostic E0309 "a dataset field can't be a matrix" sp "not allowed in a schema"
        & withNote "dataset fields are `Real` or `Vec n`"
    pure Nothing
  _ -> concreteType scope "a dataset schema" ty

-- | An input's default value: a literal of the declared shape.
checkDefault :: Scope -> Type Span -> Maybe (Shape Int) -> Literal Span -> Check (Maybe [Rational])
checkDefault scope ty declared lit = do
  value <- checkLiteral scope lit
  case (value, declared) of
    (Just (s, vs), Just s')
      | s == s' -> pure (Just vs)
      | otherwise -> do
          report scope $
            diagnostic E0308 "the default value doesn't match the declared type" (literalAnn lit) ("this is `" <> prettyShape s <> "`")
              & withSecondary (typeAnn ty) ("expected `" <> prettyShape s' <> "` because of this")
          pure Nothing
    _ -> pure Nothing

-- | A parameter's initialiser.
checkInit :: Scope -> Type Span -> Maybe (Shape Int) -> ParamInit Span -> Check (Maybe Init)
checkInit scope ty declared = \case
  InitConst lit -> do
    value <- checkLiteral scope lit
    case (value, declared) of
      (Just (Real, [v]), Just _) -> pure (Just (InitFill v))
      (Just (s, vs), Just s')
        | s == s' -> pure (Just (InitValues vs))
        | otherwise -> do
            report scope $
              diagnostic E0308 "the initial value doesn't match the declared type" (literalAnn lit) ("this is `" <> prettyShape s <> "`")
                & withSecondary (typeAnn ty) ("expected `" <> prettyShape s' <> "` because of this")
                & withHint "a single number fills the whole shape: `= 0`"
            pure Nothing
      _ -> pure Nothing
  InitRandom _ (Ident sp f) lo hi -> case f of
    "uniform" -> pure (InitUniform (numberValue lo) (numberValue hi) <$ declared)
    "normal" -> pure (InitNormal (numberValue lo) (numberValue hi) <$ declared)
    _ -> do
      report scope $
        diagnostic E0201 ("unknown initialiser `" <> f <> "`") sp "not an initialiser"
          & maybe id (\s -> withHint ("did you mean `" <> s <> "`?")) (suggestName f ["uniform", "normal"])
          & withNote "the initialisers are `uniform(lo, hi)` and `normal(mean, std)`"
      pure Nothing

-- | The shape and the values (row-major) of a literal, or E0312.
checkLiteral :: Scope -> Literal Span -> Check (Maybe (Shape Int, [Rational]))
checkLiteral scope = \case
  LitNumber n -> pure (Just (Real, [numberValue n]))
  LitVector _ ls -> do
    parts <- mapM (checkLiteral scope) ls
    case sequence parts >>= nonEmpty . zip (map literalAnn ls) of
      Nothing -> pure Nothing
      Just elems -> case vectorShape (fmap (fmap fst) elems) of
        Left d -> report scope d >> pure Nothing
        Right s -> pure (Just (s, concatMap (snd . snd) (toList elems)))

-- | The shape of a bracketed list of elements: @Vec n@ for numbers,
-- @Mat r c@ for rows of one length. Anything else is E0312.
vectorShape :: NonEmpty (Span, Shape Int) -> Either Diagnostic (Shape Int)
vectorShape elems@((sp0, s0) :| _) =
  case [e | e@(_, Mat _ _) <- toList elems] of
    (sp, s) : _ ->
      Left $
        diagnostic E0312 "a vector literal can't contain a matrix" sp ("this is `" <> prettyShape s <> "`")
          & withNote "a literal is a vector of numbers or a matrix (a vector of rows)"
    [] -> case [e | e@(_, s) <- toList elems, s /= s0] of
      [] -> Right $ case s0 of
        Vec k -> Mat (length elems) k
        _ -> Vec (length elems)
      (sp, s) : _ -> Left $ case (s0, s) of
        (Vec k, Vec k') ->
          diagnostic E0312 "matrix rows of different lengths" sp ("this row has " <> elements k')
            & withSecondary sp0 ("the first row has " <> elements k)
        _ ->
          diagnostic E0312 "a vector literal mixes numbers and rows" sp ("this is `" <> prettyShape s <> "`")
            & withSecondary sp0 ("but this is `" <> prettyShape s0 <> "`")
            & withNote "a literal is a vector of numbers or a matrix (a vector of rows)"
  where
    elements k = Text.pack (show k) <> if k == 1 then " element" else " elements"

-- ---------------------------------------------------------------------------
-- Functions

-- | A @fn@ or @model@ declaration (D8).
--
-- A function whose parameters all have concrete annotations is checked
-- here, once, and that check is its only instance. A generic one is
-- checked here with the shapes of its generic parameters unknown, which
-- finds every error that doesn't depend on them (names, kinds, arities,
-- and shape errors among known shapes); its shapes are checked at calls.
checkFunctionDecl ::
  FileInfo -> Loop -> Bool -> Ident Span -> [Binder Span] -> Maybe (Type Span) -> Expr Span -> Check (Maybe CheckedDecl, Loop)
checkFunctionDecl file loop isModel x binders res body = do
  let globals = loopGlobals loop
      scope = topScope file globals (Just (identName x))
      concreteFor = if isModel then Just "a `model` signature" else Nothing
  before <- errorCount
  when isModel $ forM_ (loopModel loop) $ \first ->
    report scope $
      diagnostic E0206 "more than one `model`" (identAnn x) "a second `model`"
        & withSecondary first "the first `model` is here"
        & withNote "a file has at most one `model`, the predictor that `hsgrad predict` runs"
  fresh <- declare scope x
  duplicates scope "parameter" [b | Binder b _ <- binders]
  params <- forM binders $ \(Binder b ann) -> do
    t <- case ann of
      Just ty -> annotation scope concreteFor ty
      Nothing -> do
        when isModel $
          report scope $
            diagnostic E0309 ("`model` parameter `" <> identName b <> "` has no type") (identAnn b) "needs a type"
              & withNote "`hsgrad predict` feeds the model, so its signature must be fully concrete"
        pure Nothing
    pure (FnParam b t (typeAnn <$> ann))
  result <- forM res $ \ty -> fmap (\t -> (t, typeAnn ty)) <$> annotation scope concreteFor ty
  let shapes = [t >>= toShape | FnParam _ t _ <- params]
      concrete = all isJust shapes
      fn0 = Function (identName x) (identAnn x) params (join result) body globals False Nothing isModel
      origins = [maybe [] (\s -> selfOrigins (identName b) s (identAnn b)) ms | (FnParam b _ _, ms) <- zip params shapes]
  outcome <- runBody file Declaring fn0 shapes origins Map.empty
  let key = InstanceKey (identName x) (mapMaybe id shapes)
  when concrete $ modify' $ \st -> st {stInstances = Map.insert key outcome (stInstances st)}
  after <- errorCount
  let resultShape = join result >>= toShape . fst
      fallback = case (resultShape, outcome) of
        (Just s, _) -> Just s
        (Nothing, Succeeded i) | concrete -> Just (instanceResult i)
        _ -> Nothing
      fn = fn0 {fnBroken = after > before, fnFallback = fallback}
      signature = Signature (identName x) [(identName b, t) | FnParam b t _ <- params] (fst <$> join result)
      out
        | isModel = case outcome of
            Succeeded i | concrete -> Just (CModel (ModelInfo (identName x) (identAnn x) [(identName b, s) | (FnParam b _ _, Just s) <- zip params shapes] (instanceResult i) key))
            _ -> Nothing
        | otherwise = Just (CFunction (FunctionInfo (identName x) (identAnn x) signature []))
      loop' = if isModel then loop {loopModel = loopModel loop <|> Just (identAnn x)} else loop
      a <|> b = maybe b Just a
  pure
    ( if after > before then Nothing else out
    , if fresh then loop' {loopGlobals = Map.insert (identName x) (Global (identAnn x) (EFunction fn)) globals} else loop'
    )

-- | How a function body is being checked.
data Mode
  = -- | At its declaration: report everything.
    Declaring
  | -- | For a call, with these context notes: report only shape errors.
    Calling [Text]

-- | Check a function's body with its parameters bound to the given shapes
-- ('Nothing': unknown) and the shape variables to the given values.
runBody :: FileInfo -> Mode -> Function -> [Maybe (Shape Int)] -> [[Maybe Origin]] -> Map Text Int -> Check Outcome
runBody file mode fn shapes origins bindings = do
  let (context, quiet) = case mode of
        Declaring -> ([], False)
        Calling notes -> (notes, True)
      scope = Scope (fnGlobals fn) Map.empty (Just bindings) context quiet (Just (fnName fn)) file
      locals = [NewLocal b FnParameter s o | (FnParam b _ _, s, o) <- zip3 (fnParams fn) shapes (origins <> repeat [])]
  t <- withLocals scope locals (`checkExpr` fnBody fn)
  let finish ms = case (ms, typedExpr t) of
        (Just s, Just e) -> Succeeded (Instance [identName b | FnParam b _ _ <- fnParams fn] e s)
        _ -> Failed ms
  case (fnResult fn, typedShape t) of
    (Nothing, ms) -> pure (finish ms)
    (Just (tmpl, _), Nothing) -> pure (Failed (substitute bindings tmpl))
    (Just (tmpl, annSp), Just s) -> case matchShape bindings tmpl s of
      Just _ -> pure (finish (Just s))
      Nothing -> do
        report scope (annotationMismatch (fnBody fn) s annSp (expected tmpl) "body doesn't match the declared result type")
        pure (Failed (substitute bindings tmpl))
  where
    expected tmpl =
      let bound = bindTemplate bindings tmpl
       in if null (templateVariables tmpl) || bound == tmpl
            then "`" <> prettyTemplate tmpl <> "`"
            else "`" <> prettyTemplate tmpl <> "`, which is `" <> prettyTemplate bound <> "` here,"

-- | E0308: an expression doesn't match the annotation at @annSp@.
annotationMismatch :: Expr Span -> Shape Int -> Span -> Text -> Text -> Diagnostic
annotationMismatch e s annSp expectedText message =
  diagnostic E0308 message (exprAnn e) ("this is `" <> prettyShape s <> "`")
    & withSecondary annSp ("expected " <> expectedText <> " because of this")
    & (if expectedText == "`Real`" && s /= Real then withHint "use `sum(…)`, `dot(…)` or an index `[i]` to get a `Real`" else id)

-- ---------------------------------------------------------------------------
-- Locals

-- | A local about to be bound.
data NewLocal = NewLocal (Ident Span) LocalSort (Maybe (Shape Int)) [Maybe Origin]

-- | Check something with locals in scope: W0201 when one shadows a name
-- (function parameters excepted), W0202 for those never used. (In a quiet
-- scope 'report' drops both: they were reported the first time.)
withLocals :: Scope -> [NewLocal] -> (Scope -> Check a) -> Check a
withLocals scope new body = do
  forM_ new $ \(NewLocal x sort _ _) -> when (sort /= FnParameter) (shadowing scope x)
  -- A repeated name (E0202, reported by the caller) binds only once.
  let distinct = nubOrdOn (\(NewLocal x _ _ _) -> identName x) new
      bind m (NewLocal (Ident sp x) sort s o) = Map.insert x (Local sp sort s o) m
  r <- body scope {scopeLocals = foldl bind (scopeLocals scope) distinct}
  used <- gets stUsedLocals
  forM_ distinct $ \(NewLocal (Ident sp x) _ _ _) ->
    unless (sp `Set.member` used || "_" `Text.isPrefixOf` x) $
      report scope $
        diagnostic W0202 ("`" <> x <> "` is never used") sp ""
          & withHint ("if that is intended, start its name with `_`: `_" <> x <> "`")
  pure r

-- | W0201 if a new local hides another name.
shadowing :: Scope -> Ident Span -> Check ()
shadowing scope (Ident sp x) = case (Map.lookup x (scopeLocals scope), Map.lookup x (scopeGlobals scope)) of
  (Just l, _) -> warn (localDescription l) (Just (localSpan l))
  (Nothing, Just g) -> warn (entryDescription (globalEntry g)) (Just (globalSpan g))
  (Nothing, Nothing)
    | isJust (lookupBuiltin x) -> warn "a built-in function" Nothing
    | otherwise -> pure ()
  where
    warn what other =
      report scope $
        diagnostic W0201 ("`" <> x <> "` shadows " <> what) sp ("this `" <> x <> "`")
          & maybe id (`withSecondary` "shadows this one") other

localDescription :: Local -> Text
localDescription l = case localSort l of
  FnParameter -> "a parameter"
  LetLocal -> "a local `let`"
  PatternVariable -> "a comprehension variable"

entryDescription :: Entry -> Text
entryDescription = \case
  EInput _ -> "an input"
  EParam _ -> "a `param`"
  ELet _ _ -> "a top-level `let`"
  EFunction fn -> if fnIsModel fn then "the model" else "a function"
  EDataset _ -> "a dataset"

-- | Each dimension of a declared name comes from its own declaration.
selfOrigins :: Text -> Shape Int -> Span -> [Maybe Origin]
selfOrigins x s sp = map (const (Just (Origin x s sp))) (shapeDims s)

-- ---------------------------------------------------------------------------
-- Names

-- | What a name refers to.
data Resolved
  = LocalRef Text Local
  | GlobalRef Text Global
  | BuiltinRef Builtin

-- | Look a name up, innermost scope first, and record the use. E0201 or
-- E0203 if it isn't there.
resolve :: Scope -> Ident Span -> Check (Maybe Resolved)
resolve scope (Ident sp x)
  | Just l <- Map.lookup x (scopeLocals scope) = useLocal l >> pure (Just (LocalRef x l))
  | Just g <- Map.lookup x (scopeGlobals scope) = useGlobal x >> pure (Just (GlobalRef x g))
  | Just b <- lookupBuiltin x = pure (Just (BuiltinRef b))
  | Just declared <- Map.lookup x (fileNames (scopeFile scope)) = do
      useGlobal x
      report scope $
        if scopeDecl scope == Just x
          then
            diagnostic E0203 ("`" <> x <> "` is used in its own declaration") sp "not declared yet"
              & withNote "declarations must come before use; recursion is not supported"
          else
            diagnostic E0203 ("`" <> x <> "` is used before its declaration") sp "used here"
              & withSecondary declared "declared later, here"
              & withNote "declarations must come before use; recursion is not supported"
      pure Nothing
  | otherwise = do
      let candidates = Map.keys (scopeLocals scope) <> Map.keys (scopeGlobals scope) <> map builtinName builtins
      report scope $
        diagnostic E0201 ("undefined name `" <> x <> "`") sp "not defined"
          & maybe id (\s -> withHint ("did you mean `" <> s <> "`?")) (suggestName x candidates)
      pure Nothing

-- | The number of single-character edits (insertions, deletions,
-- substitutions, swaps of neighbours) that turn one text into the other:
-- the "optimal string alignment" distance.
editDistance :: Text -> Text -> Int
editDistance a b = d m n
  where
    s = Text.unpack a
    t = Text.unpack b
    m = length s
    n = length t
    at xs = (LazyMap.fromList (zip [1 ..] xs) LazyMap.!)
    sAt = at s
    tAt = at t
    -- A lazy map: each cell is computed when first needed.
    table = LazyMap.fromList [((i, j), cell i j) | i <- [0 .. m], j <- [0 .. n]]
    d i j = table LazyMap.! (i, j)
    cell i 0 = i
    cell 0 j = j
    cell i j =
      minimum $
        [d (i - 1) j + 1, d i (j - 1) + 1, d (i - 1) (j - 1) + (if sAt i == tAt j then 0 else 1)]
          <> [d (i - 2) (j - 2) + 1 | i > 1, j > 1, sAt i == tAt (j - 1), sAt (i - 1) == tAt j]

-- | The candidate closest to a misspelt name, if it is close enough: at
-- most a third of the name's length away (at least 1), and fewer edits
-- than the name has characters. Ties go to the earlier candidate.
suggestName :: Text -> [Text] -> Maybe Text
suggestName x candidates = fst <$> listToMaybe (sortOn snd close)
  where
    limit = max 1 (Text.length x `div` 3)
    close = [(c, k) | c <- nub candidates, c /= x, let k = editDistance x c, k <= limit, k < Text.length x]

-- ---------------------------------------------------------------------------
-- Expressions

-- | Check an expression.
checkExpr :: Scope -> Expr Span -> Check Typed
checkExpr scope = \case
  Num sp r -> pure (Full (TExpr sp Real (NNum r)))
  Var sp x -> checkVar scope sp x
  Call sp f args -> checkCall scope sp f args
  Index sp e i j -> checkIndex scope sp e i j
  Vector sp es -> do
    ts <- mapM (checkExpr scope) es
    case traverse typedShape ts >>= nonEmpty . zip (map exprAnn es) of
      Nothing -> pure Unknown
      Just elems -> case vectorShape elems of
        Left d -> report scope d >> pure Unknown
        Right s -> pure (node sp s (NVector <$> traverse typedExpr ts))
  Comprehension sp body pat gen -> misplaced scope sp body pat gen Nothing
  Power sp e k -> do
    t <- checkExpr scope e
    pure (elementwise sp t (\x -> NPower x (numberValue k)))
  Negate sp e -> do
    t <- checkExpr scope e
    pure (elementwise sp t NNegate)
  Binary sp op l r -> checkBinary scope sp op l r
  Let sp x ann bound body -> checkLet scope sp x ann bound body
  where
    elementwise sp t f = maybe Unknown (\s -> node sp s (f <$> typedExpr t)) (typedShape t)

-- | A name used as a value.
checkVar :: Scope -> Span -> Ident Span -> Check Typed
checkVar scope sp x =
  resolve scope x >>= \case
    Nothing -> pure Unknown
    Just (LocalRef name l) -> pure (ref (RLocal name) (localShape l))
    Just (GlobalRef name g) -> case globalEntry g of
      EInput s -> pure (ref (RInput name) s)
      EParam s -> pure (ref (RParam name) s)
      ELet s _ -> pure (ref (RLet name) s)
      EFunction fn -> do
        report scope $
          diagnostic E0204 ("`" <> name <> "` is a function, not a value") sp "used as a value"
            & withSecondary (fnSpan fn) "declared here"
            & withHint ("call it: `" <> name <> "(…)`")
        pure Unknown
      EDataset fields -> do
        datasetAsValue scope x g fields "used as a value"
        pure Unknown
    Just (BuiltinRef b) -> do
      report scope $
        diagnostic E0204 ("`" <> builtinName b <> "` is a built-in function, not a value") sp "used as a value"
          & withHint ("call it: `" <> builtinName b <> "(…)`")
      pure Unknown
  where
    ref r = maybe Unknown (\s -> Full (TExpr sp s (NRef r)))

-- | E0313: a dataset anywhere but as a comprehension source.
datasetAsValue :: Scope -> Ident Span -> Global -> Maybe [(Text, Span, Shape Int)] -> Text -> Check ()
datasetAsValue scope (Ident sp x) g fields what =
  report scope $
    diagnostic E0313 ("`" <> x <> "` is a dataset, not a value") sp what
      & withSecondary (globalSpan g) "declared here"
      & withHint ("a dataset can only be iterated over: `[ … | " <> pattern <> " <- " <> x <> " ]`")
  where
    pattern = case fields of
      Just [(f, _, _)] -> f
      Just fs -> "(" <> Text.intercalate ", " [f | (f, _, _) <- fs] <> ")"
      Nothing -> "…"

-- | @e[i]@, @e[i, j]@.
checkIndex :: Scope -> Span -> Expr Span -> Nat Span -> Maybe (Nat Span) -> Check Typed
checkIndex scope sp e i j = do
  t <- checkExpr scope e
  case typedShape t of
    Nothing -> pure Unknown
    Just s -> case indexShape s (natValue i) (natValue <$> j) of
      Right r -> pure (node sp r (NIndex <$> typedExpr t <*> pure (natValue i) <*> pure (natValue <$> j)))
      Left (OutOfBounds which k size) -> do
        let what = case (s, which) of
              (Mat _ _, 0) -> "row "
              (Mat _ _, _) -> "column "
              _ -> ""
            natSp = if which == 0 then natAnn i else maybe (natAnn i) natAnn j
        notes <- becauseNotes scope [(t, which)]
        report scope $
          diagnostic E0305 (what <> "index " <> showInt k <> " is out of bounds for `" <> prettyShape s <> "`") natSp
            ("valid " <> what <> "indices are 0 to " <> showInt (size - 1))
            & withSecondary (exprAnn e) ("this is `" <> prettyShape s <> "`")
            & withNotes notes
        pure Unknown
      Left TooManyIndices -> do
        let brackets = (exprAnn e) {spanStart = spanEnd (exprAnn e), spanEnd = spanEnd sp}
        report scope $ case s of
          Real ->
            diagnostic E0306 "indexing a `Real`" brackets "a `Real` has no elements"
              & withSecondary (exprAnn e) "this is `Real`"
          _ ->
            diagnostic E0306 ("too many indices for `" <> prettyShape s <> "`") (maybe brackets natAnn j) "one index too many"
              & withSecondary (exprAnn e) ("this is `" <> prettyShape s <> "`, which takes one index")
        pure Unknown

-- | @l op r@: a matrix product or an elementwise operation.
checkBinary :: Scope -> Span -> Operator -> Expr Span -> Expr Span -> Check Typed
checkBinary scope sp op l r = do
  tl <- checkExpr scope l
  tr <- checkExpr scope r
  case (typedShape tl, typedShape tr) of
    (Just a, Just b) -> case (if op == MatMul then matMul a b else broadcast a b) of
      Just s -> pure (node sp s (NBinary op <$> typedExpr tl <*> typedExpr tr))
      Nothing -> do
        let label = "`" <> prettyShape a <> " " <> prettyOperator op <> " " <> prettyShape b <> "`"
        if op == MatMul
          then do
            notes <- becauseNotes scope [(tl, 1), (tr, 0)]
            report scope $
              diagnostic E0302 "shape mismatch in matrix product" sp label
                & withNotes notes
                & matMulHint a b
          else do
            let k = firstDifference a b
            notes <- becauseNotes scope [(tl, k), (tr, k)]
            report scope $
              diagnostic E0301 "elementwise operands with incompatible shapes" sp label
                & withNotes notes
                & withNote "elementwise operations need equal shapes, or a `Real` on one side"
                & (if op == Times && isJust (matMul a b) then withHint "`*` is elementwise; for a matrix product write `@`" else id)
        pure Unknown
    _ -> pure Unknown
  where
    matMulHint a b = case (a, b) of
      (Vec _, Vec _) -> withHint "for the inner product of two vectors write `dot(u, v)`"
      (Real, _) -> withHint "to scale by a number, use `*`"
      (_, Real) -> withHint "to scale by a number, use `*`"
      (Vec _, Mat _ _) -> withHint "for a vector times a matrix write `transpose(M) @ v`"
      _ -> id

-- | The index of the first dimension in which two shapes differ (0 if they
-- are different kinds of shape).
firstDifference :: Shape Int -> Shape Int -> Int
firstDifference a b = case [k | (k, x, y) <- zip3 [0 ..] (shapeDims a) (shapeDims b), x /= y] of
  k : _ -> k
  [] -> 0

-- | @let x : t = bound in body@.
checkLet :: Scope -> Span -> Ident Span -> Maybe (Type Span) -> Expr Span -> Expr Span -> Check Typed
checkLet scope sp x ann bound body = do
  tb <- checkExpr scope bound
  shape <- case ann of
    Nothing -> pure (typedShape tb)
    Just ty -> do
      mt <- annotation scope (maybe (Just "a `let` outside a function") (const Nothing) (scopeVariables scope)) ty
      let vars = fromMaybe Map.empty (scopeVariables scope)
      case (mt, typedShape tb) of
        (Nothing, _) -> pure Nothing
        (Just t, Nothing) -> pure (substitute vars t)
        (Just t, Just s) -> case matchShape vars t s of
          Just _ -> pure (Just s)
          Nothing -> do
            let bound' = bindTemplate vars t
                expectedText
                  | bound' == t = "`" <> prettyTemplate t <> "`"
                  | otherwise = "`" <> prettyTemplate t <> "`, which is `" <> prettyTemplate bound' <> "` here,"
            report scope (annotationMismatch bound s (typeAnn ty) expectedText "value doesn't match the `let` annotation")
            pure (substitute vars t)
  instances <- gets stInstances
  let origins = case ann of
        Just _ -> maybe [] (\s -> selfOrigins (identName x) s (identAnn x)) shape
        Nothing -> maybe [] (dimOrigins scope instances) (typedExpr tb)
  tbody <- withLocals scope [NewLocal x LetLocal shape origins] (`checkExpr` body)
  pure $ case typedShape tbody of
    Nothing -> Unknown
    Just s -> node sp s (NLet (identName x) <$> typedExpr tb <*> typedExpr tbody)

-- ---------------------------------------------------------------------------
-- Calls

-- | @f(args)@.
checkCall :: Scope -> Span -> Ident Span -> [Expr Span] -> Check Typed
checkCall scope sp f args = do
  target <- resolve scope f
  case (target, args) of
    (Just (BuiltinRef b), [Comprehension _ body pat gen])
      | builtinType b == Reduction -> reduction scope sp b body pat gen
    _ -> do
      let takesSequences = case target of
            Just (BuiltinRef b) -> builtinType b == Reduction
            _ -> False
      typed <- mapM (argument takesSequences) args
      case target of
        Nothing -> pure Unknown
        Just (LocalRef _ l) -> notAFunction (localDescription l) (Just (localSpan l))
        Just (GlobalRef _ g) -> case globalEntry g of
          EFunction fn -> callFunction scope sp f fn args typed
          EDataset fields -> datasetAsValue scope f g fields "called here" >> pure Unknown
          entry -> notAFunction (entryDescription entry) (Just (globalSpan g))
        Just (BuiltinRef b) -> callBuiltin scope sp f b args typed
  where
    argument takesSequences = \case
      Comprehension csp body pat gen
        | takesSequences -> comprehension scope body pat gen >> pure Unknown
        | otherwise -> misplaced scope csp body pat gen (Just f)
      e -> checkExpr scope e
    notAFunction what declared = do
      report scope $
        diagnostic E0204 ("`" <> identName f <> "` is not a function") (identAnn f) ("called, but it is " <> what)
          & maybe id (`withSecondary` "declared here") declared
      pure Unknown

-- | E0205, or 'Nothing' if the number of arguments is right.
arityError :: Text -> Int -> Int -> Span -> Text -> Maybe Diagnostic
arityError f expected given sp signature
  | expected == given = Nothing
  | otherwise =
      Just $
        diagnostic E0205 ("`" <> f <> "` takes " <> count expected <> ", but " <> showInt given <> verb) sp ("called with " <> count given)
          & withNote ("signature: " <> signature)
  where
    count k = showInt k <> if k == 1 then " argument" else " arguments"
    verb = if given == 1 then " was given" else " were given"

-- | How to talk about one parameter of a callee in messages: its
-- template, what it expects (in words), and its annotation's span.
data ParamDesc = ParamDesc (Maybe Template) Text (Maybe Span)

pdTemplate :: ParamDesc -> Maybe Template
pdTemplate (ParamDesc t _ _) = t

pdAnnotation :: ParamDesc -> Maybe Span
pdAnnotation (ParamDesc _ _ a) = a

-- | E0303 and E0304, for built-in and user functions alike.
reportMatch :: Scope -> Text -> Text -> [ParamDesc] -> [Expr Span] -> [Typed] -> MatchProblem -> Check ()
reportMatch scope f signature params args typed = \case
  ArgumentMismatch i -> do
    let ParamDesc _ expects annSp = params !! i
        s = maybe "?" prettyShape (typedShape (typed !! i))
    notes <- becauseNotes scope [(typed !! i, 0)]
    report scope $
      diagnostic E0304 ("argument " <> showInt (i + 1) <> " of `" <> f <> "` doesn't match its parameter") (exprAnn (args !! i))
        ("expected " <> expects <> ", found `" <> s <> "`")
        & maybe (withNote ("signature: " <> signature)) (`withSecondary` "the parameter is declared here") annSp
        & withNotes notes
  InconsistentVariable v (Binding n1 i1) (Binding n2 i2) -> do
    let argShape i = maybe "?" prettyShape (typedShape (typed !! i))
        dimOf i = case pdTemplate (params !! i) of
          Just t -> length (takeWhile (/= Variable v) (shapeDims t))
          Nothing -> 0
        so i n = "`" <> argShape i <> "`, so `" <> v <> " = " <> showInt n <> "` here"
        bothVec = all (\i -> fmap (() <$) (pdTemplate (params !! i)) == Just (Vec ())) [i1, i2]
        annotations
          | i1 == i2 = maybe id (`withSecondary` ("`" <> v <> "` appears twice here")) (pdAnnotation (params !! i1))
          | otherwise = case (pdAnnotation (params !! i1), pdAnnotation (params !! i2)) of
              (Just a1, Just a2) ->
                withSecondary a1 "" . withSecondary a2 (if bothVec then "both parameters have length `" <> v <> "`" else "both parameters use `" <> v <> "`")
              _ -> withNote ("signature: " <> signature)
    notes <- becauseNotes scope (if i1 == i2 then [(typed !! i1, dimOf i1)] else [(typed !! i1, dimOf i1), (typed !! i2, dimOf i2)])
    report scope $
      diagnostic E0303 ("shape variable `" <> v <> "` bound inconsistently") (exprAnn (args !! i2))
        ( if i1 == i2
            then "`" <> argShape i2 <> "`: `" <> v <> "` would be both " <> showInt n1 <> " and " <> showInt n2
            else so i2 n2
        )
        & (if i1 == i2 then id else withSecondary (exprAnn (args !! i1)) (so i1 n1))
        & annotations
        & withNotes notes

-- | A call of a built-in function on values.
callBuiltin :: Scope -> Span -> Ident Span -> Builtin -> [Expr Span] -> [Typed] -> Check Typed
callBuiltin scope sp f b args typed
  | Just d <- arityError name (builtinArity b) (length args) (identAnn f) signature = report scope d >> pure fallback
  | otherwise = case traverse typedShape typed of
      Nothing -> pure fallback
      Just shapes -> case applyBuiltin b (map Value shapes) of
        Right s -> pure (node sp s (NBuiltin b <$> traverse typedExpr typed))
        Left problems -> do
          forM_ problems $ \case
            BadArgument p -> reportMatch scope name signature params args typed p
            IncompatibleArguments x y -> do
              notes <- becauseNotes scope [(t, firstDifference x y) | t <- typed]
              report scope $
                diagnostic E0301 "elementwise operands with incompatible shapes" sp
                  ("`" <> name <> "(" <> prettyShape x <> ", " <> prettyShape y <> ")`")
                  & withNotes notes
                  & withNote "elementwise operations need equal shapes, or a `Real` on one side"
            -- Arity was checked above, and comprehension arguments never
            -- get here (they are E0311 already).
            WrongArity _ _ -> pure ()
            SequenceArgument _ -> pure ()
          pure fallback
  where
    name = builtinName b
    signature = builtinSignature b
    params = case builtinType b of
      Matching ps _ -> [ParamDesc (Just t) (builtinExpects b i) Nothing | (i, (_, t)) <- zip [0 ..] ps]
      _ -> [ParamDesc Nothing (builtinExpects b i) Nothing | i <- [0 .. builtinArity b - 1]]
    fallback = case builtinType b of
      Matching _ r | Just s <- toShape r -> ShapeOnly s
      _ -> Unknown

-- | A call of a @fn@ or the @model@ (D8): match the arguments against the
-- parameter templates, then check the body at those shapes, once per
-- distinct list of argument shapes.
callFunction :: Scope -> Span -> Ident Span -> Function -> [Expr Span] -> [Typed] -> Check Typed
callFunction scope sp f fn args typed
  | Just d <- arityError name (length (fnParams fn)) (length args) (identAnn f) signature = do
      report scope (d & withSecondary (fnSpan fn) ("`" <> name <> "` is declared here"))
      pure (fallback Map.empty)
  | otherwise = do
      let templates = [t | FnParam _ t _ <- fnParams fn]
          (bindings, problems) = matchArguments templates (map typedShape typed)
      mapM_ (reportMatch scope name signature params args typed) problems
      case traverse typedShape typed of
        Just shapes | null problems, not (fnBroken fn) -> do
          instances <- gets stInstances
          let key = InstanceKey name shapes
              origins = [maybe [] (dimOrigins scope instances) (typedExpr t) | t <- typed]
              notes = ("while checking the call at " <> showLocation sp) : scopeContext scope
          outcome <- case Map.lookup key instances of
            Just o -> pure o
            Nothing -> do
              o <- runBody (scopeFile scope) (Calling notes) fn (map Just shapes) origins (bindingSize <$> bindings)
              modify' $ \st -> st {stInstances = Map.insert key o (stInstances st)}
              pure o
          pure $ case outcome of
            Succeeded i -> node sp (instanceResult i) (NCall key <$> traverse typedExpr typed)
            Failed ms -> maybe Unknown ShapeOnly ms
        _ -> pure (fallback (bindingSize <$> bindings))
  where
    name = fnName fn
    signature = prettySignature (Signature name [(identName b, t) | FnParam b t _ <- fnParams fn] (fst <$> fnResult fn))
    params = [ParamDesc t (maybe "anything" (\t' -> "`" <> prettyTemplate t' <> "`") t) annSp | FnParam _ t annSp <- fnParams fn]
    fallback bindings = case fnResult fn >>= substitute bindings . fst of
      Just s -> ShapeOnly s
      Nothing -> maybe Unknown ShapeOnly (fnFallback fn)

-- ---------------------------------------------------------------------------
-- Comprehensions

-- | @sum [ … ]@ / @mean [ … ]@.
reduction :: Scope -> Span -> Builtin -> Expr Span -> Pattern Span -> Generator Span -> Check Typed
reduction scope sp b body pat gen = do
  (ms, it) <- comprehension scope body pat gen
  pure $ case ms of
    Just t | Right r <- applyBuiltin b [Seq t] -> node sp r (NReduce b <$> it)
    _ -> Unknown

-- | E0311: a comprehension that is not the argument of @sum@ or @mean@.
-- Its body is still checked, for its own errors and for the label.
misplaced :: Scope -> Span -> Expr Span -> Pattern Span -> Generator Span -> Maybe (Ident Span) -> Check Typed
misplaced scope sp body pat gen callee = do
  (ms, _) <- comprehension scope body pat gen
  report scope $
    diagnostic E0311 "a comprehension can only be the argument of `sum` or `mean`" sp
      (maybe "this is a sequence" (\s -> "this is a sequence of `" <> prettyShape s <> "`") ms)
      & maybe id (\f -> withSecondary (identAnn f) ("`" <> identName f <> "` can't take a sequence")) callee
      & withHint "reduce it to a value with `sum [ … ]` or `mean [ … ]`"
  pure Unknown

-- | Check a comprehension: the shape of its body (the @t@ of @Seq t@) and
-- its typed form.
comprehension :: Scope -> Expr Span -> Pattern Span -> Generator Span -> Check (Maybe (Shape Int), Maybe Iteration)
comprehension scope body pat gen = do
  let names = case pat of
        PVar x -> [x]
        PTuple _ xs -> xs
      patSp = case pat of
        PVar x -> identAnn x
        PTuple sp _ -> sp
  duplicates scope "comprehension variable" names
  case gen of
    FromData d -> do
      target <- resolve scope d
      schema <- case target of
        Just (GlobalRef _ g) | EDataset fields <- globalEntry g -> pure (Just (g, fields))
        Just (GlobalRef _ g) -> notADataset d (entryDescription (globalEntry g)) (Just (globalSpan g))
        Just (LocalRef _ l) -> notADataset d (localDescription l) (Just (localSpan l))
        Just (BuiltinRef _) -> notADataset d "a built-in function" Nothing
        Nothing -> pure Nothing
      bound <- case schema of
        Just (g, Just fields)
          | length fields == length names ->
              pure [NewLocal x PatternVariable (Just s) (selfOrigins f s fsp) | (x, (f, fsp, s)) <- zip names fields]
          | otherwise -> do
              let fieldList = Text.intercalate ", " ["`" <> f <> "`" | (f, _, _) <- fields]
                  suggestion = case fields of
                    [(f, _, _)] -> f
                    _ -> "(" <> Text.intercalate ", " [f | (f, _, _) <- fields] <> ")"
              report scope $
                diagnostic E0207 ("the pattern binds " <> names' (length names) <> ", but `" <> identName d <> "` has " <> fields' (length fields)) patSp
                  (names' (length names))
                  & withSecondary (globalSpan g) ("fields " <> fieldList)
                  & withHint ("write `" <> suggestion <> " <- " <> identName d <> "`")
              pure (unknowns names)
        _ -> pure (unknowns names)
      t <- withLocals scope bound (`checkExpr` body)
      instances <- gets stInstances
      forM_ (typedExpr t >>= nestedData instances) $ \(inner, what) ->
        report scope $
          diagnostic E0314 "comprehensions over datasets can't be nested" inner what
            & withSecondary (identAnn d) ("inside this comprehension over `" <> identName d <> "`")
            & withNote "a comprehension over a dataset is unrolled once per row; its body can't contain another one"
      pure (typedShape t, OverData (identName d) (map identName names) <$> typedExpr t)
    FromParams psp -> do
      let params = fileParams (scopeFile scope)
      mapM_ (\(p, _, _) -> useGlobal p) params
      when (length names /= 1) $
        report scope $
          diagnostic E0207 ("`params` gives one parameter at a time, but the pattern binds " <> names' (length names)) patSp (names' (length names))
            & withHint "write `p <- params`"
      case (names, params) of
        (x : _, []) -> do
          _ <- withLocals scope (unknowns [x]) (`checkExpr` body)
          report scope $
            diagnostic E0310 "comprehension over `params`, but the file declares no `param`" psp "there are no parameters"
              & withNote "the body has no iteration to give it a shape"
          pure (Nothing, Nothing)
        (x : _, _ : _) -> overParams x params
        ([], _) -> pure (Nothing, Nothing)
  where
    unknowns = map (\x -> NewLocal x PatternVariable Nothing [])
    names' k = showInt k <> if k == 1 then " name" else " names"
    fields' k = showInt k <> if k == 1 then " field" else " fields"
    notADataset d what declared = do
      report scope $
        diagnostic E0204 ("`" <> identName d <> "` is not a dataset") (identAnn d) ("this is " <> what)
          & maybe id (`withSecondary` "declared here") declared
          & withNote "a comprehension iterates over a dataset or over `params`"
      pure Nothing

    -- The body once per parameter. After the first parameter the names are
    -- known to be fine, so only shape errors are reported; after the first
    -- parameter with errors, the rest are skipped (they would mostly
    -- repeat them).
    overParams x params = do
      results <- iterateParams x (0 :: Int) params
      let shapes = traverse (\(_, _, t) -> typedShape t) results
      case (results, shapes) of
        ((p0, sp0, _) : _, Just ss@(s0 : _)) | length ss == length params -> case [(r, s) | (r, s) <- zip results ss, s /= s0] of
          [] -> pure (Just s0, OverParams (identName x) <$> traverse (\(p, _, t) -> (,) p <$> typedExpr t) results)
          (((pk, spk, _), sk) : _) -> do
            report scope $
              diagnostic E0310 "the comprehension body has different shapes for different parameters" (exprAnn body)
                ("`" <> prettyShape s0 <> "` for `" <> p0 <> "`, but `" <> prettyShape sk <> "` for `" <> pk <> "`")
                & withSecondary sp0 ("`" <> p0 <> " : " <> maybe "?" prettyShape (paramShapeOf p0 params) <> "`")
                & withSecondary spk ("`" <> pk <> " : " <> maybe "?" prettyShape (paramShapeOf pk params) <> "`")
                & withHint "every parameter must give the same shape; `sum(…)` makes any shape a `Real`"
            pure (Nothing, Nothing)
        _ -> pure (Nothing, Nothing)

    iterateParams _ _ [] = pure []
    iterateParams x i ((p, psp, ps) : rest) = do
      before <- errorCount
      let note = "while checking the body with `" <> identName x <> "` = `" <> p <> maybe "" (\s -> " : " <> prettyShape s) ps <> "`"
          scope' = scope {scopeQuiet = scopeQuiet scope || i > 0, scopeContext = note : scopeContext scope}
          local = NewLocal x PatternVariable ps (maybe [] (\s -> selfOrigins p s psp) ps)
      t <- withLocals scope' [local] (`checkExpr` body)
      after <- errorCount
      if after > before || isNothing (typedShape t)
        then pure [(p, psp, t)]
        else ((p, psp, t) :) <$> iterateParams x (i + 1) rest

    paramShapeOf p params = listToMaybe [s | (q, _, Just s) <- params, q == p]

-- | A comprehension over a dataset somewhere inside a typed expression:
-- written there, or in the body of a function it calls. Its span (or the
-- call's) and a label.
nestedData :: Map InstanceKey Outcome -> TExpr -> Maybe (Span, Text)
nestedData instances e = describe <$> go e
  where
    describe (sp, d, viaCall)
      | viaCall = (sp, "this call iterates over `" <> d <> "`")
      | otherwise = (sp, "a comprehension over `" <> d <> "`")

    -- The span, the dataset, and whether the span is a call.
    go (TExpr sp _ n) = case n of
      NReduce _ (OverData d _ _) -> Just (sp, d, False)
      NReduce _ (OverParams _ bodies) -> first (map (go . snd) bodies)
      NCall key args -> first (map go args) `orElse` (callee key >>= \(_, d, _) -> Just (sp, d, True))
      NNum _ -> Nothing
      NRef _ -> Nothing
      NVector es -> first (map go es)
      NIndex x _ _ -> go x
      NNegate x -> go x
      NPower x _ -> go x
      NBinary _ l r -> first [go l, go r]
      NBuiltin _ args -> first (map go args)
      NLet _ b body -> first [go b, go body]
    callee key = case Map.lookup key instances of
      Just (Succeeded i) -> go (instanceBody i)
      _ -> Nothing
    first = foldr orElse Nothing
    orElse a b = maybe b Just a

-- ---------------------------------------------------------------------------
-- Explanations: where a dimension comes from

-- | The notes that explain operands' shapes: for an operand that is a name
-- whose shape was inferred (a @let@, a parameter of a function instance),
-- the declaration that the given dimension comes from. "h1 : Vec 8 because
-- W1 : Mat 8 2 (declared at 6:7)".
becauseNotes :: Scope -> [(Typed, Int)] -> Check [Text]
becauseNotes scope operands = do
  instances <- gets stInstances
  pure (nub [n | (t, k) <- operands, Just e <- [typedExpr t], Just n <- [because instances e k]])
  where
    because instances e k = case texprNode e of
      NRef r -> do
        o <- join (listToMaybe (drop k (dimOrigins scope instances e)))
        guard (originName o /= refName r)
        Just $
          refName r <> " : " <> prettyShape (texprShape e) <> " because " <> originName o <> " : "
            <> prettyShape (originShape o) <> " (declared at " <> showPos (spanStart (originSpan o)) <> ")"
      _ -> Nothing

-- | For each dimension of an expression's shape, the declaration whose
-- type it was copied from, if any. The shape rules say which operand each
-- result dimension comes from: the rows of @W \@ x@ are the rows of @W@,
-- the length of @a + b@ is that of either operand, and so on.
dimOrigins :: Scope -> Map InstanceKey Outcome -> TExpr -> [Maybe Origin]
dimOrigins scope instances = go Map.empty
  where
    go env (TExpr _ shape n) = case n of
      NNum _ -> []
      NRef (RLocal x) -> case Map.lookup x env of
        Just os -> os
        Nothing -> maybe none localOrigins (Map.lookup x (scopeLocals scope))
      NRef r -> maybe none (globalOrigins (refName r)) (Map.lookup (refName r) (scopeGlobals scope))
      NVector es -> case (shape, es) of
        (Mat _ _, row : _) -> Nothing : take 1 (go env row)
        _ -> [Nothing]
      NIndex e _ Nothing -> drop 1 (go env e)
      NIndex {} -> []
      NNegate e -> go env e
      NPower e _ -> go env e
      NBinary MatMul l r -> take 1 (go env l) <> drop 1 (go env r)
      NBinary _ l r -> merge env l r
      NBuiltin b args -> case (b, args) of
        (BTranspose, [m]) -> reverse (go env m)
        (_, [a]) | builtinType b == Elementwise -> go env a
        (_, [a, c]) | builtinType b == Broadcasting -> merge env a c
        _ -> []
      NReduce _ (OverData d xs body) ->
        let fields = case Map.lookup d (scopeGlobals scope) of
              Just (Global _ (EDataset (Just fs))) -> [selfOrigins f s sp | (f, sp, s) <- fs]
              _ -> []
         in go (Map.union (Map.fromList (zip xs fields)) env) body
      NReduce _ (OverParams x ((p, body) : _)) ->
        let os = maybe [] (\(_, sp, s) -> maybe [] (\s' -> selfOrigins p s' sp) s) (lookupParam p)
         in go (Map.insert x os env) body
      NReduce _ (OverParams _ []) -> none
      NCall key args -> case Map.lookup key instances of
        Just (Succeeded i) -> go (Map.fromList (zip (instanceParams i) (map (go env) args))) (instanceBody i)
        _ -> none
      NLet x b body -> go (Map.insert x (go env b) env) body
      where
        none = map (const Nothing) (shapeDims shape)

    merge env l r
      | texprShape r == Real = go env l
      | texprShape l == Real = go env r
      | otherwise = zipWith orElse (go env l) (go env r)
    orElse a b = maybe b Just a

    globalOrigins x (Global sp entry) = case entry of
      EInput (Just s) -> selfOrigins x s sp
      EParam (Just s) -> selfOrigins x s sp
      ELet _ os -> os
      _ -> []
    lookupParam p = listToMaybe [e | e@(q, _, _) <- fileParams (scopeFile scope), q == p]

-- ---------------------------------------------------------------------------
-- W0203: parameters that can't affect the loss

-- | W0203 for every used parameter that the loss doesn't depend on. Only
-- run on a checked program (the typed trees are complete). Parameters
-- that are never used at all already have W0202.
--
-- The analysis follows references through @let@s, function instances and
-- comprehensions over @params@, so it is exact at the level of whole
-- declarations. (Whether every /element/ of a parameter matters, say every
-- row of a matrix of which only row 0 is used, is a question about the
-- compiled graph.)
lossInfluence :: Set Text -> Checked -> [Diagnostic]
lossInfluence used c = case checkedLoss c of
  Nothing -> []
  Just l ->
    let affecting = deps Map.empty (lossExpr l)
     in [ diagnostic W0203 ("`" <> paramName p <> "` can't affect `loss`") (paramNameSpan p) "its gradient is always zero"
            & withSecondary (lossKeywordSpan l) "`loss` doesn't depend on it"
            & withNote "training never changes it"
        | p <- checkedParams c
        , paramName p `Set.member` used
        , paramName p `Set.notMember` affecting
        ]
  where
    lets = foldl (\m li -> Map.insert (letName li) (depsWith m Map.empty (letExpr li)) m) Map.empty (checkedLets c)
    deps = depsWith lets

    -- The parameters an expression depends on; @env@ gives those of the
    -- locals in scope.
    depsWith :: Map Text (Set Text) -> Map Text (Set Text) -> TExpr -> Set Text
    depsWith letDeps env (TExpr _ _ n) = case n of
      NNum _ -> Set.empty
      NRef (RParam p) -> Set.singleton p
      NRef (RInput _) -> Set.empty
      NRef (RLet x) -> Map.findWithDefault Set.empty x letDeps
      NRef (RLocal x) -> Map.findWithDefault Set.empty x env
      NVector es -> foldMap go es
      NIndex e _ _ -> go e
      NNegate e -> go e
      NPower e _ -> go e
      NBinary _ l r -> go l <> go r
      NBuiltin _ args -> foldMap go args
      NReduce _ (OverData _ xs body) -> depsWith letDeps (foldr Map.delete env xs) body
      NReduce _ (OverParams x bodies) ->
        foldMap (\(p, body) -> depsWith letDeps (Map.insert x (Set.singleton p) env) body) bodies
      NCall key args -> case Map.lookup key (checkedInstances c) of
        Just i -> depsWith letDeps (Map.fromList (zip (instanceParams i) (map go args))) (instanceBody i)
        Nothing -> foldMap go args
      NLet x b body -> depsWith letDeps (Map.insert x (go b) env) body
      where
        go = depsWith letDeps env

-- ---------------------------------------------------------------------------
-- Small helpers

-- | The first element with each key, in order.
nubOrdOn :: (Ord k) => (a -> k) -> [a] -> [a]
nubOrdOn key = go Set.empty
  where
    go _ [] = []
    go seen (x : xs)
      | key x `Set.member` seen = go seen xs
      | otherwise = x : go (Set.insert (key x) seen) xs

showInt :: Int -> Text
showInt = Text.pack . show

withNotes :: [Text] -> Diagnostic -> Diagnostic
withNotes ns d = foldl (flip withNote) d ns
