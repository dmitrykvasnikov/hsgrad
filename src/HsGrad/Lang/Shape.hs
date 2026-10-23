{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The types of the @.hsg@ language and the rules that combine them.
--
-- Every value of the language is a 'Real', a @'Vec' n@ or a @'Mat' r c@
-- with sizes known before anything runs (@01-language.md@ §4). This module
-- knows those shapes and nothing about names or source text:
--
-- * 'Shape' is parameterised by what a dimension is. A concrete shape is a
--   @Shape Int@ (@Vec 3@); a declared signature may contain shape
--   variables, so it is a @Shape Dimension@, called a 'Template'
--   (@Vec n@).
-- * __Matching__ ('matchArguments', 'matchShape') checks concrete argument
--   shapes against templates and binds the shape variables, as the
--   "template style" checking of decision D8 needs.
-- * __Operations__: 'broadcast' (elementwise operators, §4.1), 'matMul'
--   (@\@@), 'indexShape' (@v[i]@, @M[i, j]@).
-- * __Built-in functions__ (§6): their names, their types, and
--   'applyBuiltin', which computes the result shape of a call or says
--   what is wrong with it.
-- * Pretty-printing of all of these for diagnostics.
--
-- "HsGrad.Lang.Check" walks the program and uses these rules at every
-- node; turning a 'MatchProblem' or a 'CallProblem' into a message with
-- source locations is its job.
-- See @docs/13-checker.md@.
module HsGrad.Lang.Shape
  ( -- * Shapes
    Shape (..)
  , shapeDims
  , shapeSize
  , prettyShape

    -- * Templates: shapes with shape variables
  , Dimension (..)
  , Template
  , fromShape
  , toShape
  , templateVariables
  , bindTemplate
  , prettyTemplate

    -- * Other sorts of things
  , Operand (..)
  , prettyOperand
  , Kind (..)
  , prettyKind
  , Signature (..)
  , prettySignature
  , prettyArrow

    -- * Matching
  , Binding (..)
  , Bindings
  , MatchProblem (..)
  , matchArguments
  , matchShape
  , substitute

    -- * Operations
  , broadcast
  , matMul
  , IndexProblem (..)
  , indexShape

    -- * Built-in functions
  , Builtin (..)
  , builtins
  , builtinName
  , lookupBuiltin
  , BuiltinType (..)
  , builtinType
  , builtinArity
  , builtinSignature
  , builtinExpects
  , CallProblem (..)
  , applyBuiltin
  ) where

import Control.Monad (foldM)
import Data.Foldable (toList)
import Data.List (find, nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text

-- ---------------------------------------------------------------------------
-- Shapes

-- | The shape of a value: a scalar, a vector of @n@ elements, or a matrix
-- of @r@ rows and @c@ columns (stored row-major).
--
-- The parameter is what a dimension is: an 'Int' in a concrete shape, a
-- 'Dimension' (number or shape variable) in a 'Template'. Because 'Shape'
-- is a 'Functor', @fmap 'Fixed'@ turns a concrete shape into a template and
-- 'traverse' does the opposite where possible ('toShape').
--
-- The constructor 'Real' lives in the namespace of data constructors, so
-- it does not clash with the Prelude class of the same name.
data Shape d
  = Real
  | Vec d
  | -- | Rows, then columns.
    Mat d d
  deriving (Eq, Ord, Show, Functor, Foldable, Traversable)

-- | The dimensions, outermost first: @[]@, @[n]@ or @[r, c]@.
shapeDims :: Shape d -> [d]
shapeDims = toList

-- | The number of scalars in a value of this shape: 1, @n@ or @r·c@.
shapeSize :: Shape Int -> Int
shapeSize = product . shapeDims

-- | As written in the language: @Real@, @Vec 3@, @Mat 16 2@.
prettyShape :: Shape Int -> Text
prettyShape = prettyWith (Text.pack . show)

prettyWith :: (d -> Text) -> Shape d -> Text
prettyWith dim = \case
  Real -> "Real"
  Vec n -> "Vec " <> dim n
  Mat r c -> "Mat " <> dim r <> " " <> dim c

-- ---------------------------------------------------------------------------
-- Templates

-- | A dimension in a declared type: a number (@16@) or a shape variable
-- (@n@).
data Dimension
  = Fixed Int
  | Variable Text
  deriving (Eq, Ord, Show)

-- | A declared type, possibly with shape variables: @Mat m n@, @Vec 2@.
type Template = Shape Dimension

-- | A concrete shape as a template without variables.
fromShape :: Shape Int -> Template
fromShape = fmap Fixed

-- | The concrete shape of a template, if it has no shape variables.
toShape :: Template -> Maybe (Shape Int)
toShape = substitute Map.empty

-- | The shape variables of a template, each once, in order.
templateVariables :: Template -> [Text]
templateVariables t = nub [v | Variable v <- shapeDims t]

-- | Replace the variables that have a value; leave the others. For
-- messages: with @m = 16@, @Vec m@ is shown as @Vec 16@.
bindTemplate :: Map Text Int -> Template -> Template
bindTemplate bs = fmap $ \case
  Variable v | Just n <- Map.lookup v bs -> Fixed n
  d -> d

-- | As written: @Vec n@, @Mat 16 n@.
prettyTemplate :: Template -> Text
prettyTemplate = prettyWith $ \case
  Fixed n -> Text.pack (show n)
  Variable v -> v

-- ---------------------------------------------------------------------------
-- Other sorts of things

-- | What a built-in function can receive: a value, or a sequence (the
-- result of a comprehension, @Seq t@ in @01-language.md@ §4), which only
-- @sum@ and @mean@ accept.
data Operand
  = Value (Shape Int)
  | Seq (Shape Int)
  deriving (Eq, Show)

-- | @Vec 2@, or @Seq Real@ / @Seq (Vec 2)@.
prettyOperand :: Operand -> Text
prettyOperand = \case
  Value s -> prettyShape s
  Seq Real -> "Seq Real"
  Seq s -> "Seq (" <> prettyShape s <> ")"

-- | What a name stands for (@01-language.md@ §7.2). Values have shapes;
-- functions can only be called; datasets can only be the source of a
-- comprehension.
data Kind
  = ValueKind
  | FunctionKind
  | DatasetKind
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | With an article, for messages: "a value", "a function", "a dataset".
prettyKind :: Kind -> Text
prettyKind = \case
  ValueKind -> "a value"
  FunctionKind -> "a function"
  DatasetKind -> "a dataset"

-- | The declared signature of a function: its parameters with their
-- optional annotations, and the optional result annotation.
data Signature = Signature
  { signatureName :: Text
  , signatureParams :: [(Text, Maybe Template)]
  , signatureResult :: Maybe Template
  }
  deriving (Eq, Show)

-- | As written: @layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m@,
-- @sq(z)@.
prettySignature :: Signature -> Text
prettySignature (Signature f ps r) =
  f <> "(" <> Text.intercalate ", " (map param ps) <> ")" <> annotation r
  where
    param (x, t) = x <> annotation t
    annotation = maybe "" ((" : " <>) . prettyTemplate)

-- | The type of a function at concrete shapes: @Vec 2 → Real@,
-- @Mat 16 2, Vec 16, Vec 2 → Vec 16@, @() → Real@.
prettyArrow :: [Shape Int] -> Shape Int -> Text
prettyArrow args result = params <> " → " <> prettyShape result
  where
    params
      | null args = "()"
      | otherwise = Text.intercalate ", " (map prettyShape args)

-- ---------------------------------------------------------------------------
-- Matching

-- | The value of a shape variable, and the argument (0-based) whose shape
-- gave it that value first. The position is what lets an inconsistency
-- point at both arguments involved (E0303).
data Binding = Binding
  { bindingSize :: Int
  , bindingArgument :: Int
  }
  deriving (Eq, Show)

-- | Shape variables bound so far.
type Bindings = Map Text Binding

-- | Why an argument list doesn't fit a list of templates.
data MatchProblem
  = -- | The argument at this position has the wrong kind of shape, or a
    -- wrong fixed dimension (E0304).
    ArgumentMismatch Int
  | -- | The variable was bound by one argument, and another argument (or
    -- another dimension of the same one) needs a different value (E0303).
    -- The first binding, then the conflicting one.
    InconsistentVariable Text Binding Binding
  deriving (Eq, Show)

-- | Match concrete argument shapes against parameter templates, left to
-- right, binding shape variables as they are met (@01-language.md@ §4.2).
--
-- A parameter without an annotation ('Nothing') accepts anything; an
-- argument whose shape is unknown ('Nothing', because of an earlier
-- error) is skipped, so it causes no follow-up problem. An argument that
-- doesn't fit its template binds nothing: its sizes are not trustworthy.
-- Every problem is reported, not just the first.
--
-- For @layer(W : Mat m n, b : Vec m, x : Vec n)@ called with
-- @Mat 16 2, Vec 16, Vec 2@ the result is @m = 16@ and @n = 2@, both bound
-- by argument 0, and no problems; with @Mat 16 2, Vec 8, Vec 2@ it is
-- @InconsistentVariable "m" (Binding 16 0) (Binding 8 1)@.
matchArguments :: [Maybe Template] -> [Maybe (Shape Int)] -> (Bindings, [MatchProblem])
matchArguments templates shapes = reverse <$> foldl step (Map.empty, []) (zip3 [0 ..] templates shapes)
  where
    step (bs, ps) (i, Just t, Just s) = case pairUp t s of
      Just ds | and [n == k | (Fixed k, n) <- ds] -> foldl (bind i) (bs, ps) [(v, n) | (Variable v, n) <- ds]
      _ -> (bs, ArgumentMismatch i : ps)
    step acc _ = acc

    bind i (bs, ps) (v, n) = case Map.lookup v bs of
      Nothing -> (Map.insert v (Binding n i) bs, ps)
      Just b
        | bindingSize b == n -> (bs, ps)
        | otherwise -> (bs, InconsistentVariable v b (Binding n i) : ps)

-- | Match one concrete shape against one template, extending the given
-- variable values: for result annotations and @let@ annotations. 'Nothing'
-- if it doesn't fit.
matchShape :: Map Text Int -> Template -> Shape Int -> Maybe (Map Text Int)
matchShape bs t s = pairUp t s >>= foldM bind bs
  where
    bind acc (Fixed k, n) = if k == n then Just acc else Nothing
    bind acc (Variable v, n) = case Map.lookup v acc of
      Nothing -> Just (Map.insert v n acc)
      Just m -> if m == n then Just acc else Nothing

-- | The concrete shape of a template under the given variable values, if
-- every variable has one.
substitute :: Map Text Int -> Template -> Maybe (Shape Int)
substitute bs = traverse $ \case
  Fixed n -> Just n
  Variable v -> Map.lookup v bs

-- | The dimensions of two shapes side by side, if they are the same kind
-- of shape.
pairUp :: Shape a -> Shape b -> Maybe [(a, b)]
pairUp a b = case (a, b) of
  (Real, Real) -> Just []
  (Vec m, Vec n) -> Just [(m, n)]
  (Mat r c, Mat r' c') -> Just [(r, r'), (c, c')]
  _ -> Nothing

-- ---------------------------------------------------------------------------
-- Operations

-- | The result shape of an elementwise operation (@+ - * /@, @max@,
-- @min@), following §4.1: equal shapes give that shape, and a 'Real' on
-- either side is applied to every element of the other. Nothing else is
-- broadcast.
broadcast :: Shape Int -> Shape Int -> Maybe (Shape Int)
broadcast a b = case (a, b) of
  (Real, _) -> Just b
  (_, Real) -> Just a
  _
    | a == b -> Just a
    | otherwise -> Nothing

-- | The result shape of @a \@ b@: @Mat m n \@ Vec n = Vec m@ and
-- @Mat m n \@ Mat n k = Mat m k@. Anything else (mismatched inner
-- dimensions, a vector or scalar on the left) is 'Nothing'.
matMul :: Shape Int -> Shape Int -> Maybe (Shape Int)
matMul a b = case (a, b) of
  (Mat m n, Vec n') | n == n' -> Just (Vec m)
  (Mat m n, Mat n' k) | n == n' -> Just (Mat m k)
  _ -> Nothing

-- | Why an index expression is wrong.
data IndexProblem
  = -- | Which index (0 = first, 1 = second), its value, and the size it
    -- must stay below (E0305).
    OutOfBounds Int Int Int
  | -- | More indices than dimensions: any index on a 'Real', two on a
    -- vector (E0306).
    TooManyIndices
  deriving (Eq, Show)

-- | The shape of @e[i]@ or @e[i, j]@ (§5.3): @Vec n@ gives a 'Real', a
-- matrix with one index gives a row (@Vec c@), with two a 'Real'. Indices
-- are 0-based and checked against the sizes.
indexShape :: Shape Int -> Int -> Maybe Int -> Either IndexProblem (Shape Int)
indexShape s i j = case (s, j) of
  (Vec n, Nothing) -> Real <$ within 0 i n
  (Mat r c, Nothing) -> Vec c <$ within 0 i r
  (Mat r c, Just j') -> Real <$ within 0 i r <* within 1 j' c
  _ -> Left TooManyIndices
  where
    within which k size
      | 0 <= k && k < size = Right ()
      | otherwise = Left (OutOfBounds which k size)

-- ---------------------------------------------------------------------------
-- Built-in functions

-- | The built-in functions of @01-language.md@ §6. (The constructors have
-- a @B@ in front so that they don't clash with Core's operations, which
-- the compiler imports too.)
data Builtin
  = BExp
  | BLog
  | BTanh
  | BRelu
  | BStep
  | BSigmoid
  | BSqrt
  | BAbs
  | BMax
  | BMin
  | BSum
  | BMean
  | BDot
  | BTranspose
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | All of them, in the order of the table in §6.
builtins :: [Builtin]
builtins = [minBound .. maxBound]

-- | The name a program uses.
builtinName :: Builtin -> Text
builtinName = \case
  BExp -> "exp"
  BLog -> "log"
  BTanh -> "tanh"
  BRelu -> "relu"
  BStep -> "step"
  BSigmoid -> "sigmoid"
  BSqrt -> "sqrt"
  BAbs -> "abs"
  BMax -> "max"
  BMin -> "min"
  BSum -> "sum"
  BMean -> "mean"
  BDot -> "dot"
  BTranspose -> "transpose"

-- | The built-in function with this name.
lookupBuiltin :: Text -> Maybe Builtin
lookupBuiltin x = find ((== x) . builtinName) builtins

-- | The four ways a built-in function is typed.
data BuiltinType
  = -- | @s → s@ for any shape @s@: applied to every element.
    Elementwise
  | -- | Two arguments combined elementwise, with 'broadcast'.
    Broadcasting
  | -- | @sum@ and @mean@: @Real → Real@, @Vec n → Real@,
    -- @Mat r c → Real@, @Seq t → t@. (The sum of one number is that
    -- number, so that @sum [ sum(p ^ 2) | p <- params ]@ works whatever
    -- the shapes of the parameters.)
    Reduction
  | -- | An ordinary signature, checked by matching like a user function:
    -- parameter names and templates, and the result template.
    Matching [(Text, Template)] Template
  deriving (Eq, Show)

-- | How each built-in function is typed.
builtinType :: Builtin -> BuiltinType
builtinType = \case
  BMax -> Broadcasting
  BMin -> Broadcasting
  BSum -> Reduction
  BMean -> Reduction
  BDot -> Matching [("u", Vec n), ("v", Vec n)] Real
  BTranspose -> Matching [("M", Mat r c)] (Mat c r)
  _ -> Elementwise
  where
    n = Variable "n"
    r = Variable "r"
    c = Variable "c"

-- | How many arguments it takes.
builtinArity :: Builtin -> Int
builtinArity b = case builtinType b of
  Elementwise -> 1
  Broadcasting -> 2
  Reduction -> 1
  Matching ps _ -> length ps

-- | The signature, for messages: @dot(u : Vec n, v : Vec n) : Real@,
-- @exp(e : s) : s@ (@s@ is any shape).
builtinSignature :: Builtin -> Text
builtinSignature b = case builtinType b of
  Elementwise -> f <> "(e : s) : s"
  Broadcasting -> f <> "(a : s, b : s) : s"
  Reduction ->
    Text.intercalate ", " [f <> "(e : " <> t <> ") : " <> r | (t, r) <- [("s", "Real"), ("Seq t", "t")]]
  Matching ps r -> prettySignature (Signature f [(x, Just t) | (x, t) <- ps] (Just r))
  where
    f = builtinName b

-- | What the argument at this position (0-based) must be, for messages:
-- "`Vec n`", "a `Real`, `Vec n` or `Mat r c`".
builtinExpects :: Builtin -> Int -> Text
builtinExpects b i = case builtinType b of
  Reduction -> "a `Real`, `Vec n`, `Mat r c` or a comprehension"
  Matching ps _ | (_, t) : _ <- drop i ps -> "`" <> prettyTemplate t <> "`"
  _ -> "a `Real`, `Vec n` or `Mat r c`"

-- | Why a call of a built-in function is wrong.
data CallProblem
  = -- | Expected and given number of arguments (E0205).
    WrongArity Int Int
  | -- | An argument doesn't fit (E0304), or two arguments disagree on a
    -- shape variable (E0303).
    BadArgument MatchProblem
  | -- | @max@ / @min@ of shapes that don't broadcast (E0301).
    IncompatibleArguments (Shape Int) (Shape Int)
  | -- | A sequence at this position, which only @sum@ and @mean@ accept
    -- (E0311).
    SequenceArgument Int
  deriving (Eq, Show)

-- | The result shape of a call of a built-in function, or every problem
-- with it.
--
-- >>> applyBuiltin BDot [Value (Vec 3), Value (Vec 3)]
-- Right Real
-- >>> applyBuiltin BSum [Seq (Vec 2)]
-- Right (Vec 2)
applyBuiltin :: Builtin -> [Operand] -> Either [CallProblem] (Shape Int)
applyBuiltin b ops
  | length ops /= arity = Left [WrongArity arity (length ops)]
  | builtinType b /= Reduction, not (null sequences) = Left (map SequenceArgument sequences)
  | otherwise = case (builtinType b, ops) of
      (Elementwise, [Value s]) -> Right s
      (Broadcasting, [Value x, Value y]) ->
        maybe (Left [IncompatibleArguments x y]) Right (broadcast x y)
      (Reduction, [Seq t]) -> Right t
      (Reduction, [Value _]) -> Right Real
      (Matching ps result, _) -> case matchArguments [Just t | (_, t) <- ps] (map (Just . valueShape) ops) of
        (bs, []) -> case substitute (bindingSize <$> bs) result of
          Just s -> Right s
          -- Every variable of a built-in result appears in its parameters.
          Nothing -> error ("applyBuiltin: unbound result variable in " <> show b)
        (_, problems) -> Left (map BadArgument problems)
      -- The arity and the sequences were checked above, so the operands
      -- always have one of the forms matched so far.
      _ -> error ("applyBuiltin: unexpected operands for " <> show b)
  where
    arity = builtinArity b
    sequences = [i | (i, Seq _) <- zip [0 ..] ops]
    valueShape = \case
      Value s -> s
      Seq s -> s
