{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The surface syntax of @.hsg@ files: the abstract syntax tree (AST)
-- that the parser produces, and a printer that turns it back into text.
--
-- The tree is purely syntactic. Names are just text: whether @x@ is an
-- input, a parameter or a local is the checker's business (step 12).
--
-- Every type is parameterised by an __annotation__ @a@ that sits on every
-- node. The parser fills it with the 'Span' of the source text the node
-- came from (@Program Span@); tests compare trees without positions by
-- replacing every annotation with @()@ ('forgetSpans'). All types derive
-- 'Functor', 'Foldable' and 'Traversable' over the annotation, so
-- @fmap@ changes annotations and @toList@ lists them.
--
-- The printer ('prettyProgram', 'prettyExpr', …) produces text that the
-- parser reads back into the same tree (the round-trip property of the
-- tests). It puts in exactly the parentheses that the precedence table
-- (@01-language.md@ §5.1) needs.
-- See @docs/12-parser.md@.
module HsGrad.Lang.Syntax
  ( -- * Programs and declarations
    Program (..)
  , Decl (..)
  , ParamInit (..)
  , Field (..)
  , DataSource (..)
  , Row (..)
  , ColumnMap (..)
  , Columns (..)
  , Binder (..)

    -- * Types
  , Type (..)
  , Dim (..)

    -- * Expressions
  , Expr (..)
  , Operator (..)
  , Pattern (..)
  , Generator (..)

    -- * Small pieces
  , Ident (..)
  , Nat (..)
  , Number (..)
  , Literal (..)
  , StringLit (..)

    -- * Annotations
  , forgetSpans
  , declAnn
  , typeAnn
  , exprAnn
  , setExprAnn
  , literalAnn
  , dataSourceAnn

    -- * Reserved words
  , keywords
  , typeNames
  , isReserved
  , isIdentStart
  , isIdentChar

    -- * Printing
  , prettyProgram
  , prettyDecl
  , prettyType
  , prettyExpr
  , prettyLiteral
  , prettyOperator
  , showNumber
  ) where

import Data.Char (isAsciiLower, isAsciiUpper, isDigit)
import Data.Ratio (denominator, numerator)
import Data.Text (Text)
import Data.Text qualified as Text

-- ---------------------------------------------------------------------------
-- Small pieces

-- | A name as written: @x@, @W1@, @data_loss@, @x'@.
data Ident a = Ident
  { identAnn :: a
  , identName :: Text
  }
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A natural-number literal: an index (@v[0]@) or a dimension (@Vec 16@).
data Nat a = Nat
  { natAnn :: a
  , natValue :: Int
  }
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A numeric constant that may have a minus sign: an exponent (@x ^ -1@),
-- an argument of an initialiser (@uniform(-1, 1)@), a number in a literal
-- (@-3.0@).
--
-- The value is a 'Rational', read exactly from its decimal digits: @0.1@
-- is exactly @1/10@, not the nearest 'Double'. Exponents need that
-- exactness (@x ^ 0.5@ is the square root, and Core's @PowC@ takes a
-- 'Rational'); other numbers become 'Double's only in the compiler.
data Number a = Number
  { numberAnn :: a
  , numberValue :: Rational
  }
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A literal value, the only thing allowed as an @input@ default, a
-- constant @param@ value or a field of an inline row: a signed number, or
-- a bracketed list of literals (a vector; a list of vectors is a matrix).
data Literal a
  = LitNumber (Number a)
  | -- | At least one element.
    LitVector a [Literal a]
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A string literal, with its escapes already resolved: the text between
-- the quotes of @\"data\/moons.csv\"@.
data StringLit a = StringLit
  { stringAnn :: a
  , stringValue :: Text
  }
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- ---------------------------------------------------------------------------
-- Types

-- | A type annotation: @Real@, @Vec n@, @Mat 16 2@.
data Type a
  = TReal a
  | TVec a (Dim a)
  | -- | Rows, then columns.
    TMat a (Dim a) (Dim a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A dimension in a type: a number, or a shape variable such as @n@.
data Dim a
  = DimNat (Nat a)
  | DimVar (Ident a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- ---------------------------------------------------------------------------
-- Expressions

-- | The binary operators, from @\@@ (binding tightest) to @+@ and @-@.
data Operator
  = -- | @\@@: matrix product.
    MatMul
  | -- | @*@: elementwise product.
    Times
  | -- | @/@
    Divide
  | -- | @+@
    Plus
  | -- | @-@ (binary; unary minus is 'Negate').
    Minus
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | An expression (@01-language.md@ §5). The first field of every
-- constructor is the annotation of the whole expression.
--
-- Parentheses are not nodes: @(a + b)@ is the 'Binary' node of @a + b@,
-- whose span covers the parentheses too. A matrix literal is a 'Vector' of
-- 'Vector's; telling the two apart is the checker's job.
data Expr a
  = -- | A number: @2@, @1e-4@. Never negative in a parsed tree: @-2@ is
    -- @'Negate' (Num 2)@, since @-@ is an operator (§2).
    Num a Rational
  | -- | A name: @x@.
    Var a (Ident a)
  | -- | A call: @f(e1, …, en)@, possibly without arguments. A call written
    -- with a comprehension and no parentheses, @mean [ e | x <- d ]@, is
    -- the same call with one argument.
    Call a (Ident a) [Expr a]
  | -- | Indexing: @e[i]@ or @e[i, j]@.
    Index a (Expr a) (Nat a) (Maybe (Nat a))
  | -- | A vector literal @[e1, …, en]@, at least one element.
    Vector a [Expr a]
  | -- | A comprehension: @[ body | pattern <- source ]@.
    Comprehension a (Expr a) (Pattern a) (Generator a)
  | -- | A power with a constant exponent: @e ^ 2@, @e ^ -0.5@.
    Power a (Expr a) (Number a)
  | -- | Unary minus: @-e@.
    Negate a (Expr a)
  | -- | A binary operation: @a + b@, @W \@ x@.
    Binary a Operator (Expr a) (Expr a)
  | -- | @let x : t = bound in body@ (the annotation is optional).
    Let a (Ident a) (Maybe (Type a)) (Expr a) (Expr a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | What a comprehension binds: one name, or a tuple of names that matches
-- the fields of a dataset (@(x, y)@, at least one name).
data Pattern a
  = PVar (Ident a)
  | PTuple a [Ident a]
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | Where a comprehension takes its elements from.
data Generator a
  = -- | A dataset, by name: @(x, y) <- moons@.
    FromData (Ident a)
  | -- | The keyword @params@: every parameter, in declaration order.
    FromParams a
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- ---------------------------------------------------------------------------
-- Declarations

-- | A whole file: its declarations, in order.
newtype Program a = Program {programDecls :: [Decl a]}
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A top-level declaration (@01-language.md@ §3). The first field is the
-- annotation of the whole declaration, from its keyword to its last token.
data Decl a
  = -- | @input x : t@, with an optional default @= literal@.
    InputDecl a (Ident a) (Type a) (Maybe (Literal a))
  | -- | @param w : t = literal@ or @param w : t ~ uniform(lo, hi)@.
    ParamDecl a (Ident a) (Type a) (ParamInit a)
  | -- | @data d : { field, … } = source@, at least one field.
    DataDecl a (Ident a) [Field a] (DataSource a)
  | -- | @fn f(binders) : t = body@, the result type optional.
    FnDecl a (Ident a) [Binder a] (Maybe (Type a)) (Expr a)
  | -- | @model m(binders) : t = body@, the result type required.
    ModelDecl a (Ident a) [Binder a] (Type a) (Expr a)
  | -- | @let x : t = e@, the annotation optional. No @in@.
    LetDecl a (Ident a) (Maybe (Type a)) (Expr a)
  | -- | @loss = e@. The second field annotates the keyword @loss@, the
    -- only declaration without a name to point at.
    LossDecl a a (Expr a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | How a parameter gets its initial value.
data ParamInit a
  = -- | @= literal@: a constant.
    InitConst (Literal a)
  | -- | @~ uniform(lo, hi)@ or @~ normal(mean, std)@. The annotation covers
    -- @uniform(lo, hi)@. The parser accepts any name; the checker knows
    -- which initialisers exist.
    InitRandom a (Ident a) (Number a) (Number a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A field of a dataset schema: @x : Vec 2@.
data Field a = Field (Ident a) (Type a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | Where a dataset's rows come from.
data DataSource a
  = -- | @rows [ row, … ]@, at least one row. The annotation covers the
    -- whole source, from @rows@ to @]@.
    Rows a [Row a]
  | -- | @csv "path" { mapping, … }@, at least one mapping.
    Csv a (StringLit a) [ColumnMap a]
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | An inline row: one literal per field. @([0, 1], -1)@, or for a
-- one-field schema the bare literal @-1@; both are a 'Row', so the tree
-- does not remember which way it was written.
data Row a = Row a [Literal a]
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | Which CSV columns feed a field: @x = [x1, x2]@, @y = label@.
data ColumnMap a = ColumnMap (Ident a) (Columns a)
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | One column (for a @Real@ field) or a bracketed list (for @Vec n@).
data Columns a
  = Column (Ident a)
  | -- | At least one column.
    Columns a [Ident a]
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | A parameter of a @fn@ or @model@, with an optional type: @x : Vec n@.
data Binder a = Binder (Ident a) (Maybe (Type a))
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- ---------------------------------------------------------------------------
-- Annotations

-- | Replace every annotation with @()@: the tree without its positions.
-- Two parses of differently laid out text give the same tree once their
-- spans are forgotten.
forgetSpans :: (Functor f) => f a -> f ()
forgetSpans = (() <$)

-- | The annotation of a whole declaration.
declAnn :: Decl a -> a
declAnn = \case
  InputDecl a _ _ _ -> a
  ParamDecl a _ _ _ -> a
  DataDecl a _ _ _ -> a
  FnDecl a _ _ _ _ -> a
  ModelDecl a _ _ _ _ -> a
  LetDecl a _ _ _ -> a
  LossDecl a _ _ -> a

-- | The annotation of a whole type.
typeAnn :: Type a -> a
typeAnn = \case
  TReal a -> a
  TVec a _ -> a
  TMat a _ _ -> a

-- | The annotation of a whole expression.
exprAnn :: Expr a -> a
exprAnn = \case
  Num a _ -> a
  Var a _ -> a
  Call a _ _ -> a
  Index a _ _ _ -> a
  Vector a _ -> a
  Comprehension a _ _ _ -> a
  Power a _ _ -> a
  Negate a _ -> a
  Binary a _ _ _ -> a
  Let a _ _ _ _ -> a

-- | Replace the annotation of the outermost node only. The parser uses it
-- to widen an expression's span over the parentheses around it.
setExprAnn :: a -> Expr a -> Expr a
setExprAnn a = \case
  Num _ r -> Num a r
  Var _ x -> Var a x
  Call _ f es -> Call a f es
  Index _ e i j -> Index a e i j
  Vector _ es -> Vector a es
  Comprehension _ e p g -> Comprehension a e p g
  Power _ e k -> Power a e k
  Negate _ e -> Negate a e
  Binary _ op l r -> Binary a op l r
  Let _ x t b body -> Let a x t b body

-- | The annotation of a whole literal.
literalAnn :: Literal a -> a
literalAnn = \case
  LitNumber n -> numberAnn n
  LitVector a _ -> a

-- | The annotation of a whole data source.
dataSourceAnn :: DataSource a -> a
dataSourceAnn = \case
  Rows a _ -> a
  Csv a _ _ -> a

-- ---------------------------------------------------------------------------
-- Reserved words

-- | The keywords of @01-language.md@ §2. None of them can be a name.
keywords :: [Text]
keywords = ["input", "param", "data", "fn", "let", "in", "model", "loss", "rows", "csv", "params"]

-- | The type names. They are reserved too, so @Vec@ can never be a shape
-- variable or a parameter.
typeNames :: [Text]
typeNames = ["Real", "Vec", "Mat"]

-- | Is the word a keyword or a type name?
isReserved :: Text -> Bool
isReserved w = w `elem` keywords || w `elem` typeNames

-- | Can a name start with this character? @[a-zA-Z_]@.
isIdentStart :: Char -> Bool
isIdentStart c = isAsciiLower c || isAsciiUpper c || c == '_'

-- | Can a name continue with this character? @[a-zA-Z0-9_']@.
isIdentChar :: Char -> Bool
isIdentChar c = isIdentStart c || isDigit c || c == '\''

-- ---------------------------------------------------------------------------
-- Printing

-- | A whole program, one declaration per line.
prettyProgram :: Program a -> Text
prettyProgram (Program ds) = Text.concat [prettyDecl d <> "\n" | d <- ds]

-- | One declaration, on one line.
prettyDecl :: Decl a -> Text
prettyDecl = \case
  InputDecl _ x t def ->
    "input " <> name x <> " : " <> prettyType t <> maybe "" ((" = " <>) . prettyLiteral) def
  ParamDecl _ x t ini -> "param " <> name x <> " : " <> prettyType t <> case ini of
    InitConst l -> " = " <> prettyLiteral l
    InitRandom _ f lo hi -> " ~ " <> name f <> "(" <> signed lo <> ", " <> signed hi <> ")"
  DataDecl _ x fields src ->
    "data " <> name x <> " : { " <> commas [name f <> " : " <> prettyType t | Field f t <- fields] <> " } = " <> source src
  FnDecl _ f bs t body -> "fn " <> signature f bs <> maybe "" ((" : " <>) . prettyType) t <> " = " <> prettyExpr body
  ModelDecl _ f bs t body -> "model " <> signature f bs <> " : " <> prettyType t <> " = " <> prettyExpr body
  LetDecl _ x t e -> "let " <> name x <> annotation t <> " = " <> prettyExpr e
  LossDecl _ _ e -> "loss = " <> prettyExpr e
  where
    signature f bs = name f <> "(" <> commas [name b <> annotation t | Binder b t <- bs] <> ")"
    source = \case
      Rows _ rs -> "rows [" <> commas ["(" <> commas (map prettyLiteral ls) <> ")" | Row _ ls <- rs] <> "]"
      Csv _ path ms -> "csv " <> quoted (stringValue path) <> " { " <> commas (map mapping ms) <> " }"
    mapping (ColumnMap f cols) = name f <> " = " <> case cols of
      Column c -> name c
      Columns _ cs -> "[" <> commas (map name cs) <> "]"
    quoted s = "\"" <> Text.concatMap escape s <> "\""
    escape = \case
      '"' -> "\\\""
      '\\' -> "\\\\"
      c -> Text.singleton c

-- | A type: @Real@, @Vec n@, @Mat 16 2@.
prettyType :: Type a -> Text
prettyType = \case
  TReal _ -> "Real"
  TVec _ d -> "Vec " <> dim d
  TMat _ r c -> "Mat " <> dim r <> " " <> dim c
  where
    dim = \case
      DimNat n -> nat n
      DimVar v -> name v

-- | A literal: @-2.5@, @[1, [2, 3]]@.
prettyLiteral :: Literal a -> Text
prettyLiteral = \case
  LitNumber n -> signed n
  LitVector _ ls -> "[" <> commas (map prettyLiteral ls) <> "]"

-- | How an operator is written.
prettyOperator :: Operator -> Text
prettyOperator = \case
  MatMul -> "@"
  Times -> "*"
  Divide -> "/"
  Plus -> "+"
  Minus -> "-"

-- | The precedence level of an expression's outermost form, as numbered in
-- @01-language.md@ §5.1: 1 binds tightest (atoms, calls, indexing), 7
-- loosest (@let … in@).
level :: Expr a -> Int
level = \case
  Num _ r | r < 0 -> 3 -- printed with a minus sign, like a negation
  Power {} -> 2
  Negate {} -> 3
  Binary _ op _ _ -> operatorLevel op
  Let {} -> 7
  _ -> 1

operatorLevel :: Operator -> Int
operatorLevel = \case
  MatMul -> 4
  Times -> 5
  Divide -> 5
  Plus -> 6
  Minus -> 6

-- | An expression, on one line, with the parentheses that precedence
-- needs and no others.
--
-- An operand is put in parentheses when its own form binds less tightly
-- than its position allows. The binary operators associate to the left,
-- so the left operand of @-@ may be another @-@ (@a - b - c@) but the
-- right operand may not (@a - (b - c)@).
prettyExpr :: Expr a -> Text
prettyExpr = go
  where
    go = \case
      Num _ r -> showNumber r
      Var _ x -> name x
      Call _ f [arg@Comprehension {}] -> name f <> " " <> go arg
      Call _ f args -> name f <> "(" <> commas (map go args) <> ")"
      Index _ e i j -> atMost 1 e <> "[" <> nat i <> maybe "" ((", " <>) . nat) j <> "]"
      Vector _ es -> "[" <> commas (map go es) <> "]"
      Comprehension _ body p g -> "[ " <> go body <> " | " <> pattern p <> " <- " <> generator g <> " ]"
      Power _ e k -> atMost 1 e <> " ^ " <> signed k
      Negate _ e ->
        -- "--" would start a comment: a negated negation needs parentheses.
        let operand = atMost 3 e
         in "-" <> if "-" `Text.isPrefixOf` operand then parens operand else operand
      Binary _ op l r ->
        let p = operatorLevel op
         in atMost p l <> " " <> prettyOperator op <> " " <> atMost (p - 1) r
      Let _ x t bound body -> "let " <> name x <> annotation t <> " = " <> go bound <> " in " <> go body

    -- The expression, in parentheses unless its level is at most n.
    atMost n e
      | level e <= n = go e
      | otherwise = parens (go e)

    pattern = \case
      PVar x -> name x
      PTuple _ xs -> "(" <> commas (map name xs) <> ")"
    generator = \case
      FromData d -> name d
      FromParams _ -> "params"

-- | A non-negative 'Rational' as a decimal number that reads back as
-- exactly the same value: @2@, @0.5@, @0.0001@, @1.5e-7@, @1e20@.
-- Negative numbers get a minus sign in front.
--
-- Every number the parser reads is a decimal fraction (its denominator
-- divides a power of 10), so it has an exact decimal form. A rational
-- that has none, such as @1/3@, can only come from a hand-built tree; it
-- is printed rounded to a 'Double'.
showNumber :: Rational -> Text
showNumber r
  | r < 0 = "-" <> showNumber (negate r)
  | r == 0 = "0"
  | otherwise = case decimal r of
      Nothing -> Text.pack (show (fromRational r :: Double))
      Just (digits, k) -> layout (show digits) k
  where
    -- r = c · 10^k with c not divisible by 10.
    decimal x
      | twosAndFives (denominator x) /= 1 = Nothing
      | otherwise = Just (strip (scaleUp x 0))
    scaleUp x k
      | denominator x == 1 = (numerator x, k)
      | otherwise = scaleUp (x * 10) (k - 1)
    strip (c, k)
      | c `mod` 10 == 0 = strip (c `div` 10, k + 1)
      | otherwise = (c, k)
    twosAndFives d
      | even d = twosAndFives (d `div` 2)
      | d `mod` 5 == 0 = twosAndFives (d `div` 5)
      | otherwise = d

    -- Plain decimal notation for moderate sizes, scientific otherwise.
    -- e is the power of 10 of the first digit.
    layout ds k
      | e < -5 || e > 15 = Text.pack (scientific ds) <> "e" <> Text.pack (show e)
      | k >= 0 = Text.pack (ds <> replicate k '0')
      | point > 0 = Text.pack (take point ds <> "." <> drop point ds)
      | otherwise = Text.pack ("0." <> replicate (negate point) '0' <> ds)
      where
        n = length ds
        e = n - 1 + k
        point = n + k
    scientific = \case
      d : rest@(_ : _) -> d : '.' : rest
      ds -> ds

-- Shared pieces of the printers.

name :: Ident a -> Text
name = identName

nat :: Nat a -> Text
nat = Text.pack . show . natValue

signed :: Number a -> Text
signed = showNumber . numberValue

annotation :: Maybe (Type a) -> Text
annotation = maybe "" ((" : " <>) . prettyType)

commas :: [Text] -> Text
commas = Text.intercalate ", "

parens :: Text -> Text
parens t = "(" <> t <> ")"
