{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Random syntax trees for property tests of the parser and printer.
--
-- The generators only make trees the parser can produce: numbers in an
-- 'Expr' are never negative (@-2@ is a negation), names are never
-- reserved words, lists that the grammar requires to be non-empty are
-- non-empty, and every number is a decimal fraction in the range of a
-- 'Double'. Annotations are @()@.
module HsGrad.Test.Syntax
  ( -- * Generators
    genProgram
  , genDecl
  , genExpr
  , genType
  , genLiteral
  , genName
  , genMagnitude

    -- * Shrinking and traversal
  , shrinkExpr
  , subExpressions
  , universe

    -- * Source slices
  , sliceSpan
  ) where

import Data.Ratio ((%))
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..))
import HsGrad.Lang.Syntax
import Test.QuickCheck

-- ---------------------------------------------------------------------------
-- Pieces

-- | A name: mostly random, sometimes one that starts like a keyword
-- (@inputs@, @letter@), to test that keywords are whole words.
genName :: Gen Text
genName =
  frequency
    [ (5, random `suchThat` (not . isReserved))
    , (1, elements ["inputs", "letter", "in_", "params2", "Vec3", "Real'", "x'", "_", "data_loss", "e", "E", "fn1"])
    ]
  where
    random = do
      c <- elements (['a' .. 'z'] <> ['A' .. 'Z'] <> "_")
      n <- choose (0, 5)
      cs <- vectorOf n (elements (['a' .. 'z'] <> ['A' .. 'Z'] <> ['0' .. '9'] <> "_'"))
      pure (Text.pack (c : cs))

genIdent :: Gen (Ident ())
genIdent = Ident () <$> genName

-- | A non-negative decimal number: small integers, decimals with up to
-- six digits after the point, and numbers that need an exponent.
genMagnitude :: Gen Rational
genMagnitude =
  oneof
    [ fromInteger <$> choose (0, 1000)
    , (\m k -> m % (10 ^ k)) <$> choose (0, 10 ^ (6 :: Int)) <*> choose (0, 6 :: Int)
    , (\m e -> (m % 1) * 10 ^^ e) <$> choose (1, 999) <*> choose (-30, 30 :: Int)
    ]

genSigned :: Gen (Number ())
genSigned = do
  m <- genMagnitude
  negative <- arbitrary
  pure (Number () (if negative then negate m else m))

genNat :: Gen (Nat ())
genNat = Nat () <$> choose (0, 20)

nonEmptyOf :: Int -> Gen a -> Gen [a]
nonEmptyOf most g = do
  n <- choose (1, most)
  vectorOf n g

genType :: Gen (Type ())
genType = oneof [pure (TReal ()), TVec () <$> genDim, TMat () <$> genDim <*> genDim]
  where
    genDim = oneof [DimNat <$> genNat, DimVar <$> genIdent]

-- | A literal, nested at most a few levels.
genLiteral :: Gen (Literal ())
genLiteral = sized (go . min 3)
  where
    go 0 = LitNumber <$> genSigned
    go n = frequency [(2, LitNumber <$> genSigned), (1, LitVector () <$> nonEmptyOf 3 (go (n - 1)))]

-- ---------------------------------------------------------------------------
-- Expressions

-- | An expression of every form, mixed freely, so that the printer has to
-- get the parentheses right.
genExpr :: Gen (Expr ())
genExpr = sized (go . min 12)
  where
    go 0 = leaf
    go n =
      frequency
        [ (3, leaf)
        , (2, Call () <$> genIdent <*> (choose (0, 3) >>= \k -> vectorOf k sub))
        , (1, Index () <$> sub <*> genNat <*> oneof [pure Nothing, Just <$> genNat])
        , (1, Vector () <$> nonEmptyOf 3 sub)
        , (1, Comprehension () <$> sub <*> genPattern <*> genGenerator)
        , (1, (\f c -> Call () f [c]) <$> genIdent <*> (Comprehension () <$> sub <*> genPattern <*> genGenerator))
        , (2, Power () <$> sub <*> genSigned)
        , (2, Negate () <$> sub)
        , (5, Binary () <$> elements [minBound .. maxBound] <*> sub <*> sub)
        , (1, Let () <$> genIdent <*> oneof [pure Nothing, Just <$> genType] <*> sub <*> sub)
        ]
      where
        sub = go (n `div` 2)
    leaf = oneof [Num () <$> genMagnitude, Var () <$> genIdent]
    genPattern = oneof [PVar <$> genIdent, PTuple () <$> nonEmptyOf 3 genIdent]
    genGenerator = oneof [FromData <$> genIdent, pure (FromParams ())]

-- | The immediate subexpressions.
subExpressions :: Expr a -> [Expr a]
subExpressions = \case
  Num {} -> []
  Var {} -> []
  Call _ _ es -> es
  Index _ e _ _ -> [e]
  Vector _ es -> es
  Comprehension _ body _ _ -> [body]
  Power _ e _ -> [e]
  Negate _ e -> [e]
  Binary _ _ l r -> [l, r]
  Let _ _ _ bound body -> [bound, body]

-- | The expression and all its subexpressions, at any depth.
universe :: Expr a -> [Expr a]
universe e = e : concatMap universe (subExpressions e)

-- | Shrink to a subexpression.
shrinkExpr :: Expr a -> [Expr a]
shrinkExpr = subExpressions

-- ---------------------------------------------------------------------------
-- Declarations

genDecl :: Gen (Decl ())
genDecl =
  oneof
    [ InputDecl () <$> genIdent <*> genType <*> oneof [pure Nothing, Just <$> genLiteral]
    , ParamDecl () <$> genIdent <*> genType <*> genInit
    , DataDecl () <$> genIdent <*> nonEmptyOf 3 (Field <$> genIdent <*> genType) <*> genSource
    , FnDecl () <$> genIdent <*> genBinders <*> oneof [pure Nothing, Just <$> genType] <*> genExpr
    , ModelDecl () <$> genIdent <*> genBinders <*> genType <*> genExpr
    , LetDecl () <$> genIdent <*> oneof [pure Nothing, Just <$> genType] <*> genExpr
    , LossDecl () () <$> genExpr
    ]
  where
    genInit =
      oneof
        [ InitConst <$> genLiteral
        , InitRandom () <$> genIdent <*> genSigned <*> genSigned
        ]
    genSource =
      oneof
        [ Rows () <$> nonEmptyOf 3 (Row () <$> nonEmptyOf 3 genLiteral)
        , Csv () <$> (StringLit () <$> genString) <*> nonEmptyOf 3 (ColumnMap <$> genIdent <*> genColumns)
        ]
    genColumns = oneof [Column <$> genIdent, Columns () <$> nonEmptyOf 3 genIdent]
    genBinders = choose (0, 3) >>= \k -> vectorOf k (Binder <$> genIdent <*> oneof [pure Nothing, Just <$> genType])
    -- Any characters but a newline, quotes and backslashes included.
    genString = Text.pack <$> listOf (elements ("ab/._ -\"\\ø∂" :: String))

-- | A program of a few declarations.
genProgram :: Gen (Program ())
genProgram = Program <$> resize 8 (listOf (resize 8 genDecl))

-- ---------------------------------------------------------------------------
-- Source slices

-- | The text a span covers.
sliceSpan :: Text -> Span -> Text
sliceSpan src (SrcSpan _ (Pos l1 c1) (Pos l2 c2)) =
  Text.intercalate "\n" (zipWith cut [l1 ..] covered)
  where
    covered = take (l2 - l1 + 1) (drop (l1 - 1) (Text.splitOn "\n" src))
    cut l line
      | l == l1 && l == l2 = Text.take (c2 - c1) (Text.drop (c1 - 1) line)
      | l == l1 = Text.drop (c1 - 1) line
      | l == l2 = Text.take (c2 - 1) line
      | otherwise = line
