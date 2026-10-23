{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Syntax": the printer and the helpers on
-- annotations. The round trip through the parser is in
-- "HsGrad.Lang.ParserSpec".
module HsGrad.Lang.SyntaxSpec
  ( spec
  ) where

import Data.Foldable (toList)
import Data.Ratio ((%))
import Data.Text (Text)
import HsGrad.Lang.Parser (parseExpr)
import HsGrad.Lang.Syntax
import HsGrad.Test.Syntax (genMagnitude)
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (forAll, (===))

spec :: Spec
spec = do
  describe "showNumber" numberSpec
  describe "printing" printSpec
  describe "annotations" annotationSpec

name :: Text -> Ident ()
name = Ident ()

var :: Text -> Expr ()
var = Var () . name

numberSpec :: Spec
numberSpec = do
  it "prints integers, decimals and small or large numbers" $ do
    map showNumber [0, 2, 300, 1 % 2, 25 % 10, 1 % 10000, 15 % 100000000, 10 ^ (20 :: Int), 123456 % 1000, -3]
      `shouldBe` ["0", "2", "300", "0.5", "2.5", "0.0001", "1.5e-7", "1e20", "123.456", "-3"]

  it "rounds a rational without a decimal form" $
    showNumber (1 % 3) `shouldBe` "0.3333333333333333"

  prop "prints numbers that read back as the same value" $
    forAll genMagnitude $ \r ->
      (forgetSpans <$> parseExpr "n" (showNumber r)) === Right (Num () r)

printSpec :: Spec
printSpec = do
  it "puts the right operand of a left-associative operator in parentheses" $ do
    prettyExpr (Binary () Minus (var "a") (Binary () Minus (var "b") (var "c"))) `shouldBe` "a - (b - c)"
    prettyExpr (Binary () Minus (Binary () Minus (var "a") (var "b")) (var "c")) `shouldBe` "a - b - c"

  it "puts lower-precedence operands in parentheses" $ do
    prettyExpr (Binary () Times (Binary () Plus (var "a") (var "b")) (var "c")) `shouldBe` "(a + b) * c"
    prettyExpr (Binary () Times (Num () 2) (Binary () MatMul (var "W") (var "x"))) `shouldBe` "2 * W @ x"

  it "keeps -x ^ 2 and (-x) ^ 2 apart" $ do
    prettyExpr (Negate () (Power () (var "x") (Number () 2))) `shouldBe` "-x ^ 2"
    prettyExpr (Power () (Negate () (var "x")) (Number () 2)) `shouldBe` "(-x) ^ 2"

  it "never prints `--`, which would start a comment" $ do
    prettyExpr (Negate () (Negate () (var "x"))) `shouldBe` "-(-x)"
    prettyExpr (Negate () (Num () (-2))) `shouldBe` "-(-2)"
    prettyExpr (Binary () Minus (var "a") (Negate () (var "b"))) `shouldBe` "a - -b"

  it "puts let … in operands in parentheses" $
    prettyExpr (Binary () Plus (Let () (name "x") Nothing (Num () 1) (var "x")) (var "y"))
      `shouldBe` "(let x = 1 in x) + y"

  it "prints a call with one comprehension argument without parentheses" $
    prettyExpr (Call () (name "mean") [Comprehension () (var "x") (PVar (name "x")) (FromData (name "d"))])
      `shouldBe` "mean [ x | x <- d ]"

  it "prints declarations on one line each" $
    prettyProgram
      ( Program
          [ ParamDecl () (name "W") (TMat () (DimNat (Nat () 4)) (DimVar (name "n"))) (InitRandom () (name "uniform") (Number () (-1)) (Number () 1))
          , InputDecl () (name "x") (TVec () (DimNat (Nat () 2))) (Just (LitVector () [LitNumber (Number () 1), LitNumber (Number () (-2))]))
          , DataDecl () (name "d") [Field (name "y") (TReal ())] (Csv () (StringLit () "a\"b.csv") [ColumnMap (name "y") (Column (name "label"))])
          , LossDecl () () (Num () 0)
          ]
      )
      `shouldBe` "param W : Mat 4 n ~ uniform(-1, 1)\ninput x : Vec 2 = [1, -2]\ndata d : { y : Real } = csv \"a\\\"b.csv\" { y = label }\nloss = 0\n"

annotationSpec :: Spec
annotationSpec = do
  it "lists every annotation in order with toList" $
    toList (Binary 'a' Plus (Var 'b' (Ident 'c' "x")) (Num 'd' 1)) `shouldBe` "abcd"

  it "replaces the outermost annotation only with setExprAnn" $
    setExprAnn 'z' (Negate 'a' (Num 'b' 1)) `shouldBe` Negate 'z' (Num 'b' 1)
