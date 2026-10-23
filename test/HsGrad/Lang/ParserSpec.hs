{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Parser".
--
-- * Examples: the four programs of @docs/01-language.md@ §1, kept in
--   @examples/@, parse; @intro.hsg@ to exactly the expected tree.
-- * Precedence and associativity (§5.1), the extent of @let … in@,
--   indexing versus @sum [ … ]@.
-- * Tokens: keywords as whole words, comments, numbers read exactly,
--   strings and their escapes.
-- * Spans: of tokens, operators, parentheses; tabs are one column.
-- * Round trip: random trees, printed and parsed back, are the same
--   trees; the text under every subexpression's span parses to that
--   subexpression.
-- * Errors: golden files in @test/golden/parse/@ (every E01xx code and
--   recovery from several errors), and the hand-built diagnostics of
--   step 10 reproduced by the parser.
module HsGrad.Lang.ParserSpec
  ( spec
  ) where

import Data.Foldable (toList)
import Data.List (sort)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict qualified as Map
import Data.Ratio ((%))
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Parser
import HsGrad.Lang.Span
import HsGrad.Lang.Syntax
import HsGrad.Test.Golden (goldenFiles)
import HsGrad.Test.Syntax
import System.Directory (listDirectory)
import System.FilePath (takeExtension, (</>))
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (modifyMaxSuccess, prop)
import Test.QuickCheck (conjoin, counterexample, forAllShrink, shrinkList, (===))

spec :: Spec
spec = do
  describe "examples" exampleSpec
  describe "precedence" precedenceSpec
  describe "tokens" tokenSpec
  describe "spans" spanSpec
  describe "round trip" roundTripSpec
  describe "errors" errorSpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Parse an expression and forget its spans.
expr :: Text -> Either [Code] (Expr ())
expr src = either (Left . map diagCode . toList) (Right . forgetSpans) (parseExpr "t.hsg" src)

-- | Parse a program and forget its spans.
program :: Text -> Either [Code] (Program ())
program src = either (Left . map diagCode . toList) (Right . forgetSpans) (parseProgram "t.hsg" src)

-- | Parse an expression that must parse, keeping its spans.
spannedExpr :: Text -> Expr Span
spannedExpr src = either (error . show) id (parseExpr "t.hsg" src)

-- | The codes of the diagnostics of a program that must fail.
errorCodes :: Text -> [Code]
errorCodes src = either (map diagCode . toList) (const []) (parseProgram "t.hsg" src)

name :: Text -> Ident ()
name = Ident ()

var :: Text -> Expr ()
var = Var () . name

num :: Rational -> Expr ()
num = Num ()

bin :: Operator -> Expr () -> Expr () -> Expr ()
bin = Binary ()

neg :: Expr () -> Expr ()
neg = Negate ()

pow :: Expr () -> Rational -> Expr ()
pow e k = Power () e (Number () k)

call :: Text -> [Expr ()] -> Expr ()
call f = Call () (name f)

nat :: Int -> Nat ()
nat = Nat ()

-- | The span from line:column to line:column in t.hsg.
at :: (Int, Int) -> (Int, Int) -> Span
at (l1, c1) (l2, c2) = SrcSpan "t.hsg" (Pos l1 c1) (Pos l2 c2)

-- | Render a file's diagnostics the way the driver will.
rendered :: FilePath -> Text -> Text
rendered file src = case parseProgram file src of
  Left ds -> renderDiagnostics defaultRenderOptions (Map.singleton file src) (toList ds)
  Right _ -> "parsed without errors\n"

-- ---------------------------------------------------------------------------
-- Examples

exampleSpec :: Spec
exampleSpec = do
  it "parses all four example programs of the language spec" $ do
    files <- filter ((== ".hsg") . takeExtension) <$> listDirectory "examples"
    files `shouldSatisfy` (\fs -> all (`elem` fs) ["intro.hsg", "neuron.hsg", "xor.hsg", "moons.hsg"])
    mapM_ parsesWithoutErrors files

  it "parses intro.hsg to the expected tree" $ do
    src <- Utf8.readFile "examples/intro.hsg"
    let input x v = InputDecl () (name x) (TReal ()) (Just (LitNumber (Number () v)))
        letD x e = LetDecl () (name x) Nothing e
    program src
      `shouldBe` Right
        ( Program
            [ input "a" 2
            , input "b" (-3)
            , input "c" 10
            , letD "e" (bin Times (var "a") (var "b"))
            , letD "d" (bin Plus (var "e") (var "c"))
            , letD "L" (bin Times (var "d") (neg (num 2)))
            ]
        )

  it "parses the declarations of moons.hsg, the largest example" $ do
    src <- Utf8.readFile "examples/moons.hsg"
    let kinds = map kind . programDecls <$> program src
        kind = \case
          InputDecl {} -> "input"
          ParamDecl {} -> "param"
          DataDecl {} -> "data"
          FnDecl {} -> "fn"
          ModelDecl {} -> "model"
          LetDecl {} -> "let"
          LossDecl {} -> "loss" :: Text
    kinds `shouldBe` Right ["data", "param", "param", "param", "param", "param", "param", "fn", "model", "loss"]

  it "parses the smaller snippets of the spec" $ do
    let parses src = program src `shouldSatisfy` either (const False) (const True)
    parses "input x : Vec 3"
    parses "param W : Mat 16 2 ~ uniform(-1, 1)\nparam b : Vec 16 = 0"
    parses "fn sq(z) = z ^ 2"
    parses "let h : Vec 4 = tanh(W1 @ x + b1)"
    parses "fn f() = 1"
    parses "data d : { y : Real } = rows [ -1, 1, 0.5 ]"
    parses "input M : Mat 2 2 = [[1, 2], [3, 4]]"
  where
    parsesWithoutErrors file = do
      src <- Utf8.readFile ("examples" </> file)
      rendered file src `shouldBe` "parsed without errors\n"

-- ---------------------------------------------------------------------------
-- Precedence

precedenceSpec :: Spec
precedenceSpec = do
  it "binds ^ tighter than unary minus: -x ^ 2 is -(x²)" $
    expr "-x ^ 2" `shouldBe` Right (neg (pow (var "x") 2))

  it "binds @ tighter than *: 2 * W @ x is 2 * (W @ x)" $
    expr "2 * W @ x" `shouldBe` Right (bin Times (num 2) (bin MatMul (var "W") (var "x")))

  it "allows unary minus after an operator: d * -2.0" $
    expr "d * -2.0" `shouldBe` Right (bin Times (var "d") (neg (num 2)))

  it "binds unary minus tighter than @: -A @ x is (-A) @ x" $
    expr "-A @ x" `shouldBe` Right (bin MatMul (neg (var "A")) (var "x"))

  it "binds * tighter than + and -" $
    expr "a - b * c + d" `shouldBe` Right (bin Plus (bin Minus (var "a") (bin Times (var "b") (var "c"))) (var "d"))

  it "makes binary operators left-associative" $ do
    expr "a - b - c" `shouldBe` Right (bin Minus (bin Minus (var "a") (var "b")) (var "c"))
    expr "a / b * c" `shouldBe` Right (bin Times (bin Divide (var "a") (var "b")) (var "c"))
    expr "A @ B @ x" `shouldBe` Right (bin MatMul (bin MatMul (var "A") (var "B")) (var "x"))

  it "keeps parentheses as grouping only" $
    expr "a - (b - c)" `shouldBe` Right (bin Minus (var "a") (bin Minus (var "b") (var "c")))

  it "reads exponents as signed constants" $ do
    expr "x ^ -1" `shouldBe` Right (pow (var "x") (-1))
    expr "(x - y) ^ 2" `shouldBe` Right (pow (bin Minus (var "x") (var "y")) 2)
    expr "f(x)[0] ^ 0.5" `shouldBe` Right (pow (Index () (call "f" [var "x"]) (nat 0) Nothing) (1 % 2))

  it "makes ^ non-associative" $
    errorCodes "let y = x ^ 2 ^ 3" `shouldBe` [E0101]

  it "allows repeated unary minus" $
    expr "- -x" `shouldBe` Right (neg (neg (var "x")))

  it "extends let … in as far right as possible" $ do
    expr "let x = 1 in x + 1" `shouldBe` Right (Let () (name "x") Nothing (num 1) (bin Plus (var "x") (num 1)))
    expr "let a = 1 in let b = 2 in a * b"
      `shouldBe` Right (Let () (name "a") Nothing (num 1) (Let () (name "b") Nothing (num 2) (bin Times (var "a") (var "b"))))

  it "ends a let … in body at a comma or closing bracket" $
    expr "f(let x = 1 in x, 2)" `shouldBe` Right (call "f" [Let () (name "x") Nothing (num 1) (var "x"), num 2])

  it "does not allow let … in as an operand without parentheses" $ do
    expr "a + (let x = 1 in x)" `shouldBe` Right (bin Plus (var "a") (Let () (name "x") Nothing (num 1) (var "x")))
    expr "a + let x = 1 in x" `shouldSatisfy` either (== [E0101]) (const False)

  it "indexes calls, vectors and indexes" $ do
    expr "layer(W3, b3, h2)[0]" `shouldBe` Right (Index () (call "layer" [var "W3", var "b3", var "h2"]) (nat 0) Nothing)
    expr "M[1, 2]" `shouldBe` Right (Index () (var "M") (nat 1) (Just (nat 2)))
    expr "M[0][1]" `shouldBe` Right (Index () (Index () (var "M") (nat 0) Nothing) (nat 1) Nothing)

  it "reads `name [ … | … ]` as a call with one comprehension argument" $ do
    let comp = Comprehension () (pow (var "p") 2) (PVar (name "p")) (FromParams ())
    expr "sum [ p ^ 2 | p <- params ]" `shouldBe` Right (call "sum" [comp])
    expr "sum([ p ^ 2 | p <- params ])" `shouldBe` Right (call "sum" [comp])
    expr "mean [ 0 | (x, y) <- d ]" `shouldBe` Right (call "mean" [Comprehension () (num 0) (PTuple () [name "x", name "y"]) (FromData (name "d"))])

  it "tells vector literals from comprehensions after the first element" $ do
    expr "[a * b, c]" `shouldBe` Right (Vector () [bin Times (var "a") (var "b"), var "c"])
    expr "[[1, 2], [3, 4]]" `shouldBe` Right (Vector () [Vector () [num 1, num 2], Vector () [num 3, num 4]])
    expr "[a * b | a <- d]" `shouldBe` Right (Comprehension () (bin Times (var "a") (var "b")) (PVar (name "a")) (FromData (name "d")))

  it "calls with no arguments" $
    expr "f()" `shouldBe` Right (call "f" [])

-- ---------------------------------------------------------------------------
-- Tokens

tokenSpec :: Spec
tokenSpec = do
  it "reads keywords only as whole words" $ do
    program "let inputs = 1\nlet letter = inputs\nlet in_ = 2"
      `shouldBe` Right
        ( Program
            [ LetDecl () (name "inputs") Nothing (num 1)
            , LetDecl () (name "letter") Nothing (var "inputs")
            , LetDecl () (name "in_") Nothing (num 2)
            ]
        )
    errorCodes "let input = 1" `shouldBe` [E0101]
    errorCodes "let Vec = 1" `shouldBe` [E0101]

  it "skips line comments and nested block comments" $ do
    program "-- a\nlet {- b {- c -} d -} y = 1 -- e\n{- f -}" `shouldBe` Right (Program [LetDecl () (name "y") Nothing (num 1)])
    expr "a --b" `shouldBe` Right (var "a")
    expr "a - -b" `shouldBe` Right (bin Minus (var "a") (neg (var "b")))

  it "reads numbers exactly, never through a Double" $ do
    expr "0.1" `shouldBe` Right (num (1 % 10))
    expr "1e-4" `shouldBe` Right (num (1 % 10000))
    expr "3.0E2" `shouldBe` Right (num 300)
    expr "2.5e+1" `shouldBe` Right (num 25)
    expr "0.30000000000000004" `shouldBe` Right (num (30000000000000004 % 100000000000000000))

  it "rejects malformed numbers with E0103" $
    mapM_ (\src -> (src, errorCodes ("let y = " <> src)) `shouldBe` (src, [E0103])) ["1.", ".5", "1e", "1e-", "2x", "1.2.3", "1e999", "1e-999"]

  it "resolves the escapes of strings" $ do
    let parsed = program "data d : { y : Real } = csv \"a\\\"b\\\\c.csv\" { y = label }"
        path = case parsed of
          Right (Program [DataDecl _ _ _ (Csv _ (StringLit _ s) _)]) -> Just s
          _ -> Nothing
    path `shouldBe` Just "a\"b\\c.csv"

  it "parses literals and types on their own" $ do
    forgetSpans <$> parseLiteral "v" "[1, -2, [3]]"
      `shouldBe` Right (LitVector () [LitNumber (Number () 1), LitNumber (Number () (-2)), LitVector () [LitNumber (Number () 3)]])
    forgetSpans <$> parseType "v" "Mat m 2" `shouldBe` Right (TMat () (DimVar (name "m")) (DimNat (nat 2)))
    either (map diagCode . toList) (const []) (parseLiteral "v" "[1, x]") `shouldBe` [E0101]

-- ---------------------------------------------------------------------------
-- Spans

spanSpec :: Spec
spanSpec = do
  it "gives a binary expression the span of its operands (doc 11, §1.3)" $ do
    let decl = either (error . show) id (parseProgram "t.hsg" "let L = d * -2.0")
    case programDecls decl of
      [LetDecl sp x _ e@(Binary _ _ d m)] -> do
        sp `shouldBe` at (1, 1) (1, 17)
        identAnn x `shouldBe` at (1, 5) (1, 6)
        exprAnn d `shouldBe` at (1, 9) (1, 10)
        exprAnn m `shouldBe` at (1, 13) (1, 17)
        exprAnn e `shouldBe` at (1, 9) (1, 17)
      other -> error (show other)

  it "includes parentheses in the span of what they enclose" $
    case spannedExpr "(x - y) ^ 2" of
      Power sp base _ -> do
        sp `shouldBe` at (1, 1) (1, 12)
        exprAnn base `shouldBe` at (1, 1) (1, 8)
      other -> error (show other)

  it "ends a token's span before the whitespace and comments after it" $
    exprAnn (spannedExpr "a   -- comment\n") `shouldBe` at (1, 1) (1, 2)

  it "counts a tab as one column" $
    case parseProgram "t.hsg" "let\ty = 1" of
      Right (Program [LetDecl _ x _ _]) -> identAnn x `shouldBe` at (1, 5) (1, 6)
      other -> error (show other)

  it "spans lines for multi-line declarations" $ do
    src <- Utf8.readFile "examples/xor.hsg"
    case parseProgram "xor.hsg" src of
      Right (Program (DataDecl sp _ _ _ : _)) -> (spanStart sp, spanEnd sp) `shouldBe` (Pos 1 1, Pos 6 2)
      other -> error (show other)

-- ---------------------------------------------------------------------------
-- Round trip

roundTripSpec :: Spec
roundTripSpec = do
  modifyMaxSuccess (const 1000) $
    prop "parses a printed expression back to the same tree" $
      forAllShrink genExpr shrinkExpr $ \e ->
        let text = prettyExpr e
         in counterexample (Text.unpack text) (expr text === Right e)

  modifyMaxSuccess (const 300) $
    prop "parses a printed program back to the same tree" $
      forAllShrink genProgram (\(Program ds) -> Program <$> shrinkList (const []) ds) $ \p ->
        let text = prettyProgram p
         in counterexample (Text.unpack text) (program text === Right p)

  modifyMaxSuccess (const 500) $
    prop "gives every subexpression the span of exactly its text" $
      forAllShrink genExpr shrinkExpr $ \e ->
        let text = prettyExpr e
            parsed = spannedExpr text
         in counterexample (Text.unpack text) $
              conjoin
                [ counterexample (Text.unpack piece) (expr piece === Right (forgetSpans sub))
                | sub <- universe parsed
                , let piece = sliceSpan text (exprAnn sub)
                ]

  modifyMaxSuccess (const 300) $
    prop "gives every declaration the span of exactly its text" $
      forAllShrink genProgram (\(Program ds) -> Program <$> shrinkList (const []) ds) $ \p ->
        let text = prettyProgram p
            decls = either (error . show) programDecls (parseProgram "t.hsg" text)
         in conjoin
              [ program (sliceSpan text (declAnn d)) === Right (Program [forgetSpans d])
              | d <- decls
              ]

  it "round-trips the examples" $
    mapM_
      ( \file -> do
          src <- Utf8.readFile ("examples" </> file)
          let p = either (error . show) id (parseProgram file src)
          program (prettyProgram p) `shouldBe` Right (forgetSpans p)
      )
      ["intro.hsg", "neuron.hsg", "xor.hsg", "moons.hsg"]

-- ---------------------------------------------------------------------------
-- Errors

errorSpec :: Spec
errorSpec = do
  describe "golden" $
    goldenFiles "test/golden/parse" ".hsg" $ \file src -> pure (rendered file src)

  it "has a golden file for every syntax error code" $ do
    files <- filter ((== ".err") . takeExtension) <$> listDirectory "test/golden/parse"
    outputs <- mapM (Utf8.readFile . ("test/golden/parse" </>)) files
    let mentioned code = any (Text.isInfixOf ("error[" <> codeText code <> "]")) outputs
    filter (not . mentioned) [E0101 .. E0105] `shouldBe` []

  it "reproduces the hand-built E0102 and E0104 diagnostics of step 10" $
    mapM_
      ( \file -> do
          src <- Utf8.readFile ("test/golden/diagnostics" </> file <> ".hsg")
          expected <- Utf8.readFile ("test/golden/diagnostics" </> file <> ".err")
          rendered (file <> ".hsg") src `shouldBe` expected
      )
      ["missing-in", "unterminated-comment"]

  it "reports one error per broken declaration and parses the rest" $
    errorCodes "input x : Real = [1\nlet y = 1\nparam w : Real\nlet z = y ^ x\nloss = z"
      `shouldBe` [E0101, E0101, E0105]

  it "does not resynchronise at an indented let" $
    errorCodes "model m(x : Real) : Real =\n  let a = ) in\n  let b = 1 in\n  b"
      `shouldBe` [E0101]

  it "reports errors in source order" $
    case parseProgram "t.hsg" "let a = 1.\nlet b = (\nlet c = .5" of
      Left ds -> do
        let starts = map (spanStart . diagSpan) (NonEmpty.toList ds)
        length starts `shouldBe` 3
        starts `shouldBe` sort starts
      Right _ -> error "expected errors"
