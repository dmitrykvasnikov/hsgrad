{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Pretty".
--
-- * Numbers: constants print short and read back exactly; exponents print
--   exactly.
-- * Precedence: golden outputs for every position where parentheses are
--   or are not needed, in both styles.
-- * Sharing: shared subexpressions get bindings (with labels as names
--   where possible), tiny ones are repeated, and a chain of squarings,
--   exponentially large as a tree, prints in a few lines.
-- * micrograd's examples, unsimplified.
module HsGrad.Core.PrettySpec
  ( spec
  ) where

import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Core.Pretty
import HsGrad.Test.Examples
import HsGrad.Test.Graphs (built)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (counterexample, (===), (==>))

spec :: Spec
spec = do
  describe "numbers" numberSpec
  describe "precedence" precedenceSpec
  describe "sharing" sharingSpec
  describe "micrograd examples" examplesSpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Print the expression a builder returns.
pretty :: Builder NodeId -> Text
pretty b = prettyNode g i
  where
    (i, g) = built b

-- | The same, in the ASCII style.
ascii :: Builder NodeId -> Text
ascii b = prettyNodeWith defaultPrettyOptions {prettyStyle = Ascii} g i
  where
    (i, g) = built b

-- | Three inputs, for the precedence cases.
abc :: Builder (NodeId, NodeId, NodeId)
abc = (,,) <$> input "a" <*> input "b" <*> input "c"

-- ---------------------------------------------------------------------------
-- Numbers

numberSpec :: Spec
numberSpec = do
  it "prints constants short: integers without .0, shortest decimals, scientific at the ends" $
    map showConstant [2, -2, 0, -0, 0.5, 0.1, -2.5, 1 / 3, 1e-4, 1.5e-5, 123456789, 1e15, 1.25e20, 1e-300]
      `shouldBe` [ "2"
                 , "-2"
                 , "0"
                 , "0"
                 , "0.5"
                 , "0.1"
                 , "-2.5"
                 , "0.3333333333333333"
                 , "0.0001"
                 , "1.5e-5"
                 , "123456789"
                 , "1e15"
                 , "1.25e20"
                 , "1e-300"
                 ]
  prop "constants read back as exactly the same Double" $ \x ->
    not (isNaN x || isInfinite x) ==>
      let s = showConstant x in counterexample (Text.unpack s) (read (Text.unpack s) === (x :: Double))
  it "prints exponents exactly: integers, finite decimals, fractions" $
    map showExponent [2, -1, 0, 1 / 2, -3 / 2, 1 / 8, 1 / 20, 1 / 10, 1 / 3, -2 / 3]
      `shouldBe` ["2", "-1", "0", "0.5", "-1.5", "0.125", "0.05", "0.1", "(1/3)", "(-2/3)"]
  prop "decimal exponents read back as the same Rational" $ \p q ->
    let c = toRational (p :: Integer) / 2 ^ (abs q `mod` 12 :: Int)
        s = Text.unpack (showExponent c)
     in counterexample s (toRational (read s :: Double) === c)

-- ---------------------------------------------------------------------------
-- Precedence

precedenceSpec :: Spec
precedenceSpec = do
  it "uses no parentheses where precedence and left-associativity suffice" $ do
    pretty (abc >>= \(a, b, c) -> binary Add a =<< binary Mul b c) `shouldBe` "a + b·c"
    pretty (abc >>= \(a, b, c) -> (\ab -> binary Add ab c) =<< binary Mul a b) `shouldBe` "a·b + c"
    pretty (abc >>= \(a, b, c) -> (\ab -> binary Sub ab c) =<< binary Sub a b) `shouldBe` "a - b - c"
    pretty (abc >>= \(a, b, c) -> (\ab -> binary Div ab c) =<< binary Mul a b) `shouldBe` "a·b/c"
    pretty (abc >>= \(a, b, c) -> (\ab -> binary Mul ab c) =<< binary Div a b) `shouldBe` "a/b·c"
  it "puts parentheses around right operands that do not bind tighter" $ do
    pretty (abc >>= \(a, b, c) -> binary Sub a =<< binary Sub b c) `shouldBe` "a - (b - c)"
    pretty (abc >>= \(a, b, c) -> binary Add a =<< binary Add b c) `shouldBe` "a + (b + c)"
    pretty (abc >>= \(a, b, c) -> binary Div a =<< binary Mul b c) `shouldBe` "a/(b·c)"
    pretty (abc >>= \(a, b, c) -> binary Mul a =<< binary Div b c) `shouldBe` "a·(b/c)"
    pretty (abc >>= \(a, b, c) -> (\ab -> binary Mul ab c) =<< binary Add a b) `shouldBe` "(a + b)·c"
  it "binds powers tightest, then unary minus" $ do
    pretty (input "x" >>= unary (PowC 2) >>= unary Neg) `shouldBe` "-x²"
    pretty (input "x" >>= unary Neg >>= unary (PowC 2)) `shouldBe` "(-x)²"
    pretty (input "x" >>= unary (PowC 2) >>= unary (PowC 3)) `shouldBe` "(x²)³"
    pretty (constant (-2) >>= unary (PowC 2)) `shouldBe` "(-2)²"
    pretty (abc >>= \(a, b, _) -> unary Neg =<< binary Mul a b) `shouldBe` "-(a·b)"
    pretty (input "x" >>= unary Neg >>= unary Neg) `shouldBe` "-(-x)"
    pretty (abc >>= \(a, b, _) -> unary (PowC 2) =<< binary Add a b) `shouldBe` "(a + b)²"
    pretty (abc >>= \(a, b, _) -> (\v -> binary Mul v b) =<< unary Neg a) `shouldBe` "-a·b"
  it "puts negative right operands in parentheses" $ do
    pretty (do x <- input "x"; binary Mul x =<< constant (-2)) `shouldBe` "x·(-2)"
    pretty (do x <- input "x"; binary Sub x =<< constant (-2)) `shouldBe` "x - (-2)"
    pretty (do x <- input "x"; y <- input "y"; binary Add x =<< unary Neg y) `shouldBe` "x + (-y)"
    pretty (do b <- input "b"; m <- constant (-2); binary Mul m b) `shouldBe` "-2·b"
  it "prints exponents as superscripts if integer, with ^ otherwise" $ do
    pretty (input "x" >>= unary (PowC (-1))) `shouldBe` "x⁻¹"
    pretty (input "x" >>= unary (PowC 12)) `shouldBe` "x¹²"
    pretty (input "x" >>= unary (PowC (1 / 2))) `shouldBe` "x^0.5"
    pretty (input "x" >>= unary (PowC (-1 / 2))) `shouldBe` "x^-0.5"
    pretty (input "x" >>= unary (PowC (1 / 3))) `shouldBe` "x^(1/3)"
    pretty (input "x" >>= unary Exp >>= unary (PowC (1 / 2))) `shouldBe` "exp(x)^0.5"
  it "prints functions with the language's names" $
    pretty
      ( do
          x <- input "x"
          parts <- mapM (`unary` x) [Exp, Log, Tanh, Relu, Step]
          foldl1 (\a b -> do a' <- a; b' <- b; binary Add a' b') (map pure parts)
      )
      `shouldBe` "exp(x) + log(x) + tanh(x) + relu(x) + step(x)"
  it "prints * and ^ in the ASCII style" $ do
    ascii (do x <- input "x"; m <- constant (-2); s <- unary (PowC 2) x; binary Mul m s) `shouldBe` "-2*x^2"
    ascii (input "x" >>= unary (PowC (-1))) `shouldBe` "x^-1"
    ascii (input "x" >>= unary (PowC (1 / 3))) `shouldBe` "x^(1/3)"

-- ---------------------------------------------------------------------------
-- Sharing

sharingSpec :: Spec
sharingSpec = do
  it "gives a shared subexpression a binding" $
    pretty
      ( do
          (a, b, c) <- abc
          s <- (\v -> binary Add v c) =<< binary Mul a b
          ss <- binary Mul s s
          binary Add ss =<< unary Tanh s
      )
      `shouldBe` "let t1 = a·b + c\nin t1·t1 + tanh(t1)"
  it "repeats shared subexpressions of at most three nodes" $
    pretty (do x <- input "x"; s <- unary (PowC 2) x; binary Add s =<< unary Exp s)
      `shouldBe` "x² + exp(x²)"
  it "names a binding after the node's label if it is a plain name not taken" $ do
    let shared l = do
          (a, b, c) <- abc
          s <- (\v -> binary Add v c) =<< binary Mul a b
          label l s
          binary Mul s s
    pretty (shared "h") `shouldBe` "let h = a·b + c\nin h·h"
    pretty (shared "W1[0,1]") `shouldBe` "let W1[0,1] = a·b + c\nin W1[0,1]·W1[0,1]"
    -- not a plain name
    pretty (shared "∂L/∂a") `shouldBe` "let t1 = a·b + c\nin t1·t1"
    -- the name of a leaf in the expression
    pretty (shared "a") `shouldBe` "let t1 = a·b + c\nin t1·t1"
  it "never gives a binding a leaf's name" $
    pretty
      ( do
          (a, b, _) <- abc
          t1 <- input "t1"
          s <- (\v -> binary Add v t1) =<< binary Mul a b
          binary Mul s s
      )
      `shouldBe` "let t2 = a·b + t1\nin t2·t2"
  it "lets several roots share bindings, and names a shared root" $ do
    let (roots, g) = built $ do
          (a, b, c) <- abc
          s <- (\v -> binary Add v c) =<< binary Mul a b
          u <- binary Mul s a
          v <- binary Mul s b
          pure [s, u, v]
    renderNodes defaultPrettyOptions g roots
      `shouldBe` Rendered [("t1", "a·b + c")] ["t1", "t1·a", "t1·b"]
  it "prints a chain of 30 squarings, two billion nodes as a tree, in 15 lines" $ do
    let (out, g) = built $ do
          x <- input "x"
          foldr (=<<) (pure x) (replicate 30 (\v -> binary Mul v v))
        text = prettyNode g out
    length (Text.lines text) `shouldBe` 15
    Text.length text `shouldSatisfy` (< 500)
    last (Text.lines text) `shouldBe` "in t14·t14·(t14·t14)"
    take 2 (Text.lines text) `shouldBe` ["let t1 = x·x·(x·x)", "    t2 = t1·t1·(t1·t1)"]

-- ---------------------------------------------------------------------------
-- micrograd's examples

examplesSpec :: Spec
examplesSpec = do
  it "prints intro.hsg's L" $ do
    let (ex, g) = intro
    prettyNode g (introL ex) `shouldBe` "(a·b + c)·(-2)"
  it "prints the neuron's y" $ do
    let (ex, g) = neuron
    prettyNode g (neuronY ex) `shouldBe` "tanh(w[0]·x[0] + w[1]·x[1] + w[2]·x[2] + b)"
