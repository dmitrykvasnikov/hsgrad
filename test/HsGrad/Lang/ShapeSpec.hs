{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Shape".
--
-- * Printing shapes, templates, signatures and arrows as the language
--   writes them.
-- * Matching: binding shape variables, E0303 with both bindings, E0304,
--   unknown and unannotated arguments.
-- * The shape rules: broadcasting, matrix products, indexing.
-- * Every built-in function on every kind of shape, and the table of
--   built-ins against @docs/01-language.md@ §6.
module HsGrad.Lang.ShapeSpec
  ( spec
  ) where

import Data.List (sort)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Lang.Shape
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Gen, chooseInt, forAll, oneof, (===))

spec :: Spec
spec = do
  describe "printing" printSpec
  describe "matching" matchSpec
  describe "operations" operationSpec
  describe "built-in functions" builtinSpec

-- | A random concrete shape with small dimensions.
genShape :: Gen (Shape Int)
genShape = oneof [pure Real, Vec <$> dim, Mat <$> dim <*> dim]
  where
    dim = chooseInt (1, 4)

var :: Text -> Dimension
var = Variable

-- ---------------------------------------------------------------------------
-- Printing

printSpec :: Spec
printSpec = do
  it "prints shapes as the language writes them" $
    map prettyShape [Real, Vec 3, Mat 16 2] `shouldBe` ["Real", "Vec 3", "Mat 16 2"]

  it "prints templates with their shape variables" $
    map prettyTemplate [Vec (var "n"), Mat (Fixed 16) (var "n")] `shouldBe` ["Vec n", "Mat 16 n"]

  it "prints sequences with parentheses where needed" $
    map prettyOperand [Value (Vec 2), Seq Real, Seq (Vec 2)] `shouldBe` ["Vec 2", "Seq Real", "Seq (Vec 2)"]

  it "prints signatures, with and without annotations" $ do
    prettySignature (Signature "layer" [("W", Just (Mat (var "m") (var "n"))), ("b", Just (Vec (var "m"))), ("x", Just (Vec (var "n")))] (Just (Vec (var "m"))))
      `shouldBe` "layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m"
    prettySignature (Signature "sq" [("z", Nothing)] Nothing) `shouldBe` "sq(z)"

  it "prints function types as arrows" $ do
    prettyArrow [Vec 2] Real `shouldBe` "Vec 2 → Real"
    prettyArrow [Mat 16 2, Vec 16, Vec 2] (Vec 16) `shouldBe` "Mat 16 2, Vec 16, Vec 2 → Vec 16"
    prettyArrow [] Real `shouldBe` "() → Real"

  it "prints kinds with an article" $
    map prettyKind [minBound .. maxBound] `shouldBe` ["a value", "a function", "a dataset"]

-- ---------------------------------------------------------------------------
-- Matching

matchSpec :: Spec
matchSpec = do
  let layer = [Just (Mat (var "m") (var "n")), Just (Vec (var "m")), Just (Vec (var "n"))]
      sizes = fmap bindingSize . fst

  it "binds shape variables from the first argument that has them" $
    fst (matchArguments layer [Just (Mat 16 2), Just (Vec 16), Just (Vec 2)])
      `shouldBe` Map.fromList [("m", Binding 16 0), ("n", Binding 2 0)]

  it "accepts consistent arguments without problems" $
    snd (matchArguments layer [Just (Mat 16 2), Just (Vec 16), Just (Vec 2)]) `shouldBe` []

  it "reports an inconsistent variable with both bindings (E0303)" $
    snd (matchArguments layer [Just (Mat 16 2), Just (Vec 8), Just (Vec 2)])
      `shouldBe` [InconsistentVariable "m" (Binding 16 0) (Binding 8 1)]

  it "reports every problem, not only the first" $
    snd (matchArguments layer [Just (Mat 16 2), Just (Vec 8), Just (Vec 3)])
      `shouldBe` [InconsistentVariable "m" (Binding 16 0) (Binding 8 1), InconsistentVariable "n" (Binding 2 0) (Binding 3 2)]

  it "reports a variable used twice in one argument" $
    snd (matchArguments [Just (Mat (var "n") (var "n"))] [Just (Mat 2 3)])
      `shouldBe` [InconsistentVariable "n" (Binding 2 0) (Binding 3 0)]

  it "reports a wrong kind of shape or a wrong fixed size (E0304)" $ do
    snd (matchArguments [Just (Vec (var "n"))] [Just Real]) `shouldBe` [ArgumentMismatch 0]
    snd (matchArguments [Just Real, Just (Vec (Fixed 2))] [Just Real, Just (Vec 3)]) `shouldBe` [ArgumentMismatch 1]

  it "binds nothing from an argument that doesn't fit" $
    matchArguments [Just (Mat (var "n") (Fixed 2)), Just (Vec (var "n"))] [Just (Mat 3 4), Just (Vec 5)]
      `shouldBe` (Map.fromList [("n", Binding 5 1)], [ArgumentMismatch 0])

  it "skips unknown arguments and accepts anything for unannotated parameters" $ do
    snd (matchArguments layer [Nothing, Just (Vec 8), Just (Vec 2)]) `shouldBe` []
    sizes (matchArguments [Nothing, Just (Vec (var "n"))] [Just (Mat 2 2), Just (Vec 7)]) `shouldBe` Map.fromList [("n", 7)]

  prop "matches every concrete shape against itself" $
    forAll genShape $ \s -> matchArguments [Just (fromShape s)] [Just s] === (Map.empty, [])

  it "matches one shape against a template, extending the bindings" $ do
    matchShape (Map.fromList [("m", 16)]) (Vec (var "m")) (Vec 16) `shouldBe` Just (Map.fromList [("m", 16)])
    matchShape (Map.fromList [("m", 16)]) (Vec (var "m")) (Vec 8) `shouldBe` Nothing
    matchShape Map.empty (Mat (var "m") (Fixed 2)) (Mat 3 2) `shouldBe` Just (Map.fromList [("m", 3)])
    matchShape Map.empty (Mat (var "n") (var "n")) (Mat 3 2) `shouldBe` Nothing
    matchShape Map.empty Real (Vec 1) `shouldBe` Nothing

  it "substitutes variables, and only when all are bound" $ do
    substitute (Map.fromList [("m", 4)]) (Mat (var "m") (Fixed 2)) `shouldBe` Just (Mat 4 2)
    substitute Map.empty (Vec (var "m")) `shouldBe` Nothing
    bindTemplate (Map.fromList [("m", 4)]) (Mat (var "m") (var "n")) `shouldBe` Mat (Fixed 4) (var "n")

  it "lists a template's variables once each" $
    templateVariables (Mat (var "n") (var "n")) `shouldBe` ["n"]

  prop "turns concrete shapes into templates and back" $
    forAll genShape $ \s -> toShape (fromShape s) === Just s

-- ---------------------------------------------------------------------------
-- Operations

operationSpec :: Spec
operationSpec = do
  it "broadcasts equal shapes and a Real on either side" $ do
    broadcast (Vec 3) (Vec 3) `shouldBe` Just (Vec 3)
    broadcast Real (Mat 2 3) `shouldBe` Just (Mat 2 3)
    broadcast (Vec 3) Real `shouldBe` Just (Vec 3)
    broadcast Real Real `shouldBe` Just Real

  it "broadcasts nothing else" $ do
    broadcast (Vec 3) (Vec 2) `shouldBe` Nothing
    broadcast (Mat 2 3) (Vec 3) `shouldBe` Nothing
    broadcast (Vec 2) (Mat 2 2) `shouldBe` Nothing
    broadcast (Vec 4) (Mat 2 2) `shouldBe` Nothing

  prop "broadcasts symmetrically" $
    forAll genShape $ \a -> forAll genShape $ \b -> broadcast a b === broadcast b a

  prop "broadcasts a shape with itself and with a Real to itself" $
    forAll genShape $ \s -> (broadcast s s, broadcast Real s) === (Just s, Just s)

  it "multiplies matrices with vectors and matrices" $ do
    matMul (Mat 16 2) (Vec 2) `shouldBe` Just (Vec 16)
    matMul (Mat 2 3) (Mat 3 4) `shouldBe` Just (Mat 2 4)

  it "rejects mismatched inner dimensions and non-matrices on the left" $ do
    matMul (Mat 16 16) (Vec 8) `shouldBe` Nothing
    matMul (Mat 2 3) (Mat 2 3) `shouldBe` Nothing
    matMul (Vec 2) (Vec 2) `shouldBe` Nothing
    matMul (Vec 2) (Mat 2 2) `shouldBe` Nothing
    matMul Real (Mat 2 2) `shouldBe` Nothing
    matMul (Mat 2 2) Real `shouldBe` Nothing

  it "indexes vectors and matrices" $ do
    indexShape (Vec 3) 2 Nothing `shouldBe` Right Real
    indexShape (Mat 4 2) 3 Nothing `shouldBe` Right (Vec 2)
    indexShape (Mat 4 2) 3 (Just 1) `shouldBe` Right Real

  it "checks indices against the sizes (E0305)" $ do
    indexShape (Vec 3) 3 Nothing `shouldBe` Left (OutOfBounds 0 3 3)
    indexShape (Mat 4 2) 4 Nothing `shouldBe` Left (OutOfBounds 0 4 4)
    indexShape (Mat 4 2) 1 (Just 2) `shouldBe` Left (OutOfBounds 1 2 2)

  it "rejects more indices than dimensions (E0306)" $ do
    indexShape Real 0 Nothing `shouldBe` Left TooManyIndices
    indexShape (Vec 3) 0 (Just 0) `shouldBe` Left TooManyIndices

  it "counts the scalars of a shape" $
    map shapeSize [Real, Vec 3, Mat 4 2] `shouldBe` [1, 3, 8]

-- ---------------------------------------------------------------------------
-- Built-in functions

builtinSpec :: Spec
builtinSpec = do
  it "has the built-in functions of docs/01-language.md §6" $ do
    doc <- Utf8.readFile "docs/01-language.md"
    let section = takeWhile (not . Text.isPrefixOf "## 7.") (dropWhile (not . Text.isPrefixOf "## 6.") (Text.lines doc))
        firstColumn line = Text.takeWhile (/= '|') (Text.drop 2 line)
        named =
          [ Text.takeWhile (/= '(') (Text.drop 1 w)
          | line <- section
          , "| `" `Text.isPrefixOf` line
          , w <- Text.splitOn ", " (firstColumn line)
          , "(" `Text.isInfixOf` w
          ]
    sort named `shouldBe` sort (map builtinName builtins)

  it "finds every built-in function by its name" $
    map (lookupBuiltin . builtinName) builtins `shouldBe` map Just builtins

  it "doesn't know other names" $
    map lookupBuiltin ["uniform", "params", "sigmod"] `shouldBe` [Nothing, Nothing, Nothing]

  prop "applies the elementwise functions to any shape" $
    forAll genShape $ \s ->
      [applyBuiltin b [Value s] | b <- builtins, builtinType b == Elementwise] === map (const (Right s)) [BExp, BLog, BTanh, BRelu, BStep, BSigmoid, BSqrt, BAbs]

  it "broadcasts max and min" $ do
    applyBuiltin BMax [Value (Vec 3), Value Real] `shouldBe` Right (Vec 3)
    applyBuiltin BMin [Value (Mat 2 2), Value (Mat 2 2)] `shouldBe` Right (Mat 2 2)
    applyBuiltin BMax [Value (Vec 3), Value (Vec 2)] `shouldBe` Left [IncompatibleArguments (Vec 3) (Vec 2)]

  prop "reduces every value to a Real, and a sequence to its element shape" $
    forAll genShape $ \s ->
      [applyBuiltin b [o] | b <- [BSum, BMean], o <- [Value s, Seq s]] === [Right Real, Right s, Right Real, Right s]

  it "types dot and transpose by matching" $ do
    applyBuiltin BDot [Value (Vec 3), Value (Vec 3)] `shouldBe` Right Real
    applyBuiltin BDot [Value (Vec 3), Value (Vec 2)] `shouldBe` Left [BadArgument (InconsistentVariable "n" (Binding 3 0) (Binding 2 1))]
    applyBuiltin BDot [Value Real, Value (Vec 2)] `shouldBe` Left [BadArgument (ArgumentMismatch 0)]
    applyBuiltin BTranspose [Value (Mat 2 3)] `shouldBe` Right (Mat 3 2)
    applyBuiltin BTranspose [Value (Vec 3)] `shouldBe` Left [BadArgument (ArgumentMismatch 0)]

  it "checks the number of arguments" $ do
    applyBuiltin BTanh [] `shouldBe` Left [WrongArity 1 0]
    applyBuiltin BDot [Value (Vec 2)] `shouldBe` Left [WrongArity 2 1]
    map builtinArity builtins `shouldBe` [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1]

  it "accepts sequences only in sum and mean (E0311)" $ do
    applyBuiltin BExp [Seq Real] `shouldBe` Left [SequenceArgument 0]
    applyBuiltin BDot [Value (Vec 2), Seq (Vec 2)] `shouldBe` Left [SequenceArgument 1]

  it "gives every built-in function a result for some arguments" $
    -- Guards the invariant behind the 'error' in applyBuiltin: the result
    -- variables of a matching signature appear among its parameters.
    [b | b <- builtins, Matching ps r <- [builtinType b], not (all (`elem` concatMap (templateVariables . snd) ps) (templateVariables r))]
      `shouldBe` []

  it "prints signatures for messages" $ do
    builtinSignature BDot `shouldBe` "dot(u : Vec n, v : Vec n) : Real"
    builtinSignature BTranspose `shouldBe` "transpose(M : Mat r c) : Mat c r"
    builtinSignature BExp `shouldBe` "exp(e : s) : s"
    builtinSignature BMax `shouldBe` "max(a : s, b : s) : s"
    builtinSignature BSum `shouldBe` "sum(e : s) : Real, sum(e : Seq t) : t"

  it "says what each argument must be" $ do
    builtinExpects BDot 1 `shouldBe` "`Vec n`"
    builtinExpects BTanh 0 `shouldSatisfy` ("`Real`" `Text.isInfixOf`)
