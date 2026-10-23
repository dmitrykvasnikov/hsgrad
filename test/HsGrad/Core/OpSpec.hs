{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Op".
--
-- * Forward functions: known values and a few identities.
-- * Structure: every op has one rule per argument, and rules only mention
--   arguments that exist.
-- * Derivative rules: each one agrees with central finite differences of
--   the forward function, on inputs away from singular points.
module HsGrad.Core.OpSpec
  ( spec
  ) where

import Control.Monad (forM_)
import Data.Ratio (denominator, numerator)
import HsGrad.Core.Op
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck
  ( Gen
  , Property
  , choose
  , conjoin
  , counterexample
  , forAll
  , oneof
  , property
  , (===)
  )

spec :: Spec
spec = do
  describe "forward functions" forwardSpec
  describe "structure" structureSpec
  describe "derivative rules" ruleShapeSpec
  describe "derivative rules vs finite differences" finiteDiffSpec

-- ---------------------------------------------------------------------------
-- Forward functions

forwardSpec :: Spec
forwardSpec = do
  it "computes the binary ops" $ do
    forwardBin Add 2 3 `shouldBe` 5
    forwardBin Sub 2 3 `shouldBe` (-1)
    forwardBin Mul 2 (-3) `shouldBe` (-6)
    forwardBin Div 3 4 `shouldBe` 0.75

  it "computes the unary ops at known points" $ do
    forwardUn Neg 2 `shouldBe` (-2)
    forwardUn Exp 0 `shouldBe` 1
    forwardUn Log 1 `shouldBe` 0
    forwardUn Tanh 0 `shouldBe` 0
    forwardUn Relu (-2) `shouldBe` 0
    forwardUn Relu 3 `shouldBe` 3
    forwardUn Step (-2) `shouldBe` 0
    forwardUn Step 3 `shouldBe` 1

  it "uses 0 for relu and step at exactly 0" $ do
    forwardUn Relu 0 `shouldBe` 0
    forwardUn Step 0 `shouldBe` 0

  it "computes constant powers" $ do
    forwardUn (PowC 2) (-3) `shouldBe` 9
    forwardUn (PowC 3) (-2) `shouldBe` (-8)
    forwardUn (PowC (-1)) 4 `shouldBe` 0.25
    forwardUn (PowC (1 / 2)) 9 `shouldBe` 3
    forwardUn (PowC 0) 0 `shouldBe` 1

  it "returns NaN or infinity outside the domain instead of failing" $ do
    forwardUn Log 0 `shouldBe` (-1 / 0)
    isNaN (forwardUn Log (-1)) `shouldBe` True
    isNaN (forwardUn (PowC (1 / 2)) (-4)) `shouldBe` True

  it "keeps NaN as NaN in relu and step" $ do
    isNaN (forwardUn Relu nan) `shouldBe` True
    isNaN (forwardUn Step nan) `shouldBe` True

  prop "exp and log are inverse" $
    forAll genPositive $ \x -> closeTo (forwardUn Exp (forwardUn Log x)) x

  prop "tanh is odd" $
    forAll genAny $ \x -> forwardUn Tanh (-x) === negate (forwardUn Tanh x)

  prop "relu x = x * step x" $
    forAll genAny $ \x -> forwardUn Relu x === x * forwardUn Step x

  prop "integer powers agree with repeated multiplication" $
    forAll genAny $ \x ->
      forAll (choose (0, 6)) $ \n ->
        closeTo (forwardUn (PowC (fromIntegral n)) x) (product (replicate n x))

  prop "x ^ 0.5 is the square root" $
    forAll genPositive $ \x -> closeTo (forwardUn (PowC (1 / 2)) x) (sqrt x)

-- ---------------------------------------------------------------------------
-- Structure

structureSpec :: Spec
structureSpec = do
  it "gives leaves arity 0, unary ops 1 and binary ops 2" $ do
    map arity [Leaf (Const 1), Leaf (Input "x"), Leaf (Param "w")] `shouldBe` [0, 0, 0]
    map arity [Un Tanh, Un (PowC 2)] `shouldBe` [1, 1]
    map arity [Bin Add, Bin Div] `shouldBe` [2, 2]

  it "has one rule per argument" $
    forM_ allOps $ \op -> length (partials op) `shouldBe` arity op

  it "has rules that only mention existing arguments" $
    forM_ allOps $ \op ->
      forM_ (partials op) $ \rule ->
        filter (>= arity op) (argsUsed rule) `shouldBe` []

  it "compares exponents exactly" $ do
    PowC (1 / 2) `shouldBe` PowC 0.5
    -- With Double exponents, 0.1 - 1 /= -0.9; with Rational they are equal,
    -- so the derivative of x^0.1 hash-conses with a literal x^-0.9.
    PowC (0.1 - 1) `shouldBe` PowC (-0.9)

  it "tells inputs and parameters with the same name apart" $
    (Leaf (Input "w") == Leaf (Param "w")) `shouldBe` False

-- | One op of every constructor, with several interesting exponents.
allOps :: [Op]
allOps =
  [Leaf (Const 2), Leaf (Input "x"), Leaf (Param "w")]
    ++ map Un ([Neg, Exp, Log, Tanh, Relu, Step] ++ map PowC exponents)
    ++ map Bin [minBound .. maxBound]

-- | The argument indices a rule refers to.
argsUsed :: DExpr -> [Int]
argsUsed e = case e of
  DLit _ -> []
  DArg i -> [i]
  DOut -> []
  DUn _ a -> argsUsed a
  DBin _ a b -> argsUsed a ++ argsUsed b

-- ---------------------------------------------------------------------------
-- The shape of some rules (D6)

ruleShapeSpec :: Spec
ruleShapeSpec = do
  it "differentiates relu to step" $
    unPartial Relu `shouldBe` DUn Step (DArg 0)

  it "differentiates tanh and exp by reusing the node's own output" $ do
    unPartial Tanh `shouldBe` DBin Sub (DLit 1) (DUn (PowC 2) DOut)
    unPartial Exp `shouldBe` DOut

  it "differentiates x^c to c * x^(c-1), skipping trivial powers" $ do
    unPartial (PowC 3) `shouldBe` DBin Mul (DLit 3) (DUn (PowC 2) (DArg 0))
    unPartial (PowC 2) `shouldBe` DBin Mul (DLit 2) (DArg 0)
    unPartial (PowC 1) `shouldBe` DLit 1
    unPartial (PowC 0) `shouldBe` DLit 0

  it "gives x^0 a zero derivative even at x = 0" $
    localGrads (Un (PowC 0)) [0] 1 `shouldBe` [0]

  it "evaluates micrograd's product example" $
    localGrads (Bin Mul) [2, -3] (-6) `shouldBe` [-3, 2]

-- ---------------------------------------------------------------------------
-- Derivative rules versus central finite differences

finiteDiffSpec :: Spec
finiteDiffSpec = do
  forM_ unaryCases $ \(label, op, gen) ->
    prop label $ forAll gen $ \x -> checkUnary op x

  forM_ exponents $ \c ->
    prop ("x ^ " <> showExponent c) $ forAll (genForExponent c) $ \x -> checkUnary (PowC c) x

  forM_ binaryCases $ \(label, op, genX, genY) ->
    prop label $ forAll genX $ \x -> forAll genY $ \y -> checkBinary op x y

-- | Unary ops with inputs that keep away from their singular points:
-- @log@ needs @x > 0@, and @relu@/@step@ have a kink or jump at 0 that a
-- finite difference straddling 0 would see.
unaryCases :: [(String, UnOp, Gen Double)]
unaryCases =
  [ ("neg", Neg, genAny)
  , ("exp", Exp, genAny)
  , ("log", Log, genPositive)
  , ("tanh", Tanh, genAny)
  , ("relu", Relu, genAwayFromZero)
  , ("step", Step, genAwayFromZero)
  ]

-- | Division needs a divisor away from 0.
binaryCases :: [(String, BinOp, Gen Double, Gen Double)]
binaryCases =
  [ ("add", Add, genAny, genAny)
  , ("sub", Sub, genAny, genAny)
  , ("mul", Mul, genAny, genAny)
  , ("div", Div, genAny, genAwayFromZero)
  ]

-- | Exponents to test: negative, zero, fractional and integer.
exponents :: [Rational]
exponents = [-3, -2, -1, -1 / 2, 0, 1 / 3, 1 / 2, 1, 3 / 2, 2, 3]

-- | @2@, @-1/2@ rather than 'Rational''s @2 % 1@, @(-1) % 2@.
showExponent :: Rational -> String
showExponent c
  | denominator c == 1 = show (numerator c)
  | otherwise = show (numerator c) <> "/" <> show (denominator c)

-- | Inputs where @x ^ c@ is defined and smooth: any @x@ for natural
-- exponents, @x /= 0@ for negative integers, @x > 0@ otherwise.
genForExponent :: Rational -> Gen Double
genForExponent c
  | isInteger && c >= 0 = genAny
  | isInteger = genAwayFromZero
  | otherwise = genPositive
  where
    isInteger = denominator c == 1

-- | The rule for @y = f(x)@ evaluated at @x@ agrees with the central
-- difference of 'forwardUn'.
checkUnary :: UnOp -> Double -> Property
checkUnary op x =
  case localGrads (Un op) [x] (f x) of
    [dx] -> counterexample "d/dx" (closeTo dx (centralDiff f x))
    grads -> counterexample ("expected one gradient, got " <> show grads) False
  where
    f = forwardUn op

-- | Both rules for @z = g(x, y)@ agree with central differences in @x@
-- and in @y@.
checkBinary :: BinOp -> Double -> Double -> Property
checkBinary op x y =
  case localGrads (Bin op) [x, y] (g x y) of
    [dx, dy] ->
      conjoin
        [ counterexample "d/dx" (closeTo dx (centralDiff (`g` y) x))
        , counterexample "d/dy" (closeTo dy (centralDiff (g x) y))
        ]
    grads -> counterexample ("expected two gradients, got " <> show grads) False
  where
    g = forwardBin op

-- | Central finite difference @(f(x+h) - f(x-h)) / 2h@. Its error is
-- O(h^2), against O(h) for the one-sided @(f(x+h) - f(x)) / h@. The step
-- grows with @|x|@ so that @x ± h@ stays distinguishable from @x@.
centralDiff :: (Double -> Double) -> Double -> Double
centralDiff f x = (f (x + h) - f (x - h)) / (2 * h)
  where
    h = 1e-6 * max 1 (abs x)

-- | Approximate equality with a relative tolerance (absolute near 0).
closeTo :: Double -> Double -> Property
closeTo actual expected =
  counterexample (show actual <> " is not close to " <> show expected) $
    property (abs (actual - expected) <= 1e-6 * maximum [1, abs actual, abs expected])

-- ---------------------------------------------------------------------------
-- Generators

-- | Moderate values: @exp 5 ≈ 148@ keeps finite differences well-conditioned.
genAny :: Gen Double
genAny = choose (-5, 5)

genPositive :: Gen Double
genPositive = choose (0.1, 10)

-- | At least 0.01 away from 0, much more than the finite-difference step.
genAwayFromZero :: Gen Double
genAwayFromZero = oneof [choose (-5, -0.01), choose (0.01, 5)]

nan :: Double
nan = 0 / 0
