{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Compile".
--
-- * The milestone: compiling @intro.hsg@ and @neuron.hsg@ gives exactly
--   the hand-built graphs of "HsGrad.Test.Examples" (nodes, ids, spans,
--   labels), so they evaluate and differentiate identically; @xor.hsg@
--   gives a graph structurally equal to the hand-built XOR net and trains
--   to accuracy 1 through "HsGrad.Train.Loop"; @moons.hsg@ compiles at a
--   small batch size with the documented node count.
-- * Sharing: a @let@, a @let … in@ and a function argument are compiled
--   once.
-- * Every lowering of @01-language.md@ §6 and §8, evaluated against a
--   direct Haskell computation, and the desugared built-ins compared with
--   their formulas built by hand.
-- * The handles: names, leaves, slots, defaults, the predictor, errors.
module HsGrad.Lang.CompileSpec
  ( spec
  ) where

import Control.Monad (foldM, zipWithM)
import Data.Foldable (toList)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (isJust)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Core.Backprop (LeafGrads (..), gradients, leafGrads)
import HsGrad.Core.Eval (Env (..), emptyEnv, evaluate, schedule, valueAt)
import HsGrad.Core.Graph
import HsGrad.Core.Op (BinOp (..), Leaf (..), Name (..), Op (..), UnOp (..))
import HsGrad.Data.Dataset (Dataset (..), loadDatasets, slotNames, toFeed)
import HsGrad.Lang.Check (Checked, checkProgram, checkedDatasets)
import HsGrad.Lang.Compile
import HsGrad.Lang.Diagnostic (diagCode)
import HsGrad.Lang.Parser (parseProgram)
import HsGrad.Lang.Shape qualified as Shape
import HsGrad.Test.Examples
import HsGrad.Test.Graphs (built, sameExpressions)
import HsGrad.Train.Loop
import HsGrad.Train.Optim (LrSchedule (..), Sgd (..))
import HsGrad.Train.Param (Init (..), ParamSpec (..), elementNames, initParams)
import Test.Hspec (Spec, describe, expectationFailure, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Gen, Property, choose, counterexample, forAll, vectorOf)

-- ---------------------------------------------------------------------------
-- Helpers

-- | Parse and check a program that must check.
checked :: FilePath -> Text -> Checked
checked file src = case parseProgram file src of
  Left ds -> error ("parse error in test program: " <> show (toList ds))
  Right p -> case checkProgram p of
    (_, Just c) -> c
    (ds, Nothing) -> error ("test program doesn't check: " <> show (map diagCode ds))

-- | Compile a program that must compile, with these slot counts.
compiledWith :: Map Text Int -> FilePath -> Text -> Compiled
compiledWith batch file src = case compileProgram batch (checked file src) of
  Right c -> c
  Left err -> error ("test program doesn't compile: " <> show err)

compiled :: Text -> Compiled
compiled = compiledWith Map.empty "t.hsg"

-- | An example file, compiled with its bare name as the span file (as the
-- hand-built examples are).
example :: FilePath -> IO Compiled
example name = compiledWith Map.empty name <$> Utf8.readFile ("examples/" <> name)

-- | The value of a name that must exist.
named :: Text -> Compiled -> Value
named x c = case lookupName x c of
  Just v -> v
  Nothing -> error ("no value named " <> show x)

-- | The one node of a 'Real'.
scalar :: Value -> NodeId
scalar = \case
  Real x -> x
  v -> error ("not a Real: " <> show v)

-- | Evaluate a value's elements with the inputs bound.
valuesOf :: Compiled -> Env -> Value -> [Double]
valuesOf c env v = case evaluate (compiledGraph c) env (valueNodes v) of
  Right vals -> map (valueAt vals) (valueNodes v)
  Left err -> error ("evaluation failed: " <> show err)

-- | An environment with the inputs' defaults.
defaults :: Compiled -> Env
defaults c = emptyEnv {envInputs = inputDefaults c}

-- | Equal up to rounding.
approx :: Double -> Double -> Bool
approx a b = abs (a - b) <= 1e-12 * max 1 (max (abs a) (abs b))

-- | The rows of a dataset that must load.
loaded :: FilePath -> Checked -> IO [Dataset]
loaded path c = do
  (_, result) <- loadDatasets path (checkedDatasets c)
  either (\ds -> error ("datasets don't load: " <> show (map diagCode ds))) pure result

-- ---------------------------------------------------------------------------

spec :: Spec
spec = do
  milestoneSpec
  sharingSpec
  loweringSpec
  builtinSpec
  handleSpec

-- ---------------------------------------------------------------------------
-- The milestone

milestoneSpec :: Spec
milestoneSpec = describe "the milestone: .hsg → graph equals the hand-built graph" $ do
  it "compiles intro.hsg to exactly the hand-built graph" $ do
    c <- example "intro.hsg"
    compiledGraph c `shouldBe` snd intro
    let (i, _) = intro
    named "L" c `shouldBe` Real (introL i)
    compiledLoss c `shouldBe` Nothing
    map fst (compiledLets c) `shouldBe` ["e", "d", "L"]

  it "evaluates and differentiates intro.hsg like micrograd: L = -8, gradients 6, -4, -2" $ do
    c <- example "intro.hsg"
    let l = scalar (named "L" c)
    case gradients (compiledGraph c) (defaults c) l of
      Left err -> expectationFailure (show err)
      Right (vals, grads) -> do
        valueAt vals l `shouldBe` -8
        inputGrads (leafGrads (schedule (compiledGraph c) [l]) grads)
          `shouldBe` Map.fromList [("a", 6), ("b", -4), ("c", -2)]

  it "compiles neuron.hsg to exactly the hand-built graph" $ do
    c <- example "neuron.hsg"
    compiledGraph c `shouldBe` snd neuron
    let (n, _) = neuron
    named "y" c `shouldBe` Real (neuronY n)
    named "x" c `shouldBe` Vec (neuronX n)
    named "w" c `shouldBe` Vec (neuronW n)
    valuesOf c neuronEnv (named "y" c) `shouldBe` [tanh 1.5]

  it "compiles xor.hsg to a graph structurally equal to the hand-built XOR net" $ do
    c <- example "xor.hsg"
    let ((loss, pairs), g) = xorNet 4
    case compiledLoss c of
      Nothing -> expectationFailure "no loss"
      Just l -> sameExpressions ([l], compiledGraph c) ([loss], g) `shouldBe` True
    -- The accuracy pairs are the hand-built net's predictions and labels.
    let flat = concatMap (\(p, y) -> [p, y])
    length (compiledAccuracy c) `shouldBe` 4
    sameExpressions (flat (compiledAccuracy c), compiledGraph c) (flat pairs, g) `shouldBe` True
    compiledParams c `shouldBe` xorSpecs
    compiledSlots c `shouldBe` Map.singleton "xor" (xorSlots 4)

  describe "xor.hsg trains through Train.Loop" $ do
    let config = TrainConfig 200 (Sgd 0.1 ConstantRate) 1337
        setupOf c ds l =
          Setup
            { setupGraph = compiledGraph c
            , setupLoss = l
            , setupInputs = inputDefaults c
            , setupFeeds = [toFeed d (Map.findWithDefault [] (dataName d) (compiledSlots c)) | d <- ds]
            , setupAccuracy = compiledAccuracy c
            }
    it "reaches accuracy 1 and a loss below 1e-3, like the hand-built net" $ do
      src <- Utf8.readFile "examples/xor.hsg"
      let ch = checked "examples/xor.hsg" src
          c = compiledWith Map.empty "examples/xor.hsg" src
      ds <- loaded "examples/xor.hsg" ch
      case compiledLoss c of
        Nothing -> expectationFailure "no loss"
        Just l -> do
          let reports = runReports (train config (setupOf c ds l) (initParams 1337 (compiledParams c)))
              ((hand, pairs), g) = xorNet 4
              handReports =
                runReports
                  (train config (Setup g hand Map.empty [Feed "xor" xorRows (xorSlots 4)] pairs) (initParams 1337 xorSpecs))
          map reportStep reports `shouldBe` [0 .. 199]
          map reportLoss (take 1 reports) `shouldSatisfy` all (> 1)
          map reportLoss (drop 199 reports) `shouldSatisfy` all (< 1e-3)
          map reportAccuracy (drop 199 reports) `shouldBe` [Just 1]
          -- Same graph up to node order, same seed: the same run, up to
          -- the rounding of sums taken in a different order.
          and (zipWith approxLoss reports handReports) `shouldBe` True
          length handReports `shouldBe` 200
  it "compiles moons.hsg at batch size 4 with 647·B + 1015 nodes" $ do
    src <- Utf8.readFile "examples/moons.hsg"
    let at b = compiledWith (Map.singleton "moons" b) "examples/moons.hsg" src
    map (nodeCount . compiledGraph . at) [4, 7] `shouldBe` [647 * 4 + 1015, 647 * 7 + 1015]
    let c = at 4
    Map.keys (compiledSlots c) `shouldBe` ["moons"]
    compiledSlots c `shouldBe` Map.singleton "moons" (slotNames "moons" [("x", Shape.Vec 2), ("y", Shape.Real)] 4)
    length (compiledAccuracy c) `shouldBe` 4
    fmap (nodeCount . predictorGraph) (compiledPredictor c) `shouldBe` Just (337 + 2 + 640)

  it "evaluates moons.hsg's loss on the first rows of the CSV file" $ do
    src <- Utf8.readFile "examples/moons.hsg"
    let ch = checked "examples/moons.hsg" src
        c = compiledWith (Map.singleton "moons" 4) "examples/moons.hsg" src
    ds <- loaded "examples/moons.hsg" ch
    let slots = Map.findWithDefault [] "moons" (compiledSlots c)
        rows = concatMap dataRows (take 1 ds)
        env =
          Env
            { envInputs = Map.fromList (concat (zipWith zip slots rows))
            , envParams = initParams 1337 (compiledParams c)
            }
    case compiledLoss c of
      Nothing -> expectationFailure "no loss"
      Just l -> valuesOf c env (Real l) `shouldSatisfy` all (\v -> v > 0 && v < 10)
  where
    approxLoss a b = abs (reportLoss a - reportLoss b) <= 1e-9 * max 1 (abs (reportLoss b))

-- ---------------------------------------------------------------------------
-- Sharing

sharingSpec :: Spec
sharingSpec = describe "sharing" $ do
  it "compiles a top-level let once, however often it is used" $ do
    let c = compiled "input a : Real = 1\nlet s = exp(a)\nlet y = s + s"
        s = scalar (named "s" c)
    nodeCount (compiledGraph c) `shouldBe` 3
    nodeArgs (nodeAt (compiledGraph c) (scalar (named "y" c))) `shouldBe` [s, s]

  it "compiles a let … in once; its node keeps the bound expression's span" $ do
    let c = compiled "input a : Real = 1\nlet y = let t = tanh(a) in t * t"
        g = compiledGraph c
        y = nodeAt g (scalar (named "y" c))
    nodeCount g `shouldBe` 3
    case nodeArgs y of
      [t, t'] -> do
        t `shouldBe` t'
        provSpan (nodeProv (nodeAt g t)) `shouldBe` Just (SrcSpan "t.hsg" (Pos 2 17) (Pos 2 24))
      args -> expectationFailure (show args)

  it "compiles a function argument once" $ do
    let c = compiled "input a : Real = 1\nfn sq(z) = z * z\nlet y = sq(exp(a))"
    nodeCount (compiledGraph c) `shouldBe` 3

  it "compiles a chain of 60 doublings in linear time" $ do
    -- Substituting each let into its uses would compile 2^60 copies.
    let chain = foldr (\k body -> "let a" <> showT k <> " = a" <> showT (k - 1) <> " + a" <> showT (k - 1) <> " in " <> body) "a60" [1 .. 60 :: Int]
        c = compiled ("input a0 : Real = 1\nlet y = " <> chain)
    nodeCount (compiledGraph c) `shouldBe` 61
    valuesOf c (defaults c) (named "y" c) `shouldBe` [2 ^ (60 :: Int)]

  it "gives two comprehensions over one dataset the same slot leaves" $ do
    let c =
          compiled $
            Text.unlines
              [ "data d : { v : Real } = rows [1, 2, 3]"
              , "let s = sum [ v | v <- d ]"
              , "let m = mean [ v * v | v <- d ]"
              ]
        leaves = [i | (i, n) <- nodes (compiledGraph c), null (nodeArgs n), isInput (nodeOp n)]
        isInput op = case op of
          Leaf (Input _) -> True
          _ -> False
    length leaves `shouldBe` 3
    valuesOf c emptyEnv {envInputs = Map.fromList [("d#0.v", 1), ("d#1.v", 2), ("d#2.v", 3)]} (named "m" c)
      `shouldBe` [14 / 3]
  where
    showT = Text.pack . show

-- ---------------------------------------------------------------------------
-- Lowerings (01-language §8)

loweringSpec :: Spec
loweringSpec = describe "lowering" $ do
  it "makes input and param leaves named by their elements, with the declaration's span" $ do
    let c = compiled "input x : Vec 2 = [1, 2]\nparam W : Mat 2 2 ~ uniform(-1, 1)\nlet y = W @ x"
        g = compiledGraph c
        leafAt i = (nodeOp (nodeAt g i), provSpan (nodeProv (nodeAt g i)))
    map leafAt (valueNodes (named "x" c))
      `shouldBe` [(Leaf (Input n), Just (SrcSpan "t.hsg" (Pos 1 1) (Pos 1 25))) | n <- ["x[0]", "x[1]"]]
    map (fst . leafAt) (valueNodes (named "W" c))
      `shouldBe` map (Leaf . Param) ["W[0,0]", "W[0,1]", "W[1,0]", "W[1,1]"]
    map specName (compiledParams c) `shouldBe` ["W"]
    concatMap elementNames (compiledParams c) `shouldBe` ["W[0,0]", "W[0,1]", "W[1,0]", "W[1,1]"]

  it "makes B slot copies of a comprehension over a dataset" $ do
    let c = compiledWith (Map.singleton "d" 2) "t.hsg" "data d : { x : Vec 2, y : Real } = rows [([1, 2], 3), ([4, 5], 6), ([7, 8], 9)]\nlet s = sum [ x * y | (x, y) <- d ]"
        env = emptyEnv {envInputs = Map.fromList (concat (zipWith zip (slotNames "d" [("x", Shape.Vec 2), ("y", Shape.Real)] 2) [[1, 2, 3], [4, 5, 6]]))}
    compiledSlots c `shouldBe` Map.singleton "d" [["d#0.x[0]", "d#0.x[1]", "d#0.y"], ["d#1.x[0]", "d#1.x[1]", "d#1.y"]]
    -- Slot leaves have the span of the dataset's declaration.
    let declSpan = SrcSpan "t.hsg" (Pos 1 1) (Pos 1 80)
        slotLeafSpans = [provSpan (nodeProv n) | (_, n) <- nodes (compiledGraph c), Leaf (Input _) <- [nodeOp n]]
    slotLeafSpans `shouldBe` replicate 6 (Just declSpan)
    valuesOf c env (named "s" c) `shouldBe` [1 * 3 + 4 * 6, 2 * 3 + 5 * 6]

  it "uses one slot per inline row when no batch size is given" $ do
    let c = compiled "data d : { v : Real } = rows [1, 2, 3]\nlet s = sum [ v | v <- d ]"
    fmap length (Map.lookup "d" (compiledSlots c)) `shouldBe` Just 3

  it "makes one copy per parameter of a comprehension over params, in declaration order" $ do
    let c = compiled "param p : Vec 2 = [1, 2]\nparam q : Real = 3\nlet r = sum [ sum(z ^ 2) | z <- params ]"
        env = emptyEnv {envParams = Map.fromList [("p[0]", 1), ("p[1]", 2), ("q", 3)]}
    valuesOf c env (named "r" c) `shouldBe` [(1 + 4) + 9]

  it "lets a comprehension body see the enclosing locals" $ do
    let c = compiled "data d : { v : Real } = rows [1, 2]\nfn scaled(k : Real) : Real = sum [ k * v | v <- d ]\nlet y = let j = 10 in scaled(j) + mean [ j | v <- d ]"
        env = emptyEnv {envInputs = Map.fromList [("d#0.v", 1), ("d#1.v", 2)]}
    valuesOf c env (named "y" c) `shouldBe` [30 + 10]

  it "inlines function calls with their arguments" $ do
    let c = compiled "input x : Vec 2 = [1, 2]\nfn twice(v : Vec n) : Vec n = v + v\nlet y = twice(twice(x))"
    valuesOf c (defaults c) (named "y" c) `shouldBe` [4, 8]

  it "turns a negative literal into one constant, and negates anything else" $ do
    let c = compiled "input a : Real = 2\nlet k = -2.5\nlet y = -a"
        g = compiledGraph c
    nodeOp (nodeAt g (scalar (named "k" c))) `shouldBe` Leaf (Const (-2.5))
    nodeOp (nodeAt g (scalar (named "y" c))) `shouldBe` Un Neg

  it "labels top-level lets by element, and the loss as loss" $ do
    let c = compiled "param a : Real = 1\nparam b : Real = 2\nlet h = [a * b, a + b]\nloss = h[0] * h[1]"
        labelOf i = provLabel (nodeProv (nodeAt (compiledGraph c) i))
    map labelOf (valueNodes (named "h" c)) `shouldBe` [Just "h[0]", Just "h[1]"]
    fmap labelOf (compiledLoss c) `shouldBe` Just (Just "loss")

  it "gives every node of moons.hsg a span" $ do
    src <- Utf8.readFile "examples/moons.hsg"
    let c = compiledWith (Map.singleton "moons" 2) "moons.hsg" src
    all (isJust . provSpan . nodeProv . snd) (nodes (compiledGraph c)) `shouldBe` True

-- ---------------------------------------------------------------------------
-- Built-in functions (01-language §6), against Haskell

-- | The inputs of the built-in tests and their values.
data Inputs = Inputs {inA, inB :: Double, inU, inV :: [Double], inM :: [[Double]]}
  deriving (Show)

genInputs :: (Double, Double) -> Gen Inputs
genInputs range =
  Inputs
    <$> choose range
    <*> choose range
    <*> vectorOf 3 (choose range)
    <*> vectorOf 3 (choose range)
    <*> vectorOf 2 (vectorOf 3 (choose range))

-- | A program with inputs @a, b : Real@, @u, v : Vec 3@, @M : Mat 2 3@
-- and @let y = <expr>@.
program :: Text -> Compiled
program e =
  compiled $
    Text.unlines
      [ "input a : Real"
      , "input b : Real"
      , "input u : Vec 3"
      , "input v : Vec 3"
      , "input M : Mat 2 3"
      , "let y = " <> e
      ]

inputEnv :: Inputs -> Env
inputEnv i =
  emptyEnv
    { envInputs =
        Map.fromList $
          [("a", inA i), ("b", inB i)]
            <> zip ["u[0]", "u[1]", "u[2]"] (inU i)
            <> zip ["v[0]", "v[1]", "v[2]"] (inV i)
            <> zip [Name ("M[" <> t r <> "," <> t k <> "]") | r <- [0, 1 :: Int], k <- [0, 1, 2 :: Int]] (concat (inM i))
    }
  where
    t = Text.pack . show

-- | @y@ evaluates to what the Haskell function computes.
agrees :: (Double, Double) -> Text -> (Inputs -> [Double]) -> Property
agrees range e expected =
  let c = program e
   in forAll (genInputs range) $ \i ->
        let got = valuesOf c (inputEnv i) (named "y" c)
            want = expected i
         in counterexample (show (got, want)) (length got == length want && and (zipWith approx got want))

relu, step :: Double -> Double
relu x = if x > 0 then x else 0
step x = if x > 0 then 1 else 0

dotL :: [Double] -> [Double] -> Double
dotL us vs = foldl1 (+) (zipWith (*) us vs)

builtinSpec :: Spec
builtinSpec = describe "built-in functions, evaluated against Haskell" $ do
  let anyR = (-3, 3)
      posR = (0.1, 3)
      one f i = [f i]
  prop "exp" $ agrees anyR "exp(a)" (one (exp . inA))
  prop "log" $ agrees posR "log(a)" (one (log . inA))
  prop "tanh" $ agrees anyR "tanh(u)" (map tanh . inU)
  prop "relu" $ agrees anyR "relu(u)" (map relu . inU)
  prop "step" $ agrees anyR "step(a)" (one (step . inA))
  prop "sigmoid" $ agrees anyR "sigmoid(a)" (one (\i -> 1 / (1 + exp (negate (inA i)))))
  prop "sqrt" $ agrees posR "sqrt(a)" (one (sqrt . inA))
  prop "abs" $ agrees anyR "abs(u)" (map abs . inU)
  prop "max" $ agrees anyR "max(a, b)" (one (\i -> max (inA i) (inB i)))
  prop "min" $ agrees anyR "min(u, b)" (\i -> map (`min` inB i) (inU i))
  prop "^ with integer, negative and fractional exponents" $
    agrees posR "[a ^ 3, a ^ -1, a ^ 0.5]" (\i -> [inA i ^ (3 :: Int), 1 / inA i, sqrt (inA i)])
  prop "elementwise + - * / on vectors" $
    agrees anyR "(u + v) * (u - v) / (v * v + 1)" (\i -> zipWith (\x y -> (x + y) * (x - y) / (y * y + 1)) (inU i) (inV i))
  prop "broadcasting a Real, in operand order" $
    agrees anyR "[(b - u)[0], (u - b)[0], (M * a)[1, 2]]" (\i -> [inB i - head3 (inU i), head3 (inU i) - inB i, (inM i !! 1 !! 2) * inA i])
  prop "sum and mean of a vector, a matrix and a Real" $
    agrees anyR "[sum(u), mean(u), sum(M), mean(M), sum(a), mean(a)]" $ \i ->
      let ms = concat (inM i)
       in [foldl1 (+) (inU i), foldl1 (+) (inU i) / 3, foldl1 (+) ms, foldl1 (+) ms / 6, inA i, inA i]
  prop "dot" $ agrees anyR "dot(u, v)" (one (\i -> dotL (inU i) (inV i)))
  prop "matrix-vector product" $ agrees anyR "M @ u" (\i -> map (`dotL` inU i) (inM i))
  prop "matrix-matrix product and transpose" $
    agrees anyR "M @ transpose(M)" (\i -> [dotL r s | r <- inM i, s <- inM i])
  prop "transpose and indexing" $
    agrees anyR "[transpose(M)[2, 1], M[1][0], transpose(M)[0][1]]" (\i -> [inM i !! 1 !! 2, inM i !! 1 !! 0, inM i !! 1 !! 0])
  prop "vector and matrix literals" $
    agrees anyR "[[a, b * 2], [-a, 3]] @ [b, a]" (\i -> [inA i * inB i + inB i * 2 * inA i, negate (inA i) * inB i + 3 * inA i])
  prop "transpose of a literal" $
    agrees anyR "sum(transpose([[a, b, 1]]) @ [b])" (one (\i -> foldl1 (+) [inA i * inB i, inB i * inB i, inB i]))

  describe "desugarings build exactly the formulas of §6" $ do
    let same e hand =
          let c = program e
              (h, g) = built hand
           in sameExpressions (valueNodes (named "y" c), compiledGraph c) ([h], g)
        a = input "a"
        b = input "b"
    it "sigmoid(e) = 1 / (1 + exp(-e))" $
      same "sigmoid(a)" (do k <- constant 1; e <- a >>= unary Neg >>= unary Exp; binary Add k e >>= binary Div k) `shouldBe` True
    it "sqrt(e) = e ^ 0.5" $
      same "sqrt(a)" (a >>= unary (PowC (1 / 2))) `shouldBe` True
    it "abs(e) = relu(e) + relu(-e)" $
      same "abs(a)" (do x <- a; r <- unary Relu x; r' <- unary Neg x >>= unary Relu; binary Add r r') `shouldBe` True
    it "max(a, b) = a + relu(b - a)" $
      same "max(a, b)" (do x <- a; y <- b; binary Sub y x >>= unary Relu >>= binary Add x) `shouldBe` True
    it "min(a, b) = a - relu(a - b)" $
      same "min(a, b)" (do x <- a; y <- b; binary Sub x y >>= unary Relu >>= binary Sub x) `shouldBe` True
    it "mean(v) = sum(v) / count, the sum a chain of + from the left" $
      same "mean(u)" (do xs <- mapM input ["u[0]", "u[1]", "u[2]"]; s <- foldM (binary Add) (head3 xs) (drop 1 xs); constant 3 >>= binary Div s) `shouldBe` True
    it "dot(u, v) = Σ u[i]·v[i]" $
      same "dot(u, v)" (do us <- mapM input ["u[0]", "u[1]", "u[2]"]; vs <- mapM input ["v[0]", "v[1]", "v[2]"]; ps <- zipWithM (binary Mul) us vs; foldM (binary Add) (head3 ps) (drop 1 ps)) `shouldBe` True

  it "makes no nodes for transpose, indexing, sum and mean of a Real" $ do
    let c = program "transpose(M)[2, 1] + sum(a) + mean(b)"
    -- 2 + 3 + 3 + 6 leaves, then two additions.
    nodeCount (compiledGraph c) `shouldBe` 14 + 2
  where
    head3 xs = case xs of
      x : _ -> x
      [] -> error "empty"

-- ---------------------------------------------------------------------------
-- Handles

handleSpec :: Spec
handleSpec = describe "the compiled program's handles" $ do
  it "describes the inputs, with their defaults by leaf name" $ do
    let c = compiled "input a : Real = 2\ninput x : Vec 2\ninput M : Mat 1 2 = [[3, 4]]"
    map inputSpecName (compiledInputs c) `shouldBe` ["a", "x", "M"]
    map inputSpecLeaves (compiledInputs c) `shouldBe` [["a"], ["x[0]", "x[1]"], ["M[0,0]", "M[0,1]"]]
    map inputSpecDefault (compiledInputs c) `shouldBe` [Just [2], Nothing, Just [3, 4]]
    inputDefaults c `shouldBe` Map.fromList [("a", 2), ("M[0,0]", 3), ("M[0,1]", 4)]

  it "describes the parameters as Train.Param specs, explicit values included" $ do
    let c = compiled "param w : Vec 2 = [1, 2]\nparam b : Real = 0.5\nparam W : Mat 1 2 ~ normal(0, 2)\nlet y = dot(w, W[0]) + b"
    map specInit (compiledParams c) `shouldBe` [Values [1, 2], Constant 0.5, Normal 0 2]
    initParams 7 (take 2 (compiledParams c)) `shouldBe` Map.fromList [("w[0]", 1), ("w[1]", 2), ("b", 0.5)]

  it "compiles the model into a predictor whose arguments are its own input leaves" $ do
    c <- example "xor.hsg"
    case compiledPredictor c of
      Nothing -> expectationFailure "no predictor"
      Just p -> do
        predictorName p `shouldBe` "net"
        map inputSpecLeaves (predictorArguments p) `shouldBe` [["net.x[0]", "net.x[1]"]]
        predictorSlots p `shouldBe` Map.empty
        -- On every row, the predictor gives what the training graph
        -- predicts in that row's slot.
        let ps = initParams 3 (compiledParams c)
            out = scalar (predictorOutput p)
            predicted row = case evaluate (predictorGraph p) (Env (Map.fromList (zip ["net.x[0]", "net.x[1]"] row)) ps) [out] of
              Right vals -> valueAt vals out
              Left err -> error (show err)
            slotEnv = Env (Map.fromList (concat (zipWith zip (xorSlots 4) xorRows))) ps
            inSlots = [valuesOf c slotEnv (Real pr) | (pr, _) <- compiledAccuracy c]
        map (\r -> [predicted (take 2 r)]) xorRows `shouldBe` inSlots

  it "takes the first model call of a slot as its prediction" $ do
    let c =
          compiled $
            Text.unlines
              [ "data d : { x : Real, y : Real } = rows [(1, 1), (2, -1)]"
              , "param w : Real = 1"
              , "model m(x : Real) : Real = w * x"
              , "loss = mean [ (m(x) - y) ^ 2 + 0 * m(3) | (x, y) <- d ]"
              ]
        preds = map fst (compiledAccuracy c)
    length preds `shouldBe` 2
    length (Map.fromList [(p, ()) | p <- preds]) `shouldBe` 2

  it "only pairs predictions that the loss depends on" $ do
    let c =
          compiled $
            Text.unlines
              [ "data d : { x : Real, y : Real } = rows [(1, 1), (2, -1)]"
              , "param w : Real = 1"
              , "model m(x : Real) : Real = w * x"
              , "let p = sum [ m(x) | (x, y) <- d ]"
              , "loss = sum [ w * y | (x, y) <- d ]"
              ]
    compiledAccuracy c `shouldBe` []
    fmap length (Map.lookup "d" (compiledSlots c)) `shouldBe` Just 2

  it "compiles the lets the model uses into the predictor graph" $ do
    let c = compiled "input k : Real = 2\nparam w : Real = 3\nlet s = w * k\nmodel m(x : Real) : Real = s * x"
    case compiledPredictor c of
      Nothing -> expectationFailure "no predictor"
      Just p -> do
        let out = scalar (predictorOutput p)
        fmap (`valueAt` out) (evaluate (predictorGraph p) (Env (Map.fromList [("k", 2), ("m.x", 5)]) (Map.singleton "w" 3)) [out])
          `shouldBe` Right 30

  it "says when a CSV dataset has no batch size, or a bad one" $ do
    src <- Utf8.readFile "examples/moons.hsg"
    let c = checked "examples/moons.hsg" src
    compileProgram Map.empty c `shouldSatisfy` either (== MissingBatchSize "moons") (const False)
    compileProgram (Map.singleton "moons" 0) c `shouldSatisfy` either (== BadBatchSize "moons" 0) (const False)

  it "computes batch sizes from --batch or the datasets' sizes" $ do
    src <- Utf8.readFile "examples/xor.hsg"
    ds <- loaded "examples/xor.hsg" (checked "examples/xor.hsg" src)
    batchSizes Nothing ds `shouldBe` Map.singleton "xor" 4
    batchSizes (Just 2) ds `shouldBe` Map.singleton "xor" 2

  it "describes values by shape" $ do
    let c = program "M"
    valueShape (named "M" c) `shouldBe` Shape.Mat 2 3
    length (valueNodes (named "M" c)) `shouldBe` 6
