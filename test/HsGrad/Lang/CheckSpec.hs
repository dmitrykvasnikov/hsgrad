{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Check".
--
-- * Examples: the four programs of @docs/01-language.md@ §1 check without
--   any diagnostic, with the expected types, and their typed trees have
--   the expected structure.
-- * Names: define before use, @params@, shadowing, unused names,
--   suggestions.
-- * Functions (D8): concrete functions checked at their declaration,
--   generic ones at their calls, once per instance; annotations stop
--   cascades.
-- * W0203.
-- * Invariants on mutated example programs: the checker never crashes,
--   returns a checked program exactly when there is no error, and every
--   node of that program has the shape the rules give it ('lint').
-- * Golden files in @test/golden/check/@: every E02xx, E03xx and W02xx
--   code, and a file with several errors and no cascades.
module HsGrad.Lang.CheckSpec
  ( spec
  ) where

import Control.Monad (foldM)
import Data.Foldable (toList)
import Data.List (isPrefixOf)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (isJust)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Lang.Check
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Parser (parseProgram)
import HsGrad.Lang.Shape
import HsGrad.Lang.Span (Pos (..), SrcSpan (..))
import HsGrad.Lang.Syntax (Operator (..), isIdentChar, isReserved)
import HsGrad.Test.Golden (goldenFiles)
import System.Directory (listDirectory)
import System.FilePath (takeExtension, (</>))
import Test.Hspec (Spec, describe, expectationFailure, it, runIO, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (modifyMaxSuccess, prop)
import Test.QuickCheck (Gen, Property, chooseInt, counterexample, discard, elements, forAll, oneof, (.&&.), (===))

spec :: Spec
spec = do
  describe "examples" exampleSpec
  describe "names" nameSpec
  describe "functions" functionSpec
  describe "W0203" influenceSpec
  describe "suggestions" suggestionSpec
  describe "invariants" invariantSpec
  describe "errors" errorSpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Parse (which must succeed) and check.
check :: Text -> ([Diagnostic], Maybe Checked)
check src = case parseProgram "t.hsg" src of
  Left ds -> error ("parse error in test program: " <> show (toList ds))
  Right p -> checkProgram p

-- | The codes of a program's diagnostics, in source order.
codes :: Text -> [Code]
codes = map diagCode . fst . check

-- | The checked program of a program that must check.
checked :: Text -> Checked
checked src = case check src of
  (_, Just c) -> c
  (ds, Nothing) -> error ("test program doesn't check: " <> show (map diagCode ds))

-- | Render a file's diagnostics the way the driver will.
rendered :: FilePath -> Text -> Text
rendered file src = case parseProgram file src of
  Left ds -> renderDiagnostics defaultRenderOptions (Map.singleton file src) (toList ds)
  Right p -> case checkProgram p of
    ([], _) -> "checked without diagnostics\n"
    (ds, _) -> renderDiagnostics defaultRenderOptions (Map.singleton file src) ds

readExample :: FilePath -> IO Text
readExample name = Utf8.readFile ("examples" </> name)

-- ---------------------------------------------------------------------------
-- Examples

exampleSpec :: Spec
exampleSpec = do
  it "checks all four example programs without any diagnostic" $
    mapM_
      ( \name -> do
          src <- readExample name
          rendered name src `shouldBe` "checked without diagnostics\n"
      )
      ["intro.hsg", "neuron.hsg", "xor.hsg", "moons.hsg"]

  it "types intro.hsg" $ do
    src <- readExample "intro.hsg"
    describeChecked (checked src)
      `shouldBe` ["input a : Real", "input b : Real", "input c : Real", "let e : Real", "let d : Real", "let L : Real"]

  it "types neuron.hsg, with one instance of neuron" $ do
    src <- readExample "neuron.hsg"
    describeChecked (checked src)
      `shouldBe` [ "input x : Vec 3"
                 , "param w : Vec 3"
                 , "param b : Real"
                 , "fn neuron(w : Vec n, b : Real, x : Vec n) : Real"
                 , "  at Vec 3, Real, Vec 3 → Real"
                 , "let y : Real"
                 ]

  it "types xor.hsg: net : Vec 2 → Real, loss : Real" $ do
    src <- readExample "xor.hsg"
    describeChecked (checked src)
      `shouldBe` [ "data xor : { x : Vec 2, y : Real }"
                 , "param W1 : Mat 4 2"
                 , "param b1 : Vec 4"
                 , "param w2 : Vec 4"
                 , "param b2 : Real"
                 , "model net : Vec 2 → Real"
                 , "loss : Real"
                 ]

  it "types moons.hsg: mlp : Vec 2 → Real, three instances of layer" $ do
    src <- readExample "moons.hsg"
    let c = checked src
    describeChecked c `shouldSatisfy` elem "model mlp : Vec 2 → Real"
    describeChecked c `shouldSatisfy` elem "loss : Real"
    map instanceArguments (concatMap functionInstances (checkedFunctions c))
      `shouldBe` [[Mat 1 16, Vec 1, Vec 16], [Mat 16 2, Vec 16, Vec 2], [Mat 16 16, Vec 16, Vec 16]]

  it "resolves the names of intro.hsg" $ do
    src <- readExample "intro.hsg"
    map (texprNode . letExpr) (take 1 (checkedLets (checked src)))
      `shouldBe` [NBinary Times (TExpr (at 6 9 10) Real (NRef (RInput "a"))) (TExpr (at 6 13 14) Real (NRef (RInput "b")))]

  it "keeps parameters' shapes and initialisers, and inputs' defaults" $ do
    xor <- readExample "xor.hsg"
    [(paramName p, paramShape p, paramInit p) | p <- checkedParams (checked xor)]
      `shouldBe` [ ("W1", Mat 4 2, InitUniform (-1) 1)
                 , ("b1", Vec 4, InitFill 0)
                 , ("w2", Vec 4, InitUniform (-1) 1)
                 , ("b2", Real, InitFill 0)
                 ]
    neuron <- readExample "neuron.hsg"
    map inputDefault (checkedInputs (checked neuron)) `shouldBe` [Just [1, -2, 0.5]]

  it "keeps the dataset schemas, and the model's signature and instance" $ do
    src <- readExample "xor.hsg"
    let c = checked src
    map datasetFields (checkedDatasets c) `shouldBe` [[("x", Vec 2), ("y", Real)]]
    case checkedModel c of
      Just m -> do
        (modelName m, modelParams m, modelResult m) `shouldBe` ("net", [("x", Vec 2)], Real)
        fmap instanceParams (Map.lookup (modelInstance m) (checkedInstances c)) `shouldBe` Just ["x"]
      Nothing -> expectationFailure "no model"

  it "types the loss of moons.hsg: a mean over the dataset plus a sum over params" $ do
    src <- readExample "moons.hsg"
    case fmap (texprNode . lossExpr) (checkedLoss (checked src)) of
      Just (NLet "data_loss" (TExpr _ Real (NReduce BMean (OverData "moons" ["x", "y"] _))) (TExpr _ Real (NLet "reg" reg _))) ->
        case texprNode reg of
          NBinary Times _ (TExpr _ Real (NReduce BSum (OverParams "p" bodies))) ->
            [(p, texprShape b) | (p, b) <- bodies] `shouldBe` [(p, Real) | p <- ["W1", "b1", "W2", "b2", "W3", "b3"]]
          other -> expectationFailure ("unexpected reg: " <> show other)
      other -> expectationFailure ("unexpected loss: " <> show other)
  where
    at l c1 c2 = SrcSpan "t.hsg" (Pos l c1) (Pos l c2)

-- ---------------------------------------------------------------------------
-- Names

nameSpec :: Spec
nameSpec = do
  it "requires declarations before use, also in function bodies" $ do
    codes "let y = x\ninput x : Real = 1" `shouldBe` [E0203]
    codes "fn f(a) = a * x\ninput x : Real = 1\nlet y = f(x)" `shouldBe` [E0203]

  it "reports recursion as use before declaration" $
    codes "fn f(a : Real) : Real = f(a)\nlet y = f(1)" `shouldBe` [E0203]

  it "lets params mean every parameter, even those declared later" $
    codes "let r = sum [ sum(p) | p <- params ]\nparam w : Vec 2 = 0\nparam b : Real = 0" `shouldBe` []

  it "lets function parameters reuse global names without a warning" $
    codes "input x : Real = 1\nfn f(x) = x * 2\nlet y = f(x)" `shouldBe` []

  it "warns when a let or a comprehension variable shadows a name" $ do
    codes "input x : Real = 1\nlet y = let x = 2 in x" `shouldBe` [W0202, W0201]
    codes "input y : Real = 1\ndata d : { x : Real, y : Real } = rows [ (1, 2) ]\nlet s = sum [ x * y | (x, y) <- d ]" `shouldBe` [W0202, W0201]

  it "allows a local to shadow a built-in function, but not to be called" $ do
    codes "let y = let exp = 2 in exp" `shouldBe` [W0201]
    codes "let y = let exp = 2 in exp(1)" `shouldBe` [W0201, E0204]

  it "doesn't report names starting with an underscore as unused" $
    codes "input _x : Real = 1\nlet y = let _t = 2 in 3" `shouldBe` []

  it "never reports a top-level let or the model as unused" $
    codes "let y = 1\nmodel m(x : Real) : Real = x" `shouldBe` []

  it "reports a top-level name that repeats a built-in function" $
    codes "let tanh = 1" `shouldBe` [E0202]

  it "rejects unknown initialisers" $
    codes "param w : Real ~ gauss(0, 1)\nlet y = w" `shouldBe` [E0201]

  it "fills a parameter from a single number, or takes every value" $ do
    map paramInit (checkedParams (checked "param w : Mat 2 2 = 1.5\nlet y = w")) `shouldBe` [InitFill 1.5]
    map paramInit (checkedParams (checked "param w : Mat 2 2 = [[1, 2], [3, 4]]\nlet y = w")) `shouldBe` [InitValues [1, 2, 3, 4]]

-- ---------------------------------------------------------------------------
-- Functions

functionSpec :: Spec
functionSpec = do
  it "checks a concrete function at its declaration, even if it is never called" $
    codes "fn f(x : Vec 2) : Real = x" `shouldBe` [W0202, E0308]

  it "checks a generic function's shapes only at its calls (D8)" $ do
    codes "fn f(x) = x @ x" `shouldBe` [W0202]
    codes "fn f(x) = x @ x\nlet y = f([1, 2])" `shouldBe` [E0302]

  it "reports errors that don't depend on the shape variables at the declaration" $ do
    codes "fn f(x) = [1, 2] + [1, 2, 3] + x" `shouldBe` [W0202, E0301]
    codes "fn f(x) = x + nope\nlet y = f(1)" `shouldBe` [E0201]

  it "checks each instance once" $ do
    codes "fn f(x) = x @ x\nlet a = f([1, 2])\nlet b = f([3, 4])" `shouldBe` [E0302]
    codes "fn f(x) = x @ x\nlet a = f([1, 2])\nlet b = f([3, 4, 5])" `shouldBe` [E0302, E0302]

  it "notes the calls that led to an error in an instance" $ do
    let (ds, _) = check "fn f(x) = x @ x\nfn g(x) = f(x)\nlet a = g([1, 2])"
    map diagNotes ds
      `shouldBe` [["while checking the call at t.hsg:2:11", "while checking the call at t.hsg:3:9"]]

  it "reports a name problem in a generic function once, whatever its instances" $ do
    codes "fn f(x) = let exp = x in exp\nlet a = f(1)\nlet b = f([1, 2])" `shouldBe` [W0201]
    codes "param a : Real = 1\nparam b : Vec 2 = 0\nlet r = sum [ sum(let exp = p in exp) | p <- params ]" `shouldBe` [W0201]

  it "checks the body against a result variable bound by the arguments" $
    codes "param W : Mat 4 3 ~ uniform(-1, 1)\nfn f(v : Vec n) : Vec n = W @ v\nlet y = f([1, 2, 3])" `shouldBe` [E0308]

  it "types the calls of a broken function by its result annotation" $
    codes "fn f(v : Vec 3) : Vec 4 = v @ v\nlet r = f([1, 2, 3]) + [1, 2]" `shouldBe` [E0302, E0301]

  it "types a broken let by its annotation" $
    codes "let k : Vec 3 = nope\nlet q = k + [1, 2]" `shouldBe` [E0201, E0301]

  it "binds a function's shape variables in local annotations" $ do
    codes "fn f(x : Vec n) : Real = let y : Vec n = x * 2 in sum(y)\nlet a = f([1, 2])" `shouldBe` []
    codes "fn f(x : Vec n) : Real = let y : Vec n = [1, 2, 3] in sum(x * y)\nlet a = f([1, 2])" `shouldBe` [E0308]

  it "gives a result variable its value from the body" $
    describeChecked (checked "fn f(x) : Vec m = [x, x]\nlet a = f(1)") `shouldSatisfy` elem "  at Real → Vec 2"

  it "rejects nested comprehensions over datasets, also through calls" $
    codes "data d : { x : Real } = rows [ 1, 2 ]\nfn g(a) = a * mean [ x | x <- d ]\nlet s = mean [ g(x) | x <- d ]" `shouldBe` [E0314]

-- ---------------------------------------------------------------------------
-- W0203

influenceSpec :: Spec
influenceSpec = do
  it "is silent for the examples" $ do
    srcs <- mapM readExample ["xor.hsg", "moons.hsg"]
    concatMap codes srcs `shouldBe` []

  it "warns about a used parameter that the loss doesn't depend on" $
    codes "param a : Real = 1\nparam b : Real = 2\nlet shown = b\nloss = a ^ 2" `shouldBe` [W0203]

  it "follows lets, calls and let-in locals" $ do
    codes "param a : Real = 1\nfn f(x) = x * 2\nlet h = f(a)\nloss = let t = h in t ^ 2" `shouldBe` []
    codes "param a : Real = 1\nfn f(x, _y) = x\nlet h = f(1, a)\nloss = h ^ 2" `shouldBe` [W0203]

  it "counts a comprehension over params only where the body uses the variable" $ do
    codes "param a : Real = 1\nloss = sum [ p ^ 2 | p <- params ]" `shouldBe` []
    codes "param a : Real = 1\nloss = sum [ 1 | _p <- params ]" `shouldBe` [W0203]

  it "says nothing without a loss" $
    codes "param a : Real = 1\nlet y = a" `shouldBe` []

-- ---------------------------------------------------------------------------
-- Suggestions

suggestionSpec :: Spec
suggestionSpec = do
  it "counts edits, with a swap of neighbours as one" $ do
    editDistance "sigmod" "sigmoid" `shouldBe` 1
    editDistance "tnah" "tanh" `shouldBe` 1
    editDistance "kitten" "sitting" `shouldBe` 3
    editDistance "" "abc" `shouldBe` 3

  prop "is symmetric and zero only for equal texts" $
    forAll genWord $ \a -> forAll genWord $ \b ->
      (editDistance a b === editDistance b a) .&&. ((editDistance a b == 0) === (a == b))

  it "suggests a close name, and nothing for a far one" $ do
    suggestName "sigmod" ["exp", "sigmoid", "sum"] `shouldBe` Just "sigmoid"
    suggestName "wieghts" ["weights", "w"] `shouldBe` Just "weights"
    suggestName "frobnicate" ["exp", "x"] `shouldBe` Nothing

  it "doesn't suggest a different one-letter name" $
    suggestName "y" ["x"] `shouldBe` Nothing
  where
    genWord = Text.pack <$> (chooseInt (0, 6) >>= \n -> mapM (const (elements "abc")) [1 .. n])

-- ---------------------------------------------------------------------------
-- Invariants

invariantSpec :: Spec
invariantSpec = do
  examples <- runIO (mapM readExample ["intro.hsg", "neuron.hsg", "xor.hsg", "moons.hsg"])
  modifyMaxSuccess (const 400) $
    prop "on mutated examples: no crash, a checked program exactly without errors, and it passes lint" $
      forAll (elements examples >>= mutant) invariants

  it "lints the examples" $
    mapM_ (\src -> lint (checked src) `shouldBe` []) examples

-- | The properties every result of the checker has.
invariants :: Text -> Property
invariants src = case parseProgram "t.hsg" src of
  Left _ -> discard
  Right p ->
    let (ds, c) = checkProgram p
        out = renderDiagnostics defaultRenderOptions (Map.singleton "t.hsg" src) ds
     in counterexample (Text.unpack src <> "\n" <> Text.unpack out) $
          (isJust c === not (hasErrors ds))
            .&&. (Text.length out >= 0)
            .&&. (maybe [] lint c === [])
            .&&. (maybe 0 (length . describeChecked) c >= 0)

-- | A program with one to three random edits: delete a line, swap two
-- lines, replace a name by another name of the file, change a number.
mutant :: Text -> Gen Text
mutant src = do
  k <- chooseInt (1, 3)
  foldM (\t _ -> mutate t) src [1 .. k]
  where
    mutate t = do
      let ls = Text.lines t
          names = [w | w <- concatMap tokens ls, isName w]
      i <- chooseInt (0, length ls - 1)
      j <- chooseInt (0, length ls - 1)
      change <- oneof [pure 0, pure 1, pure 2, pure (3 :: Int)]
      case change of
        0 -> pure (Text.unlines [l | (n, l) <- zip [0 ..] ls, n /= i])
        1 -> pure (Text.unlines [pick n l | (n, l) <- zip [0 ..] ls])
          where
            pick n l
              | n == i = ls !! j
              | n == j = ls !! i
              | otherwise = l
        2 | not (null names) -> do
          new <- elements names
          line <- replaceOne isName new (ls !! i)
          pure (Text.unlines [if n == i then line else l | (n, l) <- zip [0 ..] ls])
        _ -> do
          new <- Text.pack . show <$> chooseInt (0, 4)
          line <- replaceOne (Text.all (`elem` ['0' .. '9'])) new (ls !! i)
          pure (Text.unlines [if n == i then line else l | (n, l) <- zip [0 ..] ls])
    tokens = Text.groupBy (\a b -> isIdentChar a == isIdentChar b)
    isName w = case Text.uncons w of
      Just (c, _) -> c `elem` (['a' .. 'z'] <> ['A' .. 'Z'] <> "_") && not (isReserved w)
      Nothing -> False
    replaceOne p new line = do
      let ts = tokens line
          candidates = [n | (n, w) <- zip [0 :: Int ..] ts, p w, not (Text.null w)]
      if null candidates
        then pure line
        else do
          n <- elements candidates
          pure (Text.concat [if m == n then new else w | (m, w) <- zip [0 ..] ts])

-- | Problems with a checked program: nodes whose stored shape is not the
-- one the shape rules give them from their parts, references to unknown
-- names, calls of missing instances. The compiler relies on there being
-- none.
lint :: Checked -> [Text]
lint c =
  concatMap decl (checkedDecls c)
    <> concat [inst k i | (k, i) <- Map.toList (checkedInstances c)]
  where
    globals =
      Map.fromList $
        [(inputName i, inputShape i) | i <- checkedInputs c]
          <> [(paramName p, paramShape p) | p <- checkedParams c]
          <> [(letName l, letShape l) | l <- checkedLets c]
    datasets = Map.fromList [(datasetName d, map snd (datasetFields d)) | d <- checkedDatasets c]
    params = [(paramName p, paramShape p) | p <- checkedParams c]

    decl = \case
      CLet l -> expect "let" (letShape l) (letExpr l) <> expr Map.empty (letExpr l)
      CLoss l -> expect "loss" Real (lossExpr l) <> expr Map.empty (lossExpr l)
      CModel m -> [Text.pack ("missing model instance " <> show (modelInstance m)) | not (Map.member (modelInstance m) (checkedInstances c))]
      CInput i -> [inputName i <> ": wrong default" | Just vs <- [inputDefault i], length vs /= shapeSize (inputShape i)]
      CParam p -> [paramName p <> ": wrong initial values" | InitValues vs <- [paramInit p], length vs /= shapeSize (paramShape p)]
      _ -> []

    inst k i =
      expect "instance result" (instanceResult i) (instanceBody i)
        <> [Text.pack ("instance arity " <> show k) | length (instanceParams i) /= length (instanceArguments k)]
        <> expr (Map.fromList (zip (instanceParams i) (instanceArguments k))) (instanceBody i)

    expect what s e = [what <> ": " <> prettyShape s <> " /= " <> prettyShape (texprShape e) | s /= texprShape e]

    expr :: Map Text (Shape Int) -> TExpr -> [Text]
    expr locals e@(TExpr _ s n) = here <> children
      where
        sh = texprShape
        here = case rule of
          Just s' | s' == s -> []
          other -> [Text.pack ("bad node " <> show (fmap prettyShape other) <> " for " <> take 200 (show e))]
        rule = case n of
          NNum _ -> Just Real
          NRef (RLocal x) -> Map.lookup x locals
          NRef r -> Map.lookup (refName r) globals
          NVector es -> case map sh es of
            ss@(Real : _) | all (== Real) ss -> Just (Vec (length ss))
            ss@(Vec k : _) | all (== Vec k) ss -> Just (Mat (length ss) k)
            _ -> Nothing
          NIndex x i j -> either (const Nothing) Just (indexShape (sh x) i j)
          NNegate x -> Just (sh x)
          NPower x _ -> Just (sh x)
          NBinary MatMul l r -> matMul (sh l) (sh r)
          NBinary _ l r -> broadcast (sh l) (sh r)
          NBuiltin b args -> either (const Nothing) Just (applyBuiltin b (map (Value . sh) args))
          NReduce b (OverData d xs body)
            | fmap length (Map.lookup d datasets) /= Just (length xs) -> Nothing
            | otherwise -> either (const Nothing) Just (applyBuiltin b [Seq (sh body)])
          NReduce b (OverParams _ bodies)
            | map fst bodies /= map fst params -> Nothing
            | otherwise -> case map (sh . snd) bodies of
                ss@(t : _) | all (== t) ss -> either (const Nothing) Just (applyBuiltin b [Seq t])
                _ -> Nothing
          NCall k args
            | map sh args /= instanceArguments k -> Nothing
            | otherwise -> instanceResult <$> Map.lookup k (checkedInstances c)
          NLet _ _ body -> Just (sh body)
        children = case n of
          NVector es -> concatMap (expr locals) es
          NIndex x _ _ -> expr locals x
          NNegate x -> expr locals x
          NPower x _ -> expr locals x
          NBinary _ l r -> expr locals l <> expr locals r
          NBuiltin _ args -> concatMap (expr locals) args
          NReduce _ (OverData d xs body) ->
            expr (Map.union (Map.fromList (zip xs (Map.findWithDefault [] d datasets))) locals) body
          NReduce _ (OverParams x bodies) ->
            concat [expr (Map.insert x ps locals) body | ((_, body), (_, ps)) <- zip bodies params]
          NCall _ args -> concatMap (expr locals) args
          NLet x b body -> expr locals b <> expr (Map.insert x (sh b) locals) body
          _ -> []

-- ---------------------------------------------------------------------------
-- Errors

errorSpec :: Spec
errorSpec = do
  describe "golden" $
    goldenFiles "test/golden/check" ".hsg" $ \file src -> pure (rendered file src)

  it "has a golden file for every name, shape and warning code" $ do
    files <- filter ((== ".err") . takeExtension) <$> listDirectory "test/golden/check"
    outputs <- mapM (Utf8.readFile . ("test/golden/check" </>)) files
    let mentioned code = any (\o -> any (`Text.isInfixOf` o) ["error[" <> codeText code <> "]", "warning[" <> codeText code <> "]"]) outputs
        wanted = [c | c <- [minBound .. maxBound], any (`isPrefixOf` show c) ["E02", "E03", "W02"]]
    filter (not . mentioned) wanted `shouldBe` []

  it "reproduces the hand-built E0201 diagnostic of step 10" $ do
    src <- Utf8.readFile "test/golden/diagnostics/tabs.hsg"
    expected <- Utf8.readFile "test/golden/diagnostics/tabs.err"
    rendered "tabs.hsg" src `shouldBe` expected

  it "reports errors in source order" $ do
    let (ds, _) = check "let a = x\nlet b = [1, 2] + [1]\nlet c = y"
    map diagCode ds `shouldBe` [E0201, E0301, E0201]
