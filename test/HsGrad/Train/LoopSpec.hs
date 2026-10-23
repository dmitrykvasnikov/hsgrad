{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Train.Loop".
--
-- * The milestone: a 2-4-1 tanh MLP, built with the Core 'Builder' the way
--   the compiler will build @xor.hsg@ (dataset rows enter through slots,
--   D10), learns XOR: the loss goes down and all four points end up on
--   the right side of zero.
-- * Mini-batches: 'sampleBatch' is deterministic, picks distinct rows in
--   range, and the full batch is every row in order; whole runs are
--   deterministic per seed.
-- * The loop's bookkeeping: step numbers, learning rates, loss before the
--   update, feeds that cannot work, a step that fails.
module HsGrad.Train.LoopSpec
  ( spec
  ) where

import Control.Monad (foldM, forM, replicateM, zipWithM)
import Control.Monad.State.Strict (evalState, runState)
import Data.List (nub)
import Data.Map.Strict qualified as Map
import Data.String (fromString)
import HsGrad.Core.Backprop (GradError (..))
import HsGrad.Core.Eval (Env (..), EvalError (..), bindInput, emptyEnv, evaluate, valueAt)
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Test.Graphs (built)
import HsGrad.Train.Loop
import HsGrad.Train.Optim
import HsGrad.Train.Param
import Test.Hspec (Spec, describe, expectationFailure, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Positive (..), choose, forAll, (.&&.), (===))

-- ---------------------------------------------------------------------------
-- xor.hsg, by hand

-- | The parameters of @xor.hsg@ (@docs/01-language.md@ §1.3).
w1Spec, b1Spec, w2Spec, b2Spec :: ParamSpec
w1Spec = ParamSpec "W1" (Matrix 4 2) (Uniform (-1) 1)
b1Spec = ParamSpec "b1" (Vector 4) (Constant 0)
w2Spec = ParamSpec "w2" (Vector 4) (Uniform (-1) 1)
b2Spec = ParamSpec "b2" Scalar (Constant 0)

xorSpecs :: [ParamSpec]
xorSpecs = [w1Spec, b1Spec, w2Spec, b2Spec]

-- | The dataset: inputs in {0, 1}², label -1 or 1. Columns @x[0]@, @x[1]@,
-- @y@.
xorRows :: [[Double]]
xorRows = [[0, 0, -1], [0, 1, 1], [1, 0, 1], [1, 1, -1]]

-- | The leaf name of a column in slot @i@ (D10): @xor#2.x[0]@.
slot :: Int -> String -> Name
slot i col = fromString ("xor#" <> show i <> "." <> col)

-- | The slots of a batch of @b@ rows, columns in row order.
xorSlots :: Int -> [[Name]]
xorSlots b = [[slot i "x[0]", slot i "x[1]", slot i "y"] | i <- [0 .. b - 1]]

-- | The graph of @xor.hsg@ with @b@ dataset slots:
--
-- > model net(x : Vec 2) : Real = dot(w2, tanh(W1 @ x + b1)) + b2
-- > loss = mean [ (net(x) - y) ^ 2 | (x, y) <- xor ]
--
-- Returns the loss node and, per slot, the prediction and label nodes.
xorNet :: Int -> ((NodeId, [(NodeId, NodeId)]), Graph)
xorNet b = built $ do
  w1 <- leaves w1Spec
  b1 <- leaves b1Spec
  w2 <- leaves w2Spec
  b2 <- param "b2"
  let rows = [take 2 (drop (2 * r) w1) | r <- [0 .. 3]]
  perSlot <- forM [0 .. b - 1] $ \i -> do
    x <- mapM input [slot i "x[0]", slot i "x[1]"]
    y <- input (slot i "y")
    h <- forM (zip rows b1) $ \(wr, bi) -> do
      s <- dot wr x
      binary Add s bi >>= unary Tanh
    out <- dot w2 h
    p <- binary Add out b2
    sq <- binary Sub p y >>= unary (PowC 2)
    pure ((p, y), sq)
  total <- sumOf (map snd perSlot)
  count <- constant (fromIntegral b)
  loss <- binary Div total count
  pure (loss, map fst perSlot)
  where
    leaves = mapM param . elementNames
    dot us vs = zipWithM (binary Mul) us vs >>= sumOf
    sumOf (n : ns) = foldM (binary Add) n ns
    sumOf [] = constant 0

-- | Train the XOR net with @b@ slots.
xorRun :: Int -> TrainConfig -> Run
xorRun b cfg = train cfg setup (initParams (trainSeed cfg) xorSpecs)
  where
    ((loss, pairs), g) = xorNet b
    setup =
      Setup
        { setupGraph = g
        , setupLoss = loss
        , setupInputs = Map.empty
        , setupFeeds = [Feed "xor" xorRows (xorSlots b)]
        , setupAccuracy = pairs
        }

xorConfig :: TrainConfig
xorConfig = TrainConfig {trainSteps = 200, trainSgd = Sgd 0.1 ConstantRate, trainSeed = 1337}

-- | The predictions of the XOR net for all four rows, with these
-- parameters (a fresh forward pass, after training).
xorPredictions :: Params -> [Double]
xorPredictions ps = case evaluate g env (map fst pairs) of
  Right vals -> map (valueAt vals . fst) pairs
  Left err -> error (show err)
  where
    ((_, pairs), g) = xorNet 4
    env = foldr (uncurry bindInput) emptyEnv {envParams = ps} (concat (zipWith zip (xorSlots 4) xorRows))

-- ---------------------------------------------------------------------------
-- Small graphs

-- | @loss = (w - a)²@ for a parameter @w@ and an input @a@.
quadratic :: ((NodeId, NodeId), Graph)
quadratic = built $ do
  w <- param "w"
  a <- input "a"
  l <- binary Sub w a >>= unary (PowC 2)
  pure (l, w)

-- | Train 'quadratic' from @w = 0@ towards @a@.
quadraticRun :: Double -> TrainConfig -> Run
quadraticRun a cfg = train cfg setup (Map.singleton "w" 0)
  where
    ((l, _), g) = quadratic
    setup = Setup g l (Map.singleton "a" a) [] []

config :: Int -> Double -> LrSchedule -> TrainConfig
config n eta sched = TrainConfig {trainSteps = n, trainSgd = Sgd eta sched, trainSeed = 0}

-- | A setup with one feed and no real loss: for the checks of the feed.
withFeed :: Feed -> Run
withFeed feed = train (config 1 0.1 ConstantRate) setup Map.empty
  where
    (l, g) = built (constant 1)
    setup = Setup g l Map.empty [feed] []

-- ---------------------------------------------------------------------------

spec :: Spec
spec = do
  describe "the milestone: an MLP learns XOR" $ do
    let run = xorRun 4 xorConfig
        reports = runReports run
        losses = map reportLoss reports
    it "runs all 200 steps" $ do
      map reportStep reports `shouldBe` [0 .. 199]
      runResult run `shouldSatisfy` either (const False) (const True)
    it "decreases the loss" $ do
      take 1 losses `shouldSatisfy` all (> 1)
      last losses `shouldSatisfy` (< 1e-3)
      -- Every 50 steps the loss is smaller than 50 steps before.
      let checkpoints = everyNth 50 reports
      zipWith (>) (map reportLoss checkpoints) (drop 1 (map reportLoss checkpoints))
        `shouldBe` replicate (length checkpoints - 1) True
    it "ends up classifying all four points correctly" $ do
      reportAccuracy (last reports) `shouldBe` Just 1
      case runResult run of
        Left err -> expectationFailure (show err)
        Right ps -> do
          let labels = map last xorRows
          zipWith (\p y -> signum p == signum y) (xorPredictions ps) labels
            `shouldBe` [True, True, True, True]
    it "starts from the initialised parameters, not a solved net" $
      map reportAccuracy (take 1 reports) `shouldSatisfy` all (< Just 1)
    it "learns it from other seeds as well" $
      [ reportAccuracy (last (runReports (xorRun 4 xorConfig {trainSeed = s})))
      | s <- [1 .. 5]
      ]
        `shouldBe` replicate 5 (Just 1)

  describe "sampleBatch" $ do
    let draws n b seed = evalState (sampleBatch n b) (streamGen seed BatchStream)
    prop "picks b distinct rows in range" $
      forAll (choose (1, 60)) $ \n -> forAll (choose (0, n)) $ \b -> \seed ->
        let picked = draws n b seed
         in length picked === b
              .&&. length (nub picked) === b
              .&&. all (\r -> r >= 0 && r < n) picked
    prop "is deterministic for a seed" $ \(Positive n) seed ->
      let b = n `div` 2 in draws n b seed === draws n b seed
    prop "a full batch is every row, in order, and draws nothing" $ \(Positive n) seed ->
      let gen = streamGen seed BatchStream
          (picked, gen') = runState (sampleBatch n n) gen
       in (picked, show gen') === ([0 .. n - 1], show gen)
    it "picks every row about equally often" $ do
      let batches = evalState (mapM (const (sampleBatch 4 2)) [1 .. 4000 :: Int]) (streamGen 7 BatchStream)
          counts = Map.elems (Map.fromListWith (+) [(r, 1 :: Int) | r <- concat batches])
      counts `shouldSatisfy` all (\c -> abs (c - 2000) < 150)
    it "depends on the seed" $
      draws 100 10 1 `shouldSatisfy` (/= draws 100 10 2)

  describe "mini-batch runs" $ do
    it "bind the sampled rows, a new batch in every step" $ do
      -- The loss is slot 0's value itself, so each report shows which row
      -- the step bound: row r holds the number r.
      let (l, g) = built (input "d#0.v")
          feed = Feed "d" [[fromIntegral r] | r <- [0 .. 9 :: Int]] [["d#0.v"]]
          run = train (config 40 0.1 ConstantRate) {trainSeed = 3} (Setup g l Map.empty [feed] []) Map.empty
          expected = evalState (replicateM 40 (sampleBatch 10 1)) (streamGen 3 BatchStream)
      map reportLoss (runReports run) `shouldBe` map fromIntegral (concat expected)
      length (nub (concat expected)) `shouldSatisfy` (> 5)
    let run seed = xorRun 2 xorConfig {trainSeed = seed}
    it "are deterministic for a seed" $
      run 5 `shouldBe` run 5
    it "differ between seeds" $
      map reportLoss (runReports (run 5)) `shouldSatisfy` (/= map reportLoss (runReports (run 6)))
    it "full-batch runs use no batch randomness" $ do
      -- Same initial parameters, different batch seeds: identical runs.
      let ps = initParams 1337 xorSpecs
          ((l, pairs), g) = xorNet 4
          setup = Setup g l Map.empty [Feed "xor" xorRows (xorSlots 4)] pairs
          cfg seed = xorConfig {trainSteps = 20, trainSeed = seed}
      train (cfg 1) setup ps `shouldBe` train (cfg 2) setup ps

  describe "train" $ do
    it "reports the loss before each step's update" $
      map reportLoss (runReports (quadraticRun 3 (config 2 0.25 ConstantRate)))
        -- w: 0 → 0 − 0.25·(−6) = 1.5; losses (0−3)², (1.5−3)².
        `shouldBe` [9, 2.25]
    it "uses the schedule's learning rate at every step" $ do
      let cfg = config 10 1 (StepDecay 3 0.5)
      map reportRate (runReports (quadraticRun 3 cfg))
        `shouldBe` map (learningRate (trainSgd cfg) 10) [0 .. 9]
    it "uses the fixed inputs and converges" $
      case runResult (quadraticRun 5 (config 100 0.1 ConstantRate)) of
        Right ps -> Map.lookup "w" ps `shouldSatisfy` maybe False (\w -> abs (w - 5) < 1e-6)
        Left err -> expectationFailure (show err)
    it "with zero steps returns the parameters unchanged" $
      quadraticRun 3 (config 0 0.1 ConstantRate) `shouldBe` Done (Map.singleton "w" 0)
    it "reports accuracy only when asked" $
      map reportAccuracy (runReports (quadraticRun 3 (config 1 0.1 ConstantRate)))
        `shouldBe` [Nothing]

  describe "errors" $ do
    it "an empty dataset" $
      withFeed (Feed "d" [] []) `shouldBe` Failed (EmptyFeed "d")
    it "more slots than rows" $
      withFeed (Feed "d" [[1]] [["d#0.y"], ["d#1.y"]]) `shouldBe` Failed (BatchTooLarge "d" 2 1)
    it "a row of the wrong width" $
      withFeed (Feed "d" [[1, 2], [3]] [["d#0.x", "d#0.y"]])
        `shouldBe` Failed (RowWidthMismatch "d" 1 1 2)
    it "a parameter without a value fails the first step" $
      case train (config 5 0.1 ConstantRate) quadSetup Map.empty of
        Failed (StepFailed 0 (ForwardError (UnboundLeaves _))) -> pure ()
        other -> expectationFailure (show other)
    it "a non-finite value stops the run, keeping the reports before it" $ do
      -- loss = log w from w = 0.5 with η = 1: w becomes 0.5 − 1·2 = −1.5.
      let (l, g) = built (param "w" >>= unary Log)
          run = train (config 5 1 ConstantRate) (Setup g l Map.empty [] []) (Map.singleton "w" 0.5)
      map reportStep (runReports run) `shouldBe` [0]
      case runResult run of
        Left (StepFailed 1 (ForwardError (NonFiniteValue i [x] _))) -> (i, x) `shouldBe` (l, -1.5)
        other -> expectationFailure (show other)

  describe "everyNth" $ do
    let reports = runReports (quadraticRun 3 (config 10 0.1 ConstantRate))
    it "keeps steps 0, n, 2n, …" $
      map reportStep (everyNth 4 reports) `shouldBe` [0, 4, 8]
    it "keeps every step for n < 1" $
      map reportStep (everyNth 0 reports) `shouldBe` [0 .. 9]
  where
    quadSetup = let ((l, _), g) = quadratic in Setup g l (Map.singleton "a" 3) [] []
