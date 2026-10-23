-- | The training loop.
--
-- micrograd's demo trains like this:
--
-- > for k in range(100):
-- >     total_loss, acc = loss()          # forward pass on a mini-batch
-- >     model.zero_grad()
-- >     total_loss.backward()             # backward pass
-- >     for p in model.parameters():      # update (SGD)
-- >         p.data -= learning_rate * p.grad
-- >     print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
--
-- micrograd rebuilds the whole expression graph in every step. Here the
-- graph is built __once__ (decision D10): dataset rows enter it through
-- /slots/, input leaves such as @xor#2.x[0]@, and every step only rebinds
-- the slots to the rows of a new mini-batch and the parameter leaves to the
-- current parameter store. So one step is:
--
-- 1. __bind__: the store, the fixed inputs and a batch of rows go into an
--    'Env' (the batch is sampled with the seed when there are fewer slots
--    than rows, and is the whole dataset in order otherwise);
-- 2. __forward and backward__: 'forwardBackward' on a 'Schedule' that was
--    also computed once;
-- 3. __update__: 'sgdStep' with the step's learning rate;
-- 4. __report__: the loss (and, optionally, the accuracy) of this step.
--
-- The loop is a pure function. It does not print: it returns the run as a
-- lazy stream of 'Report's ending in the final parameters or an error
-- ('Run'). Since the stream is lazy, a caller that prints each report as
-- it arrives sees them while training proceeds, and a test can simply look
-- at the list.
-- See @docs/10-train.md@.
module HsGrad.Train.Loop
  ( -- * What to train
    Feed (..)
  , Setup (..)
  , TrainConfig (..)

    -- * Running
  , Run (..)
  , Report (..)
  , TrainError (..)
  , train
  , runReports
  , runResult
  , everyNth

    -- * Mini-batches
  , sampleBatch
  ) where

import Control.Monad (foldM)
import Control.Monad.State.Strict (State, runState, state)
import Data.IntMap.Strict qualified as IntMap
import Data.List (find)
import Data.Map.Strict (Map)
import Data.Sequence (Seq)
import Data.Sequence qualified as Seq
import Data.Text (Text)
import HsGrad.Core.Backprop (GradError, LeafGrads (..), forwardBackward, leafGrads)
import HsGrad.Core.Eval (Env (..), Values, bindInput, schedule, valueAt)
import HsGrad.Core.Graph (Graph, NodeId)
import HsGrad.Core.Op (Name)
import HsGrad.Train.Optim (Sgd, learningRate, sgdStep)
import HsGrad.Train.Param (Params, Seed, Stream (..), streamGen)
import System.Random (StdGen, uniformR)

-- ---------------------------------------------------------------------------
-- What to train

-- | One dataset and the slots its rows are bound to (D10).
--
-- The rows are plain numbers: every row lists one value per /column/, in a
-- fixed order (for the schema @{ x : Vec 2, y : Real }@ the columns are
-- @x[0]@, @x[1]@, @y@). Reading them from a CSV file or an inline @rows@
-- literal is "HsGrad.Data"'s job (step 13).
--
-- The slots are the input leaves the compiler created for a comprehension
-- over the dataset (step 14): for every slot, the leaf name of each column,
-- in the same order as the row values. Their number is the batch size.
-- Binding slot @i@ to row @r@ is just @zip (feedSlots !! i) (rows !! r)@,
-- so this module knows nothing about how slots are named.
data Feed = Feed
  { feedName :: !Text
  -- ^ The dataset's name, for error messages.
  , feedRows :: ![[Double]]
  -- ^ The dataset: one list of column values per row.
  , feedSlots :: ![[Name]]
  -- ^ The batch: for every slot, one leaf name per column.
  }
  deriving (Eq, Show)

-- | Everything about /what/ is trained: the graph and the data.
data Setup = Setup
  { setupGraph :: !Graph
  , setupLoss :: !NodeId
  -- ^ The loss node: a scalar, minimised.
  , setupInputs :: !(Map Name Double)
  -- ^ Values for the inputs that are not dataset slots (@input@
  -- declarations: defaults and @--set@). The same in every step.
  , setupFeeds :: ![Feed]
  , setupAccuracy :: ![(NodeId, NodeId)]
  -- ^ For sign accuracy (@01-language.md@ §11): one pair per slot, the
  -- model's prediction node and the label's input leaf. A slot counts as
  -- correct when both are positive or both are not, as in micrograd. Leave
  -- the list empty for no accuracy.
  }

-- | Everything about /how/ to train (the CLI flags of D11).
data TrainConfig = TrainConfig
  { trainSteps :: !Int
  -- ^ @--steps@: how many updates. Zero or less means none.
  , trainSgd :: !Sgd
  -- ^ @--lr@ and @--lr-schedule@.
  , trainSeed :: !Seed
  -- ^ @--seed@, used here for the mini-batches ('BatchStream'). The
  -- initial parameters come from the same seed through
  -- 'HsGrad.Train.Param.initParams'.
  }
  deriving (Eq, Show)

-- ---------------------------------------------------------------------------
-- Runs

-- | What one step measured. Loss and accuracy are those of the forward
-- pass of this step, so they belong to the parameters /before/ the
-- step's update (micrograd prints the same).
data Report = Report
  { reportStep :: !Int
  -- ^ 0-based step number.
  , reportRate :: !Double
  -- ^ The learning rate this step's update used.
  , reportLoss :: !Double
  , reportAccuracy :: !(Maybe Double)
  -- ^ Fraction of correctly classified slots, between 0 and 1;
  -- 'Nothing' if the setup has no accuracy pairs.
  }
  deriving (Eq, Show)

-- | A training run: one 'Report' per step, then the final parameters or
-- the error that stopped the run.
--
-- It is a list with two kinds of end. Like a list it is lazy: the next
-- step is only computed when something looks at it.
data Run
  = Step !Report Run
  | Done !Params
  | Failed !TrainError
  deriving (Eq, Show)

-- | Why training could not start, or stopped.
data TrainError
  = -- | The dataset has no rows (E0405).
    EmptyFeed Text
  | -- | More slots than rows: dataset, slots, rows (E0405, @--batch@
    -- larger than the dataset).
    BatchTooLarge Text Int Int
  | -- | A row does not have one value per slot column: dataset, row index
    -- (0-based), the row's length, the slot's length.
    RowWidthMismatch Text Int Int Int
  | -- | The forward or backward pass of a step failed (a non-finite value
    -- or gradient, or a leaf without a value): the step and the error.
    StepFailed Int GradError
  deriving (Eq, Show)

-- | The reports of a run, in step order.
runReports :: Run -> [Report]
runReports r = case r of
  Step rep rest -> rep : runReports rest
  _ -> []

-- | How a run ended: the final parameters, or the error.
runResult :: Run -> Either TrainError Params
runResult r = case r of
  Step _ rest -> runResult rest
  Done ps -> Right ps
  Failed e -> Left e

-- | The reports of steps @0, n, 2n, …@: what @--report-every n@ prints.
-- An @n@ below @1@ keeps every report.
everyNth :: Int -> [Report] -> [Report]
everyNth n = filter (\rep -> reportStep rep `mod` max 1 n == 0)

-- ---------------------------------------------------------------------------
-- The loop

-- | Train: starting from the given parameters, run the configured number
-- of steps.
--
-- The feeds are checked first (rows present, enough rows for the slots,
-- every row as wide as the slots); a problem there is an immediate
-- 'Failed'. Then each step binds, runs forward and backward, updates, and
-- reports. The first step whose forward or backward pass fails ends the
-- run with 'StepFailed'; the reports before it are kept.
train :: TrainConfig -> Setup -> Params -> Run
train cfg setup params0 = case mapM checkFeed (setupFeeds setup) of
  Left e -> Failed e
  Right feeds -> go feeds 0 params0 (streamGen (trainSeed cfg) BatchStream)
  where
    n = trainSteps cfg
    loss = setupLoss setup
    pairs = setupAccuracy setup
    -- Computed once (D10). The prediction and label nodes are outputs too,
    -- so their values are there even if the loss did not need them.
    sched = schedule (setupGraph setup) (loss : concat [[p, l] | (p, l) <- pairs])

    go feeds t ps gen
      | t >= n = Done ps
      | otherwise = case forwardBackward sched env loss of
          Left e -> Failed (StepFailed t e)
          Right (vals, grads) ->
            let eta = learningRate (trainSgd cfg) n t
                ps' = sgdStep eta (paramGrads (leafGrads sched grads)) ps
                rep = Report t eta (valueAt vals loss) (accuracy vals pairs)
             in Step rep (go feeds (t + 1) ps' gen')
      where
        (env, gen') = runState (bindFeeds feeds base) gen
        base = Env {envInputs = setupInputs setup, envParams = ps}

-- | A checked feed: its rows in a sequence, ready for indexing, and its
-- slots.
data Ready = Ready !(Seq [Double]) ![[Name]]

-- | Check a feed and turn its rows into a sequence (constant-time-ish
-- indexing for the sampled batches).
checkFeed :: Feed -> Either TrainError Ready
checkFeed (Feed name rows slots)
  | null rows = Left (EmptyFeed name)
  | length slots > length rows = Left (BatchTooLarge name (length slots) (length rows))
  | Just (i, rowLen, slotLen) <- find mismatched widths =
      Left (RowWidthMismatch name i rowLen slotLen)
  | otherwise = Right (Ready (Seq.fromList rows) slots)
  where
    widths = [(i, length r, length s) | s <- slots, (i, r) <- zip [0 ..] rows]
    mismatched (_, rowLen, slotLen) = rowLen /= slotLen

-- | Bind every feed's slots to a batch of its rows, drawing the batches
-- from the generator in the state (one feed after the other).
bindFeeds :: [Ready] -> Env -> State StdGen Env
bindFeeds feeds env0 = foldM bindFeed env0 feeds
  where
    bindFeed env (Ready rows slots) = do
      picked <- sampleBatch (Seq.length rows) (length slots)
      let row r = Seq.index rows r
          bindings = concat [zip names (row r) | (names, r) <- zip slots picked]
      pure (foldr (uncurry bindInput) env bindings)

-- | The row indices for one batch of @b@ slots from @n@ rows.
--
-- * @b == n@ (full batch, the default): every row once, in order:
--   @[0 .. n−1]@. Nothing is drawn, so the generator is unchanged.
-- * @b < n@ (mini-batch): @b@ distinct rows chosen uniformly at random,
--   in the order they were drawn. This is the first @b@ steps of a
--   Fisher–Yates shuffle (see below), so it costs @O(b log b)@, not
--   @O(n)@.
--
-- Rows are drawn /without/ replacement, like micrograd's
-- @np.random.permutation(n)[:batch_size]@: a batch never uses a row twice.
-- The caller ensures @0 <= b <= n@.
sampleBatch :: Int -> Int -> State StdGen [Int]
sampleBatch n b
  | b >= n = pure [0 .. n - 1]
  | otherwise = go 0 IntMap.empty
  where
    -- Fisher–Yates on the array a = [0 .. n−1]: at step i, pick j from
    -- [i, n−1], swap a[i] and a[j]; a[i] is the i-th sample. The array is
    -- represented by the positions that differ from the identity
    -- ('moved'), so it never has to be built. After the swap, a[i] is
    -- never read again (later steps only look at positions > i), so only
    -- a[j] has to be recorded.
    go i moved
      | i >= b = pure []
      | otherwise = do
          j <- state (uniformR (i, n - 1))
          let at k = IntMap.findWithDefault k k moved
              moved' = IntMap.insert j (at i) moved
          (at j :) <$> go (i + 1) moved'

-- | The fraction of slots whose prediction has the label's sign, or
-- 'Nothing' without any pairs.
accuracy :: Values -> [(NodeId, NodeId)] -> Maybe Double
accuracy _ [] = Nothing
accuracy vals pairs = Just (fromIntegral (length correct) / fromIntegral (length pairs))
  where
    correct = [() | (p, l) <- pairs, (valueAt vals p > 0) == (valueAt vals l > 0)]

