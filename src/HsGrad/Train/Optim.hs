-- | The optimiser: plain stochastic gradient descent with a learning-rate
-- schedule.
--
-- One step of gradient descent moves every parameter a little against its
-- gradient:
--
-- > p ← p − η · ∂L/∂p
--
-- where @η@ (eta) is the learning rate. The gradient points in the
-- direction in which the loss grows fastest, so a small enough step the
-- other way makes the loss smaller. This is micrograd's
--
-- > for p in model.parameters():
-- >     p.data -= learning_rate * p.grad
--
-- "Stochastic" refers to the gradient, not to this module: when the loss
-- is computed on a random mini-batch ("HsGrad.Train.Loop"), its gradient is
-- a noisy estimate of the full one.
--
-- The learning rate may change over the run. 'LrSchedule' mirrors the
-- planned CLI flag @--lr-schedule const|linear:END|step:K:F@
-- (@docs/00-architecture.md@ §5); parsing the flag is step 15's job.
-- Everything here is pure.
-- See @docs/10-train.md@.
module HsGrad.Train.Optim
  ( -- * Learning-rate schedules
    LrSchedule (..)
  , Sgd (..)
  , learningRate

    -- * One step
  , sgdStep
  ) where

import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import HsGrad.Core.Op (Name)
import HsGrad.Train.Param (Params)

-- ---------------------------------------------------------------------------
-- Learning-rate schedules

-- | How the learning rate changes over a run of @n@ steps, numbered
-- @t = 0 … n−1@, starting from a base rate @η₀@.
data LrSchedule
  = -- | @const@: @η₀@ at every step.
    ConstantRate
  | -- | @linear:END@: from @η₀@ at step @0@ in a straight line towards
    -- @END@, which is reached at step @n@, just after the last step:
    --
    -- > η(t) = (1 − s) · η₀ + s · END      where s = t / n
    --
    -- (Written as a weighted average, rather than @η₀ + (END − η₀)·s@, so
    -- that @t = 0@ gives exactly @η₀@ and @t = n@ exactly @END@ in floating
    -- point.)
    --
    -- This is exactly micrograd's demo, @learning_rate = 1.0 - 0.9*k/100@
    -- for 100 steps, which is @--lr 1.0 --lr-schedule linear:0.1@.
    LinearTo !Double
  | -- | @step:K:F@: multiply the rate by @F@ every @K@ steps:
    --
    -- > η(t) = η₀ · F ^ (t div K)
    --
    -- @K@ should be positive (the CLI rejects anything else); a @K@ below
    -- @1@ is treated as @1@, to keep the function total.
    StepDecay !Int !Double
  deriving (Eq, Show)

-- | The optimiser's settings: base learning rate (@--lr@) and schedule
-- (@--lr-schedule@).
data Sgd = Sgd
  { sgdRate :: !Double
  , sgdSchedule :: !LrSchedule
  }
  deriving (Eq, Show)

-- | The learning rate at step @t@ (0-based) of a run of @n@ steps.
--
-- A pure function of the step number, so a run can be resumed or replayed
-- from any step without keeping optimiser state.
learningRate :: Sgd -> Int -> Int -> Double
learningRate (Sgd base sched) n t = case sched of
  ConstantRate -> base
  LinearTo end
    | n <= 0 -> base
    | otherwise ->
        let s = fromIntegral t / fromIntegral n
         in (1 - s) * base + s * end
  StepDecay k f -> base * f ^ (max 0 t `div` max 1 k)

-- ---------------------------------------------------------------------------
-- One step

-- | One step of gradient descent with learning rate @η@: every parameter
-- moves by @−η · ∂L/∂p@.
--
-- The gradients are by leaf name, as 'HsGrad.Core.Backprop.paramGrads'
-- gives them. A parameter without a gradient keeps its value: backprop
-- leaves out exactly the parameters the loss does not depend on, whose
-- gradient is @0@ (warning W0203). Gradients of names that are not in the
-- store are ignored; nothing is ever added to the store.
sgdStep :: Double -> Map Name Double -> Params -> Params
sgdStep eta grads = Map.mapWithKey move
  where
    move n p = p - eta * Map.findWithDefault 0 n grads
