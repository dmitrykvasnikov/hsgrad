{-# LANGUAGE OverloadedStrings #-}

-- | Test helpers shared by the specs of the Core modules: running builders
-- that must succeed, and random graphs for QuickCheck.
--
-- A random graph is described by a 'Recipe' (see @docs/04-core-graph.md@
-- §4.1). It was introduced for the "HsGrad.Core.Graph" tests and moved here
-- in step 4, when "HsGrad.Core.Eval" needed random graphs too. Step 5 added
-- 'SmoothRecipe', a variant using every op, for gradient checks, and moved
-- 'genEnv' and 'withGraph' here from the "HsGrad.Core.Eval" spec.
module HsGrad.Test.Graphs
  ( built
  , Recipe (..)
  , SmoothRecipe (..)
  , Step (..)
  , genStep
  , genSmoothStep
  , runRecipe
  , builtRecipe
  , genEnv
  , withGraph
  ) where

import Control.Monad (foldM)
import Data.Map.Strict qualified as Map
import HsGrad.Core.Eval (Env (..))
import HsGrad.Core.Graph
import HsGrad.Core.Op
import Test.QuickCheck
  ( Arbitrary (..)
  , Gen
  , Property
  , choose
  , counterexample
  , elements
  , forAll
  , frequency
  , listOf1
  , shrinkList
  , sized
  , vectorOf
  )

-- | Run a builder that is expected to succeed.
built :: Builder a -> (a, Graph)
built b = case build b of
  Right r -> r
  Left err -> error ("unexpected build error: " <> show err)

-- | A recipe for a random graph: a list of steps, each creating (or
-- finding) one node. Arguments are indices into the ids produced by the
-- steps before, taken modulo their number, so every list of steps is a
-- valid recipe, and so is every sublist (which keeps shrinking simple).
newtype Recipe = Recipe [Step]
  deriving (Show)

data Step
  = SLeaf Leaf
  | SUn UnOp Int
  | SBin BinOp Int Int
  deriving (Show)

instance Arbitrary Recipe where
  arbitrary = sized $ \n -> do
    len <- choose (1, max 1 (2 * n))
    Recipe <$> vectorOf len genStep
  shrink (Recipe steps) = map Recipe (shrinkList (const []) steps)

-- | Few distinct leaves and ops, so that random graphs contain many
-- structurally identical nodes for hash-consing to merge.
--
-- The leaves are @Input "x"@, @Input "y"@, @Param "w"@ and three constants.
-- Some ops are partial (@x ^ (1/2)@ of a negative number, division by zero,
-- @exp@ overflowing), so a random graph can produce non-finite values.
genStep :: Gen Step
genStep =
  frequency
    [ (2, SLeaf <$> elements leaves)
    , (3, SUn <$> elements unOps <*> genIndex)
    , (5, SBin <$> elements [minBound .. maxBound] <*> genIndex <*> genIndex)
    ]
  where
    leaves = [Input "x", Input "y", Param "w", Const 0, Const 1, Const (-2.5)]
    unOps = [Neg, Exp, Tanh, Relu, PowC 2, PowC (1 / 2)]
    genIndex = choose (0, 1000)

-- | A recipe for gradient checks (step 5): the same kind of 'Step's, but
-- drawn from 'genSmoothStep', which uses every op of the op set.
newtype SmoothRecipe = SmoothRecipe [Step]
  deriving (Show)

instance Arbitrary SmoothRecipe where
  arbitrary = sized $ \n -> do
    len <- choose (1, max 1 (2 * n))
    SmoothRecipe <$> vectorOf len genSmoothStep
  shrink (SmoothRecipe steps) = map SmoothRecipe (shrinkList (const []) steps)

-- | Steps for gradient checks: every unary op (including @log@, @step@ and
-- negative and fractional powers), every binary op, and no constant @0@,
-- which mostly creates divisions by zero.
--
-- Nothing here keeps the graph away from singular points (@log@ of a
-- negative number, a @relu@ at its kink); a random graph can contain all
-- of them. The gradient check itself skips the samples where a finite
-- difference cannot be trusted (see @docs/06-core-backprop.md@ §5.1).
genSmoothStep :: Gen Step
genSmoothStep =
  frequency
    [ (2, SLeaf <$> elements leaves)
    , (3, SUn <$> elements unOps <*> genIndex)
    , (5, SBin <$> elements [minBound .. maxBound] <*> genIndex <*> genIndex)
    ]
  where
    leaves = [Input "x", Input "y", Param "w", Const 1, Const (-2.5), Const 0.5]
    unOps =
      [Neg, Exp, Log, Tanh, Relu, Step, PowC 2, PowC 3, PowC (-1), PowC (1 / 2), PowC (-1 / 2)]
    genIndex = choose (0, 1000)

-- | Run a recipe. Returns the id produced by every step, in step order.
runRecipe :: [Step] -> Builder [NodeId]
runRecipe = fmap reverse . foldM step []
  where
    -- @done@ holds the ids produced so far, most recent first. A step that
    -- needs an argument when there is none yet becomes a leaf.
    step done s = (: done) <$> case s of
      SLeaf l -> node (Leaf l) []
      SUn op i
        | null done -> input "x"
        | otherwise -> unary op (pick done i)
      SBin op i j
        | null done -> input "y"
        | otherwise -> binary op (pick done i) (pick done j)
    pick done i = done !! (i `mod` length done)

-- | Build a recipe; random recipes never fail.
builtRecipe :: [Step] -> ([NodeId], Graph)
builtRecipe = built . runRecipe

-- | Values for the three non-constant leaves of random graphs ('genStep',
-- 'genSmoothStep'): inputs @x@ and @y@, parameter @w@, each in @[-3, 3]@.
genEnv :: Gen Env
genEnv = do
  x <- genVal
  y <- genVal
  w <- genVal
  pure
    Env
      { envInputs = Map.fromList [("x", x), ("y", y)]
      , envParams = Map.fromList [("w", w)]
      }
  where
    genVal = choose (-3, 3)

-- | A random graph, some outputs of it, and a check on them.
withGraph :: (Graph -> [NodeId] -> Property) -> Recipe -> Property
withGraph check (Recipe steps) =
  forAll (listOf1 (choose (0, 1000))) $ \picks ->
    let (ids, g) = builtRecipe steps
        outs = [ids !! (p `mod` length ids) | p <- picks]
     in counterexample (show (nodes g)) (check g outs)
