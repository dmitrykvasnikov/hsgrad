{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Core.Eval".
--
-- * micrograd's examples, built by hand, give micrograd's numbers.
-- * Sharing and reachability: each needed node is computed once, and
--   nodes the outputs do not depend on are not computed at all.
-- * Environments and schedules: rebinding leaves without rebuilding.
-- * Errors: unbound leaves (all of them, at once) and the first
--   non-finite value, with the provenance of the node that produced it.
-- * Properties over random graphs: the forward pass agrees with a naive
--   recursive evaluator, and reports exactly the right errors.
module HsGrad.Core.EvalSpec
  ( spec
  ) where

import Data.IntMap.Strict qualified as IntMap
import Data.List (find)
import Data.List.NonEmpty (NonEmpty (..), nonEmpty)
import Data.Map.Strict qualified as Map
import Data.Maybe (catMaybes, isNothing)
import Data.Set qualified as Set
import HsGrad.Core.Eval
import HsGrad.Core.Graph
import HsGrad.Core.Op
import HsGrad.Test.Graphs (Recipe (..), built, builtRecipe)
import Test.Hspec
  ( Expectation
  , Spec
  , describe
  , expectationFailure
  , it
  , shouldBe
  , shouldSatisfy
  )
import Test.Hspec.QuickCheck (modifyMaxSize, modifyMaxSuccess, prop)
import Test.QuickCheck
  ( Gen
  , Property
  , choose
  , classify
  , conjoin
  , counterexample
  , elements
  , forAll
  , listOf1
  , property
  , (===)
  )

spec :: Spec
spec = do
  describe "micrograd examples" examplesSpec
  describe "sharing and reachability" sharingSpec
  describe "environments and schedules" envSpec
  describe "errors" errorSpec
  describe "random graphs" propertySpec

-- ---------------------------------------------------------------------------
-- Helpers

-- | Evaluate, expecting success.
evaluated :: Graph -> Env -> [NodeId] -> Values
evaluated g env outs = case evaluate g env outs of
  Right vals -> vals
  Left err -> error ("unexpected evaluation error: " <> show err)

-- | Check that evaluation fails, and pass the error to a check.
evalFailsWith :: Either EvalError Values -> (EvalError -> Expectation) -> Expectation
evalFailsWith r check = case r of
  Left err -> check err
  Right vals -> expectationFailure ("expected an error, got " <> show vals)

-- | An environment with only inputs.
inputs :: [(Name, Double)] -> Env
inputs bs = emptyEnv {envInputs = Map.fromList bs}

-- | Approximate equality, for results of transcendental functions.
shouldBeNear :: Double -> Double -> Expectation
shouldBeNear x y =
  x `shouldSatisfy` \v -> abs (v - y) <= 1e-12 * max 1 (abs y)

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

-- | Equal as IEEE values, except that NaN counts as equal to NaN.
sameDouble :: Double -> Double -> Bool
sameDouble x y = x == y || (isNaN x && isNaN y)

-- | A span on line @l@ of @m.hsg@, from column @a@ to column @b@.
spanAt :: Int -> Int -> Int -> SrcSpan
spanAt l a b = SrcSpan "m.hsg" (Pos l a) (Pos l b)

-- ---------------------------------------------------------------------------
-- micrograd examples

examplesSpec :: Spec
examplesSpec = do
  it "evaluates intro.hsg: L = -8" $ do
    -- docs/01-language.md §1.1
    let ((a, b, c, e, d, l), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          c' <- input "c"
          e' <- binary Mul a' b'
          d' <- binary Add e' c'
          l' <- binary Mul d' =<< constant (-2)
          pure (a', b', c', e', d', l')
        vals = evaluated g (inputs [("a", 2), ("b", -3), ("c", 10)]) [l]
    map (valueAt vals) [a, b, c, e, d, l] `shouldBe` [2, -3, 10, -6, 4, -8]

  it "evaluates micrograd's neuron: tanh(x1*w1 + x2*w2 + b)" $ do
    -- The bias is chosen so that the pre-activation is atanh(1/√2).
    let ((n, o), g) = built $ do
          x1 <- input "x1"
          x2 <- input "x2"
          w1 <- param "w1"
          w2 <- param "w2"
          bias <- param "b"
          x1w1 <- binary Mul x1 w1
          x2w2 <- binary Mul x2 w2
          s <- binary Add x1w1 x2w2
          n' <- binary Add s bias
          o' <- unary Tanh n'
          pure (n', o')
        env =
          Env
            { envInputs = Map.fromList [("x1", 2), ("x2", 0)]
            , envParams = Map.fromList [("w1", -3), ("w2", 1), ("b", 6.881373587019543)]
            }
        vals = evaluated g env [o]
    valueAt vals n `shouldBeNear` 0.881373587019543
    valueAt vals o `shouldBeNear` (1 / sqrt 2)

  it "evaluates every kind of op" $ do
    let (outs, g) = built $ do
          x <- input "x" -- 4
          y <- param "y" -- -0.5
          sequence
            [ unary Neg x
            , unary (PowC 3) y
            , unary (PowC (1 / 2)) x
            , unary Exp y
            , unary Log x
            , unary Tanh y
            , unary Relu y
            , unary Step x
            , binary Add x y
            , binary Sub x y
            , binary Mul x y
            , binary Div x y
            ]
        env = bindParam "y" (-0.5) (bindInput "x" 4 emptyEnv)
        vals = evaluated g env outs
    map (valueAt vals) outs
      `shouldBe` [-4, -0.125, 2, exp (-0.5), log 4, tanh (-0.5), 0, 1, 3.5, 4.5, -2, -8]

  it "evaluates a graph of constants with the empty environment" $ do
    let (out, g) = built $ do
          two <- constant 2
          three <- constant 3
          binary Mul two three
    valueAt (evaluated g emptyEnv [out]) out `shouldBe` 6

-- ---------------------------------------------------------------------------
-- Sharing and reachability

sharingSpec :: Spec
sharingSpec = do
  it "computes a shared node once (L = d * d)" $ do
    let ((d, l), g) = built $ do
          a <- input "a"
          b <- input "b"
          d' <- binary Add a b
          l' <- binary Mul d' d'
          pure (d', l')
        s = schedule g [l]
    map fst (scheduleNodes s) `shouldBe` nodeIds g
    let vals = evaluated g (inputs [("a", 1), ("b", 2)]) [l]
    valueAt vals d `shouldBe` 3
    valueAt vals l `shouldBe` 9

  it "stays linear on a chain of doublings (2^40 paths, 41 nodes)" $ do
    -- d₀ = x, dₖ₊₁ = dₖ + dₖ. As a tree this would have 2^40 leaves; a
    -- recursive evaluator without sharing would never finish.
    let (out, g) = built $ do
          x <- input "x"
          let double d = binary Add d d
          foldr (=<<) (pure x) (replicate 40 double)
        s = schedule g [out]
    length (scheduleNodes s) `shouldBe` 41
    case evalSchedule s (inputs [("x", 1)]) of
      Right vals -> do
        valueAt vals out `shouldBe` 2 ^ (40 :: Int)
        IntMap.size vals `shouldBe` 41
      Left err -> expectationFailure (show err)

  it "skips nodes the outputs do not depend on" $ do
    -- The dead node would be log(-1) = NaN, and the dead input has no
    -- value. Neither matters, because neither is needed.
    let ((out, dead, deadIn), g) = built $ do
          x <- input "x"
          out' <- unary Exp x
          dead' <- unary Log x
          deadIn' <- input "unused"
          pure (out', dead', deadIn')
        vals = evaluated g (inputs [("x", -1)]) [out]
    valueAt vals out `shouldBe` exp (-1)
    lookupValue dead vals `shouldBe` Nothing
    lookupValue deadIn vals `shouldBe` Nothing
    IntMap.size vals `shouldBe` 2

  it "evaluates several outputs together, sharing their common part" $ do
    let ((p, q), g) = built $ do
          x <- input "x"
          t <- unary Tanh x
          p' <- binary Mul t t
          q' <- unary Neg t
          pure (p', q')
        vals = evaluated g (inputs [("x", 0.5)]) [p, q]
    IntMap.size vals `shouldBe` 4
    valueAt vals p `shouldBeNear` (tanh 0.5 * tanh 0.5)
    valueAt vals q `shouldBe` negate (tanh 0.5)

  it "evaluates nothing for no outputs, or for ids of another graph" $ do
    let (_, g) = built (input "x")
        (bigIds, _) = built (mapM input ["p", "q", "r"])
    evaluate g emptyEnv [] `shouldBe` Right IntMap.empty
    evaluate g emptyEnv [last bigIds] `shouldBe` Right IntMap.empty

-- ---------------------------------------------------------------------------
-- Environments and schedules

envSpec :: Spec
envSpec = do
  -- loss = (w * x - y)^2, the shape of one term of a training loss.
  let ((w, x, y, loss), g) = built $ do
        w' <- param "w"
        x' <- input "x"
        y' <- input "y"
        wx <- binary Mul w' x'
        err <- binary Sub wx y'
        loss' <- unary (PowC 2) err
        pure (w', x', y', loss')

  it "evaluates one schedule with many environments (D10)" $ do
    let s = schedule g [loss]
        env0 = bindParam "w" 0.5 emptyEnv
        row env (xv, yv) = bindInput "y" yv (bindInput "x" xv env)
        lossOn env = valueAt <$> evalSchedule s env <*> pure loss
    -- Rebind the dataset slot to different rows...
    lossOn (row env0 (2, 3)) `shouldBe` Right 4
    lossOn (row env0 (4, 1)) `shouldBe` Right 1
    -- ...and replace all parameters at once, as an optimiser step would.
    let env1 = (row env0 (2, 3)) {envParams = Map.fromList [("w", 1.5)]}
    lossOn env1 `shouldBe` Right 0

  it "lists the leaves that need a value, in id order" $
    scheduleLeaves (schedule g [loss])
      `shouldBe` [(w, Param "w"), (x, Input "x"), (y, Input "y")]

  it "does not list constants as leaves that need a value" $ do
    let (out, g2) = built $ do
          c <- constant 3
          v <- input "v"
          binary Add c v
    map snd (scheduleLeaves (schedule g2 [out])) `shouldBe` [Input "v"]

  it "ignores names the graph does not use" $ do
    let env = bindInput "zzz" 1 (bindParam "w" 1 (inputs [("x", 1), ("y", 1)]))
    valueAt (evaluated g env [loss]) loss `shouldBe` 0

  it "keeps the last value bound to a name" $ do
    let env = bindParam "w" 2 (bindParam "w" 1 (inputs [("x", 1), ("y", 0)]))
    valueAt (evaluated g env [loss]) loss `shouldBe` 4

-- ---------------------------------------------------------------------------
-- Errors

errorSpec :: Spec
errorSpec = do
  it "reports every unbound leaf at once, in id order (E0502)" $ do
    let ((a, b, w), g) = built $ do
          a' <- input "a"
          b' <- input "b"
          w' <- param "w"
          c <- input "c"
          s1 <- binary Add a' b'
          s2 <- binary Add s1 w'
          _ <- binary Add s2 c
          pure (a', b', w')
        out = last (nodeIds g)
    evaluate g (inputs [("c", 1)]) [out]
      `evalFailsWith` (`shouldBe` UnboundLeaves ((a, Input "a") :| [(b, Input "b"), (w, Param "w")]))

  it "does not confuse an input and a parameter with the same name" $ do
    let (w, g) = built (param "w")
    evaluate g (inputs [("w", 1)]) [w]
      `evalFailsWith` (`shouldBe` UnboundLeaves ((w, Param "w") :| []))
    evaluate g (bindParam "w" 1 emptyEnv) [w] `shouldBe` Right (IntMap.singleton (nodeIndex w) 1)

  it "reports a NaN at the node that produced it, with its arguments (E0501)" $ do
    let ((lg, out), g) = built $ do
          x <- input "x"
          lg' <- unary Log x
          out' <- binary Mul lg' x
          pure (lg', out')
    evaluate g (inputs [("x", -1)]) [out] `evalFailsWith` \err -> case err of
      NonFiniteValue i args v -> do
        i `shouldBe` lg
        args `shouldBe` [-1]
        v `shouldSatisfy` isNaN
      other -> expectationFailure ("wrong error: " <> show other)

  it "reports infinities: division by zero and overflow" $ do
    let ((dv, ex), g) = built $ do
          x <- input "x"
          z <- input "z"
          dv' <- binary Div x z
          ex' <- unary Exp x
          pure (dv', ex')
    evaluate g (inputs [("x", 1), ("z", 0)]) [dv]
      `evalFailsWith` (`shouldBe` NonFiniteValue dv [1, 0] (1 / 0))
    evaluate g (inputs [("x", 1000)]) [ex]
      `evalFailsWith` (`shouldBe` NonFiniteValue ex [1000] (1 / 0))

  it "reports where the problem starts, not where the NaN spread to" $ do
    -- out = tanh(sqrt(x) + 1) * 2: every node from sqrt(x) on is NaN for
    -- x < 0, and the error names sqrt(x).
    let ((sq, out), g) = built $ do
          x <- input "x"
          sq' <- unary (PowC (1 / 2)) x
          one <- constant 1
          t <- unary Tanh =<< binary Add sq' one
          out' <- binary Mul t =<< constant 2
          pure (sq', out')
    evaluate g (inputs [("x", -4)]) [out] `evalFailsWith` \err -> case err of
      NonFiniteValue i _ _ -> i `shouldBe` sq
      other -> expectationFailure ("wrong error: " <> show other)

  it "even when the non-finite value would disappear further on" $ do
    -- 1 / exp(1000) = 1 / ∞ = 0 is finite, but its derivative is not: an
    -- infinity on the way is an error.
    let ((ex, out), g) = built $ do
          x <- input "x"
          ex' <- unary Exp x
          one <- constant 1
          out' <- binary Div one ex'
          pure (ex', out')
    evaluate g (inputs [("x", 1000)]) [out]
      `evalFailsWith` (`shouldBe` NonFiniteValue ex [1000] (1 / 0))

  it "reports a non-finite leaf value" $ do
    let (x, g) = built (input "x")
    evaluate g (inputs [("x", 1 / 0)]) [x]
      `evalFailsWith` (`shouldBe` NonFiniteValue x [] (1 / 0))

  it "reports unbound leaves before computing anything" $ do
    -- log(x) would be NaN, but y has no value: that is reported first.
    let ((y, out), g) = built $ do
          x <- input "x"
          y' <- input "y"
          lg <- unary Log x
          out' <- binary Add lg y'
          pure (y', out')
    evaluate g (inputs [("x", -1)]) [out]
      `evalFailsWith` (`shouldBe` UnboundLeaves ((y, Input "y") :| []))

  it "leads back to the source span of the failing node" $ do
    -- What the driver will do for E0501: look up the node's provenance.
    let (out, g) = built $ do
          x <- withSpan (spanAt 1 7 8) (input "x")
          withSpan (spanAt 3 9 15) (unary Log x)
    evaluate g (inputs [("x", 0)]) [out] `evalFailsWith` \err -> case err of
      NonFiniteValue i _ _ ->
        provSpan (nodeProv (nodeAt g i)) `shouldBe` Just (spanAt 3 9 15)
      other -> expectationFailure ("wrong error: " <> show other)

-- ---------------------------------------------------------------------------
-- Random graphs

-- | The reference evaluator: the definition of a node's value, followed
-- recursively down to the leaves. No schedule, no map of values, no
-- sharing: a node used twice is computed twice. That makes it slow (it
-- takes exponential time on graphs with a lot of sharing), and obviously
-- right. Every leaf must have a value.
refValue :: Graph -> Env -> NodeId -> Double
refValue g env i = case (nodeOp n, nodeArgs n) of
  (Leaf (Const c), []) -> c
  (Leaf (Input x), []) -> envInputs env Map.! x
  (Leaf (Param x), []) -> envParams env Map.! x
  (Un op, [a]) -> forwardUn op (ref a)
  (Bin op, [a, b]) -> forwardBin op (ref a) (ref b)
  _ -> error "refValue: wrong number of arguments"
  where
    n = nodeAt g i
    ref = refValue g env

-- | Values for the three non-constant leaves of random graphs
-- ('HsGrad.Test.Graphs.genStep'): inputs @x@ and @y@, parameter @w@.
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

-- | Like 'genEnv', but each leaf is bound only half of the time. Also binds
-- @Input "w"@ sometimes, which must not count as a value for @Param "w"@.
genPartialEnv :: Gen Env
genPartialEnv = do
  let maybeBind n = do
        bound <- elements [False, True]
        v <- choose (-3, 3)
        pure (if bound then Just (n, v) else Nothing)
  ins <- catMaybes <$> mapM maybeBind ["x", "y", "w"]
  ps <- catMaybes <$> mapM maybeBind ["w"]
  pure (Env (Map.fromList ins) (Map.fromList ps))

-- | A random graph, some outputs of it, and a check on them.
withGraph :: (Graph -> [NodeId] -> Property) -> Recipe -> Property
withGraph check (Recipe steps) =
  forAll (listOf1 (choose (0, 1000))) $ \picks ->
    let (ids, g) = builtRecipe steps
        outs = [ids !! (p `mod` length ids) | p <- picks]
     in counterexample (show (nodes g)) (check g outs)

propertySpec :: Spec
propertySpec = do
  -- Small graphs only: the reference evaluator is exponential in the
  -- depth of sharing. Size 8 means recipes of at most 16 steps. Only a few
  -- percent of random cases produce a non-finite value, so run more cases
  -- than the default 100 to exercise that branch often.
  modifyMaxSize (const 8) . modifyMaxSuccess (const 500) $
    prop "agrees with a naive recursive evaluator, or names the first non-finite node" $
      withGraph $ \g outs -> forAll genEnv $ \env ->
        let needed = Set.toAscList (reachable g outs)
            ref = [(i, refValue g env i) | i <- needed]
            result = evaluate g env outs
         in classify (either (const True) (const False) result) "non-finite" $
              case find (not . isFinite . snd) ref of
                Nothing ->
                  result === Right (IntMap.fromList [(nodeIndex i, v) | (i, v) <- ref])
                Just (bad, v) -> case result of
                  Left (NonFiniteValue i args v') ->
                    conjoin
                      [ i === bad
                      , args === map (refValue g env) (nodeArgs (nodeAt g bad))
                      , counterexample "arguments are finite" (property (all isFinite args))
                      , counterexample (show (v, v')) (property (sameDouble v v'))
                      ]
                  other -> counterexample (show other) False

  prop "evaluates exactly the reachable nodes, each once" $
    withGraph $ \g outs -> forAll genEnv $ \env ->
      let needed = reachable g outs
       in conjoin
            [ map fst (scheduleNodes (schedule g outs)) === Set.toAscList needed
            , case evaluate g env outs of
                Right vals -> IntMap.keys vals === map nodeIndex (Set.toAscList needed)
                Left (NonFiniteValue i _ _) -> property (i `Set.member` needed)
                Left err -> counterexample (show err) False
            ]

  prop "reports exactly the reachable leaves that have no value" $
    withGraph $ \g outs -> forAll genPartialEnv $ \env ->
      let unbound l = case l of
            Input n -> isNothing (Map.lookup n (envInputs env))
            Param n -> isNothing (Map.lookup n (envParams env))
            Const _ -> False
          missing =
            [ (i, l)
            | i <- Set.toAscList (reachable g outs)
            , Leaf l <- [nodeOp (nodeAt g i)]
            , unbound l
            ]
       in case (nonEmpty missing, evaluate g env outs) of
            (Just m, result) -> result === Left (UnboundLeaves m)
            (Nothing, Left (UnboundLeaves m)) -> counterexample (show m) False
            (Nothing, _) -> property True
