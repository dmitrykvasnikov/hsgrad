{-# LANGUAGE OverloadedStrings #-}

-- | The example programs of @docs/01-language.md@ §1, built by hand the way
-- the compiler (step 14) will build them: every leaf records the span of
-- its declaration, every expression's nodes the span of that expression,
-- and @let@-bound results carry their name as a label (D4).
--
-- Added in step 6 for the "HsGrad.Viz.Dot" tests and pictures. Later steps
-- can reuse them: the symbolic gradients of 'intro' (step 7), and "the
-- compiled @.hsg@ file equals the hand-built graph" (step 14).
module HsGrad.Test.Examples
  ( -- * @intro.hsg@
    Intro (..)
  , intro
  , introEnv

    -- * @neuron.hsg@
  , Neuron (..)
  , neuron
  , neuronEnv
  ) where

import Control.Monad (foldM, zipWithM)
import Data.Text qualified as Text
import HsGrad.Core.Eval (Env, bindInput, bindParam, emptyEnv)
import HsGrad.Core.Graph
import HsGrad.Core.Op (BinOp (..), Name (..), UnOp (..))
import HsGrad.Test.Graphs (built)

-- | A span on one line of a file: line, first column, column after the
-- last.
lineSpan :: FilePath -> Int -> Int -> Int -> SrcSpan
lineSpan file l c1 c2 = SrcSpan file (Pos l c1) (Pos l c2)

-- ---------------------------------------------------------------------------
-- intro.hsg

-- | The nodes of @intro.hsg@, by their names in the source.
data Intro = Intro
  { introA, introB, introC, introE, introD, introF, introL :: NodeId
  }
  deriving (Show)

-- | micrograd's first example, @docs/01-language.md@ §1.1:
--
-- > 1  -- Scalar expression; used to explore derivatives.
-- > 2  input a : Real = 2.0
-- > 3  input b : Real = -3.0
-- > 4  input c : Real = 10.0
-- > 5
-- > 6  let e = a * b
-- > 7  let d = e + c
-- > 8  let L = d * -2.0
--
-- The literal @-2.0@ becomes one constant node (named @f@ in micrograd)
-- with the span of the literal, inside the span of @d * -2.0@.
intro :: (Intro, Graph)
intro = built $ do
  a <- withSpan (sp 2 1 21) (input "a")
  b <- withSpan (sp 3 1 22) (input "b")
  c <- withSpan (sp 4 1 22) (input "c")
  e <- withSpan (sp 6 9 14) (binary Mul a b)
  label "e" e
  d <- withSpan (sp 7 9 14) (binary Add e c)
  label "d" d
  (f, l) <- withSpan (sp 8 9 17) $ do
    f <- withSpan (sp 8 13 17) (constant (-2))
    l <- binary Mul d f
    pure (f, l)
  label "L" l
  pure (Intro a b c e d f l)
  where
    sp = lineSpan "intro.hsg"

-- | The inputs' default values: @a = 2@, @b = -3@, @c = 10@.
introEnv :: Env
introEnv = bindInput "a" 2 (bindInput "b" (-3) (bindInput "c" 10 emptyEnv))

-- ---------------------------------------------------------------------------
-- neuron.hsg

-- | The nodes of @neuron.hsg@ that have a name in the source, and the
-- output of @dot(w, x)@.
data Neuron = Neuron
  { neuronX, neuronW :: [NodeId]
  , neuronB, neuronDot, neuronY :: NodeId
  }
  deriving (Show)

-- | A single tanh neuron, @docs/01-language.md@ §1.2:
--
-- > 1  input x : Vec 3 = [1.0, -2.0, 0.5]
-- > 2  param w : Vec 3 ~ uniform(-1, 1)
-- > 3  param b : Real  = 0
-- > 4
-- > 5  fn neuron(w : Vec n, b : Real, x : Vec n) : Real =
-- > 6    tanh(dot(w, x) + b)
-- > 7
-- > 8  let y = neuron(w, b, x)
--
-- Unrolling (D5) turns the vectors into the leaves @x[0]@ … @w[2]@, all
-- with the span of their declaration, and @dot(w, x)@ into
-- @(w[0]·x[0] + w[1]·x[1]) + w[2]·x[2]@, all with the span of the call in
-- the function body. The call is inlined, so the nodes of the body get the
-- body's spans; only the result is labelled @y@.
neuron :: (Neuron, Graph)
neuron = built $ do
  xs <- withSpan (sp 1 1 36) (mapM (input . element "x") [0 .. 2])
  ws <- withSpan (sp 2 1 34) (mapM (param . element "w") [0 .. 2])
  b <- withSpan (sp 3 1 20) (param "b")
  (dt, y) <- withSpan (sp 6 3 22) $ do
    (dt, s) <- withSpan (sp 6 8 21) $ do
      dt <- withSpan (sp 6 8 17) $ do
        products <- zipWithM (binary Mul) ws xs
        case products of
          p : ps -> foldM (binary Add) p ps
          [] -> constant 0 -- not reached: the vectors have three elements
      s <- binary Add dt b
      pure (dt, s)
    y <- unary Tanh s
    pure (dt, y)
  label "y" y
  pure (Neuron xs ws b dt y)
  where
    sp = lineSpan "neuron.hsg"
    element v i = Name (v <> "[" <> Text.pack (show (i :: Int)) <> "]")

-- | @x = [1, -2, 0.5]@ (the default), and parameters chosen so that the
-- numbers are easy to follow: @w = [0.5, -0.25, 1]@, @b = 0@. Then
-- @dot(w, x) = 0.5 + 0.5 + 0.5 = 1.5@.
neuronEnv :: Env
neuronEnv =
  foldr
    (\(n, v) -> bindInput n v)
    (foldr (\(n, v) -> bindParam n v) emptyEnv [("w[0]", 0.5), ("w[1]", -0.25), ("w[2]", 1), ("b", 0)])
    [("x[0]", 1), ("x[1]", -2), ("x[2]", 0.5)]
