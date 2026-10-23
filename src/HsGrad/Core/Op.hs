-- | The fixed set of scalar operations of the core graph.
--
-- Every node of a graph ("HsGrad.Core.Graph", step 3) is one 'Op' applied to
-- zero, one or two argument nodes. This module says, for each op:
--
-- * its 'arity': how many arguments it takes;
-- * its forward function on 'Double' ('forwardUn', 'forwardBin');
-- * its local derivative rule ('partials'): the partial derivative of the
--   op's output with respect to each argument.
--
-- Derivative rules are /data/, not functions: small expressions ('DExpr')
-- built from the very same ops. Numeric backprop (step 5) evaluates them to
-- numbers with 'evalDExpr'; symbolic differentiation (step 7) will turn them
-- into new graph nodes. Because a 'DExpr' can only mention ops of this module,
-- the op set is closed under differentiation by construction (decision D6).
--
-- The module knows nothing about graphs, node ids or evaluation order.
-- See @docs/03-core-op.md@.
module HsGrad.Core.Op
  ( -- * Leaf names
    Name (..)

    -- * Operations
  , Op (..)
  , Leaf (..)
  , UnOp (..)
  , BinOp (..)
  , arity

    -- * Forward functions
  , forwardUn
  , forwardBin

    -- * Local derivative rules
  , DExpr (..)
  , unPartial
  , binPartials
  , partials
  , evalDExpr
  , localGrads
  ) where

import Data.Ratio (denominator, numerator)
import Data.String (IsString (..))
import Data.Text (Text)
import Data.Text qualified as Text

-- ---------------------------------------------------------------------------
-- Leaf names

-- | The name of an input or parameter leaf, for example @a@, @W1[3,0]@ or
-- @moons#3.x[1]@ (see @01-language.md@ §8).
--
-- Core treats names as opaque labels: two leaves are the same leaf exactly
-- when their names are equal. How names are spelled is up to the compiler.
newtype Name = Name Text
  deriving (Eq, Ord, Show)

-- | With @OverloadedStrings@, a string literal can stand for a name:
-- @Input "x"@ instead of @Input (Name (Text.pack "x"))@.
instance IsString Name where
  fromString = Name . Text.pack

-- ---------------------------------------------------------------------------
-- Operations

-- | A scalar operation: a leaf, a unary op or a binary op.
--
-- The three groups have arity 0, 1 and 2. Splitting them into separate types
-- lets the forward functions below be total: 'forwardUn' only ever sees
-- unary ops, and needs no "wrong number of arguments" case.
--
-- 'Eq' and 'Ord' are derived so that @(Op, [NodeId])@ can be the key of the
-- hash-consing table in step 3 (decision D3). Note that 'Const' compares its
-- 'Double' with the usual IEEE rules: @0.0 == -0.0@ and @NaN /= NaN@. A NaN
-- constant would therefore break the ordering a 'Data.Map.Map' relies on, so
-- graph constants must be finite (see @docs/03-core-op.md@ §3.5).
data Op
  = Leaf Leaf
  | Un UnOp
  | Bin BinOp
  deriving (Eq, Ord, Show)

-- | Leaves: nodes without arguments.
data Leaf
  = -- | A literal number, such as the @-2.0@ in @d * -2.0@.
    Const Double
  | -- | A value supplied from outside: an @input@ or a dataset slot (D10).
    Input Name
  | -- | A trainable parameter: training changes it, and gradients are
    -- usually taken with respect to these.
    Param Name
  deriving (Eq, Ord, Show)

-- | Operations with one argument @x@.
data UnOp
  = -- | @-x@
    Neg
  | -- | @x ^ c@ for a constant exponent @c@. The exponent is a 'Rational' so
    -- that exponent arithmetic in derivative rules is exact (see 'unPartial').
    PowC Rational
  | -- | @exp x@
    Exp
  | -- | @log x@, the natural logarithm
    Log
  | -- | @tanh x@
    Tanh
  | -- | @max(0, x)@
    Relu
  | -- | The Heaviside step: @1@ if @x > 0@, else @0@. It is in the op set
    -- because it is the derivative of 'Relu' (D6).
    Step
  deriving (Eq, Ord, Show)

-- | Operations with two arguments @x@ and @y@, in that order.
data BinOp
  = -- | @x + y@
    Add
  | -- | @x - y@
    Sub
  | -- | @x * y@
    Mul
  | -- | @x / y@
    Div
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | How many argument nodes an op takes.
arity :: Op -> Int
arity (Leaf _) = 0
arity (Un _) = 1
arity (Bin _) = 2

-- ---------------------------------------------------------------------------
-- Forward functions

-- | The forward function of a unary op.
--
-- Like the 'Double' arithmetic they are built from, forward functions never
-- throw: outside an op's domain they return @NaN@ or an infinity (for example
-- @log 0 = -Infinity@). Reporting those is the job of the forward pass
-- (error E0501, step 4), which knows where in the source the node came from.
forwardUn :: UnOp -> Double -> Double
forwardUn op x = case op of
  Neg -> negate x
  PowC c -> powC c x
  Exp -> exp x
  Log -> log x
  Tanh -> tanh x
  Relu -> relu x
  Step -> step x

-- | The forward function of a binary op.
forwardBin :: BinOp -> Double -> Double -> Double
forwardBin op x y = case op of
  Add -> x + y
  Sub -> x - y
  Mul -> x * y
  Div -> x / y

-- | @x ^ c@. Integer exponents use repeated multiplication ('^^'), which is
-- exact for small powers and defined for negative @x@: @(-3) ^ 2 = 9@.
-- Other exponents use '**', which gives @NaN@ for negative @x@, as it should
-- (@(-4) ^ 0.5@ has no real value).
powC :: Rational -> Double -> Double
powC c x
  | denominator c == 1 = x ^^ numerator c
  | otherwise = x ** fromRational c

-- | @max(0, x)@. Written with guards rather than 'max' so that a @NaN@
-- argument stays @NaN@ instead of silently turning into @0@.
relu :: Double -> Double
relu x
  | x > 0 = x
  | x <= 0 = 0
  | otherwise = x -- only NaN gets here

-- | The Heaviside step, with @step 0 = 0@. This matches micrograd, where the
-- gradient of relu at exactly 0 is 0. @NaN@ stays @NaN@, as in 'relu'.
step :: Double -> Double
step x
  | x > 0 = 1
  | x <= 0 = 0
  | otherwise = x -- only NaN gets here

-- ---------------------------------------------------------------------------
-- Local derivative rules

-- | A local derivative rule: an expression for one partial derivative of a
-- node, in terms of the node's arguments and the node's own output.
--
-- For example, the rule for @y = tanh x@ is @1 - y^2@:
--
-- > DBin Sub (DLit 1) (DUn (PowC 2) DOut)
--
-- Referring to the output with 'DOut' lets a rule reuse a value that is
-- already computed (for numeric backprop) or a node that already exists (for
-- symbolic gradients) instead of computing @tanh x@ a second time.
--
-- Interior nodes can only be 'UnOp's and 'BinOp's, so every rule is
-- expressible with the op set itself: the set is closed under
-- differentiation (D6), and the type checker guarantees it.
data DExpr
  = -- | A constant.
    DLit Double
  | -- | The value of argument @i@ of the node (0-based: 0 is @x@, 1 is @y@).
    DArg Int
  | -- | The value of the node itself.
    DOut
  | -- | A unary op applied to a sub-expression.
    DUn UnOp DExpr
  | -- | A binary op applied to two sub-expressions.
    DBin BinOp DExpr DExpr
  deriving (Eq, Show)

-- | The derivative rule of a unary op @y = f(x)@: an expression for @dy/dx@.
unPartial :: UnOp -> DExpr
unPartial op = case op of
  -- d(-x) = -1
  Neg -> DLit (-1)
  -- d(x^c) = c * x^(c-1). The small cases avoid creating pointless nodes
  -- such as @x^0@ or @x^1@; @c = 0@ also avoids @0 * 0^(-1) = NaN@ at @x = 0@.
  PowC c
    | c == 0 -> DLit 0
    | c == 1 -> DLit 1
    | c == 2 -> DBin Mul (DLit 2) x
    | otherwise -> DBin Mul (DLit (fromRational c)) (DUn (PowC (c - 1)) x)
  -- d(exp x) = exp x, which is the node itself
  Exp -> DOut
  -- d(log x) = x^(-1)
  Log -> DUn (PowC (-1)) x
  -- d(tanh x) = 1 - (tanh x)^2, reusing the node itself
  Tanh -> DBin Sub (DLit 1) (DUn (PowC 2) DOut)
  -- d(relu x) = step x
  Relu -> DUn Step x
  -- d(step x) = 0 (everywhere except at 0, where it is undefined)
  Step -> DLit 0
  where
    x = DArg 0

-- | The derivative rules of a binary op @z = g(x, y)@: expressions for
-- @(dz/dx, dz/dy)@.
binPartials :: BinOp -> (DExpr, DExpr)
binPartials op = case op of
  Add -> (DLit 1, DLit 1)
  Sub -> (DLit 1, DLit (-1))
  Mul -> (y, x)
  -- d(x/y)/dx = y^(-1);  d(x/y)/dy = -x/y^2 = -z/y, reusing the node itself
  Div -> (DUn (PowC (-1)) y, DUn Neg (DBin Div DOut y))
  where
    x = DArg 0
    y = DArg 1

-- | The derivative rules of any op: one expression per argument, in argument
-- order. Leaves have no arguments, so they have no rules.
--
-- prop> length (partials op) == arity op
partials :: Op -> [DExpr]
partials (Leaf _) = []
partials (Un op) = [unPartial op]
partials (Bin op) = [dx, dy]
  where
    (dx, dy) = binPartials op

-- | Evaluate a rule to a number, given the values of the node's arguments
-- and of the node itself.
--
-- This is the /numeric/ interpretation of a 'DExpr', used by backprop. It
-- evaluates interior ops with the same 'forwardUn' and 'forwardBin' that the
-- forward pass uses. The /symbolic/ interpretation (step 7) has the same
-- shape, but emits a graph node for each 'DUn' and 'DBin' instead.
--
-- Every rule only mentions arguments below the op's arity (this is tested),
-- so the list lookup cannot fail when @args@ has 'arity' elements.
evalDExpr :: [Double] -> Double -> DExpr -> Double
evalDExpr args out = go
  where
    go e = case e of
      DLit c -> c
      DArg i -> args !! i
      DOut -> out
      DUn op a -> forwardUn op (go a)
      DBin op a b -> forwardBin op (go a) (go b)

-- | The numeric local gradients of a node: the value of each of its
-- 'partials', given the argument values and the node's own value.
--
-- >>> localGrads (Bin Mul) [2, -3] (-6)
-- [-3.0,2.0]
localGrads :: Op -> [Double] -> Double -> [Double]
localGrads op args out = map (evalDExpr args out) (partials op)
