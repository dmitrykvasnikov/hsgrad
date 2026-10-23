# 03 — `Core.Op`: the scalar operations

This step adds `HsGrad.Core.Op`, the fixed set of scalar operations that
every graph is built from. For each op the module defines three things:

- its **arity**: how many argument nodes it takes;
- its **forward function** on `Double`;
- its **local derivative rule**: the partial derivative of its output with
  respect to each argument.

There is no graph yet: no node ids, no builder, no evaluation order. Those
come in steps 3–5. `Core.Op` is the vocabulary they are all written in, so
the main work of this step is getting that vocabulary right, in particular
the representation of derivative rules, which has to serve both numeric and
symbolic differentiation (D6).

---

## 1. Theory: local derivatives and the chain rule

### 1.1 What backprop needs from an op

Reverse-mode AD (micrograd's `backward()`) walks the graph from the output
`L` back to the leaves. For every node `y = f(x₁, …, xₙ)` it knows the
*upstream* gradient `∂L/∂y` and passes a share of it to each argument:

```
∂L/∂xᵢ  +=  ∂L/∂y · ∂y/∂xᵢ
```

The factor `∂y/∂xᵢ` is the **local derivative**: it only depends on the op
and on the values at this one node, not on the rest of the graph. The chain
rule glues the local pieces together. So the only thing an op has to
provide for differentiation is one local derivative per argument. That is
exactly what micrograd's `_backward` closures compute; for `tanh`:

```python
def _backward():
    self.grad += (1 - t**2) * out.grad
```

Here `1 - t**2` is the local derivative and `out.grad` is the upstream
gradient.

### 1.2 The op set

| Op | Arity | Forward | Local derivative(s) |
|---|---|---|---|
| `Const c` | 0 | `c` | — |
| `Input n` | 0 | value bound to `n` | — |
| `Param n` | 0 | current value of `n` | — |
| `Add` | 2 | `x + y` | `1`, `1` |
| `Sub` | 2 | `x − y` | `1`, `−1` |
| `Mul` | 2 | `x · y` | `y`, `x` |
| `Div` | 2 | `z = x / y` | `y⁻¹`, `−z / y` |
| `Neg` | 1 | `−x` | `−1` |
| `PowC c` | 1 | `xᶜ` | `c · xᶜ⁻¹` |
| `Exp` | 1 | `y = eˣ` | `y` |
| `Log` | 1 | `ln x` | `x⁻¹` |
| `Tanh` | 1 | `y = tanh x` | `1 − y²` |
| `Relu` | 1 | `max(0, x)` | `step(x)` |
| `Step` | 1 | `1` if `x > 0`, else `0` | `0` |

Everything the language offers lowers to these (`01-language.md` §6):
`sigmoid(e)` is `1 / (1 + exp(-e))`, `sqrt(e)` is `e ^ 0.5`, `abs`, `max`
and `min` go through `relu`, and `sum`, `dot` and `@` are chains of `+` and
`·`.

Three rules are written in terms of the node's **own output** (`y` or `z`)
rather than its argument: `exp`, `tanh` and the second partial of `div`.
§3.2 explains why.

### 1.3 Closed under differentiation (D6)

Look at the last column: every local derivative is again built from ops in
the table (`Mul`, `Sub`, `Neg`, `PowC`, `Step`, constants), from the node's
arguments and from the node itself. That is the property D6 requires.
Step 7 (`Core.Symbolic`) runs the backprop sweep but, instead of computing
numbers, emits graph nodes for each local derivative and each `·` and `+`
of the chain rule. If some derivative needed an op outside the set, the
gradient could not be written as a graph of the same kind.

Two ops are in the set *only* because of this:

- `Step` is the derivative of `Relu`. Without it, `d relu(x)` has no
  expression.
- The derivative of `Step` is `0` everywhere except at `0` (where, strictly,
  it is a Dirac delta, not a number). Using `0` closes the chain: no further
  op is needed.

## 2. Code walkthrough

The module lives at `src/HsGrad/Core/Op.hs`.

### 2.1 Ops in three groups

```haskell
data Op
  = Leaf Leaf
  | Un UnOp
  | Bin BinOp
  deriving (Eq, Ord, Show)

data Leaf  = Const Double | Input Name | Param Name
data UnOp  = Neg | PowC Rational | Exp | Log | Tanh | Relu | Step
data BinOp = Add | Sub | Mul | Div

arity :: Op -> Int
arity (Leaf _) = 0
arity (Un _)   = 1
arity (Bin _)  = 2
```

A node of the graph (step 3) will be an `Op` plus a list of argument ids.
Grouping the ops by arity costs one extra constructor in patterns
(`Un Tanh` instead of `Tanh`), but it lets every function that only makes
sense for one group take that group's type. `forwardUn :: UnOp -> Double ->
Double` has no "wrong number of arguments" case, because a leaf or a binary
op can't be passed to it.

### 2.2 Forward functions

```haskell
forwardUn :: UnOp -> Double -> Double
forwardUn op x = case op of
  Neg -> negate x
  PowC c -> powC c x
  Exp -> exp x
  Log -> log x
  Tanh -> tanh x
  Relu -> relu x
  Step -> step x

forwardBin :: BinOp -> Double -> Double -> Double
```

Leaves have no forward function here: a constant's value is in the op, and
input and parameter values come from outside the graph (D1, D10). The
forward pass (step 4) looks them up.

`powC`, `relu` and `step` are small helpers:

```haskell
powC c x
  | denominator c == 1 = x ^^ numerator c
  | otherwise = x ** fromRational c

relu x
  | x > 0 = x
  | x <= 0 = 0
  | otherwise = x -- only NaN gets here
```

- **Integer exponents use `^^`** (repeated multiplication, with a reciprocal
  for negative powers). That is exact for small powers and works for
  negative bases: `(-3) ^ 2 = 9`. Other exponents use `**`, which returns
  `NaN` for a negative base, as it should: `(-4) ^ 0.5` has no real value.
- **`relu` and `step` keep `NaN`.** `max 0 x` would turn `NaN` into `0`
  (every comparison with `NaN` is false), hiding a problem from the forward
  pass's non-finite check.
- **`step 0 = 0`**, so `relu'(0) = 0`. Any value in `[0, 1]` is a valid
  subgradient of `relu` at `0`; `0` is what micrograd uses
  (`(out.data > 0) * out.grad`).

Forward functions never throw. Outside an op's domain they return what IEEE
arithmetic returns (`log 0 = -Infinity`, `log (-1) = NaN`, `1 / 0 =
Infinity`). Detecting those is the forward pass's job (error E0501), because
only it knows which source span produced the node (D4).

### 2.3 Derivative rules as data

```haskell
data DExpr
  = DLit Double            -- a constant
  | DArg Int               -- the value of argument i (0 = x, 1 = y)
  | DOut                   -- the value of the node itself
  | DUn UnOp DExpr         -- a unary op applied to a sub-expression
  | DBin BinOp DExpr DExpr -- a binary op applied to two sub-expressions
```

A `DExpr` is a tiny expression language for local derivatives. The rules
are a direct transcription of the table in §1.2:

```haskell
unPartial :: UnOp -> DExpr
unPartial op = case op of
  Neg -> DLit (-1)
  PowC c
    | c == 0 -> DLit 0
    | c == 1 -> DLit 1
    | c == 2 -> DBin Mul (DLit 2) x
    | otherwise -> DBin Mul (DLit (fromRational c)) (DUn (PowC (c - 1)) x)
  Exp -> DOut
  Log -> DUn (PowC (-1)) x
  Tanh -> DBin Sub (DLit 1) (DUn (PowC 2) DOut)
  Relu -> DUn Step x
  Step -> DLit 0
  where
    x = DArg 0

binPartials :: BinOp -> (DExpr, DExpr)
binPartials op = case op of
  Add -> (DLit 1, DLit 1)
  Sub -> (DLit 1, DLit (-1))
  Mul -> (y, x)
  Div -> (DUn (PowC (-1)) y, DUn Neg (DBin Div DOut y))
  where
    x = DArg 0
    y = DArg 1
```

`partials :: Op -> [DExpr]` puts both behind one interface, one rule per
argument in argument order (`[]` for leaves). Its length is always the op's
arity, which matches the graph's `[NodeId]` argument lists.

The special cases of `PowC` keep the rule from creating pointless nodes:
`x²` differentiates to `2 · x`, not `2 · x¹`, and `x¹` to `1`, not `1 · x⁰`.
`x⁰` differentiates to `0`, not `0 · x⁻¹`, which would also be wrong at
`x = 0`, where it evaluates to `0 · ∞ = NaN`.

The numeric interpretation evaluates a rule given the argument values and
the node's value:

```haskell
evalDExpr :: [Double] -> Double -> DExpr -> Double
evalDExpr args out = go
  where
    go e = case e of
      DLit c -> c
      DArg i -> args !! i
      DOut -> out
      DUn op a -> forwardUn op (go a)
      DBin op a b -> forwardBin op (go a) (go b)

localGrads :: Op -> [Double] -> Double -> [Double]
localGrads op args out = map (evalDExpr args out) (partials op)
```

For micrograd's `2 · −3`, `localGrads (Bin Mul) [2, -3] (-6)` is `[-3, 2]`.

## 3. Design decisions

### 3.1 Why rules are data and not functions

micrograd stores each derivative as a Python closure that computes a
number. The Haskell equivalent would be:

```haskell
unPartial :: UnOp -> Double -> Double -> Double   -- x, y ↦ dy/dx
```

That is enough for numeric backprop (step 5), but useless for symbolic
differentiation (step 7): a function from numbers to numbers can only be
*run*, not *inspected*. Step 7 needs to know that `d tanh` is "one minus the
square of this node" so that it can create those nodes.

With rules as data (`DExpr`), there is **one** definition of each rule and
**two interpreters**:

| | `DLit c` | `DArg i` | `DOut` | `DUn op a` / `DBin op a b` |
|---|---|---|---|---|
| numeric (`evalDExpr`, step 5) | `c` | value of argument `i` | value of the node | `forwardUn` / `forwardBin` on the results |
| symbolic (step 7) | a `Const c` node | the argument's `NodeId` | the node's own `NodeId` | a new `Un op` / `Bin op` node |

Both interpreters are short structural recursions over the same tree. The
numeric one even evaluates interior ops with the very functions the forward
pass uses, so the two can't drift apart. And because the tests check the
rules numerically against finite differences (§4), the symbolic gradients
of step 7 inherit that check.

**Closure is guaranteed by the types.** A `DExpr` can only contain `UnOp`s,
`BinOp`s, constants and references to existing nodes. It is impossible to
write a rule that needs an op outside the set; the module wouldn't compile.

Alternatives considered:

- **Tagless final**: a type class `class Expr e where lit :: Double -> e;
  un :: UnOp -> e -> e; …`, with rules polymorphic in `e`, instantiated once
  with `Double` and once with a builder action. It avoids the intermediate
  tree, but it's more abstract, and the rules can't be printed, compared or
  tested as values. The data type is easier to follow and to test.
- **A polymorphic `DExpr v`** with a type of variables per arity (to rule
  out `DArg 1` in a unary rule statically). More type safety, but more
  machinery for a fact that one unit test already checks: every
  rule only mentions arguments below the op's arity.

### 3.2 Why rules may refer to the node's own output (`DOut`)

`d tanh(x) = 1 − tanh(x)²` could be written in terms of `x` alone:
`DBin Sub (DLit 1) (DUn (PowC 2) (DUn Tanh (DArg 0)))`. That computes
`tanh x` a second time, in backprop as a number and in the symbolic
gradient as a node. With `DOut` the rule says "the node itself", and:

- numeric backprop reuses the value from the forward pass, as micrograd
  does with `t` and `out.data`;
- the symbolic gradient points at the existing `tanh` node, which is exactly
  the sharing D6 describes. In `neuron.hsg` (`01-language.md` §1.2), all
  three `∂y/∂w[i]` share one `1 − tanh(…)²` node, which hangs off the
  forward graph's `tanh` node.

(With hash-consing, D3, a second `tanh(x)` node would be merged with the
first anyway, but only in step 7, and numeric backprop would still compute
it twice. `DOut` says what we mean.)

`exp` (`d eˣ = eˣ`) and the second partial of `div`
(`−x/y² = −z/y`) use the same trick.

### 3.3 How `PowC` carries its exponent: `Rational`

The exponent of `x ^ c` is a constant, but of what type?

- **`Double`** is the obvious choice, but derivative rules do arithmetic on
  exponents (`c − 1`), and floating point makes that inexact: `0.1 − 1` is
  not the same `Double` as `-0.9`. The derivative of `x ^ 0.1` would then not
  hash-cons (D3) with a literal `x ^ -0.9`, and repeated differentiation
  would drift.
- **`Int`** would make the simplifier's job simplest (D7 wants integer
  exponents in monomials), but `sqrt(e)` lowers to `e ^ 0.5`.
- **`Rational`** is exact: `0.1 − 1 == -0.9`, `1/3 − 1 == -2/3`. Testing
  whether an exponent is an integer is exact too (`denominator c == 1`),
  which is how `powC` picks `^^` and how step 8 will decide whether `x ^ c`
  is a monomial or an atom. And unlike `Double`, its `Eq` and `Ord` are
  total, with no `NaN`.

A Haskell literal such as `0.5 :: Rational` is exactly `1 % 2`, and the
parser (step 11) can read `0.5` into a `Rational` without rounding.
Converting to `Double` happens only at evaluation time (`fromRational c`).

### 3.4 Leaf names

```haskell
newtype Name = Name Text
  deriving (Eq, Ord, Show)

instance IsString Name where
  fromString = Name . Text.pack
```

- **Opaque and flat.** Core doesn't know that `W1[3,0]` is an element of a
  matrix or that `moons#3.x[1]` is a dataset slot (D10). It only needs to
  know when two leaves are the same leaf: exactly when their names are
  equal. The compiler (step 14) decides how to spell names, so the language's
  structure doesn't leak into Core (layer rule, `00-architecture.md` §2).
  The human-readable label of D4 is a separate piece of provenance on the
  node; the name is the leaf's identity.
- **`Text`, not `String`.** `Text` is the idiomatic type for names, the
  parser (megaparsec over `Text`) and the diagnostics will use it, and names
  are compared on every hash-consing lookup of a leaf. This step adds the
  `text` dependency (`^>=2.1`; it ships with GHC, so nothing is downloaded).
- **A `newtype`**, so a name can't be confused with other text, such as a
  source snippet or a label.
- **`IsString`** lets code with `OverloadedStrings` write `Input "x"`; the
  tests use it.

`Input` and `Param` are separate leaves even though both are "a named value
from outside": training updates parameters and not inputs, gradients are
usually taken with respect to parameters, and warning W0203 ("parameter
can't affect `loss`") needs to find them. `Input "w"` and `Param "w"` are
different ops, so they are different leaves.

### 3.5 `Eq` and `Ord` for hash-consing

All op types derive `Eq` and `Ord`, so that step 3 can use
`(Op, [NodeId])` as the key of a `Map` (D3). Deriving is fine for every
constructor except one subtlety in `Const Double`:

- **`NaN`**: `NaN == NaN` is `False`, and `compare` with `NaN` is
  inconsistent. A `NaN` key would break the ordering invariant a `Map` relies
  on, and lookups of *other* keys could then fail. So **graph constants must
  be finite**. Step 3's builder should reject a non-finite constant, and
  step 8's constant folding must not fold into one (`1/0`).
- **`-0.0 == 0.0`**: the two zeros would be merged into one node. The only
  observable difference is the sign of `1/c`, which is an infinity, i.e. an
  error anyway.

`PowC`'s `Rational` has none of these problems (§3.3).

### 3.6 Why `Sub`, `Div` and `Neg` are ops

They are not needed: `x − y = x + (−1)·y`, `x / y = x · y⁻¹`,
`−x = (−1)·x`. micrograd implements them that way. We keep them as ops
anyway, because graphs and symbolic gradients are meant to be *read*
(Graphviz, `--format text`): `a / b` should show up as one `/` node, not as
a `·` and a `^-1`. The cost is four more cases in each function of this
module. The simplifier (D7) normalises them into its sum-of-products form
regardless.

## 4. Tests

`test/HsGrad/Core/OpSpec.hs` has four groups:

- **Forward functions**: known values (`(-3)^2 = 9`, `9^0.5 = 3`, `relu 0 = 0`,
  `log 0 = -Infinity`, NaN stays NaN), and properties such as
  `exp (log x) ≈ x`, `tanh` is odd, `relu x = x · step x`, integer powers
  equal repeated multiplication.
- **Structure**: every op has exactly `arity op` rules, rules only mention
  arguments `< arity op`, exponents compare exactly
  (`PowC (0.1 − 1) == PowC (−0.9)`), and `Input "w" /= Param "w"`.
- **Rule shapes**: a few rules checked literally, documenting D6
  (`relu ↦ step`, `tanh` and `exp` use `DOut`, the `PowC` special cases).
- **Rules versus finite differences**: the heart of this step.

### 4.1 Central finite differences

For each op, QuickCheck picks random inputs, evaluates the rule with
`localGrads`, and compares it with

```
f'(x) ≈ (f(x + h) − f(x − h)) / 2h,     h = 10⁻⁶ · max(1, |x|)
```

The **central** difference is accurate to O(h²); the one-sided
`(f(x + h) − f(x)) / h` only to O(h). Rounding error grows like
`ε·|f|/h` as `h` shrinks, so `h` can't be tiny either; `10⁻⁶` balances the
two, and scaling it with `|x|` keeps `x ± h` distinguishable from `x`. The
comparison uses a relative tolerance of `10⁻⁶` (absolute near zero).

Binary ops are checked in both arguments, each partial against a
difference in that argument with the other one held fixed. `PowC` is
checked for eleven exponents (`−3 … 3`, including `−1/2`, `1/3`, `3/2` and
the special cases `0`, `1`, `2`).

### 4.2 Avoiding singular points

A finite difference only approximates the derivative where the function is
smooth on `[x − h, x + h]`. The generators keep away from where it isn't:

| Op | Inputs | Why |
|---|---|---|
| `log`, non-integer `x ^ c` | `[0.1, 10]` | undefined for `x ≤ 0` |
| negative integer `x ^ c`, `x / y` (in `y`) | `|x| ≥ 0.01` | pole at `0` |
| `relu`, `step` | `|x| ≥ 0.01` | kink / jump at `0`; `h` is far smaller than `0.01` |
| everything else | `[−5, 5]` | `e⁵ ≈ 148` keeps `exp` well-conditioned |

The behaviour *at* the singular points (`relu'(0) = 0`, `d x⁰ = 0` at
`x = 0`) is tested with unit tests instead.

As a sanity check of the tests themselves, breaking the `tanh` rule (to
`1 − y`) or the `div` rule (to `−z/x`) makes the corresponding properties
fail.

## 5. What later steps build on

- **Step 3 (`Core.Graph`)**: a node is an `Op` plus `[NodeId]` of length
  `arity op`; the hash-consing key is `(Op, [NodeId])`; constants must be
  finite (§3.5).
- **Step 4 (`Core.Eval`)**: leaves are looked up, other nodes use
  `forwardUn` / `forwardBin`; non-finite results become E0501.
- **Step 5 (`Core.Backprop`)**: for each node in descending id order,
  `zipWith (\a d -> grad a += g * d) args (localGrads op argVals outVal)`.
- **Step 7 (`Core.Symbolic`)**: the same sweep, with a second interpreter of
  `DExpr` that emits nodes (`DArg i` ↦ argument id, `DOut` ↦ the node's id).
- **Step 8 (`Core.Simplify`)**: `PowC` with `denominator c == 1` is a
  monomial factor; other ops (`exp`, `log`, `tanh`, `relu`, `step`,
  fractional powers) are atoms.
