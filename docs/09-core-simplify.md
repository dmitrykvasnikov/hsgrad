# 09 — `Core.Simplify` and `Core.Pretty`: readable gradients

This step adds two modules that turn the gradient graphs of step 7 into
something a person can read.

- `HsGrad.Core.Simplify` rewrites expressions into a **canonical sum of
  products** (decision D7) and builds the result as a new graph. That
  gathers like terms (`x + x → 2·x`), merges powers (`x·x·x → x³`), folds
  coefficients and constants (`2·x·3 → 6·x`, `1 + 1 → 2`) and applies the
  identities of `0` and `1`, without ever distributing a product over a
  sum.
- `HsGrad.Core.Pretty` prints the expression rooted at a node as text:
  precedence-aware, `·` for products, superscripts for integer powers, and
  named bindings for shared subexpressions, so a DAG never prints
  exponentially long.

Together they reach the milestone of the roadmap. The gradients of
`intro.hsg` (`01-language.md` §1.1), simplified and printed:

```
∂L/∂a = -2·b
∂L/∂b = -2·a
∂L/∂c = -2
```

and those of the tanh neuron (§1.2), which all share one subexpression:

```
let t1 = 1 - tanh(x[0]·w[0] + x[1]·w[1] + x[2]·w[2] + b)²
∂y/∂b    = t1
∂y/∂w[0] = x[0]·t1
∂y/∂w[1] = x[1]·t1
…
```

---

## 1. Theory

### 1.1 Simplification is evaluation in another number system

The forward pass (step 4) walks the graph from the leaves up and computes a
`Double` for every node from the `Double`s of its arguments. The
simplifier does exactly the same walk, but its "numbers" are **canonical
forms**: symbolic values that can be added, multiplied and raised to powers
like numbers, and that make equal what is obviously equal.

| | forward pass | simplifier |
|---|---|---|
| a constant `c` | `c` | the form `c` |
| a leaf `x` | its value from the `Env` | the form `x` (an atom) |
| `a + b` | `va + vb` | the sum of two forms |
| `a · b` | `va · vb` | the product of two forms |
| `exp a` | `exp va` | the atom `exp(…)` of the simplified `a` |
| memoised per node | `IntMap Double` | `IntMap Poly` |

Once every node has a form, the forms of the outputs are built back into
nodes. Nothing in the simplifier is a "rewrite rule" that searches for a
pattern like `x + 0`: the identities fall out of the arithmetic on forms,
the way `2 + 0 = 2` falls out of arithmetic on numbers.

### 1.2 The canonical form: a sum of products

A form is a **constant plus terms**, each term a **coefficient** times a
**monomial**, and each monomial a product of **atoms** raised to non-zero
integer powers:

```
c + k₁·m₁ + k₂·m₂ + …          e.g.   1 - 2·x²·y⁻¹ + 3·exp(x)
```

In Haskell (§3.2) that is a `Double` plus a `Map Monomial Double`, and a
monomial is a `Map Atom Integer`. Using maps keyed by monomials and atoms is
what does the work:

- **Like terms** have the same monomial, so adding two forms adds their
  coefficients: `x + x` is `{x ↦ 1} + {x ↦ 1} = {x ↦ 2}`, which is `2·x`.
  A coefficient that becomes `0` is dropped: `x - x = 0`, and
  `x·y - y·x = 0`, because `x·y` and `y·x` are the same monomial
  `{x ↦ 1, y ↦ 1}`.
- **Powers** of the same atom have the same key, so multiplying monomials
  adds exponents: `x·x·x = {x ↦ 3}`, `x·x⁻¹ = {}` = `1`.
- **Coefficients and constants** are just numbers: `2·x·3` multiplies
  `2·{x ↦ 1}` by `3`, giving `6·{x ↦ 1}`; `exp(0)` is the constant `1`.
- **The identities of 0 and 1** need no special case at all: `x + 0` adds
  the constant `0`, `1·x` multiplies the coefficient by `1`, `x¹` is `x`.

Because exponents can be negative, this is a *Laurent polynomial* in the
atoms; division is multiplication by a negative power: `x/y = x·y⁻¹`.

### 1.3 Atoms: what the form cannot look into

An atom is anything the canonical form treats as an opaque variable:

- the **leaves**: inputs and parameters (constants are not atoms, they are
  folded);
- the **non-polynomial ops** `exp`, `log`, `tanh`, `relu`, `step` and
  **fractional powers** (`x^0.5`), each applied to its *simplified*
  argument, so `tanh(x)` and `tanh(x + 0)` are the same atom;
- **sums used as a factor**, such as the `x + y` in `x·(x + y)` (§1.4);
- and, rarely, a node whose folding would overflow (`1/0`, `exp(1000)`),
  which stays as it is (§1.6).

`x ^ c` is a monomial factor when `c` is an integer and an atom otherwise.
`PowC` carries its exponent as a `Rational` (`docs/03-core-op.md` §3.3),
so "is it an integer?" is an exact test, `denominator c == 1`, and
`(x²)³ = x⁶` is exact arithmetic on integers.

Atoms are nodes of the *new* graph. Hash-consing (D3) makes structurally
equal expressions the same node, so "the same atom" is simply "the same
node id": the two `tanh` atoms above are one node because their simplified
arguments are one node.

### 1.4 Products are not distributed over sums

The step that is *not* taken is expanding products of sums:

```
(a + b)·(c + d)   ↛   a·c + a·d + b·c + b·d
```

Expansion multiplies the numbers of terms, so a product of `n` sums of two
terms each expands to `2ⁿ` terms; the gradient of a deep network is full of
such products. Instead, a sum that is multiplied by something that is not a
constant becomes **one atom**:

```
x·(y + z)          stays     x·(y + z)
(x + y)·(y + x)    becomes   (x + y)²        one atom, squared
(x + y)/(x + y)    becomes   1
```

Multiplying by a **constant** is distributed, because it never adds terms:
`2·(x + y)` becomes `2·x + 2·y`, and `-(x - y)` becomes `-x + y`. This lets
constants from different places meet and fold: `2·(x + y) - 2·y` is `2·x`.

The price is that the form is canonical only up to expansion:
`x·(y + z)` and `x·y + x·z` are equal but have different forms, and the
simplifier will not notice. That is the deliberate trade of D7. A full
canonical form for rational expressions needs expansion (and polynomial
GCDs), which is computer algebra, not this module.

One invariant keeps the forms canonical anyway: **a term is never a sum
atom on its own**. A product can leave one behind, as in `(x + y)·z/z`, and
`k·(x + y)` with the sum as an atom would differ from `k·x + k·y`, the form
of the very same expression written with a constant factor. Such a term is
put back as the scaled sum (`termS`, §3.4). Without it, simplifying twice
could give a different result than simplifying once (§5.1).

### 1.5 Sharing: memoised per node

A graph is a DAG. The chain `x₀ = x`, `xₖ₊₁ = xₖ·xₖ` has 31 nodes, but
written out as a tree `x₃₀` has two billion (`docs/08-core-symbolic.md`
§1.3). A simplifier that recursed over the expression as a tree would take
two billion steps; this one computes the form of each **node** once
(memoised by `NodeId`) and gets `{x ↦ 2³⁰}`, which it builds as the two
nodes of `x¹⁰⁷³⁷⁴¹⁸²⁴`.

The forms themselves stay small too: a form has at most one term per
distinct monomial, and monomials only come from nodes of the graph, so no
form is larger than the graph. Exponents can grow exponentially in value
(`2³⁰` above), which `Integer` handles.

The result is built with the hash-consing builder, so equal forms become
the same node, and nodes built for one output are reused for the others.
Rebuilding does lose some sharing in one situation: a long sum
`s₁ = x₁, sₖ = sₖ₋₁ + xₖ` in which *every* partial sum is also used inside an
atom (`exp(sₖ)`) is rebuilt once per partial sum, in canonical order, which
can cost quadratic size instead of linear. Without expansion there is no
exponential case. The tests check deep shared chains (§5).

### 1.6 Domains: where the result is defined

Every identity the simplifier uses holds for real numbers **wherever both
sides are defined**. But some make an expression defined in more places
than before, because a subexpression that cancels or is multiplied by zero
is no longer computed at all. "Defined" means what `Core.Eval` means: every
intermediate value is finite (`docs/05-core-eval.md` §3.4).

| Identity | The original is undefined, the result defined, at |
|---|---|
| `x·x⁻¹ → 1`, `x/x → 1`, `x²/x → x` | `x = 0` |
| `(x⁻¹)⁻¹ → x` | `x = 0` |
| `(x + y)/(x + y) → 1` | `x + y = 0` |
| `e - e → 0`, like terms cancelling | wherever `e` is undefined: `log x - log x` at `x ≤ 0` |
| `0·e → 0`, `0/e → 0` | wherever `e` is undefined (and `e = 0` for `0/e`) |
| `e⁰ → 1` | wherever `e` is undefined |

So the guarantee is one-sided: **wherever the original is defined, the
simplified expression is defined and has the same value** (up to rounding,
§1.7); it may also be defined where the original is not. This is the
`x · x⁻¹ → 1` caveat of D7, and the soundness tests sample away from the
singular points (§5.2). `x⁰ → 1` is exact even at `x = 0`, because the
forward pass computes `0 ^^ 0 = 1`.

Constant folding uses the forward pass's own functions (`forwardUn`,
`forwardBin`), so a folded constant is bit for bit what evaluation would
give. A fold whose result is not finite is **refused**: `x/0`, `log(0)`,
`(-4)^0.5` and `exp(1000)` stay as nodes, evaluating them still reports the
error where it happens, and the graph never gets a NaN or infinite
constant, which the builder would reject (`NonFiniteConst`, step 3).

Fractional powers are atoms, and their exponents are never merged with
others: `x^0.5·x^0.5` is `(x^0.5)²`, not `x`. Merging is not always valid
(`(x²)^0.5` is `|x|`, not `x`), and where it is valid it would only extend
the domain once more (`(x^0.5)² = x` is defined at `x < 0`, the original
is not). Keeping fractional powers as atoms avoids both.

### 1.7 Floating point

The simplified expression computes the same real number in a different
order: `(2·x)·3` becomes `6·x`, a sum is added in canonical order,
`x/3` becomes `0.3333333333333333·x`. In floating point that changes the
rounding, so values agree up to a few ulps of the *largest intermediate
value*, not of the result: `(x + y) - x` simplifies to `y`, which is
*more* accurate than the original when `x` is much larger than `y`.

The tests compare with a tolerance of `10⁻¹⁰` times the largest magnitude
in the original computation (§5.2). Coefficients are `Double`s; exact
`Rational` coefficients were considered (`x/3` would stay `x/3`) and
rejected: constants in graphs are `Double`s anyway, every `exp` or `tanh`
of a constant goes back to `Double`, and products of many coefficients like
`0.1 = 3602879701896397/2⁵⁵` grow huge denominators.

Rearranging could in theory overflow where the original does not
(`(x·y)ⁿ → xⁿ·yⁿ` with a huge `x` and a tiny `y`). The soundness properties
treat "the simplified expression fails where the original did not" as a
failure; it did not happen in 60,000 random graphs per property.

### 1.8 Printing a DAG

`Core.Pretty` prints the expression rooted at a node in the usual notation,
with the precedence of the `.hsg` language (`01-language.md` §5.1):
powers bind tightest, then unary minus, then `·` and `/`, then `+` and
`-`, all binary ones left-associative. It is **faithful**: it shows the
nodes as they are and adds exactly the parentheses the structure needs,
`x·(y·z)` included. Simplifying is the other module's job.

Printing a node by printing its arguments is printing the tree, which is
exponential on DAGs. The printer gives a **binding** to every shared
subexpression that is not tiny, and prints it once:

```
let t1 = a·b + c
in t1·t1 + tanh(t1)
```

A subexpression gets a binding when it is used at least twice and its
text would have more than three nodes (`prettyInlineSize`); `x²` and
`a·b` are simply repeated. Every node is then printed once, except small
shared ones, which are printed once per use with at most three nodes each,
so the text is linear in the size of the graph. The chain of 30 squarings
prints in 15 lines instead of two billion nodes.

## 2. Simplify and print together

| | `Core.Symbolic` (step 7) | `Core.Simplify` | `Core.Pretty` |
|---|---|---|---|
| input | a graph and an output | a graph and some nodes | a graph and some nodes |
| result | gradient nodes, *added* to the graph | simplified nodes, in a *new* graph | text |
| sharing | hash-consing, one sweep | memoised per node, hash-consing | bindings for shared nodes |
| provenance | spans and `∂L/∂a` labels | kept on the simplified nodes | labels name bindings |

The typical use, `hsgrad derive --simplify --format text` (step 15):

```haskell
do (gns, g')  <- symbolicGradients g loss
   let wrt = [gns Map.! a, gns Map.! b, gns Map.! c]
   (simp, g'') <- simplify g' wrt
   pure (renderNodes defaultPrettyOptions g'' (map (simp Map.!) wrt))
```

## 3. Code walkthrough

`src/HsGrad/Core/Simplify.hs` and `src/HsGrad/Core/Pretty.hs`.

### 3.1 The API

```haskell
simplifyNodes :: Graph -> [NodeId] -> Builder (Map NodeId NodeId)
simplify      :: Graph -> [NodeId] -> Either GraphError (Map NodeId NodeId, Graph)
```

`simplifyNodes g outs` reads the source graph `g` and adds the simplified
expressions to the graph being built; the result maps each given node to
its simplified node. It is a `Builder` action, like `backwardNodes`, so it
composes with other builder code. `simplify` is the one-off version: it
builds into a fresh graph and then copies only what the results use into
another fresh graph (`copyReachable`), because building forms can leave
dead nodes behind (the atom `exp(x)` of `exp(x) - exp(x)`). Its graph is
exactly the simplified expressions: `simplify` of `x - x` gives a graph
with the single node `0`.

Several outputs are simplified together, so they share nodes; this is what
`derive --wrt a,b,c` needs. Ids that are not nodes of `g` are left out of
the map. The `Either` is the builder's; it cannot be a `Left`, since the
simplifier never builds a non-finite constant or an ill-formed node.

### 3.2 Atoms, monomials, forms

```haskell
data Atom = Atom !NodeId !Op [Atom]

newtype Monomial = Monomial (Map Atom Integer)   -- x²·y⁻¹ = {x ↦ 2, y ↦ -1}
  deriving (Eq, Ord)

data Poly = Poly !Double !(Map Monomial Double)  -- c + Σ k·m
```

An atom is identified by its node; two atoms are equal exactly when their
ids are equal (hash-consing). The op and the arguments (themselves as
atoms) are there for one purpose: **ordering atoms by structure**.

```haskell
instance Ord Atom where
  compare (Atom i op as) (Atom j op' bs)
    | i == j = EQ
    | otherwise = compare op op' <> compare as bs
```

Why not just order by id? Ids record the order in which nodes happened to
be created, which depends on the order the source graph was walked. Terms
would come out as `y + x` from one graph and `x + y` from another, and
worse, simplifying the result again could reorder it. With a structural
order the layout of a form depends on the form alone, so it is canonical
across graphs and the simplifier is idempotent.

The comparison is cheap: equal ids stop it immediately, and two different
nodes with the same op must differ in some argument, so the lexicographic
comparison of the argument lists recurses into exactly one pair. It takes
time proportional to the depth of the graph, never to the size of the
expression as a tree. The order is `Op`'s derived one: leaves before
unary ops before binary ops, constants before inputs before parameters,
names alphabetically (as `Text`, so `x[10]` sorts before `x[2]`). Hence
`x + w + exp(x)`: input, parameter, application.

`Monomial` and `Poly` use the maps' own `Ord`, so terms are ordered by
their monomials' sorted atom lists: `a + a·b + b`, `x + x²`.

The operations are short:

```haskell
addP   (Poly c ts) (Poly d us) = Poly (c + d) (addNonZero ts us)
negP   (Poly c ts)             = Poly (negate c) (Map.map negate ts)
scaleP 0 _                     = constP 0
scaleP k (Poly c ts)           = Poly (k * c) (Map.filter (/= 0) (Map.map (k *) ts))
mulM (Monomial a) (Monomial b) = Monomial (addNonZero a b)
powM n (Monomial a)            = Monomial (Map.map (* n) a)
```

`addNonZero` merges two maps, adding the values of common keys and
dropping zeros: it gathers like terms in `addP` and merges powers in
`mulM`. It uses `Data.Map.Merge.Strict`, which is fast when one map is much
smaller than the other, as in a long chain of `+`. `scaleP` filters zeros
because a coefficient can underflow.

### 3.3 Forms of nodes

```haskell
canonicalOf :: Graph -> NodeId -> Simp Poly
canonicalOf g i = memo stForms (\m s -> s {stForms = m}) (nodeIndex i) $
  inSpanOf n $ do
    p <- canonical g n
    if finiteP p then pure p else asAtom g n
```

`Simp` is a `StateT` of memo tables over the `Builder` of the new graph.
`canonicalOf` is memoised by source node, which is what preserves sharing
(§1.5). A form with a non-finite constant or coefficient is replaced by
the node itself as an atom: the refused fold of §1.6. `canonical` is the
"forward pass" of §1.1, one case per op:

| op | form |
|---|---|
| `Const c` | the constant `c` |
| `Input`, `Param` | an atom: the same leaf in the new graph |
| any op, all arguments constant | folded with `forwardUn` / `forwardBin` |
| `-p` | `negP p` |
| `p ^ n`, `n` integer | `1` for `n = 0`, `p` for `n = 1`, else `(k·m)ⁿ = kⁿ·mⁿ` |
| `p + q`, `p - q` | `addP p q`, `addP p (negP q)` |
| `p · q` with a constant side | the other side scaled (distributed) |
| `p · q` | the product of `factor p` and `factor q` |
| `p / q` with a constant `q` | `p` scaled by `1/q` |
| `p / q` | `factor p` times `factor q` to the power `-1` |
| `exp`, `log`, `tanh`, `relu`, `step`, `p ^ (1/2)` | an atom: the op applied to the simplified argument |

`asAtom` is the atom case: it builds the node's op over the simplified
arguments (`nodeOf`, §3.5) and takes the new node as an atom. The arity
cases are exhaustive in practice (the builder guarantees `arity`), and the
fall-through is `asAtom` too, so `canonical` is total.

### 3.4 Refusing to distribute: `factor` and `termS`

```haskell
factor :: Graph -> NodeId -> Poly -> Simp (Double, Monomial)
factor g i (Poly c ts) = case Map.toList ts of
  [] -> pure (c, unitM)                -- a constant: c · 1
  [(m, k)] | c == 0 -> pure (k, m)     -- a single term: itself
  _ -> do                              -- a sum: one atom
    x <- nodeOf g i
    modify' (\s -> s {stSums = IntMap.insert (nodeIndex x) (Poly c ts) (stSums s)})
    a <- atomOf x
    pure (1, atomM a)
```

To multiply, each side is seen as a single product `k·m`. A sum becomes the
node built for it (`nodeOf`), taken as an atom. This is the whole of "do
not distribute": products only ever multiply monomials.

The product goes through `termS`, which keeps the invariant of §1.4:

```haskell
termS :: Double -> Monomial -> Simp Poly
termS k m@(Monomial factors) = case Map.toList factors of
  [(a, 1)] -> do
    known <- gets (IntMap.lookup (nodeIndex (atomNode a)) . stSums)
    pure (maybe (termP k m) (scaleP k) known)
  _ -> pure (termP k m)
```

If the monomial is a single atom to the power 1 and that atom is a sum
`factor` made, the term is the sum itself, scaled. `stSums` remembers the
form of every sum atom for exactly this.

### 3.5 Building forms back into nodes

```haskell
nodeOf :: Graph -> NodeId -> Simp NodeId   -- memoised per source node
rebuild :: Poly -> Builder NodeId
```

`nodeOf` builds the form of a source node as nodes, once per source node;
it is used for outputs, for the arguments of atoms, and for sums that are
factors. `rebuild` fixes the layout:

- the constant first, if it is not `0`, then the terms in canonical order:
  `1 - tanh(x)²`, `a·b + c`, `1.5 + x`;
- after the first part, a negative coefficient becomes `-`: `x - 2·y`;
- a term is its coefficient (left out for `1`, a `Neg` for `-1`), then its
  factors left to right, then `/` and the factors with negative exponents:
  `-2·a·b`, `-x·y`, `0.5·x/y`, `1/x`, `x/(y·z)`;
- if the first factor is a sum, the coefficient multiplies the whole
  product: `2·((1 + y)·(2 + y))`, `-((x + y)/z)`.

The last rule is the other half of idempotence. Left-nested, `2·S₁·S₂` is
`(2·S₁)·S₂`, and simplifying *that* again would distribute the `2` into
`S₁` (a constant times a sum) and make `2 + 2·y` a different atom. Since
atoms order leaves and applications before binary ops, the first factor is
a sum only when all numerator factors are sums, so the rule is rarely seen.

Constant first was chosen because the most common sum in gradients is
`tanh`'s `1 - y²`; the same rule prints `-1 + x²` for `x² - 1`.

Every part of the layout simplifies back to the form it came from, with no
arithmetic on coefficients except multiplying by `1`, so simplifying twice
gives the same graph (§5).

### 3.6 Provenance

The nodes built for a source node, including the atoms made while
computing its form, get its source span (`withSpan`, D4), so the simplified
`∂L/∂a` of `intro.hsg` keeps the span of `a * b`. Constants built while
laying out a form get the span of whatever needed them first, which is
best effort: a folded constant has no single origin.

The simplified node of a labelled source node gets its label, so the
simplified outputs are still `L` and `∂L/∂a`, unless it is a leaf: `x + 0`
labelled `d` simplifies to the leaf `x`, which must not be renamed. Labels
are first-wins (step 3), and `copyReachable` copies spans and labels into
the final graph.

### 3.7 Printing: precedence

```haskell
data Level = Atomic | Power | Prefix | Product | Sum
type Doc = (Text, Level)

nodeDoc :: Style -> (NodeId -> Doc) -> Node -> Doc
```

Each node prints to a text and the level it binds at; an argument is
wrapped in parentheses unless its level is one of those allowed in its
position:

| position | allowed without parentheses | so |
|---|---|---|
| left of `+`, `-` | anything | `a - b - c` |
| right of `+`, `-` | atomic, power, product | `a - (b - c)`, `x + (-y)` |
| left of `·`, `/` | atomic, power, prefix, product | `-a·b`, `a·b/c` |
| right of `·`, `/` | atomic, power | `a/(b·c)`, `a·(b/c)`, `x·(-2)` |
| operand of `-` | atomic, power | `-x²`, `-(a·b)`, `-(-x)` |
| base of a power | atomic | `(-x)²`, `(x²)³`, `(a + b)²` |
| argument of `f(…)` | anything | `tanh(a + b)` |

Negative constants bind like unary minus. The precedence table alone
would allow `x·-2` and `a - -b` (the language parses them), but a negative
right operand gets parentheses for readability: `x·(-2)`, which is why
`intro.hsg`'s `L` prints as `(a·b + c)·(-2)`.

### 3.8 Printing: powers and numbers

Two styles:

| | `Unicode` (default) | `Ascii` |
|---|---|---|
| product | `-2·b` | `-2*b` |
| integer power | `x²`, `x⁻¹`, `x¹²` | `x^2`, `x^-1` |
| other power | `x^0.5`, `x^(1/3)` | `x^0.5`, `x^(1/3)` |

Superscripts were chosen for the default because they are the notation of
the docs and of the milestone (`∂L/∂a = -2·b`, `1 − tanh(…)²`), and
integer exponents are by far the most common (`x²`, `y²`, `x⁻¹` from
`log`). Unicode has superscript digits and a superscript minus, but no
superscript point or slash, so fractional exponents use `^` in both styles.
The ASCII style is for terminals without Unicode, and is close to what a
`.hsg` file would say (`x ^ -1`, `x ^ 0.5`).

Exponents are `Rational` and print **exactly** (`showExponent`): an
integer as itself, a number with a finite decimal expansion (denominator
`2ᵃ5ᵇ`) as that expansion (`0.5`, `-1.5`, `0.125`), anything else as a
fraction (`(1/3)`, `(-2/3)`).

Constants are `Double` and print **as short as possible while reading
back as the same number** (`showConstant`): `-2` (not `-2.0`), `0.5`,
`0.1`, `0.0001`, `1.5e-5`, `1e15`. The digits come from `floatToDigits`,
the shortest digit string that identifies the `Double`, so `0.1` prints as
`0.1` although the number stored is not exactly a tenth. Magnitudes from
`10⁻⁴` up to (not including) `10¹⁵` are written out, others use scientific
notation. This is deliberately different from `Viz.Dot`'s `%.4f`: that one
shows *values* to a readable precision, this one shows *constants of an
expression*, which must not be rounded.

### 3.9 Printing: sharing

```haskell
data Rendered = Rendered
  { renderedBindings :: ![(Text, Text)]
  , renderedRoots :: ![Text]
  }

renderNodes    :: PrettyOptions -> Graph -> [NodeId] -> Rendered
prettyNode     :: Graph -> NodeId -> Text
prettyNodeWith :: PrettyOptions -> Graph -> NodeId -> Text
```

`renderNodes` renders several roots together, so they share bindings, and
returns them separately: `derive` lines them up as `∂L/∂a = … = 6`.
`prettyNode` lays one root out as a single line, or as `let … in …`.

`sharedNodes` decides which nodes get a binding, in one ascending pass
(arguments before users, D2). It counts uses (argument positions, plus one
per root) and computes each node's inline size, counting a named argument
as one node. An interior node with two or more uses and inline size above
`prettyInlineSize` gets a binding. A root used by another root gets one
too, and is then printed as its name: the neuron's `∂y/∂b = t1`.

`bindingNames` names bindings in ascending order: the node's label if it
is a plain name (starts with a letter; letters, digits, `_'[],#.`), such as
`h` or `W1[0,1]`, otherwise the first free `t1`, `t2`, …. Names already
used, including those of the leaves in the expression, are skipped, so a
binding never looks like a leaf. `∂L/∂a` is not a plain name: in
`∂L/∂a·x` the `/` would read as a division.

`renderNodes` is partial like `nodeAt`: every root must be a node of the
graph.

## 4. Design decisions

### 4.1 A new graph, not rewrites in place

The simplified expressions go into a fresh graph. A graph is immutable
(D1), and the forward graph is still needed as it is: for training, for
the unsimplified picture, for the ids the caller holds. Building a new
graph also makes the result clean (§3.1) and lets hash-consing merge
everything the canonical forms made equal. The `Builder` action
`simplifyNodes` is there for callers that want to build more on top.

### 4.2 Forms, not rewrite rules

A rule-based simplifier (`x + 0 → x`, `x·1 → x`, `x + x → 2x`, …) needs a
strategy (which rule, where, how often), can loop, and misses cases nobody
wrote a rule for (`x·y - y·x`). Computing a normal form makes every
identity of commutative rings fall out of `Map` arithmetic, terminates by
construction (one pass over the nodes), and is idempotent by design.

### 4.3 Constants are distributed, products are not

Distributing a constant never adds terms, and it lets constants that came
from different places fold (`-2` from the loss, `2` from a `^2` rule).
Distributing a product of sums multiplies terms (§1.4). The one thing lost
is that `x·(y + z)` and `x·y + x·z` stay different.

### 4.4 Structural order, not creation order

See §3.2. The cost is a small `Atom` structure per node of the new graph,
memoised (`stAtoms`), and comparisons that walk down one path of the graph.
The benefit is a canonical layout, which is what makes the idempotence and
commutation properties (§5) hold exactly, not only up to reordering.

### 4.5 Refuse non-finite folds, keep the node

Folding `1/0` to a constant is impossible (the builder rejects infinite
constants) and would be wrong anyway: the error belongs at the node, with
its span, when the graph is evaluated. Keeping the node as an atom
preserves exactly the behaviour of the original.

### 4.6 Bindings for shared subexpressions, with a size threshold

A depth or size cutoff (`…` below some depth) would keep the text short
but lose information. Bindings keep everything and stay linear. The
threshold avoids the opposite problem: naming every shared `x²` would turn
`x² + exp(x²)` into two lines.

## 5. Tests

`test/HsGrad/Core/SimplifySpec.hs` has seven groups:

- **Canonical forms**, checked through `prettyNode`: `x + x = 2·x`,
  `x·x·x = x³`, `2·x·3 = 6·x`, `x - x = 0`, `x·y - y·x = 0`, `x/x = 1`,
  `x·y/x = y`, the identities of 0 and 1, structural order (`y + x` gives
  `x + y` whichever leaf was created first), no distribution
  (`x·(y + z)`, `(x + y)²`), distributed constants, the lone-sum
  invariant, the layout rules, `(2·x·y)² = 4·x²·y²`, atoms with simplified
  arguments, fractional powers.
- **Constant folding**: `exp 0`, `2³`, `4^0.5`, `1 + 1`, `relu(-3)`; the
  refused folds `x/0`, `log(0)`, `(-4)^0.5`, `exp(1000)`, which still fail
  when evaluated.
- **Domain**: `x/x = 1` evaluates at `x = 0` and `0·log x = 0` at `x < 0`,
  where the originals fail (§1.6).
- **Gradients**: the milestone (`-2·b`, `-2·a`, `-2` for `intro.hsg` and
  for the same expression built without names), `L = -2·a·b - 2·c`,
  `∂L/∂f = a·b + c`, `a + a` gives `2`, the neuron's shared
  `1 - tanh(…)²`, the derivatives of `x³` (`3·x²`, `6·x`, `6`) and `tanh''`
  (`-2·tanh(x)·(1 - tanh(x)²)`).
- **Provenance and the result graph**: labels `L` and `∂L/∂a` kept, no
  label on a leaf, spans kept, the result graph is exactly the reachable
  nodes, foreign ids are left out.
- **Sharing**: the chain of 30 squarings becomes `x¹⁰⁷³⁷⁴¹⁸²⁴` (2 nodes),
  its gradient `1073741824·x¹⁰⁷³⁷⁴¹⁸²³` (4 nodes); a 200-level chain
  `xₖ₊₁ = tanh(xₖ·xₖ - 1)` (2²⁰⁰ copies of `x` as a tree) simplifies to at
  most 3 nodes per level with the same value, and its gradient (1601
  nodes) to 1205 nodes with backprop's value.
- **Random graphs** (1000 samples each): values are preserved for step 3's
  generator (constants 0, divisions by zero), for the every-op generator of
  step 5, and for symbolic gradients; simplifying twice gives structurally
  the same expressions and node count as once; swapping the arguments of
  every `+` and `·` gives structurally the same result; and the result
  has no multiplication by 0 or 1 and no addition or subtraction of 0
  left, and every node whose arguments are all constants has a non-finite
  value (a refused fold).

"Structurally the same" is tested by copying both expressions into one
fresh graph: hash-consing makes them the same ids exactly when they are
equal as trees.

`test/HsGrad/Core/PrettySpec.hs` has four groups: numbers (golden
constants and exponents; every finite `Double` reads back exactly; decimal
exponents read back as the same `Rational`), precedence (every row of the
table in §3.7, both styles, the function names), sharing (a binding, a
repeated tiny subexpression, labels as names, a label that is not plain or
is a leaf's name, a binding that must skip the leaf name `t1`, several
roots sharing and a shared root, the chain of 30 squarings in 15 lines),
and `intro.hsg`'s `L` and the neuron's `y` unsimplified.

### 5.1 Testing the tests

Eight deliberate bugs in `Simplify.hs`, each caught:

| Bug | Failing tests |
|---|---|
| keep terms whose coefficients cancel to 0 | 6 |
| order atoms by id instead of by structure | 5 |
| no lone-sum flattening in `termS` | 2 (a hand case, idempotence) |
| left-nest the coefficient also before a sum | 2 (a hand case, idempotence) |
| no finiteness check: fold `1/0` | 5 |
| no memo in `canonicalOf` | the chain of 30 squarings never finishes |
| `Sub` with the negative coefficient (`x - -2·y`) | 9 |
| `p⁰ = 0` | 1 |

The two idempotence bugs were real: the first version of this module had
neither the lone-sum rule nor the layout rule, and the idempotence property
found `2·(1 + y)·(2 + y)`, whose rebuilt `(2·(1 + y))·(2 + y)` simplified
again to the atom `2 + 2·y`. The hand cases were added afterwards. The
id-order bug shows that the structural order matters for more than looks:
it breaks idempotence and commutation.

### 5.2 Sampling and tolerance

The value properties evaluate the original first and discard the sample
if it fails (outside the domain) or is **ill-conditioned**: some value
the outputs depend on is above `10⁶`, or a division by less than `10⁻²`, a
`log`, negative or fractional power of something below `10⁻²`, or a `relu`
or `step` within `10⁻⁶` of its kink. Near those points a rounding error of
one ulp is amplified without bound, or flips a `step`. About 60% of
samples are discarded for random forward graphs (there are many outputs,
and one bad node discards the sample) and about 20% for gradients.

On the rest, the simplified outputs must evaluate (simplification must not
lose the domain), and agree within `10⁻¹⁰ · scale`, where `scale` is the
largest magnitude of any value in the original computation (at least 1).
`10⁻¹²` passes at 20,000 samples per property, `10⁻¹⁴` fails occasionally,
and exact equality fails at once (`(2·x)·3` is not always `6·x` in the
last bit), so `10⁻¹⁰` leaves a margin of 100. All three value properties,
idempotence and commutation also pass at 20,000 samples with three
different seeds.

The full suite has 252 tests and runs in under 3 seconds.

## 6. What later steps build on

- **Step 9 (`Train.*`)**: nothing; training uses numeric backprop.
  `simplify` could shrink a loss graph before training, but evaluating the
  canonical form is not cheaper in general, and it would change rounding.
- **Step 15 (`Driver`)**: `hsgrad derive FILE --of L --wrt a,b,c
  --simplify --format text` is `symbolicGradients g l`, then the gradient
  node of each `--wrt` name (`gradNodeOrZero` for unrelated ones, W0203),
  then `simplify g' those` if `--simplify`, then `renderNodes
  defaultPrettyOptions` on the (simplified) nodes: print the bindings first
  (`let t1 = …`), then one line per name with `gradientName` (or the
  label, which `simplify` keeps) and the root text, and the value from
  `evaluate` with `showConstant`. Without `--simplify`, the same printer
  works on the raw gradient nodes. `--dot` with `--simplify` draws the
  simplified graph with `renderDotFrom` and the simplified ids; labels are
  kept, so `highlight` still shows `∂L/∂a`. An ASCII fallback is
  `prettyStyle = Ascii`.
- **Later**: `prettyInlineSize` and the binding names are options if the
  CLI ever wants them. Further identities (`log(exp x) = x`, `tanh(-x) =
  -tanh(x)`, merging fractional powers) would go into the per-op cases of
  §3.3, each with its domain caveat added to §1.6.
