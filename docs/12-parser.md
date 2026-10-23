# 12 — `Lang.Syntax` and `Lang.Parser`: from text to a tree

This step reads `.hsg` files. Two modules:

- `HsGrad.Lang.Syntax`: the **abstract syntax tree** (AST) of the whole
  language: declarations, types, expressions, literals. Every node carries
  an annotation, which the parser fills with the node's `Span`. It also has
  a **printer** that turns a tree back into text the parser reads back as
  the same tree.
- `HsGrad.Lang.Parser`: a **megaparsec** parser that follows the grammar of
  `01-language.md` §9 and the precedence table of §5.1, and turns every
  syntax error into a `Diagnostic` with a code from E0101–E0105. It does
  not stop at the first error: it reports every independent one.

The milestone of the roadmap is "all example files parse, good parse
errors". The four programs of `01-language.md` §1 now live in `examples/`
(`intro.hsg`, `neuron.hsg`, `xor.hsg`, `moons.hsg`) and parse, and a file
with several mistakes gets one message for each
(`test/golden/parse/recovery.err`, shortened):

```
error[E0101]: unexpected keyword `param`
 --> recovery.hsg:3:1
  |
3 | param w : Vec 2 ~ uniform(-1, 1)
  | ^^^^^ expected `,` or `]`

error[E0101]: parameter `b` has no initial value
 --> recovery.hsg:4:15
  |
4 | param b : Real
  |               ^ expected `=` or `~` here
  |
  = hint: write `= 0` for a constant, or `~ uniform(-1, 1)` for random values

error[E0104]: `let` expression without `in`
 --> recovery.hsg:7:24
  |
7 |   let s = dot(w, x) + b
  |   ---                  ^ expected `in` here
  |   |
  |   this `let` is inside an expression
  |
  = hint: inside an expression write `let s = … in body`;
          only a top-level `let` needs no `in`

error[E0101]: unexpected `x` after `b`
  --> recovery.hsg:10:41
   |
10 | model m(x : Vec 2) : Real = neuron(w, b x)
   |                                         ^ a call needs parentheses
   |
   = hint: to call `b`, write `b(x)`
   = hint: to multiply, write `b * x`

4 errors
```

The parser also reproduces, character for character, the two syntax
errors that step 10 built by hand to design the renderer
(`missing-in.err` and `unterminated-comment.err`).

---

## 1. Theory

### 1.1 What a parser does

A parser turns a flat string into a **tree** that shows the structure the
string only implies. `a - b * c` is seven characters; its structure is "a
subtraction whose right operand is a product":

```
      Binary Minus
       /        \
    Var a    Binary Times
              /        \
           Var b      Var c
```

Everything after the parser (name resolution, shape checking, compiling
to a graph) works on trees, never on text. A tree has no spaces, no
comments and no parentheses: `(a - b)` and `a-b` are the same tree. What
it keeps from the text is **where** each node came from, its span, so that
later errors can point at the source.

This tree is called *abstract* syntax: it records structure, not
spelling. The *concrete* syntax, with its parentheses, commas and
keywords, is only in the text.

### 1.2 Parser combinators

There are two classic ways to write a parser: generate it from a grammar
with a tool (yacc, happy), or write it by hand as **recursive descent**:
one function per grammar rule, each calling the functions of the rules it
mentions. `expr` calls `additive`, which calls `multiplicative`, and so on
down to `atom`, which for `( e )` calls `expr` again.

**Parser combinators** are recursive descent with the plumbing factored
out. A parser is a value of type `Parser a`: something that consumes a
prefix of the input and produces an `a`, or fails. Small parsers are
combined with ordinary functions:

| Combinator | Meaning | Grammar |
|---|---|---|
| `p *> q`, `do` | `p` then `q` | `p q` |
| `p <|> q` | `p`, or else `q` | `p \| q` |
| `many p`, `some p` | zero / one or more `p` | `p*`, `p+` |
| `optional p` | `p` or nothing | `p?` |
| `sepBy1 p s` | `p`, separated by `s` | `p ( s p )*` |

so the parser reads almost like the EBNF of §9:

```haskell
-- binder ::= ident ( ":" type )?
binder = Binder <$> identifier <*> optional (symbol ":" *> typeP)
```

[megaparsec](https://hackage.haskell.org/package/megaparsec) is the
combinator library we use (00-architecture §8). Its main strength, and the
reason for choosing it, is error reporting (§1.5).

### 1.3 Tokens without a lexer

Many parsers have two stages: a **lexer** cuts the text into tokens
(`let`, `x`, `=`, `2.5`), and the parser works on tokens. With
combinators one stage is enough. Each token parser reads its token and
then skips the whitespace and comments after it:

```haskell
lexeme p = spanned p <* whitespace     -- the token, its span, then skip
symbol t = fst <$> lexeme (string t)   -- "(", "<-", "+", …
```

The rule "every token parser starts at a token and leaves the input at the
next one" keeps whitespace out of the grammar entirely. The span of a
token is taken *before* the whitespace, so it ends where the token ends.

Keywords need care. `in` is a keyword, but `input` and `inputs` are not
"`in` followed by something". The parser therefore looks at the **whole
word** first and only then decides whether it is the keyword asked for, a
name, or neither (§3.4).

### 1.4 Precedence and associativity

`2 * W @ x` could be `(2 * W) @ x` or `2 * (W @ x)`. The precedence table
of `01-language.md` §5.1 decides:

| Level | Form | Associativity |
|---|---|---|
| 1 | `f(…)`, `e[i]`, literals, `( e )` | — |
| 2 | `e ^ k` | non-associative |
| 3 | `-e` | prefix |
| 4 | `@` | left |
| 5 | `*` `/` | left |
| 6 | `+` `-` | left |
| 7 | `let … in` | extends to the right |

In recursive descent, each level is a rule whose operands are the next
tighter level, the classic grammar of §9:

```
additive       ::= multiplicative ( ( "+" | "-" ) multiplicative )*
multiplicative ::= matmul ( ( "*" | "/" ) matmul )*
```

`*` is "inside" `+`, so it binds tighter. **Left associativity** comes from
folding the repetition to the left: `a - b - c` reads `a`, then `- b`,
then `- c`, and builds `(a - b) - c`. A **non-associative** operator
allows at most one: `x ^ 2 ^ 3` is an error, not a guess.

Writing these loops for every level is mechanical, so the
`parser-combinators` package does it: `makeExprParser term table` takes
the operand parser and a table of levels, tightest first, and builds the
whole tower. Our table has levels 3–6; levels 1–2 are hand-written because
`^` is not a normal binary operator (its right side must be a constant,
§3.6).

### 1.5 Error messages from a combinator parser

When `p <|> q` fails, *both* alternatives failed, and a good message says
what either would have accepted. megaparsec keeps, for each error:

- the **offset** where it happened;
- the **unexpected** item;
- the set of **expected** items: tokens (`)`) or **labels** (`an
  expression`), attached with `label "an expression" p`.

Three rules make these sets precise:

1. **Longest match.** Of two errors, the one further into the input wins;
   at the same offset, their expected sets are merged. So `[1.0, 2.0 3.0]`
   reports what was expected *after* `2.0`, not at the start.
2. **Committing.** Once an alternative has consumed input, `<|>` does not
   try the next one: `let x = )` fails inside `let`, it does not go on to
   ask whether the text might be some other declaration. This is also what
   makes errors point at the right place.
3. **Hints.** A parser that succeeded without consuming input, such as
   `optional (symbol "^")` when there is no `^`, leaves a hint: "a `^`
   would have been fine here". If the next error happens at the same
   offset, the hints join its expected set. That is how
   `let y = x # 2` says "expected an operator or a declaration": the
   operators were optional continuations of `x`, the declaration what
   could have come next.

Labels decide how readable the result is. Without them the set after `x`
would be `(`, `[`, `^`, `@`, `*`, `/`, `+`, `-`; with every operator
labelled "an operator" and the brackets hidden, it is one phrase.

### 1.6 Recovering from errors

Stopping at the first syntax error is simple and common, but it turns
fixing a file into a loop of edit, run, edit. The checker reports all
errors (01-language §7); the parser should too. The difficulty is
**cascades**: after a mistake, the parser is out of step with the text,
and naive continuing produces a stream of bogus follow-up errors.

The standard cure is **panic-mode recovery** with **synchronisation
points**: on an error, skip input up to a token that certainly starts
something new, and resume there. `.hsg` makes this easy because every
declaration starts with a keyword (01-language §9, "keyword-led
declarations"). After an error in a declaration, the parser skips to the
next `input`, `param`, `data`, `fn`, `model` or `loss`, which can never
appear inside a declaration, or to a `let` at the start of a line, and
parses on from there (§3.9).

Some errors don't need recovery at all, because they are inside one token
and leave a perfectly usable token behind: `2x` is a malformed number but
clearly a number, an unterminated comment simply ends at the end of the
file, `x ^ y` has a non-constant exponent but is clearly a power. These
are **registered** (megaparsec collects them) and parsing continues as if
nothing happened, so even the rest of the same declaration is checked.

## 2. The tree

### 2.1 Annotations as a type parameter

Every node needs a span. One way is a field `exprSpan :: Span` on every
constructor. We make it a **type parameter** instead:

```haskell
data Expr a
  = Num a Rational
  | Var a (Ident a)
  | Binary a Operator (Expr a) (Expr a)
  | …
  deriving (Eq, Show, Functor, Foldable, Traversable)
```

The parser produces `Expr Span`. Because `Expr` is a `Functor` over its
annotation, `fmap` changes all annotations at once, and

```haskell
forgetSpans :: Functor f => f a -> f ()
forgetSpans = (() <$)
```

gives the same tree with `()` everywhere. Two trees "equal modulo spans"
are simply equal after `forgetSpans`, with the derived `Eq`. That is what
the round-trip test needs (§5.3), and what unit tests use to write
expected trees without positions:

```haskell
expr "-x ^ 2" `shouldBe` Right (Negate () (Power () (Var () (Ident () "x")) (Number () 2)))
```

Later steps can reuse the parameter: the checker could annotate nodes with
their shapes (`Expr (Span, Shape)`) without a second tree type. `Foldable`
comes for free too: `toList e` lists every span in the tree.

### 2.2 What the tree records, and what it doesn't

The tree is **purely syntactic**: a name is its text. Whether `x` is an
input, a local or a typo is for the checker (step 12). Some consequences:

- **Parentheses are not nodes.** `(a + b) * c` is a `Binary Times` whose
  left operand is the `Binary Plus`. The parser does widen the span of the
  inner node over the parentheses, so the span of `(x - y) ^ 2` starts at
  `(` (§3.7).
- **Numbers are exact and unsigned.** `Num a Rational` is read digit by
  digit (step 2's hand-over: never through `Double`), and `-2` is
  `Negate (Num 2)`, because `-` is an operator (01-language §2). Where the
  grammar takes a *signed constant* (exponents, initialiser arguments,
  literals) the sign is part of a `Number`.
- **Matrices are vectors of vectors.** `[[1, 2], [3, 4]]` is a `Vector` of
  two `Vector`s. Whether the rows agree (E0312) is a shape question.
- **`mean [ e | x <- d ]` is a call.** It is `Call mean [Comprehension …]`,
  exactly the tree of `mean([ e | x <- d ])` (§3.8).
- **A bare row is a one-element row.** `rows [ -1, 1 ]` and
  `rows [ (-1), (1) ]` give the same `Row`s.
- **Initialisers are names.** `~ uniform(-1, 1)` stores the name
  `uniform`; that only `uniform` and `normal` exist is for the checker.

### 2.3 The types at a glance

```haskell
newtype Program a = Program { programDecls :: [Decl a] }

data Decl a
  = InputDecl a (Ident a) (Type a) (Maybe (Literal a))
  | ParamDecl a (Ident a) (Type a) (ParamInit a)       -- = literal | ~ f(lo, hi)
  | DataDecl  a (Ident a) [Field a] (DataSource a)     -- rows [ … ] | csv "…" { … }
  | FnDecl    a (Ident a) [Binder a] (Maybe (Type a)) (Expr a)
  | ModelDecl a (Ident a) [Binder a] (Type a) (Expr a)
  | LetDecl   a (Ident a) (Maybe (Type a)) (Expr a)
  | LossDecl  a a (Expr a)                             -- second a: the keyword

data Type a = TReal a | TVec a (Dim a) | TMat a (Dim a) (Dim a)
data Dim a  = DimNat (Nat a) | DimVar (Ident a)        -- 16 | n

data Expr a
  = Num a Rational | Var a (Ident a) | Call a (Ident a) [Expr a]
  | Index a (Expr a) (Nat a) (Maybe (Nat a))           -- e[i] | e[i, j]
  | Vector a [Expr a]
  | Comprehension a (Expr a) (Pattern a) (Generator a)
  | Power a (Expr a) (Number a)                        -- constant exponent
  | Negate a (Expr a)
  | Binary a Operator (Expr a) (Expr a)                -- @ * / + -
  | Let a (Ident a) (Maybe (Type a)) (Expr a) (Expr a)
```

The first field of every constructor is the annotation of the whole node,
and `exprAnn`, `declAnn`, `typeAnn`, `literalAnn` read it. Small pieces
(`Ident`, `Nat`, `Number`, `StringLit`) have their own annotation, so a
diagnostic can point at a name, an index or an exponent alone: E0305
"index out of bounds" points at the `3` of `v[3]`, not at all of it.

`LossDecl` keeps the span of its keyword because it has no name: "more
than one `loss`" (E0206) should mark `loss`, not a whole multi-line
declaration.

Lists that the grammar requires to be non-empty (vector elements, schema
fields, rows, tuple patterns) are plain lists, with the requirement in a
comment. `NonEmpty` would be more precise, but every later step would pay
for it with conversions, and the parser is the only producer.

## 3. Code walkthrough

The modules are `src/HsGrad/Lang/Syntax.hs` and
`src/HsGrad/Lang/Parser.hs`.

### 3.1 The printer

```haskell
prettyProgram  :: Program a -> Text    -- one declaration per line
prettyDecl     :: Decl a -> Text
prettyExpr     :: Expr a -> Text
prettyType     :: Type a -> Text       -- "Mat 16 n"
prettyLiteral  :: Literal a -> Text
prettyOperator :: Operator -> Text
showNumber     :: Rational -> Text
```

The printer exists for the round-trip test, but it is also what later
messages use to show a type (`Mat 16 16 @ Vec 8`) or an expression (the
E0105 hint below).

Parentheses come from precedence. Each form has a `level`, the row of the
precedence table it belongs to (atoms 1, `^` 2, unary minus 3, `@` 4,
`* /` 5, `+ -` 6, `let` 7). An operand is printed in parentheses when its
level is higher (binds less tightly) than its position allows:

```haskell
Binary _ op l r ->
  let p = operatorLevel op
   in atMost p l <> " " <> prettyOperator op <> " " <> atMost (p - 1) r
```

The left operand of a left-associative operator may be at the same level
(`a - b - c`), the right one may not (`a - (b - c)`). The base of a power
and of an index must be an atom (`(-x) ^ 2`, `(a + b)[0]`); a `let` is
level 7, so as an operand it always gets parentheses. Nothing else gets
any: the printer is minimal, which is what makes the round trip a strong
test of both precedence tables, the parser's and the printer's.

One lexical trap: `Negate (Negate x)` would print as `--x`, which starts a
comment. A negation whose operand text starts with `-` gets parentheses:
`-(-x)`.

`showNumber` prints a `Rational` as a decimal that reads back as exactly
the same value. Every number the parser produces is a decimal fraction
(`c · 10^k`), so it has an exact decimal form: `0.5`, `0.0001`, `1.5e-7`,
`1e20`, `123.456`. Very large or small numbers use an exponent, others
plain digits. A rational without a decimal form (`1/3`) can only come from
a hand-built tree and is printed rounded.

### 3.2 The parser type and its errors

```haskell
type Parser = Parsec Problem Text

newtype Problem = Problem Diagnostic
```

`Parsec e s` is megaparsec's parser over the stream type `s` (`Text`) with
custom errors of type `e`. Errors are either **trivial** ("unexpected X,
expected Y or Z", assembled by megaparsec from labels) or **fancy**,
carrying our own value. Ours is a finished `Diagnostic`: whenever the
parser knows better than "expected …" (an unterminated comment, a missing
`in`, a variable exponent) it writes the whole message itself:

```haskell
report   :: Diagnostic -> Parser ()   -- register it and go on
failWith :: Diagnostic -> Parser a    -- fail with it
```

megaparsec keeps errors in sets, so `Problem` needs `Ord`, and so
`Diagnostic` and `Label` now derive `Ord` too (the only change to step
10's module). The newtype exists because `ShowErrorComponent`, the class
megaparsec requires, would otherwise be an orphan instance on
`Diagnostic`.

The entry points run a parser on a whole text:

```haskell
parseProgram :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Program Span)
parseExpr    :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Expr Span)
parseLiteral :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Literal Span)
parseType    :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Type Span)
```

The file name only goes into spans. `NonEmpty` says what `Left` means: at
least one error. `runHsg` starts megaparsec with a hand-made initial state
whose **tab width is 1** (megaparsec's default is 8), so that megaparsec's
columns are character columns like ours (docs/11 §4.3), and converts each
`SourcePos` with `Pos (unPos (sourceLine p)) (unPos (sourceColumn p))`.

### 3.3 Whitespace and comments

```haskell
whitespace = hidden (skipMany (void space1 <|> lineComment <|> blockComment))
```

`hidden` keeps whitespace out of expected sets ("expected a comment" is
never useful). Block comments nest, so `blockComment` counts levels: `{-`
one deeper, `-}` one up, done at zero. Between those it skips everything
that is not `-` or `{` in one `takeWhileP`, so long comments are cheap.

At the end of the file inside a comment it **reports** E0102 (primary
span: the opening `{-`; secondary: the end of the file) and succeeds. The
comment ends there, and parsing goes on at the end of the file. If the
comment had nested comments inside, a note says that each `{-` needs its
own `-}`, the usual reason a comment that looks closed is not.

### 3.4 Words: keywords and names

```haskell
peekWord   :: Parser (Maybe Text)   -- the word starting here, without consuming
keyword    :: Text -> Parser Span   -- that exact word
identifier :: Parser (Ident Span)   -- a word that is not reserved
```

Both look at the whole word with `peekWord` and only then either consume
it or fail **without consuming anything**. The obvious alternative,
`try (string "in" <* notFollowedBy identChar)`, has a subtle flaw in
megaparsec: on `input`, the failure happens after `in`, and an error
keeps its offset even through `try`. By the longest-match rule (§1.5) that
error, in the middle of a word, would win over every honest error at the
start of the word. Deciding before consuming keeps every error at a token
boundary.

The type names `Real`, `Vec` and `Mat` are reserved like keywords, so
`Vec` can never be a shape variable (a clarification now written in
01-language §2).

### 3.5 Numbers and strings

```haskell
numberToken :: Parser NumberToken
readNumber  :: Text -> Either NumberProblem (Rational, Bool)
```

`numberToken` first takes the whole **number-like run** of characters:
digits, letters, `_`, `'`, `.`, and a sign right after the `e` of an
exponent. Then `readNumber` checks it against the number syntax
`digits ( "." digits )? ( (e|E) (+|-)? digits )?` and computes its value
exactly:

```
"1.5e-3"  →  mantissa 15, 1 fraction digit, exponent -3  →  15 · 10^(-3-1) = 3/2000
```

Taking the whole run first is what makes the message useful: `2x` is
"malformed number literal `2x`" with the hint "to multiply, write
`2 * x`", not "unexpected `x`" after a valid `2`. Other E0103 cases: `1.`
(no digits after the point), `.5` (hint: `0.5`), `1e` (no exponent
digits), `1.2.3`, and numbers outside the range of a `Double` (`1e999`, or
`1e-999`, which would silently become 0). Every E0103 is registered, and
the number is read as 0, so parsing continues.

A natural number (an index or a dimension) is any number token that is
digits only. `Vec 2.5` and `v[1.5]` are reported as "expected a natural
number" (E0101) at that token, again without stopping.

Strings are short (CSV paths), stay on **one line**, and know two escapes,
`\"` and `\\`. Reaching the end of the line or the file first is E0102
with the opening quote as primary span; an unknown escape such as `\n` is
an E0101 at the escape. Both are registered.

### 3.6 Expressions

The expression parser is the precedence table, top to bottom:

```haskell
expr = letExpr <|> operatorExpr                 -- level 7

operatorExpr = E.makeExprParser powerExpr       -- levels 3–6
  [ [E.Prefix (foldr1 (.) <$> some negation)]
  , [E.InfixL (binary MatMul)]
  , [E.InfixL (binary Times), E.InfixL (binary Divide)]
  , [E.InfixL (binary Plus), E.InfixL (binary Minus)]
  ]

powerExpr   = postfixExpr >>= raise              -- level 2
postfixExpr = atom >>= indexes                   -- level 1
```

Each operator parser returns the function that builds the node, with the
span covering both operands: `\l r -> Binary (mergeSpan (exprAnn l)
(exprAnn r)) op l r`. `makeExprParser` applies only one prefix operator
per level, so negation is `some negation` composed into one function,
which makes `- -x` work.

**Powers.** After `^` comes a signed number. If something else comes, the
parser still reads it as an operand (to know its span), reports E0105 and
uses the exponent 1 to go on. The hint is built with the printer, so it
is correct for any base and exponent:

```
error[E0105]: exponent must be a numeric constant
 --> variable-exponent.hsg:4:19
  |
4 | let w = (x + 1) ^ (y - 1)
  |                   ^^^^^^^ this exponent is not a number
  |
  = hint: `^` takes a number: `x ^ 2`, `x ^ -1`, `x ^ 0.5`
  = hint: for a variable exponent write `exp((y - 1) * log(x + 1))` (needs a positive base)
```

A second `^` right after a power is reported ("a power of a power needs
parentheses") and parsed anyway, so `^` is non-associative with a message
rather than a puzzling "unexpected `^`".

**`let … in`.** The bound expression is parsed; then `in` must follow.
When it doesn't, the E0104 points at the empty span right after the bound
expression, where `in` belongs, not at the next token, which is often on
the next line and unrelated (`missing-in.err`). The `let` keyword is the
secondary label. The keyword parser is `hidden` at the start of `expr`, so
expected sets say "an expression", not "`let` or an expression".

### 3.7 Atoms and spans

```haskell
atom = label "an expression" (choice [numberAtom, nameAtom, parenAtom, bracketAtom])
```

- `parenAtom` parses `( e )` and **widens** the span of `e` over the
  parentheses with `setExprAnn`. Without that, the span of `(x - y) ^ 2`
  would be the merge of `x - y` and `2`, starting inside the parentheses.
- `bracketAtom` reads `[` and the first expression, and only then decides:
  `|` makes it a comprehension, `,` or `]` a vector. One expression of
  lookahead, as the task of this step put it, "after the first
  expression"; no backtracking is needed.

Spans of compound nodes are merges of their parts, and token spans end
before trailing whitespace, so the span of every node covers exactly its
text. A property test checks precisely that (§5.3).

### 3.8 Names: variables, calls, and `sum [ … ]`

After a name, the next character decides:

- `(`: a call, `f(e1, …, en)`.
- `[`: either an index (`v[0]`) or a call with a comprehension argument
  (`sum [ p ^ 2 | p <- params ]`, as the examples of §1 write it). It is an
  index when a number and then `]` or `,` follow. This is checked with a
  `lookAhead` that consumes nothing, and the brackets are then parsed for
  real by the chosen branch.
- anything else: a variable.

The comprehension form was missing from the grammar of §9, which only had
`ident "(" … ")"`; under that grammar the examples `mean [ … ]` and
`sum [ … ]` did not parse. It is now in the grammar, with the rule above
(§4.6).

Two targeted errors live here. `name [ e ]` where `e` is not a number is
"an index must be a natural number"; `name [ e, …` is "a comprehension
needs `|` here", with the hint to call with parentheses
(`sum([a, b])`). And a name followed **on the same line** by another name
or a number is a call without parentheses or a forgotten `*`:

```
error[E0101]: unexpected `x` after `tanh`
 --> call-without-parens.hsg:2:14
  |
2 | let y = tanh x
  |              ^ a call needs parentheses
  |
  = hint: to call `tanh`, write `tanh(x)`
  = hint: to multiply, write `tanh * x`
```

This check can never reject a valid program: in the grammar, a name is
never followed by another name or a number.

### 3.9 Declarations and recovery

```haskell
program = Program . catMaybes <$> declarations
  where
    declarations = ([] <$ eof) <|> ((:) <$> recovering (declaration <* nextDeclaration) <*> declarations)
```

`declaration` is a `choice` of the seven keyword-led parsers, labelled "a
declaration". Each one is a direct transcription of its grammar line. Two
details:

- `nextDeclaration` is a `lookAhead` for a declaration keyword or the end
  of the file. It is part of the declaration so that the hints of the
  declaration's last parsers are still in the error: `let y = x # 2` says
  "expected an operator or a declaration". (Outside the declaration, the
  error would reach `recovering` first, and the hints would be lost.)
- `param w : Vec 3` without `=` or `~` gets its own E0101, "parameter `w`
  has no initial value", pointing just after the type, rather than an
  "unexpected" error at whatever starts the next line.

`recovering` wraps each declaration with megaparsec's `withRecovery`: when
the declaration fails, the error is registered, the input is skipped to
the next **synchronisation point**, and the loop continues with `Nothing`
for the broken declaration:

```haskell
recovering p = do
  start <- lineNumber
  withRecovery (recover start) (Just <$> p)
  where
    recover start err = do
      registerParseError err
      now <- lineNumber
      done <- atEof
      when (now == start && not done) skipToken
      skipToDeclaration
      pure Nothing
```

`skipToDeclaration` skips token by token (strings, words, numbers, single
characters, each followed by whitespace) until it sees one of `input
param data fn model loss`, a `let` **in column 1**, or the end of the
file. Going token by token means a keyword inside a comment or a string,
or `input` inside `inputs`, never counts.

`let` is the delicate one: it also starts expressions, and synchronising
at an expression `let` would parse `let h2 = … in …` as a top-level
declaration and report its `in` as a second error. Expression `let`s are
normally indented, so only a `let` at the start of a line counts. This
uses layout only for recovery: the grammar itself has no layout rules, and
an unusually indented file loses some recovery (fewer errors reported),
never correctness.

The one-token skip handles a declaration keyword used by mistake on the
line where the broken declaration started: in `let input = 1` the error is
at `input`, and resuming there would report `input = 1` as a second error.

### 3.10 From megaparsec errors to diagnostics

```haskell
bundleDiagnostics :: FilePath -> Text -> ParseErrorBundle Text Problem -> NonEmpty Diagnostic
```

At the end, megaparsec hands over all registered errors and, if the parse
failed, the final one. Fancy errors already are diagnostics. Trivial ones
become E0101 in `unexpectedToken`:

- megaparsec knows the offset, but its "unexpected" item is just the first
  character. The converter looks at the source again and finds the whole
  **token** there (`nextToken`), so the message says "unexpected keyword
  `in`" or "unexpected number `3.0`" and the marks cover the whole token.
  At the end of the file, the empty span right after the last
  non-whitespace character is used, so the caret appears where something
  is missing rather than on an empty line.
- The expected items become the label: tokens in backticks first, then
  labels, "a declaration" (the widest) last, "end of file" only when it is
  the only possibility: "expected `)`, `,` or an operator".
- A few combinations get hints: a reserved word where a name was expected
  ("`in` is a keyword, so it can't be a name"), `in` where a declaration
  was expected ("a top-level `let` has no `in`"), a name where an index or
  dimension was expected.

One cascade is removed: an unterminated comment or string swallows the
rest of the file, so a trivial error at the end of the file is dropped
when there is an E0102 (`comment-in-expression.err`). Finally the
diagnostics are de-duplicated and sorted by position.

## 4. Design decisions

### 4.1 megaparsec, no separate lexer

The architecture chose megaparsec (§8) for its error messages: labels,
hints and the longest-match rule produce precise "expected" sets with
little effort, and `withRecovery` / `registerParseError` support
multi-error reporting directly. A separate lexer would give cleaner token
boundaries, but it doubles the machinery (a token type, positions on
tokens, a stream instance), and the one place where token boundaries
matter, keywords versus names, is handled by `peekWord` (§3.4).

### 4.2 Our own diagnostics, not megaparsec's messages

megaparsec can print its errors, but in its own format and words
("unexpected 'x' expecting ..."). Every error goes through `Diagnostic`
instead (00-architecture §6), so syntax errors look like every other
error and are tested the same way, with golden files. The custom error
type is the finished `Diagnostic` because the parser is the place that
knows the context ("this `let` is inside an expression"), and carrying
that context out in some intermediate type would only move the message
writing elsewhere.

### 4.3 An annotation parameter rather than a span field

§2.1: one parameter gives `forgetSpans` for free, so the round-trip
property is literally `parse (print t) == t` with derived equality, and it
leaves room for later annotations. The cost is a type parameter on every
type, and that the annotation is the first constructor field, accessed
with `exprAnn` and friends rather than a record field (records with fields
shared by some constructors would violate `-Wpartial-fields`).

### 4.4 Report and continue, or fail and resynchronise

Errors that leave a usable token are registered; the rest abandon the
declaration (§1.6). The line between them is "can parsing continue
meaningfully *right here*?". A malformed number is still a number; a
missing `in` leaves no body to parse, so it fails. Registering more
aggressively (inventing a missing `)`, say) is how some compilers
recover inside expressions, but it guesses, and wrong guesses are the
cascades recovery is meant to avoid. Declarations are small, so losing the rest of one costs little.

### 4.5 Column 1 for `let`, and only for recovery

The language has no layout rule (01-language §9: "unambiguous without
layout rules"), and this step keeps it that way. Recovery is a heuristic
by nature, though, and a heuristic may use layout: §3.9.

### 4.6 Changes to the language spec

Three genuine gaps in `01-language.md`, fixed minimally:

- **`name [ … | … ]`** (§9, §5.6). The examples write
  `mean [ (net(x) - y) ^ 2 | (x, y) <- xor ]` and
  `sum [ sum(p ^ 2) | p <- params ]`, but the grammar only allowed calls
  with parentheses, and `mean [` would have started an index. The grammar
  now has `ident comprehension` as a call with one argument, and states
  when a `[` after a name is an index (a number and then `]` or `,`). The
  alternative, changing the examples to `mean([ … ])`, would have made the
  most common form in the language the clumsiest.
- **Signed initialiser arguments** (§9). `"~" ident "(" number "," number
  ")"` did not allow `uniform(-1, 1)`, used in every example. A `signed`
  rule (`"-"? number`) is now used there, in exponents and in literals.
- **Lexical details** (§2): type names are reserved; the number syntax is
  spelled out, including an optional `+` in exponents; strings cannot span
  lines.

## 5. Tests

`test/HsGrad/Lang/SyntaxSpec.hs`, `test/HsGrad/Lang/ParserSpec.hs`, and
the generators in `test/HsGrad/Test/Syntax.hs`.

### 5.1 Unit tests

- **Examples**: every file in `examples/` parses; `intro.hsg` to exactly
  the expected tree; the declarations of `moons.hsg` in order; the small
  snippets of 01-language §3 (`fn sq(z) = z ^ 2`, an `input` without a
  default, a `let` with an annotation, a one-field dataset with bare
  rows, …).
- **Precedence**: `-x ^ 2` is `-(x²)`, `2 * W @ x` is `2 * (W @ x)`,
  `d * -2.0`, `-A @ x` is `(-A) @ x`, left associativity of `-`, `/` and
  `@`, `x ^ 2 ^ 3` rejected, `- -x`, the extent of `let … in` (to the end,
  nested, ended by `,` in a call, not allowed as a bare operand), indexing
  calls and indexes, `sum [ … ]` equal to `sum([ … ])`, vector versus
  comprehension.
- **Tokens**: keywords only as whole words (`inputs`, `letter`, `in_` are
  names; `input` and `Vec` are not), nested comments, `a --b` versus
  `a - -b`, exact numbers (`0.1` is `1/10`, `0.30000000000000004` keeps
  every digit), each malformed number gives E0103, string escapes,
  `parseLiteral`, `parseType`.
- **Spans**: the example of docs/11 §1.3 (`d * -2.0` at `1:9–1:17`),
  parentheses inside the span, no trailing whitespace or comments in a
  span, a tab is one column, a multi-line `data` declaration.
- **Printer**: `showNumber` on integers, decimals, tiny and huge numbers,
  and a rational without a decimal form; parentheses for right operands,
  lower precedence, `(-x) ^ 2`, `let` operands; never `--`.

### 5.2 Golden files

`test/golden/parse/` has one `.hsg` file per case and the expected output
as `.err`, rendered exactly as the driver will. A test checks that every
code E0101–E0105 appears in some `.err` file, and another that the parser
reproduces the two step-10 golden files that were built by hand.

| File | What it pins down |
|---|---|
| `unexpected-token` | E0101, the whole token marked, an expected list |
| `unexpected-eof` | the caret after the last token, not on an empty line |
| `unknown-character` | "character `#`", hints from the declaration's end |
| `keyword-as-name` | a reserved word as a name, with its hint |
| `top-level-in` | `in` after a top-level `let` |
| `call-without-parens` | `tanh x` |
| `missing-init` | a `param` without `=` / `~` |
| `not-natural` | `Vec 2.5`, `v[1.5]`, both reported |
| `variable-index` | `v[i]` and `sum [v, v]` |
| `chained-power` | `x ^ 2 ^ 3` |
| `unknown-escape` | `\n` in a string |
| `unterminated-comment` | E0102, identical to step 10's |
| `nested-comment` | E0102 with the note about nesting |
| `unterminated-string` | E0102 at the end of the line |
| `comment-in-expression` | E0102 without the cascade at the end of the file |
| `bad-numbers` | six different E0103s in one file |
| `missing-in` | E0104, identical to step 10's |
| `variable-exponent` | E0105 with the printed `exp(… * log(…))` hint |
| `recovery` | four independent errors in one file |
| `recovery-inner-let` | no resynchronisation at indented `let`s |

### 5.3 Properties

- **Round trip, expressions**: a random expression (every form, mixed
  freely, names that look like keywords, numbers with and without
  exponents), printed and parsed back, gives the same tree. 1000 samples;
  20 000 samples with two seeds pass too.
- **Round trip, programs**: the same for random programs of every
  declaration kind, with strings containing quotes and backslashes.
- **Spans, expressions**: for every subexpression of a parsed printed
  expression, the text under its span parses back to that subexpression.
  This checks at once that spans start and end at the right characters,
  include parentheses, and exclude whitespace.
- **Spans, declarations**: the same for whole declarations.
- **Numbers**: `showNumber` output parses back to the same `Rational`.

The generators only produce trees the parser can produce (§2.2): no
negative `Num`, no reserved names, non-empty lists where the grammar
requires them. Shrinking goes to subexpressions.

### 5.4 Testing the tests

Deliberate bugs, each tried on its own:

| Bug | Failing tests |
|---|---|
| binary `-` right-associative | 6 |
| `@` on the same level as `*` and `/` | 3 |
| printer: right operand allowed at the same level | 3 |
| token spans include the whitespace after them | 9 |
| parentheses not part of the span | 4 |
| megaparsec's default tab width 8 | 1 |
| keywords match the start of longer words | 4 |
| no one-token skip on the line of the broken declaration | 1 |
| recovery resynchronises at any `let` | 2 |
| no cascade filter after E0102 | 2 |
| numbers read through `Double` | 5 |
| printer may emit `--` | 5 |
| E0104 at the `let` keyword instead of after the bound expression | 3 |
| index lookahead without the closing `]` / `,` | 5 |

All are killed. The first version of this list had a "mutant" that
changed nothing; it was replaced by the E0104 one above.

The full suite has 446 tests and runs in about 3 seconds.

## 6. What later steps build on

- **Step 12 (`Lang.Shape`, `Lang.Check`)**: walk `Program Span`.
  Every name is an `Ident` with its own span (E0201, E0202, W0201 point
  there); `LossDecl`'s second field is the `loss` keyword (E0206); `Nat`
  spans for E0305; `typeAnn` / `exprAnn` for E0304 and E0308. The parser
  accepts any initialiser name (`~ foo(1, 2)`) and any dimension name; the
  checker decides. `Call f [Comprehension …]` is how `sum [ … ]` arrives:
  E0311 is "a comprehension that is not the argument of `sum` / `mean`".
  `prettyType` prints types for messages. Golden files go to
  `test/golden/check/`, with the same harness line as `ParserSpec`.
- **Step 13 (`Data.*`)**: `DataSource` gives `Rows` (literals with spans,
  for E0404) and `Csv` (the path as `StringLit`, already unescaped, and the
  column mapping with spans, for E0402).
- **Step 14 (`Lang.Compile`)**: AST spans go straight into `withSpan`.
  Numbers are exact `Rational`s: `fromRational` for `Const`, and the
  `Power` exponent as it is for `PowC`.
- **Step 15 (`Driver`)**: `parseProgram file text`; on `Left`, render
  with the same `Sources` and stop (no point checking a broken tree).
  `parseLiteral` reads `--set x=[1,2,3]` values, with E0503 left to the
  driver.
