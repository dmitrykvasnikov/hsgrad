# 11 — `Lang.Span` and `Lang.Diagnostic`: positions and error messages

This step starts the language layer, and starts it at the end: with the
error messages. There is no parser yet. What exists after this step is
everything a later stage needs to *report* a problem:

- `HsGrad.Lang.Span`: **positions** (line and column) and **spans**
  (regions of a file), merging spans, and the conversion into the
  provenance that Core nodes carry (D4);
- `HsGrad.Lang.Diagnostic`: one **`Diagnostic`** type for every error and
  warning, a closed type of **codes** taken from the catalogue in
  `01-language.md` §10, and a pure **renderer** that prints diagnostics
  with source snippets in the style of `00-architecture.md` §6;
- a small **golden-file harness** (`test/HsGrad/Test/Golden.hs`) that
  compares rendered output with files in `test/golden/`. The parser
  (step 11) and the checker (step 12) will reuse it for whole files.

Writing the reporting side first has a practical reason: from step 11 on,
every test of a failure case is a test of a rendered message, and those
tests need a renderer to exist and to be trusted.

The milestone is the example the architecture page showed when this step
was written, reproduced exactly from a hand-built diagnostic
(`test/golden/diagnostics/matrix-product.err`). The program behind it
was imagined (`moons.hsg` has no such line); `00-architecture.md` §6 now
shows a real one, from `examples/errors.hsg`:

```
error[E0302]: shape mismatch in matrix product
  --> examples/moons.hsg:14:17
   |
14 |   let h2 = relu(W2 @ h1 + b2) in
   |                 ^^^^^^^ Mat 16 16 @ Vec 8
   |
   = note: h1 : Vec 8 because W1 : Mat 8 2 (declared at 6:7)
```

---

## 1. Theory

### 1.1 What a good error message does

An error message has one job: to get the reader from "something is wrong"
to "I know what to change" as fast as possible. Compilers that do this
well, rustc and Elm above all, agree on a few things.

- **Say where, by showing it.** `moons.hsg:14:17` is precise but makes the
  reader open the file and count columns. Quoting the line and underlining
  the exact expression answers "where" at a glance.
- **Say what is wrong at that place**, in its own words. The headline
  ("shape mismatch in matrix product") is the general problem; the label
  next to the underline ("`Mat 16 16 @ Vec 8`") is what is wrong *here*.
- **Point at every place involved.** Many errors are a disagreement
  between two places: a duplicate declaration and the first one, a call
  and the signature it violates. The error should show both, and say which
  is which.
- **Explain, then suggest.** Notes give the reasoning the reader would
  otherwise have to reconstruct ("`h1 : Vec 8` because …"); hints propose
  a fix ("did you mean `sigmoid`?").
- **Have a stable name.** A code like `E0302` can be searched for, listed
  in documentation, and tested for.

The `Diagnostic` type has exactly these parts, and the renderer lays them
out the way rustc does, since that layout is widely known and survives
copying into a plain-text issue or chat.

### 1.2 Positions: lines and columns, from 1

A position in a file can be stored in two ways:

- as an **offset**: the number of characters before it;
- as a **line and column**.

Offsets are compact and make "is this span inside that one?" a comparison
of two numbers. But every place that *shows* a position needs a line and
column, and turning an offset into one means scanning the file. Since
positions in this project exist to be shown (in messages, in Graphviz
cluster labels), we store lines and columns. Comparing them is still easy:
`Pos` orders by line, then column, which is the order in the file.

Both count **from 1**, as every editor, compiler and terminal does:
`moons.hsg:14:17` means the 14th line, 17th character, and clicking on it
in a terminal or an editor's error list jumps there. Counting from 0 would
make every printed position off by one compared to the editor.

What is "a column"? We count **characters** (Unicode code points): a tab
is one column, and so are `ø` and `∂`. The alternative, counting *display*
columns (a tab jumps to the next multiple of 8, say), makes a position
depend on a tab-width setting that the file does not record. Character
columns are what the parser naturally produces, and what editors use for
"go to line:column". How wide things are on screen is a question for the
renderer alone (§1.5).

### 1.3 Spans

A **span** is a region of a file: a start position (inclusive) and an end
position (exclusive). Exclusive ends make lengths and adjacency simple:
`a` in `let a` is `1:5–1:6`, one character; an **empty** span has
`start == end` and means "between two characters", which is exactly what
"expected `in` here" at the end of a line needs.

Parsers build spans bottom-up. A literal or a name gets the span of its
characters; a compound expression gets the span that **covers** its parts:

```
d * -2.0          span of d:     1:9–1:10
                  span of -2.0:  1:13–1:17
                  span of d * -2.0 = merge: 1:9–1:17
```

`mergeSpan` takes the earlier start and the later end. It is commutative,
associative and idempotent, and the result is the smallest span that
contains both: the properties a "covering" operation should have, and what
the tests check.

### 1.4 Anatomy of a rendered diagnostic

```
error[E0303]: shape variable `n` bound inconsistently          ← header: severity, code, message
 --> shape-variable.hsg:8:22                                   ← location of the primary span
  |
5 | fn neuron(w : Vec n, b : Real, x : Vec n) : Real =         ← source line, with its number
  |               -----                ----- both parameters…  ← secondary labels: ---
...                                                            ← skipped lines
8 | let y = neuron(w, b, x)
  |                -     ^ `Vec 2`, so `n = 2` here            ← primary label: ^^^
  |                |
  |                `Vec 3`, so `n = 3` here                    ← a label hanging below
  |
  = note: while checking the call at shape-variable.hsg:8:9    ← notes, then hints
```

- The **header** has the severity (`error` or `warning`), the code and the
  message.
- `-->` gives the **location** of the primary span in `file:line:column`
  form, which terminals and editors make clickable.
- The **gutter** (line numbers and `|`) is as wide as the largest line
  number shown, so the source lines stay aligned.
- Each shown line is followed by its **marks**: `^` under the primary span
  (the place the message is about), `-` under secondary spans (related
  places). The label goes right after the marks; when several labels
  share a line, all but the rightmost **hang** below, connected by `|`.
- Lines between two labelled lines are **skipped** with `...` if there
  are two or more of them; a single line is shown instead, since `...`
  would take the same room.
- Labels in **another file** (a CSV header, say) get their own section,
  introduced by `::: file:line:column`.
- **Notes** (`= note:`) explain, **hints** (`= hint:`) suggest.

A span over several lines is drawn with a line in a margin, from where it
starts to where it ends:

```
4 |   fn layer(x : Vec 2) : Real =
  |                         ---- expected `Real` because of this
5 | /   let z = W @ x + b in
6 | |   tanh(z)
  | |_________^ this is `Vec 2`
```

When the span starts after other text on its first line, a connector
underlines up to the start instead of the `/`:

```
7 |   let errors = [ (dot(w, x) - y) ^ 2
  |  ______________^
8 | |              | (x, y) <- xor ]
  | |______________________________^ this is a sequence of `Real`
```

### 1.5 Display columns: tabs and wide characters

The renderer has to put a caret *under* a character, so it needs display
columns. Tabs are the problem: a tab is one character but several screen
cells, and how many depends on where it is. The renderer expands each tab
to the next **tab stop** (every 4 columns by default) in the quoted line,
and computes caret positions with the same rule, so they agree:

```
4 |     relu(z) +   sigmod(z)       ← "\trelu(z) +\tsigmod(z)"
  |                 ^^^^^^ not defined
```

Every other character counts as one cell. That is right for ASCII and for
most letters and symbols (`ø`, `∂`, `·`), but not for East Asian wide
characters, emoji or combining accents, which would need Unicode width
tables. `.hsg` identifiers are ASCII, so these can only appear in comments
and strings; the cost of a slightly misplaced caret there is accepted.

## 2. Where the style comes from

The layout follows rustc's closely, including details that are easy to
overlook: line numbers left-aligned in the gutter, `-->` for the primary
location and `:::` for other files, `/` versus `__^` at the start of a
multi-line span, labels hanging right to left. What differs:

| rustc | hsgrad | Why |
|---|---|---|
| `help:` | `hint:` | the catalogue says "hint" |
| `error: aborting due to 2 previous errors; 1 warning emitted` | `2 errors, 1 warning` | shorter; we do not abort early (D8, §7 of the language) |
| severity chosen per diagnostic | severity follows the code | §4.4 |
| diagnostics without a span | always a primary span | §4.5 |
| Unicode width tables | one cell per character, tabs expanded | §1.5 |

Elm contributes the tone rather than the layout: messages in plain words,
hints that say what to write.

## 3. Code walkthrough

The modules live at `src/HsGrad/Lang/Span.hs` and
`src/HsGrad/Lang/Diagnostic.hs`.

### 3.1 Positions and spans

```haskell
-- from HsGrad.Core.Graph (step 3), re-exported
data Pos = Pos { posLine :: !Int, posCol :: !Int }                  -- 1-based
data SrcSpan = SrcSpan { spanFile :: !FilePath, spanStart, spanEnd :: !Pos }

type Span = SrcSpan
```

The span type *is* Core's `SrcSpan` (§4.1). `Lang.Span` adds what the
language side needs:

```haskell
startPos    :: Pos                          -- 1:1
advance     :: Pos -> Char -> Pos           -- '\n' → next line, column 1; else column + 1
advanceText :: Pos -> Text -> Pos

mkSpan    :: FilePath -> Pos -> Pos -> Span  -- either order
pointSpan :: FilePath -> Pos -> Span         -- empty
textSpan  :: FilePath -> Pos -> Text -> Span -- covers the text

mergeSpan  :: Span -> Span -> Span           -- earlier start, later end
mergeSpans :: NonEmpty Span -> Span

spanContains, spanIsEmpty, spanIsMultiLine, showPos, showLocation
spanProvenance :: Span -> Provenance         -- Provenance (Just sp) Nothing
```

`advance` is the definition of a column: one per character, tabs included,
and a newline starts the next line at column 1. A `\r` before the newline
is a character like any other, so CRLF files have the same line numbers;
the renderer drops it when it shows a line. The hand-written CSV reader
(step 13) can track positions with `advance` directly; tests use
`advanceText startPos prefix` to find where a piece of text starts.

`mergeSpan` keeps the left span's file. Merging spans of two files has no
meaning, and making it a `Maybe` would burden every use in the parser, where
both spans always come from the same file.

`spanIsMultiLine` treats a span that ends at column 1 of a line as ending
with the newline before it: `"ab\n"` covers one line. The renderer applies
the same rule (§3.5).

### 3.2 Codes and severities

```haskell
data Code = E0101 | E0102 | … | E0504 | W0201 | W0202 | W0203
  deriving (Eq, Ord, Show, Enum, Bounded)

data Severity = Error | Warning

codeText     :: Code -> Text       -- "E0302"
codeSeverity :: Code -> Severity   -- from the first letter
codeSummary  :: Code -> Text       -- the catalogue's one-line meaning
```

The constructors *are* the codes, so code that creates a diagnostic reads
like the catalogue: `diagnostic E0201 …`. A test reads the tables of
`01-language.md` §10 and checks that they list exactly these codes, so the
documentation and the type cannot drift apart.

This step adds one code. Step 5 found that backprop can fail with a
non-finite *gradient* even when every forward value is finite (`√x` at
`0`), which no code described. It is now **E0504**, "non-finite gradient",
reported like E0501 at the span of the node that passes the gradient on.

### 3.3 Building a diagnostic

```haskell
data Label = Label { labelSpan :: !Span, labelText :: !Text }

data Diagnostic = Diagnostic
  { diagCode :: !Code
  , diagMessage :: !Text
  , diagPrimary :: !Label
  , diagSecondary :: ![Label]
  , diagNotes :: ![Text]
  , diagHints :: ![Text]
  }

diagnostic    :: Code -> Text -> Span -> Text -> Diagnostic
withSecondary :: Span -> Text -> Diagnostic -> Diagnostic
withNote      :: Text -> Diagnostic -> Diagnostic
withHint      :: Text -> Diagnostic -> Diagnostic
severity      :: Diagnostic -> Severity
diagSpan      :: Diagnostic -> Span
```

`diagnostic` makes the required part; the `with…` functions add the
optional parts at the end, in order, and read well with `&`:

```haskell
diagnostic E0202 "duplicate declaration of `w`" second "declared again here"
  & withSecondary first "first declared here"
  & withHint "rename one of them"
```

A label's text may be empty: then only the span is marked. A note or hint
may have several lines; continuation lines are indented under the first.

### 3.4 Rendering: painted lines, then text

```haskell
type Sources = Map FilePath Text

data RenderOptions = RenderOptions
  { renderColour :: !Bool    -- ANSI colours
  , renderTabWidth :: !Int   -- tab stops, default 4
  }

renderDiagnostic  :: RenderOptions -> Sources -> Diagnostic -> Text
renderDiagnostics :: RenderOptions -> Sources -> [Diagnostic] -> Text
```

The renderer is pure: source text in, text out. It never fails. A span
past the end of the text shows an empty line, a span in a file missing
from `Sources` is reported by position only (`= 3:5: not a number`), and
reversed spans or line 0 are put in order and clamped to 1.

Internally the output is first a list of **lines of painted segments**:

```haskell
data Paint = Plain | Title | Accent Severity | Secondary | Gutter | NoteKind
type Line  = [(Paint, Text)]
```

A segment says *what* it is (the header, a primary mark, the gutter), not
how it looks. The last step, `renderLines`, turns each line into text: it
drops trailing whitespace and, only if colour is on, wraps segments in ANSI
escape codes (errors bold red, warnings bold yellow, secondary marks and
the gutter bold blue, the message bold). So coloured output is exactly the
plain output with escape codes inserted, which is what one of the tests
checks by stripping them.

`diagnosticLines` produces the header, one section per file (the primary
span's file first, then other files in the order their labels appear),
and the notes and hints. The gutter width is computed once, from the
largest line number of any section.

### 3.5 Which lines to show

Inside a section, each label becomes an `Anno`: its role (primary or
secondary), its text, and its span with the positions in order. Then:

- **End normalisation.** A span that ends at column 1 of a line ends with
  the newline before it, since the end is exclusive. It is redrawn as
  ending after the last character of the previous line, so "the whole
  line `let a = 1`" does not reach into the next line.
- **Shown lines** are the lines where some label starts or ends, plus any
  single line between two of them. Longer stretches become a `...` row.
- A file that ends with a newline has an empty last line, and that is
  where a span "at the end of the file" points (`4 |` in
  `unterminated-comment.err`).

### 3.6 Single-line labels

The rows under a source line are drawn on a **canvas**, a map from display
column to character and paint (`IntMap (Char, Paint)`). Display columns
come from `dispCol`, which expands tabs exactly as the quoted line does
(§1.5); a column past the end of a line continues after it, which is how
a caret appears after the last character.

1. **Marks.** Every label on the line writes `^` or `-` over its columns;
   an empty span gets one mark. Secondary labels are written first, so
   where a primary and a secondary span overlap, the primary marks win:
   `--^----`.
2. **The inline label.** The label whose marks end furthest to the right
   gets its text on the mark row, after all the marks.
3. **Hanging labels.** The others with text, ordered by start column, get
   a row of `|` connectors and then one row each, rightmost first, with
   `|` still standing for the ones further left:

```
1 | let y = f(a, b, c)
  |           -  -  ^ third
  |           |  |
  |           |  second
  |           first
```

### 3.7 Multi-line labels and lanes

A multi-line label is drawn in a **margin** between the gutter and the
source. Each label gets a **lane**, a column of the margin:

```haskell
data Phase = PSingle | PConnect !Int | PEnd !(Down Int)
```

- On its **first line** the lane shows `/` if only whitespace comes before
  the start; otherwise a **connector row** under that line, `__…^`, runs
  from the lane to the start column.
- Lines in between show `|` in the lane, as do `...` rows.
- Under its **last line**, an **end row** `|__…^ text` runs from the lane
  to the last character of the span.

The rows under one source line come in a fixed order, the `Phase`: marks
of single-line labels, then connectors in lane order, then end rows in
*reverse* lane order. The derived `Ord` of `Phase` encodes that order
(`Down` reverses the ends), and a lane's character on any row follows from
comparing the row's phase with the label's own: on its first line a lane
is `|` only after its connector row, on its last line only up to its end
row. Ending inner labels first keeps lines from crossing when spans nest:

```
2 | /    let s = dot(x,
  | | ___________-
3 | ||               x) in
  | ||________________- `Real`
4 | |    s + [1,
  | | _______-
5 | ||        2]
  | ||_________- `Vec 2`
  | |__________^ this is `Vec 2`
```

Lanes are assigned greedily in order of start: a label takes the lowest
lane whose previous label ended on an earlier line. So spans that do not
overlap share one lane (above, `dot(…)` and `[1, 2]` both use the second
lane), and the margin is only as wide as the deepest nesting. Without
multi-line labels there is no margin at all.

### 3.8 Lists of diagnostics

```haskell
sortDiagnostics :: [Diagnostic] -> [Diagnostic]   -- by primary span; stable
hasErrors       :: [Diagnostic] -> Bool
summary         :: [Diagnostic] -> Text           -- "2 errors, 1 warning"
```

`renderDiagnostics` sorts, renders each diagnostic followed by a blank
line, and ends with the summary; an empty list renders as nothing. The
checker finds problems in whatever order its traversal visits them; the
reader wants them top to bottom. The sort is stable, so two diagnostics at
the same span keep the order in which they were reported.

## 4. Design decisions

### 4.1 One span type, owned by Core

Core nodes record where they came from (D4), so Core needs a span type,
and Core cannot import Lang (§2 of the architecture). Step 3 therefore
defined `Pos` and `SrcSpan` in `Core.Graph`. This step could have defined
a second, "richer" `Span` in Lang and converted between the two. It
doesn't: the representation step 3 chose (file, 1-based line and column,
exclusive end) is exactly the one argued for in §1.2–1.3, so a second type
would hold the same data and add conversions in both directions: Lang →
Core when the compiler wraps code in `withSpan` (step 14), Core → Lang
when a runtime error (E0501) is reported at a node's span (step 15).
`type Span = SrcSpan` makes both directions free. The "conversion into
Core" that remains is `spanProvenance`, for code that builds a
`Provenance` value directly.

The cost is that `Span` has no instances of its own (a `Semigroup` for
merging would be an orphan instance), hence the plain `mergeSpan`.

### 4.2 Line and column, not offsets

§1.2 has the main argument: positions exist to be shown. Two smaller ones:
megaparsec (step 11) produces line and column directly, and the renderer
needs the text of a line anyway, which it gets by splitting the source,
not by searching for an offset. Comparing positions is a derived `Ord`.

### 4.3 Character columns; tabs only expanded on screen

Columns count characters, so the parser, the CSV reader and editors agree
on them without a tab-width setting. The renderer expands tabs to tab
stops, configurable with `renderTabWidth`, both in the quoted line and
when placing marks. One consequence for step 11: megaparsec's default tab
width is 8, which makes a column after a tab jump to the next multiple of
8. The parser must set it to 1 (`pstateTabWidth = mkPos 1`) so that its
columns are character columns.

### 4.4 Codes are a closed type; severity follows the code

A `Text` code would let any stage invent `"E0399"`. With a closed type,
the compiler rejects unknown codes, and the catalogue test keeps the type
and `01-language.md` in step. Severity is not a separate field because the
code already determines it (`E…` or `W…`): a separate field could only
disagree with it.

### 4.5 Every diagnostic has a primary span

rustc allows diagnostics without a location. Every code in our catalogue
has a natural place in a file: even "input has no value" (E0502) points
at the `input` declaration, and "batch larger than the dataset" (E0405)
at the `data` declaration. Problems that are not about any file (an
unknown command-line flag) are reported by the CLI parser, not as
diagnostics. Making the primary label mandatory keeps the renderer simple
and guarantees every message says where.

### 4.6 `Sources` is a map, not one text

The simplest renderer would take one source text: "source text →
diagnostic → text". Most diagnostics indeed live in one `.hsg` file. But
E0402 ("column
missing from the CSV header") is best shown with a label on the mapping in
the `.hsg` file *and* one on the header line of the CSV file. A map from
file name to text covers both; one file is `Map.singleton name text`.

### 4.7 Paint, not escape codes, in the layout

Colour is decided at the very end (§3.4). The layout code never sees
escape codes, so column arithmetic is never confused by invisible
characters, and the plain output used by every golden test is the same
code path as the coloured one. The driver (step 15) will turn colour on
when standard error is a terminal.

### 4.8 No trailing whitespace

Every rendered line has its trailing whitespace removed. The mark rows
are built on canvases that are padded with spaces, and an empty label
would leave a trailing space. Golden files with invisible trailing spaces
are fragile: editors strip them, and diffs do not show them.

### 4.9 Two corrections to earlier docs

- The architecture's example (§6) was labelled `E0301`, but in the
  catalogue `E0301` is "elementwise operands with incompatible shapes" and
  a matrix product with mismatched dimensions is `E0302`. The example now
  says `E0302`.
- `E0504` "non-finite gradient" was added to the catalogue (§3.2).

## 5. Tests

`test/HsGrad/Lang/SpanSpec.hs` and `test/HsGrad/Lang/DiagnosticSpec.hs`:

- **Positions**: columns per character, tabs and Unicode included;
  newlines; CRLF; advancing over `a <> b` is advancing over `a`, then `b`.
- **Spans**: `mkSpan` orders, `textSpan` covers its text, empty and
  multi-line spans (including the column-1 rule); a span contains exactly
  the spans between its ends; `mergeSpan` is the earlier start to the
  later end, commutative, associative, idempotent, contains both; spans
  of other files are not contained; locations print as `file:line:col`;
  `spanProvenance`; a span passed to the builder's `withSpan` ends up on
  the node; the order is file, start, end.
- **Codes**: the type lists exactly the codes of the catalogue (read from
  `docs/01-language.md`), severities follow the letter, every code has a
  summary.
- **Golden files**: eleven files, below.
- **Layout**, with the expected text inline: an empty label; a gutter that
  grows to two digits; `...` versus a single skipped line; an empty span
  at the end of a file without a final newline; a span ending at column 1;
  overlapping primary and secondary marks; three labels on one line;
  multi-line notes and hints; a label in a second file; a file without
  source; a tab width of 2.
- **Colour**: stripping the escape codes gives the plain output; errors
  red, warnings yellow, secondary marks blue; off by default.
- **Lists**: stable sorting by position, files by name; summaries; no
  output for no diagnostics; blank lines between diagnostics.
- **Robustness**: random sources (tabs, Unicode, CRLF, blank lines) with
  random labels, many of them nonsensical (reversed, past the end, line 0,
  in an unknown file), always render, start with the header and have no
  trailing whitespace.

### 5.1 Golden files

`test/golden/diagnostics/` holds one `.hsg` file per case and the expected
output next to it as `.err`. There is no parser yet, so the diagnostics of
each case are built by hand in `DiagnosticSpec.goldenCases`, with spans
found by searching the source (`at src "W2 @ h1"`, `nth src 4 "x"`,
`fromTo src "let z" "tanh(z)"`, `endOfFile src`). Each case is rendered
the way the driver will render a file: `renderDiagnostics`, plain, with
the summary line.

| File | What it pins down |
|---|---|
| `matrix-product` | the architecture's example, character for character |
| `shape-variable` | two labels on one line, hanging labels, `...` |
| `body-mismatch` | a multi-line span with `/`, a hint |
| `comprehension` | a multi-line span with a `__^` connector |
| `two-lanes` | two multi-line spans sharing a lane |
| `nested-spans` | nested multi-line spans, inner closing first |
| `tabs` | tabs before and inside the line |
| `unicode` | non-ASCII characters: one caret each |
| `missing-in` | an empty span at the end of a line, a two-line hint |
| `unterminated-comment` | a label at the end of the file |
| `several` | four diagnostics out of order: sorted, gutter width per diagnostic, summary |

**The harness** (`test/HsGrad/Test/Golden.hs`) is meant for the later
steps:

```haskell
goldenFiles :: FilePath -> String -> (FilePath -> Text -> IO Text) -> Spec
goldenText  :: FilePath -> Text -> Expectation
```

`goldenFiles dir ".hsg" run` makes one test per `.hsg` file in `dir`
(sorted by name): it reads the file, calls `run` with its name and
contents, and compares the result with the `.err` file next to it. A
missing `.err` file is a failure, not silently created. On a mismatch the
failure shows the first differing line and the whole actual output.

**Regenerating** after an intended change of output:

```bash
HSGRAD_ACCEPT=1 cabal test
```

With `HSGRAD_ACCEPT` set (to anything but `0`), every golden test writes
its actual output to its `.err` file and passes. Then read the change with
`git diff test/golden` before committing it: the point of a golden test is
that a person has looked at the output once. All files are read and
written as UTF-8 whatever the locale.

**Adding a case** in step 11 or later is: drop `name.hsg` into the stage's
directory (`test/golden/parse/`, `test/golden/check/`, …), run with
`HSGRAD_ACCEPT=1`, review `name.err`, commit both. The stage's spec has
one line:

```haskell
goldenFiles "test/golden/parse" ".hsg" $ \name src ->
  pure (renderDiagnostics defaultRenderOptions (Map.singleton name src) (parseDiagnostics name src))
```

Use the bare file name as the span file, as above, so the `-->` lines do
not depend on the directory the tests run in.

### 5.2 Testing the tests

Deliberate bugs in the renderer, each tried on its own:

| Bug | Failing tests |
|---|---|
| tabs count as one display column | 2 |
| the leftmost label gets the inline text | 4 |
| no end normalisation (column 1 of the next line) | 1 |
| no single skipped line, always `...` | 2 |
| secondary marks drawn over primary ones | 1 |
| sort by the end of the span | 1 |
| outer multi-line labels close first | 1 |
| no lane reuse | 2 |
| never the `/` style | 2 |
| trailing whitespace kept | 5 |
| hanging labels emitted left to right | 1 |
| summary counts every diagnostic as an error | 3 |

The second-to-last survived the first round: no test had two hanging
labels on one line. The "three labels on one line" test exists because of
it.

The full suite has 372 tests and runs in under 3 seconds.

## 6. What later steps build on

- **Step 11 (`Lang.Syntax`, `Lang.Parser`)**: annotate every AST node with
  a `Span`; compound nodes get `mergeSpan` of their parts (or the span from
  the first to the last token). Set megaparsec's tab width to 1
  (`pstateTabWidth = mkPos 1`) so `SourcePos` columns are character
  columns, and convert with `Pos (unPos (sourceLine p)) (unPos (sourceColumn p))`.
  Turn parse errors into diagnostics yourself (E0101–E0105) rather than
  printing megaparsec's own messages: `E0101` with the unexpected token
  as primary span and "expected …" as label; `E0102` with the opening
  `{-` or `"` as primary and `pointSpan` at the end of the file as
  secondary; `E0104` as an empty span at the end of the `let` expression
  plus the `let` keyword as secondary (`missing-in.err` is the target).
  Golden files go to `test/golden/parse/` (§5.1).
- **Step 12 (`Lang.Check`)**: collect diagnostics in a list (a `Writer`
  or an accumulator); `hasErrors` decides whether to continue; let the
  driver sort. Use secondary labels for the "other place" of E0202,
  E0303, E0308 and W0201 (the golden files show the intended look), and
  `withNote` for "while checking the call at …" (`showLocation` gives the
  `file:line:col`).
- **Step 13 (`Data.*`)**: track CSV positions with `advance`; give
  `Sources` both the `.hsg` text and the CSV text so E0402/E0403 can
  quote both files.
- **Step 14 (`Lang.Compile`)**: AST spans go straight into
  `withSpan` (`Span` is `SrcSpan`).
- **Step 15 (`Driver`)**: read the file once, keep its text for `Sources`,
  render with `renderDiagnostics` to standard error, colour on when it is a
  terminal (`hIsTerminalDevice stderr`), write it as UTF-8, exit with
  failure when `hasErrors`. Runtime errors: `NonFiniteValue` → E0501 and
  `NonFiniteGrad` → E0504 at `provSpan (nodeProv (nodeAt g i))`, which is
  already a `Span`; `UnboundLeaves` on an input → E0502 at the `input`
  declaration.
