# 14 — `Data.Csv` and `Data.Dataset`: datasets from CSV files and inline rows

A `data` declaration names a dataset, gives its **schema** and says where
the rows come from (`01-language.md` §3):

```
data xor   : { x : Vec 2, y : Real } = rows [ ([0, 0], -1), ([0, 1], 1), … ]
data moons : { x : Vec 2, y : Real } = csv "data/moons.csv" { x = [x1, x2], y = label }
```

The checker (step 12) has made sure the schema is well formed, but it
never looks at the source. This step adds the two modules that do:

- `HsGrad.Data.Csv`: a small, hand-written reader for CSV files of
  numbers. It checks the **structure** of a file (a header, then one line
  per row with one cell per column, quotes, line ends) and keeps the
  position of every cell, so a problem can be shown in a snippet of the
  CSV file itself. It reads cells as numbers **exactly**.
- `HsGrad.Data.Dataset`: turns a checked `data` declaration into a
  **dataset in memory**, rows of numbers in column order, after checking
  that the source fits the schema. Every problem is a diagnostic with one
  of the codes E0401–E0405. It also provides the adapter to the training
  loop of step 9 (`toFeed`, `slotNames`) and the `--batch` check.

The milestone of the roadmap is "CSV and inline datasets". Both examples
load: `xor.hsg` gives the four rows the hand-built XOR net of step 9 was
trained on (and it trains on them again, §5), and `moons.hsg` reads the
100 points of the new `examples/data/moons.csv`. A problem inside a CSV
file is shown in that file, with the declaration that reads it
(`test/golden/data/non-numeric.err`):

```
error[E0403]: cell `abc` of column `x2` is not a number
 --> test/golden/data/non-numeric.csv:3:5
  |
3 | 0.5,abc,-1,a typo
  |     ^^^ not a number
  |
 ::: test/golden/data/non-numeric.hsg:4:36
  |
4 |   csv "non-numeric.csv" { x = [x1, x2], y = label }
  |                                    -- read for field `x`
  |
  = note: column `x2` has 2 more bad cells, on lines 5 and 6
```

---

## 1. Theory

### 1.1 Rows, columns and fields

A dataset is a table. Each **row** is one example (a point and its
label); the schema says what a row consists of: **fields**, each a `Real`
or a `Vec n`. A field of type `Vec n` is `n` numbers, so a row of
`{ x : Vec 2, y : Real }` is three numbers. We call those the row's
**columns** and name them like the elements of a parameter
(`10-train.md` §3.1):

```
schema        { x : Vec 2,       y : Real }
columns         x[0]    x[1]     y
row 0           0       0        -1
```

The column order is the schema order with vectors expanded. Everything
after this step sees a dataset only as rows of numbers in that order: the
training loop binds row `r` to slot `i` by zipping the row with the
slot's leaf names (`moons#i.x[0]`, `moons#i.x[1]`, `moons#i.y`, D10).

In a CSV file the columns are named by the file's **header**, which knows
nothing about the schema. The **mapping** `{ x = [x1, x2], y = label }`
connects the two: field `x` gets the file's columns `x1` and `x2`, field
`y` its column `label`. The file may order its columns differently and
have columns nobody asked for.

### 1.2 CSV: a format with many dialects

CSV ("comma-separated values") is older than any standard for it. RFC 4180
(2005) describes the common core, and real files differ from it in the
separator (`;` where `,` is the decimal mark), line ends, quoting,
encodings and header conventions. A reader has to pick a dialect. Ours is written
by hand, like the DOT writer (no CSV library, `00-architecture.md` §8),
and reads the one that spreadsheets, pandas (`to_csv`) and numpy (`savetxt` with
`delimiter=","`) write by default:

| Rule | Example |
|---|---|
| UTF-8 text; a byte-order mark at the start is skipped | Excel's "CSV UTF-8" |
| the first line is the header: one name per column | `x1,x2,label` |
| every further line is a row with one cell per column | `0.912,0.334,-1` |
| cells are separated by `,`; spaces and tabs around a cell don't count | `0.5 , 1.5` |
| a cell may be quoted; `""` is a quote inside quotes | `"x1"`, `"say ""hi"""` |
| a quoted cell may contain commas, but ends on its line | `"a, b"` |
| lines end with `\n` or `\r\n` | Windows files |
| blank lines may only end the file | an editor's trailing newline |

What it leaves out: other separators, line breaks inside quoted cells,
files without a header, comment lines, and encodings other than UTF-8.
None of them is needed for numeric data, each would make the reader and
its error messages harder to explain, and each is easy to convert away
with any tool. A file that uses another separator gets a hint (§3.6).

### 1.3 Reading a number exactly

A cell holds text such as `0.1`. The obvious conversion, Haskell's `read`,
is the wrong tool: it rejects forms that CSV writers print (`.5`, `1.`,
`+3`) and accepts Haskell-only ones. So `readNumber` has its own small
grammar:

```
number ::= sign? ( digits ( "." digits? )? | "." digits ) ( ("e" | "E") sign? digits )?
sign   ::= "+" | "-"
```

The text is read as an **exact** fraction first: `-1.25e-3` is
`-(125 · 10^(-2-3))`, an `Integer` and an exponent. Then it is rounded
**once** to the nearest `Double` with `fromRational`, which rounds
correctly. Rounding once matters: computing `125 / 100` in `Double` and
then dividing by `1000` rounds twice, and the result can be one unit in
the last place off. Reading exactly means that `0.1` in a file is the same
`Double` as the literal `0.1` in Haskell, and that `show` of any `Double`
reads back as the same `Double` (a property test checks it).

Two edges need care. A huge exponent such as `1e999999999` would make
`fromRational` build an enormous number first, so exponents far outside
the range of `Double` (about 4.9e-324 to 1.8e308) are decided before:
too small is 0, too large is an error. And `nan`, `inf` and overflowing
numbers are rejected: every value in a graph must be finite
(`05-core-eval.md` §3.4), so a non-finite value in a dataset would only
fail later, at the first training step, far from its cause.

### 1.4 Errors in a second file

Every diagnostic so far pointed into the `.hsg` file. A CSV problem is
in the CSV file: its primary span is a cell (`non-numeric.csv:3:5`).
Nothing new is needed for that. A `Span` already has a file name, and
the renderer of step 10 takes a `Sources` map from file names to texts; a
diagnostic whose labels are in two files is shown as two snippets, the
second introduced by `:::` (`11-diagnostics.md` §4.6). So the only job of
this step is to hand the renderer the CSV text under the name used in the
spans, which is why `loadDatasets` returns the texts it read.

A secondary label then ties the problem to the declaration: the column in
the mapping that reads the bad cell, or the path of the file. Problems
that are about the declaration (an unreadable file, a column the header
doesn't have) are the other way round: primary in the `.hsg` file, a
secondary label on the CSV header.

### 1.5 Validation in stages

A CSV source can go wrong in many ways, and they depend on each other: a
column can only be looked up if the file was read, and a cell can only be
read as a number if its column was found. So a source is checked in
**stages**. Each stage reports *all* of its problems, and a later stage
runs only if the earlier ones succeeded:

1. the mapping against the schema (E0404), which needs no file;
2. reading the file (E0401);
3. its structure (E0405 for an empty file, E0403 for bad lines);
4. the mapped columns in the header (E0402);
5. at least one row (E0405);
6. numbers in the mapped columns (E0403).

This is the checker's "all independent errors" (`13-checker.md` §1.5)
at the granularity of stages: the problems a user sees at once are the
ones that can be fixed at once.

---

## 2. Design

### 2.1 Two modules: structure, then meaning

`Data.Csv` knows CSV and numbers, nothing about schemas, mappings or
error codes. Its results are plain data: a `Table` of text cells with
spans, or a list of `CsvProblem`s; a list of rows of numbers, or a list of
`BadCell`s. `Data.Dataset` knows `data` declarations: it decides which
columns are needed, in which order, and turns every problem into a
`Diagnostic` with a code and words. The same split as `Lang.Shape` and
`Lang.Check` (`13-checker.md` §4.1): the problem types are the interface,
precise enough to choose a message, with no text in them.

### 2.2 The table: text cells with positions

```haskell
data Cell = Cell { cellText :: !Text, cellSpan :: !Span }
data Record = Record { recordLine :: !Int, recordSpan :: !Span, recordCells :: ![Cell] }
data Table = Table { tableHeader :: !Record, tableRecords :: ![Record] }
```

A **record** is a line of the file; the header is a record too. Cells are
kept as **text**, not numbers, for two reasons. A file may have columns
that are not numbers at all (an id, a comment), and only the dataset knows
which columns it reads (§4.1). And the header is text anyway. Reading
numbers is a second, separate function, `numericColumns`, that takes the
positions of the columns to read.

A cell's text has no surrounding whitespace and no quotes; its span
covers what the user wrote, quotes included, so a caret points at the
cell as it appears in the file.

### 2.3 The dataset: rows in column order

```haskell
data Dataset = Dataset
  { dataName :: !Text
  , dataNameSpan :: !Span
  , dataFields :: ![(Text, Shape Int)]
  , dataRows :: ![[Double]]       -- at least one row; finite values, column order
  }
```

The rows are exactly `Feed`'s rows (`10-train.md` §3.6). The name span is
where dataset-wide problems point, such as a `--batch` larger than the
dataset. A list of lists is plenty for datasets of hundreds of rows; the
graph is the expensive part of training (D5), not the rows.

### 2.4 A pure core, IO at the edge

```haskell
type Files = Map FilePath (Either ReadProblem Text)

buildDataset :: FilePath -> Files -> DatasetInfo -> Either [Diagnostic] Dataset
loadDatasets :: FilePath -> [DatasetInfo] -> IO (Sources, Either [Diagnostic] [Dataset])
```

`buildDataset` is pure: it receives the contents of the CSV files as a
map from path to text (or to the reason it could not be read). All the
logic, and every diagnostic, is in this function, and tests can call it
with a map they make up. `loadDatasets` is the thin IO part: it collects
the paths, reads each file once (`readFiles`), and calls the pure core.
It also returns the texts as `Sources` for the renderer (§1.4); the
driver merges them with the `.hsg` file's own text.

### 2.5 Paths

```haskell
csvPath :: FilePath -> DatasetInfo -> Maybe FilePath
csvPath hsg info = case datasetSource info of
  Csv _ path _ -> Just (normalise (takeDirectory hsg </> Text.unpack (stringValue path)))
  Rows _ _ -> Nothing
```

A path in a `csv` source is relative to the directory of the `.hsg` file,
not to the directory `hsgrad` runs in (§3), so a file and its data can be
moved together. `examples/moons.hsg` reading `"data/moons.csv"` reads
`examples/data/moons.csv`; `</>` keeps an absolute path as it is, and
`normalise` turns `./data/moons.csv` (for a `.hsg` file in the current
directory) into `data/moons.csv`. The resolved path is the CSV file's
name in spans, so diagnostics show the path that was actually read,
relative to where the user is, just like the `.hsg` file's own name.

---

## 3. Code walkthrough

### 3.1 Splitting a line into cells

```haskell
splitCells :: FilePath -> Int -> Text -> Either CsvProblem [Cell]
splitCells file l = cellsFrom 1
  where
    cellsFrom c t =
      let (ws, t1) = Text.span isBlank t
          c1 = c + Text.length ws
       in case Text.uncons t1 of
            Just ('"', t2) -> quoted c1 t2
            _ ->
              let (raw, t2) = Text.break (== ',') t1
                  content = Text.dropWhileEnd isBlank raw
               in (Cell content (sp c1 (c1 + Text.length content)) :) <$> afterCell (c1 + Text.length raw) t2

    afterCell c t = case Text.uncons t of
      Nothing -> Right []
      Just (_comma, rest) -> cellsFrom (c + 1) rest
```

The scan goes left to right and carries the **column** it is at, which
is all a span needs (the line is fixed). A cell starts after optional
blanks. A `"` there starts a quoted cell; anything else is an unquoted
cell running to the next comma, with its trailing blanks dropped. After a
cell comes the end of the line or a comma and another cell. So `1,2,` is
three cells, the last one empty, and `1,,3` has an empty cell (with an
empty span at column 3, which the renderer draws as one caret).

A quoted cell is scanned by `closingQuote`, which looks for a `"` that is
not doubled, turning each `""` into `"`; it also returns how many
characters it used, so the column after the closing quote is known. Then
only blanks may follow before the comma or the end of the line: `"a"b` is
`TextAfterQuote`, and a quote that is never closed is
`UnterminatedQuote`, spanning from the quote to the end of the line.

### 3.2 Lines

```haskell
parseCsv :: FilePath -> Text -> Either [CsvProblem] Table
parseCsv file text = case dropWhileEnd (isBlank . snd) numbered of
  [] -> Left [NoHeader]
  content -> case partitionEithers (map line content) of
    ([], header : records) ->
      let width = length (recordCells header)
       in case [WrongCellCount r header | r <- records, length (recordCells r) /= width] of
            [] -> Right (Table header records)
            ragged -> Left ragged
    …
    (problems, _) -> Left problems
  where
    numbered = zip [1 ..] (map dropCR (Text.splitOn "\n" (dropBom text)))
    line (l, t)
      | isBlank t = Left (BlankLine (lineSpan file l t))
      | otherwise = lineRecord file l t
```

The text is split at `\n`, and a `\r` at the end of a line is dropped:
the same rule the diagnostics renderer uses to show lines, so line
numbers agree. Trailing blank lines are dropped next (an editor's
final newline makes one, and so does `print` in a loop); nothing left
means no header. Every remaining line is a record, except a blank line,
which is a problem: in the middle of numeric data a blank line usually
means something was deleted (§4.2).

The cell count is checked last, and only when every line is fine by
itself: a line with an unterminated quote has no meaningful number of
cells. A byte-order mark becomes a space, so it is skipped like
whitespace before the first cell but still counts as column 1: the
renderer, which sees the original text, counts it too.

### 3.3 Numbers

```haskell
readNumber :: Text -> Either NumberProblem Double
readNumber t
  | Text.null t = Left EmptyCell
  | Text.map toLower (Text.dropWhile (`elem` ['+', '-']) t) `elem` ["nan", "inf", "infinity"] = Left NotFinite
  | otherwise = maybe (Left NotANumber) toDouble (scanDecimal t)
```

`scanDecimal` follows the grammar of §1.3 with `Text.span isDigit` for
each part (`isDigit` accepts ASCII digits only, so `١` is not a digit
here) and gives `Decimal negative digits exponent`. `toDouble` does the
rounding:

```haskell
toDouble (Decimal negative m e)
  | m == 0 = Right (signed 0)
  | magnitude > 310 = Left NotFinite
  | magnitude < -330 = Right (signed 0)
  | isInfinite value = Left NotFinite
  | otherwise = Right (signed value)
  where
    magnitude = e + fromIntegral (length (show m))      -- 10^(magnitude-1) ≤ |x| < 10^magnitude
    value = fromRational (fromIntegral m * 10 ^^ e) :: Double
```

The two cut-offs only decide the cases that are certain (beyond 1e310 is
infinite, below 1e-330 rounds to 0); the band in between is left to
`fromRational` and `isInfinite`, so `1.7976931348623157e308` is the
largest `Double` and `1.8e308` is too large.

`NumberProblem` has three cases, `EmptyCell`, `NotANumber` and
`NotFinite`, which become three different messages.

### 3.4 Columns of numbers

```haskell
numericColumns :: [Int] -> Table -> Either [BadCell] [[Double]]
```

Given the positions of the columns to read, in the order wanted, it reads
those cells of every record as numbers. A column may be listed twice (a
mapping may use one column for two fields). Every bad cell is reported,
record by record, with its column: grouping them is the dataset's job.
`columnIndices name table` finds the positions of a header name; it
returns a list, since a header may repeat a name.

### 3.5 Inline rows

```haskell
inlineRows :: DatasetInfo -> Span -> [Row Span] -> Either [Diagnostic] [[Double]]
```

The parser has already turned each row into literals. Each row needs one
literal per field (`row has 3 values, but dataset points has 2 fields`),
and each literal the field's shape: a number for a `Real`, a vector of
exactly `n` numbers for a `Vec n`. A mismatch names both shapes
(`field x is a Vec 2, but this value is a Vec 3`), at the literal, with a
secondary label showing the schema at the dataset's name. Every row is
checked, so all bad rows are reported at once.

One case gets a hint. A row with as many numbers as the schema has
columns has most likely lost the brackets of its vector fields; the hint
shows the row with them:

```
4 |   (0, 1, 1),
  |   ^^^^^^^^^ 3 values
  |
  = hint: vector fields are written in brackets: `([0, 1], 1)`
```

### 3.6 The CSV stages

```haskell
csvRows ctx sp maps contents = do
  mapped <- checkMapping (ctxInfo ctx) sp maps
  text <- first (\p -> [readError ctx p]) contents
  table <- first (structureDiagnostics ctx mapped) (parseCsv (ctxFile ctx) text)
  columns <- resolveColumns ctx table mapped
  stage [noRecords ctx table | null (tableRecords table)] ()
  first (cellDiagnostics table columns) (numericColumns [i | (i, _) <- columns] table)
```

The stages of §1.5 are a `do` block in `Either [Diagnostic]`: each line
either gives its result to the next or stops with its diagnostics.
`first` (from `Data.Bifunctor`) turns a stage's own problem type into
diagnostics. `CsvContext` carries what every message needs: the
declaration, the resolved path and the path literal.

- **Mapping** (`checkMapping`, E0404). It walks the mapping and the
  schema: a name that is not a field (with a "did you mean"), a field
  mapped twice (secondary at the first), a field never mapped (primary at
  the source, hint `add y = … to the mapping`), and the shape of the
  columns: a `Real` takes one column written without brackets, a `Vec n`
  a bracketed list of exactly `n`. The result is the mapped columns in
  **schema order**, each with the field it feeds: `[("x", x1), ("x", x2),
  ("y", label)]`. That order is the column order of the dataset, whatever
  order the mapping or the file use.
- **Reading** (`readError`, E0401) at the path literal: "file not found",
  "permission denied", "not UTF-8 text", or the system's words. When the
  file was not found and the resolved path differs from what was written,
  a note shows the resolved path: the most likely confusion is a path
  meant relative to the working directory.
- **Structure** (`structureDiagnostics`): an empty file is E0405 with a
  note showing a header made from the mapping (`x1,x2,label`); the other
  problems are E0403 in the CSV file, with a secondary label at the path
  literal (`dataset points reads this file`). A row with the wrong number
  of cells marks the extra cells, or the end of the row for missing ones,
  and has a secondary label on the header.
- **Columns** (`resolveColumns`, E0402) at the column's name in the
  mapping, with the header as a secondary label and a suggestion by edit
  distance among the header names (the checker's `suggestName`, step
  12), so `lable` gets `did you mean label?`. Names that can't be written
  in a mapping (§4.6) are not suggested. A name the header has twice is
  ambiguous, with both header cells marked. A header that is one single
  column containing `;` or a tab gets the hint that the file seems to use
  another separator.
- **No rows** (E0405) at the path literal, the header marked in the CSV.
- **Cells** (`cellDiagnostics`, E0403) at the cell, with a secondary label
  at the mapped column: `read for field x`.

### 3.7 One diagnostic per kind of problem

A file with a wrong separator in every line, or a whole column of text,
would otherwise produce hundreds of errors that all say the same thing.
So structural problems are grouped by kind, and bad cells by column: the
first of each group is a diagnostic, and a note lists the lines of the
others (`the same problem on lines 4 and 5`; after five numbers,
`… and 12 more`). Different kinds and different columns are still
separate diagnostics, since they are separate things to fix.

### 3.8 Reading files

```haskell
readCsvFile :: FilePath -> IO (Either ReadProblem Text)
readCsvFile path =
  (Right <$> (Utf8.readFile path >>= evaluate))
    `catches` [ Handler (pure . Left . ioProblem)
              , Handler (\(_ :: UnicodeException) -> pure (Left NotUtf8))
              ]
```

`Data.Text.IO.Utf8.readFile` reads UTF-8 whatever the locale (as the
golden harness does, `11-diagnostics.md` §5). Failures arrive as
exceptions of two types: `IOException` for missing files, permissions and
directories, and `UnicodeException` for bytes that are not UTF-8. Both are
caught here and become a `ReadProblem`, so nothing else in the module
deals with exceptions. `evaluate` makes sure a decoding error is raised
inside `catches`, not later when the text is first used.

### 3.9 Training: slots, feeds, batches

```haskell
slotNames :: Text -> [(Text, Shape Int)] -> Int -> [[Name]]
slotNames name fields b =
  [[Name (name <> "#" <> showT i <> "." <> c) | c <- fieldColumns fields] | i <- [0 .. b - 1]]

toFeed :: Dataset -> [[Name]] -> Feed
toFeed d = Feed (dataName d) (dataRows d)

checkBatch :: Int -> Dataset -> [Diagnostic]
```

`slotNames` spells out the slot convention of `01-language.md` §8
(`moons#3.x[1]`). It needs only the name and the schema, so the compiler
can name its leaves with it before any data is read (`derive` never
reads data). `toFeed` pairs the rows with the slots the compiler made;
the training loop of step 9 does the rest. `checkBatch` is E0405 for
`--batch` larger than the dataset, at the dataset's name:

```
error[E0405]: `--batch 10` is larger than dataset `xor`
 --> test/golden/data/batch.hsg:3:6
  |
3 | data xor : { x : Vec 2, y : Real } = rows [
  |      ^^^ `xor` has 4 rows
  |
  = hint: use `--batch 4` or less, or leave `--batch` out to use every row in every step
```

A flag has no span, so the primary span is the declaration the flag
doesn't fit (decision §4.5 of `11-diagnostics.md`: every diagnostic points
somewhere). A batch size below 1 is the command line's to reject.

### 3.10 `moons.csv`

`examples/data/moons.csv` is made by `scripts/MakeMoons.hs`:

```
cabal exec -v0 -- runghc --ghc-arg=-package=hsgrad scripts/MakeMoons.hs > examples/data/moons.csv
```

It is scikit-learn's `make_moons(n_samples=100, noise=0.1)`, written out
by hand: two interleaving half circles of 50 points, `(cos t, sin t)` with
label −1 and `(1 − cos t, 0.5 − sin t)` with label +1, for 50 values of
`t` evenly spaced in `[0, π]`, plus Gaussian noise with standard deviation
0.1 on each coordinate. micrograd's demo uses the same data and turns
scikit-learn's labels 0/1 into −1/+1 for the SVM loss. The noise comes
from step 9's `standardNormal` with `mkStdGen 1337`, so the file is
reproducible as long as the `random` package is the pinned one. Rows
alternate between the moons; values have six decimals.

---

## 4. Design decisions

### 4.1 Only mapped columns must be numbers

A file is checked for structure as a whole, but for numbers only in the
columns a dataset reads. Real data files often carry an id, a name or a
comment column; requiring the whole file to be numeric would force users
to strip those first. The cost is that a typo in an unused column goes
unnoticed, which is harmless since nothing reads it.

### 4.2 Blank lines only at the end

pandas skips blank lines anywhere; we report one that is followed by more
data. In a file of numbers, a blank line in the middle is almost always
an accident (a deleted row, two files pasted together), and the error is
cheap to fix. Blank lines at the end are allowed because nearly every
tool writes at least one.

### 4.3 Strict shapes in the mapping

`y = [label]` for a `Real` field, or `x = x1` for a `Vec 1`, could be
accepted: the number of columns is right. They are rejected (E0404, with
a hint that shows the right form) because the language keeps `Real` and
`Vec 1` apart everywhere else (§4 of the spec), and the mapping reads
best when it looks like the schema: brackets exactly where the schema has
a vector.

### 4.4 Stop at the first failing stage

Within a stage, all problems are reported; across stages, the first
failing stage ends the check (§1.5). The alternative, running every stage
as far as it can, would report, for example, missing columns of a file
that has the wrong separator, which is one problem described several
times. The stages that are truly independent (the mapping and reading
the file) are cheap to fix in two rounds.

### 4.5 One diagnostic per kind, with a note for the rest

§3.7. The note keeps the information (every line number, up to a count)
without the noise, and the first occurrence is shown in full, which is
usually enough to see the pattern.

### 4.6 Codes, and changes to the language spec

The catalogue had one code per "obvious" problem; the corners were fitted
to the nearest code, and the catalogue texts (and `codeSummary`) widened:

- **E0402** also covers a mapped column that the header has **twice**:
  which one to read would be a guess.
- **E0403** also covers a **malformed line**: a row with the wrong number
  of cells, quoting problems, a blank line before the end. All are "the
  file's content is broken at file:row:column".
- **E0404** also covers the **mapping** against the schema: like an inline
  row, it is the source not fitting the schema. The step-12 notes had
  left it without a code.

Other additions to `01-language.md`: §3 now spells out the mapping rule
("exactly `n` columns, every field mapped once"), the CSV format of §1.2
in a few lines, and an example of the relative path. §11 has a new open
point: a CSV column whose name is not an identifier (`x 1`, or a keyword
such as `data`) can't be mapped, since the mapping takes identifiers; a
quoted form `x = ["x 1"]` would fix that. `00-architecture.md` §8 now lists `filepath`, a
new library dependency (a boot package).

### 4.7 Where `moons.csv` lives

The roadmap called it `data/moons.csv`, and that is what `moons.hsg`
says; since paths are relative to the `.hsg` file, the file is
`examples/data/moons.csv`. Changing the spec's rule to "relative to the
working directory" would make `hsgrad train examples/moons.hsg` and
`cd examples; hsgrad train moons.hsg` read different files.

### 4.8 The XOR net moved to the shared test examples

Step 9's hand-built XOR net (`xorNet`, `xorRows`, `xorSlots`,
`xorSpecs`) moved from `LoopSpec` to `HsGrad.Test.Examples`, so the
dataset tests can check that the feed made from `xor.hsg` is exactly the
one step 9 trained with, and train on it. Step 14 will compare its
compiled `xor.hsg` with the same net.

---

## 5. Tests

`test/HsGrad/Data/CsvSpec.hs`, `test/HsGrad/Data/DatasetSpec.hs`, and the
golden files in `test/golden/data/`.

### 5.1 Unit tests

- **Csv, structure**: header and records, cell spans without the
  whitespace around cells, record spans, quoted cells (spans with the
  quotes, commas and `""` inside), `\r\n`, the byte-order mark (skipped,
  still column 1), trailing blank lines in five variants, a header alone,
  trailing commas, empty cells, `columnIndices` with a repeated name.
- **Csv, problems**: no header (four variants), blank lines in the middle
  and at the start, too few and too many cells, unterminated quotes, text
  after a quote, several problems in line order, cell counts only checked
  when every line is fine.
- **Csv, numbers**: every accepted form; exact rounding
  (`0.30000000000000004`, the largest `Double`, the smallest denormal, a
  30-digit integer); underflow to 0 and overflow to `NotFinite`, both
  fast for absurd exponents; sixteen rejected texts (`1.2.3`, `0x10`,
  `1,5`, `١`, `1_000`, …); `nan`/`inf` spellings.
- **Csv, columns**: columns in the given order and repeated, unmapped
  columns with text, every bad cell with its column and problem.
- **Dataset, examples**: `xor.hsg` gives step 9's rows; `moons.hsg` reads
  `examples/data/moons.csv` (100 rows of 3, 50 points per label, values
  near the moons, the first rows exactly), and returns its text as a
  source.
- **Dataset, building**: path resolution (relative, current directory,
  `..`, absolute, inline rows), columns in schema order whatever the
  header and mapping order, one column for several fields, unmapped
  columns, one-field datasets, every `ReadProblem`, a file missing from
  the map, the mapping checked before reading, mappings with too few and
  too many columns for a vector, problems of a stage together, stage
  order, every bad inline row (arity and shapes), several datasets.
- **Training**: slot names, the feed equals step 9's
  `Feed "xor" xorRows (xorSlots 4)`, the XOR net trained on the rows read
  from the declaration reaches accuracy 1 (seed 1337, 200 steps, as in
  step 9), batch sizes 1, 4 and 5 against 4 rows.

### 5.2 Golden files

`test/golden/data/` has one `.hsg` file per case, most with a CSV file
next to it. The harness of step 10 needed no change: the output function
(`rendered` in `DatasetSpec`) loads the datasets with `loadDatasets` and
renders with a `Sources` map that holds the `.hsg` text and the texts of
the CSV files read. Two differences from the other golden directories:

- spans use the path under the package directory
  (`test/golden/data/x.hsg`), not the bare name, so that the CSV file is
  found relative to the `.hsg` file exactly as the driver will find it;
  the `-->` lines show those paths;
- a line `-- golden: --batch N` in a program also checks that batch size;
  a file that loads without errors prints its datasets (`loads.err`).

Checker warnings are left out (most of these files have unused datasets).
A test checks that every E04xx code appears in some `.err` file.
`loads.csv` has `\r\n` line ends and a byte-order mark on purpose;
`.gitattributes` marks it `-text` so git doesn't convert it.

| File | What it pins down |
|---|---|
| `not-found` | E0401, with the note on the resolved path |
| `missing-column` | E0402 with suggestions (`x_2` → `x2`, `lable` → `label`), header as secondary |
| `duplicate-column` | E0402 for a column the header has twice |
| `separator` | E0402 with the separator hint |
| `non-numeric` | E0403 per column with notes; text in an unmapped column is fine |
| `ragged` | E0403 once for three ragged rows (missing and extra cells) |
| `malformed` | E0403 for text after a quote, a blank line, an unterminated quote |
| `rows` | E0404 for inline rows: arity (with the brackets hint), `Vec` length, a vector for a `Real`, a nested vector |
| `mapping` | E0404 in all its mapping forms, before any file is read |
| `empty`, `no-header` | E0405 for a header without rows and an empty file |
| `batch` | E0405 for `--batch` larger than the dataset |
| `loads` | no error: quotes, spaces, `\r\n`, byte-order mark, unmapped column, columns in another order |

### 5.3 Properties

- **`show` reads back**: `readNumber (show d) == Right d` for every finite
  `Double` QuickCheck generates.
- **Round trip**: random tables (1–5 columns with random identifier names,
  0–8 rows of random finite `Double`s including extremes such as the
  largest `Double` and the smallest denormal), written in a random valid
  style (quoted or bare header, random padding with spaces and tabs,
  `\n` or `\r\n`, random trailing blank lines), parse back to the same
  names and exactly the same numbers.

### 5.4 Testing the tests

Deliberate bugs, each tried on its own (number of failing tests):

| Bug | Failing tests |
|---|---|
| unquoted cells keep their trailing blanks | 3 |
| a doubled quote ends a quoted cell | 1 |
| trailing blank lines are not dropped | 39 |
| `\r` is kept at line ends | 4 |
| a number starting with `.` is rejected | 2 |
| fraction digits shift the exponent the wrong way | 7 |
| no overflow check between the two cut-offs | 1 |
| rows with the wrong number of cells are not detected | 2 |
| a `Vec n` mapping of the wrong length is accepted | 2 |
| CSV paths relative to the working directory | 12 |
| columns in header order instead of schema order | 2 |
| `--batch` equal to the dataset size rejected | 2 |
| slots numbered from 1 | 3 |
| the length of inline rows not checked | 2 |
| a header without rows accepted | 1 |
| every structural problem reported separately | 1 |

All are killed. The first run caught the inline-row and mapping-length
mutants only through golden files; each now also has a unit test. The
mutant that drops trailing blank lines fails 39 tests because almost
every file ends with a newline, which is the point of the rule.

The full suite has 621 tests and runs in about 3 seconds.

---

## 6. What later steps build on

- **Step 14 (`Lang.Compile`)**: name the slot leaves of a comprehension
  over a dataset with `slotNames (datasetName info) (datasetFields info) b`
  (or the same convention), one list per slot in column order: that is
  the other half of a `Feed`. `fieldColumns` gives the column names of a
  schema. The compiler needs no data: `B` comes from `--batch`, defaulting
  to the dataset's size, so the driver loads datasets before compiling
  when it trains.
- **Step 15 (`Driver`)**: after checking, `loadDatasets hsgPath
  (checkedDatasets c)` (only for commands that need data: `train`; maybe
  `eval`), merge the returned `Sources` with the `.hsg` text, render
  diagnostics, stop on errors. Then `checkBatch b` for each dataset a
  comprehension uses, and `toFeed dataset slots` for `Setup`'s feeds. The
  training loop's own `EmptyFeed`, `BatchTooLarge` and `RowWidthMismatch`
  can then only be internal errors.
- **Later**: other separators or quoted column names in mappings would
  extend `splitCells` and the grammar; a held-out test set would be a
  second dataset with its own slots.
