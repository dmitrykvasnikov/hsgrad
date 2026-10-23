{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Diagnostic".
--
-- * Codes: the type agrees with the catalogue in @docs/01-language.md@
--   §10, and severities follow the letter.
-- * Golden files: @test/golden/diagnostics/*.hsg@, each with diagnostics
--   built by hand below (there is no parser yet), rendered and compared
--   with the @.err@ file next to it.
-- * Layout details with the expected text inline: gutter width, gaps,
--   empty labels, ends of lines and files, spans ending at column 1,
--   multi-line notes, labels in other files and in files without source.
-- * Colour: coloured output is the plain output plus escape codes.
-- * Lists: sorting, the summary line, 'hasErrors'.
-- * Robustness: random (even nonsensical) spans always render, with the
--   header first and no trailing whitespace.
module HsGrad.Lang.DiagnosticSpec
  ( spec
  ) where

import Data.Function ((&))
import Data.List (sort)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Span
import HsGrad.Test.Golden (goldenFiles)
import System.Directory (listDirectory)
import System.FilePath (takeExtension)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Gen, choose, counterexample, elements, forAll, listOf, (.&&.))

spec :: Spec
spec = do
  describe "codes" codeSpec
  describe "golden" goldenSpec
  describe "layout" layoutSpec
  describe "colour" colourSpec
  describe "lists" listSpec
  describe "robustness" robustnessSpec

-- ---------------------------------------------------------------------------
-- Helpers: spans by searching the source

-- | A source file: the name diagnostics use, and its text.
data Src = Src FilePath Text

-- | The span of the @n@-th (1-based) occurrence of a piece of text.
nth :: Src -> Int -> Text -> Span
nth (Src file text) n needle = case drop (n - 1) (Text.breakOnAll needle text) of
  (before, _) : _ -> textSpan file (advanceText startPos before) needle
  [] -> error ("not in the source: " <> show needle)

-- | The first occurrence.
at :: Src -> Text -> Span
at src = nth src 1

-- | From the start of the first occurrence of @from@ to the end of the
-- first occurrence of @to@ after it.
fromTo :: Src -> Text -> Text -> Span
fromTo src@(Src file text) from to =
  let s = at src from
      (before, _) = Text.breakOn from text
      rest = Text.drop (Text.length before) text
      (middle, _) = Text.breakOn to rest
   in mkSpan file (spanStart s) (advanceText (advanceText (spanStart s) middle) to)

-- | The empty span right after the first occurrence.
after :: Src -> Text -> Span
after src needle = pointSpan (spanFile s) (spanEnd s)
  where
    s = at src needle

-- | The empty span at the very end of the text.
endOfFile :: Src -> Span
endOfFile (Src file text) = pointSpan file (advanceText startPos text)

-- | Render as the driver will: plain, sorted, with the summary.
renderAll :: FilePath -> Text -> [Diagnostic] -> Text
renderAll file text = renderDiagnostics defaultRenderOptions (Map.singleton file text)

-- | Render one diagnostic, plain.
render1 :: FilePath -> Text -> Diagnostic -> Text
render1 file text = renderDiagnostic defaultRenderOptions (Map.singleton file text)

-- ---------------------------------------------------------------------------
-- Codes

codeSpec :: Spec
codeSpec = do
  it "has exactly the codes of the catalogue in docs/01-language.md §10" $ do
    doc <- Utf8.readFile "docs/01-language.md"
    let catalogue =
          [ code
          | line <- Text.lines doc
          , Just rest <- [Text.stripPrefix "| " line]
          , let code = Text.takeWhile (/= ' ') rest
          , Text.length code == 5
          , Text.head code `elem` ['E', 'W']
          ]
    sort catalogue `shouldBe` sort (map codeText [minBound .. maxBound])

  it "prints codes as written" $ do
    codeText E0302 `shouldBe` "E0302"
    codeText W0201 `shouldBe` "W0201"

  it "makes E codes errors and W codes warnings" $
    [(Text.take 1 (codeText c), codeSeverity c) | c <- [minBound .. maxBound]]
      `shouldSatisfy` all (\(letter, sev) -> (letter == "W") == (sev == Warning))

  it "has a summary for every code" $
    [c | c <- [minBound .. maxBound], Text.null (codeSummary c)] `shouldBe` []

-- ---------------------------------------------------------------------------
-- Golden files

-- | The diagnostics of each golden file: the file name the diagnostics use
-- (shown in @-->@ lines) and how to make them from the source.
goldenCases :: Map FilePath (FilePath, Src -> [Diagnostic])
goldenCases =
  Map.fromList
    [ -- The example of 00-architecture.md §6, exactly.
      ( "matrix-product.hsg"
      , ( "examples/moons.hsg"
        , \src ->
            [ diagnostic E0302 "shape mismatch in matrix product" (at src "W2 @ h1") "Mat 16 16 @ Vec 8"
                & withNote "h1 : Vec 8 because W1 : Mat 8 2 (declared at 6:7)"
            ]
        )
      )
    , -- Several labels on one line, a skipped stretch, a note.
      ( "shape-variable.hsg"
      , ( "shape-variable.hsg"
        , \src ->
            [ diagnostic E0303 "shape variable `n` bound inconsistently" (nth src 4 "x") "`Vec 2`, so `n = 2` here"
                & withSecondary (nth src 4 "w") "`Vec 3`, so `n = 3` here"
                & withSecondary (nth src 1 "Vec n") ""
                & withSecondary (nth src 2 "Vec n") "both parameters have length `n`"
                & withNote "while checking the call at shape-variable.hsg:8:9"
            ]
        )
      )
    , -- A multi-line span that starts after indentation: '/' style.
      ( "body-mismatch.hsg"
      , ( "body-mismatch.hsg"
        , \src ->
            [ diagnostic E0308 "body doesn't match the declared result type" (fromTo src "let z" "tanh(z)") "this is `Vec 2`"
                & withSecondary (nth src 1 "Real") "expected `Real` because of this"
                & withHint "use `sum(…)`, `dot(…)` or an index `[i]` to get a `Real`"
            ]
        )
      )
    , -- A multi-line span that starts mid-line: connector style.
      ( "comprehension.hsg"
      , ( "comprehension.hsg"
        , \src ->
            [ diagnostic E0311 "a comprehension can only be the argument of `sum` or `mean`" (fromTo src "[ (dot" "xor ]") "this is a sequence of `Real`"
                & withSecondary (nth src 2 "errors") "used as a value here"
                & withHint "to average the errors, write `mean [ … ]`"
            ]
        )
      )
    , -- Two multi-line labels in two lanes, and a single-line primary.
      ( "two-lanes.hsg"
      , ( "two-lanes.hsg"
        , \src ->
            [ diagnostic E0301 "elementwise operands with incompatible shapes" (at src "+") "`Vec 3 + Vec 2`"
                & withSecondary (fromTo src "[1" "3]") "`Vec 3`"
                & withSecondary (fromTo src "[4" "5]") "`Vec 2`"
                & withNote "elementwise operations need equal shapes, or a `Real` on one side"
            ]
        )
      )
    , -- Nested multi-line labels: lanes cross, inner ones close first,
      -- a lane is reused once its label has ended.
      ( "nested-spans.hsg"
      , ( "nested-spans.hsg"
        , \src ->
            [ diagnostic E0308 "body doesn't match the declared result type" (fromTo src "let s" "2]") "this is `Vec 2`"
                & withSecondary (nth src 1 "Real") "expected `Real` because of this"
                & withSecondary (fromTo src "dot(" "x)") "`Real`"
                & withSecondary (fromTo src "[1" "2]") "`Vec 2`"
            ]
        )
      )
    , -- Tabs before and inside the line.
      ( "tabs.hsg"
      , ( "tabs.hsg"
        , \src ->
            [ diagnostic E0201 "undefined name `sigmod`" (at src "sigmod") "not defined"
                & withHint "did you mean `sigmoid`?"
            ]
        )
      )
    , -- A label at the end of the file (after the last newline).
      ( "unterminated-comment.hsg"
      , ( "unterminated-comment.hsg"
        , \src ->
            [ diagnostic E0102 "unterminated block comment" (at src "{-") "this comment is never closed"
                & withSecondary (endOfFile src) "the file ends here"
                & withHint "close it with `-}`"
            ]
        )
      )
    , -- An empty span at the end of a line, with another label on it.
      ( "missing-in.hsg"
      , ( "missing-in.hsg"
        , \src ->
            [ diagnostic E0104 "`let` expression without `in`" (after src "tanh(x)") "expected `in` here"
                & withSecondary (nth src 1 "let") "this `let` is inside an expression"
                & withHint "inside an expression write `let h = … in body`;\nonly a top-level `let` needs no `in`"
            ]
        )
      )
    , -- Non-ASCII characters: one column, one caret each.
      ( "unicode.hsg"
      , ( "examples/unicode.hsg"
        , \src ->
            [ diagnostic E0401 "cannot read CSV file `data/møøns.csv`" (at src "\"data/møøns.csv\"") "no such file"
                & withSecondary (at src "csv") ""
                & withNote "paths are relative to the .hsg file: looked for examples/data/møøns.csv"
            ]
        )
      )
    , -- Errors and warnings, given out of order: sorted, then summarised.
      ( "several.hsg"
      , ( "several.hsg"
        , \src ->
            [ diagnostic E0201 "undefined name `xx`" (at src "xx") "not defined"
                & withHint "did you mean `x`?"
            , diagnostic W0202 "`unused` is never used" (at src "unused") ""
            , diagnostic E0202 "duplicate declaration of `w`" (nth src 4 "w") "declared again here"
                & withSecondary (nth src 1 "w") "first declared here"
            , diagnostic W0201 "`x` shadows a parameter" (nth src 3 "x") "this `x`"
                & withSecondary (nth src 2 "x") "shadows this one"
            ]
        )
      )
    ]

goldenSpec :: Spec
goldenSpec = do
  goldenFiles "test/golden/diagnostics" ".hsg" $ \name text ->
    pure $ case Map.lookup name goldenCases of
      Nothing -> "no diagnostics for " <> Text.pack name <> " in DiagnosticSpec.goldenCases\n"
      Just (file, mk) -> renderAll file text (mk (Src file text))

  it "has a golden file for every case" $ do
    files <- filter ((== ".hsg") . takeExtension) <$> listDirectory "test/golden/diagnostics"
    Set.fromList files `shouldBe` Map.keysSet goldenCases

-- ---------------------------------------------------------------------------
-- Layout

layoutSpec :: Spec
layoutSpec = do
  let file = "t.hsg"
      src text = Src file text

  it "renders a warning with an empty label as marks only" $ do
    let text = "param unused : Real = 0\n"
    render1 file text (diagnostic W0202 "`unused` is never used" (at (src text) "unused") "")
      `shouldBe` Text.unlines
        [ "warning[W0202]: `unused` is never used"
        , " --> t.hsg:1:7"
        , "  |"
        , "1 | param unused : Real = 0"
        , "  |       ^^^^^^"
        ]

  it "widens the gutter for every line when a line number has two digits" $ do
    let text = Text.unlines [Text.pack ("let a" <> show i <> " = 1") | i <- [1 .. 10 :: Int]]
        s = src text
    render1 file text (diagnostic E0202 "dup" (at s "a10") "here" & withSecondary (at s "a9") "and here")
      `shouldBe` Text.unlines
        [ "error[E0202]: dup"
        , "  --> t.hsg:10:5"
        , "   |"
        , "9  | let a9 = 1"
        , "   |     -- and here"
        , "10 | let a10 = 1"
        , "   |     ^^^ here"
        ]

  it "skips long stretches with ..., but shows a single line in between" $ do
    let text = "a\nb\nc\nd\ne\nf\n"
        s = src text
    render1 file text (diagnostic E0201 "m" (at s "a") "" & withSecondary (at s "c") "" & withSecondary (at s "f") "")
      `shouldBe` Text.unlines
        [ "error[E0201]: m"
        , " --> t.hsg:1:1"
        , "  |"
        , "1 | a"
        , "  | ^"
        , "2 | b"
        , "3 | c"
        , "  | -"
        , "..."
        , "6 | f"
        , "  | -"
        ]

  it "marks an empty span at the end of a file without a final newline" $ do
    let text = "loss = (1 +"
    render1 file text (diagnostic E0101 "unexpected end of file" (endOfFile (src text)) "expected an expression")
      `shouldBe` Text.unlines
        [ "error[E0101]: unexpected end of file"
        , " --> t.hsg:1:12"
        , "  |"
        , "1 | loss = (1 +"
        , "  |            ^ expected an expression"
        ]

  it "draws a span that ends at column 1 of the next line on its own line" $ do
    let text = "let a = 1\nlet b = 2\n"
        sp = textSpan file startPos "let a = 1\n"
    render1 file text (diagnostic E0202 "m" sp "the whole line")
      `shouldBe` Text.unlines
        [ "error[E0202]: m"
        , " --> t.hsg:1:1"
        , "  |"
        , "1 | let a = 1"
        , "  | ^^^^^^^^^ the whole line"
        ]

  it "puts the primary mark over a secondary one where they overlap" $ do
    let text = "let y = f(a, b)\n"
        s = src text
    render1 file text (diagnostic E0205 "m" (at s "a") "" & withSecondary (fromTo s "f(" ")") "call")
      `shouldBe` Text.unlines
        [ "error[E0205]: m"
        , " --> t.hsg:1:11"
        , "  |"
        , "1 | let y = f(a, b)"
        , "  |         --^---- call"
        ]

  it "hangs several labels of one line right to left" $ do
    let text = "let y = f(a, b, c)\n"
        s = src text
    render1 file text (diagnostic E0205 "m" (at s "c") "third" & withSecondary (at s "a") "first" & withSecondary (at s "b") "second")
      `shouldBe` Text.unlines
        [ "error[E0205]: m"
        , " --> t.hsg:1:17"
        , "  |"
        , "1 | let y = f(a, b, c)"
        , "  |           -  -  ^ third"
        , "  |           |  |"
        , "  |           |  second"
        , "  |           first"
        ]

  it "indents the continuation lines of notes and hints" $ do
    let text = "x\n"
    render1 file text (diagnostic E0201 "m" (at (src text) "x") "" & withNote "one\ntwo" & withHint "three")
      `shouldBe` Text.unlines
        [ "error[E0201]: m"
        , " --> t.hsg:1:1"
        , "  |"
        , "1 | x"
        , "  | ^"
        , "  |"
        , "  = note: one"
        , "          two"
        , "  = hint: three"
        ]

  it "shows labels in another file in their own section" $ do
    let hsg = "data d : { x : Real } = csv \"d.csv\" { x = x1 }\n"
        csv = "a,b\n1,2\n"
        d =
          diagnostic E0402 "column `x1` is not in `d.csv`" (at (Src "m.hsg" hsg) "x1") "no such column"
            & withSecondary (textSpan "d.csv" startPos "a,b") "the header has `a`, `b`"
    renderDiagnostic defaultRenderOptions (Map.fromList [("m.hsg", hsg), ("d.csv", csv)]) d
      `shouldBe` Text.unlines
        [ "error[E0402]: column `x1` is not in `d.csv`"
        , " --> m.hsg:1:43"
        , "  |"
        , "1 | data d : { x : Real } = csv \"d.csv\" { x = x1 }"
        , "  |                                           ^^ no such column"
        , "  |"
        , " ::: d.csv:1:1"
        , "  |"
        , "1 | a,b"
        , "  | --- the header has `a`, `b`"
        ]

  it "reports positions only for a file whose source is not given" $ do
    let d =
          diagnostic E0403 "non-numeric cell" (textSpan "d.csv" (Pos 3 5) "abc") "not a number"
            & withSecondary (textSpan "d.csv" (Pos 1 1) "x") ""
    renderDiagnostic defaultRenderOptions Map.empty d
      `shouldBe` Text.unlines
        [ "error[E0403]: non-numeric cell"
        , " --> d.csv:3:5"
        , "  = 1:1"
        , "  = 3:5: not a number"
        ]

  it "uses the tab width of the options" $ do
    let text = "\tx\n"
        d = diagnostic E0201 "m" (at (src text) "x") "here"
    renderDiagnostic defaultRenderOptions {renderTabWidth = 2} (Map.singleton file text) d
      `shouldBe` Text.unlines
        [ "error[E0201]: m"
        , " --> t.hsg:1:2"
        , "  |"
        , "1 |   x"
        , "  |   ^ here"
        ]

-- ---------------------------------------------------------------------------
-- Colour

colourSpec :: Spec
colourSpec = do
  let text = "let y = f(a, b)\n"
      s = Src "t.hsg" text
      d =
        diagnostic E0205 "wrong number of arguments" (at s "b") "extra"
          & withSecondary (at s "f") "takes one"
          & withNote "f(x)"
      coloured = renderDiagnostic defaultRenderOptions {renderColour = True} (Map.singleton "t.hsg" text) d

  it "is the plain output plus escape codes" $
    stripAnsi coloured `shouldBe` render1 "t.hsg" text d

  it "paints errors red, secondary labels and the gutter blue" $ do
    coloured `shouldSatisfy` Text.isInfixOf "\ESC[1;31merror[E0205]\ESC[0m"
    coloured `shouldSatisfy` Text.isInfixOf "\ESC[1;34m-\ESC[0m"
    coloured `shouldSatisfy` Text.isInfixOf "\ESC[1;31m^\ESC[0m"

  it "paints warnings yellow" $
    renderDiagnostic defaultRenderOptions {renderColour = True} Map.empty (diagnostic W0202 "m" (at s "y") "")
      `shouldSatisfy` Text.isInfixOf "\ESC[1;33mwarning[W0202]\ESC[0m"

  it "is off by default" $
    render1 "t.hsg" text d `shouldSatisfy` (not . Text.isInfixOf "\ESC")

-- | Remove ANSI SGR sequences (@ESC [ … m@).
stripAnsi :: Text -> Text
stripAnsi t = case Text.breakOn "\ESC[" t of
  (before, rest)
    | Text.null rest -> before
    | otherwise -> before <> stripAnsi (Text.drop 1 (Text.dropWhile (/= 'm') rest))

-- ---------------------------------------------------------------------------
-- Lists

listSpec :: Spec
listSpec = do
  let sp l c = textSpan "t.hsg" (Pos l c) "x"
      e1 = diagnostic E0201 "first" (sp 1 1) ""
      e2 = diagnostic E0202 "second" (sp 2 1) ""
      w2 = diagnostic W0201 "second too" (sp 2 1) ""
      w3 = diagnostic W0202 "third" (sp 3 1) ""

  it "sorts by position, keeping the reported order at the same span" $
    map diagMessage (sortDiagnostics [w3, w2, e2, e1]) `shouldBe` ["first", "second too", "second", "third"]

  it "sorts files by name" $
    map diagMessage (sortDiagnostics [diagnostic E0201 "b" (textSpan "b" startPos "x") "", diagnostic E0201 "a" (textSpan "a" (Pos 9 1) "x") ""])
      `shouldBe` ["a", "b"]

  it "summarises" $ do
    summary [] `shouldBe` "no errors"
    summary [e1] `shouldBe` "1 error"
    summary [w2, w3] `shouldBe` "2 warnings"
    summary [e1, w2, e2] `shouldBe` "2 errors, 1 warning"

  it "knows whether there are errors" $ do
    hasErrors [w2, w3] `shouldBe` False
    hasErrors [w2, e1] `shouldBe` True

  it "renders nothing for no diagnostics" $
    renderDiagnostics defaultRenderOptions Map.empty [] `shouldBe` ""

  it "separates diagnostics by blank lines and ends with the summary" $
    renderDiagnostics defaultRenderOptions (Map.singleton "t.hsg" "x\nx\n") [e2, w2]
      `shouldBe` Text.unlines
        [ "error[E0202]: second"
        , " --> t.hsg:2:1"
        , "  |"
        , "2 | x"
        , "  | ^"
        , ""
        , "warning[W0201]: second too"
        , " --> t.hsg:2:1"
        , "  |"
        , "2 | x"
        , "  | ^"
        , ""
        , "1 error, 1 warning"
        ]

-- ---------------------------------------------------------------------------
-- Robustness

robustnessSpec :: Spec
robustnessSpec =
  prop "renders any labels, even nonsensical ones, without trailing whitespace" $
    forAll genSource $ \text ->
      forAll (genDiagnostic text) $ \d ->
        let out = renderDiagnostic defaultRenderOptions (Map.singleton "r.hsg" text) d
            ls = Text.lines out
         in counterexample (Text.unpack out) $
              (take 1 ls == [severityWord d <> "[" <> codeText (diagCode d) <> "]: " <> diagMessage d])
                .&&. all (\l -> Text.stripEnd l == l) ls
                .&&. (Text.length out > 0)
  where
    severityWord d = if severity d == Error then "error" else "warning"

-- | Short sources with tabs, Unicode, blank and CRLF lines.
genSource :: Gen Text
genSource = Text.concat <$> listOf (elements ["let", " ", "\t", "x", "∂", "\n", "\r\n", "(", "\n\n"])

-- | Labels anywhere: inside the text, past the ends of lines and of the
-- file, empty, reversed, with line or column 0, in another file.
genDiagnostic :: Text -> Gen Diagnostic
genDiagnostic text = do
  let lineCount = length (Text.lines text) + 1
      genPos = Pos <$> choose (0, lineCount + 2) <*> choose (0, 12)
      genSpan = do
        file <- elements ["r.hsg", "r.hsg", "r.hsg", "other.csv"]
        SrcSpan file <$> genPos <*> genPos
      genLabel = elements ["", "label", "a much longer label text"]
  code <- elements [minBound .. maxBound]
  primary <- (,) <$> genSpan <*> genLabel
  secondaries <- listOf ((,) <$> genSpan <*> genLabel)
  notes <- listOf (elements ["note", "two\nlines"])
  let withLabels = foldl (\d (s, l) -> withSecondary s l d) (uncurry (diagnostic code "message") primary) secondaries
  pure (foldl (flip withNote) withLabels notes)
