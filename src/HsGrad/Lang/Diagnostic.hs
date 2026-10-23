{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Errors and warnings about @.hsg@ files, and the renderer that prints
-- them with source snippets, Rust style (@00-architecture.md@ §6):
--
-- @
-- error[E0302]: shape mismatch in matrix product
--   --> examples/moons.hsg:14:17
--    |
-- 14 |   let h2 = relu(W2 \@ h1 + b2) in
--    |                 ^^^^^^^ Mat 16 16 \@ Vec 8
--    |
--    = note: h1 : Vec 8 because W1 : Mat 8 2 (declared at 6:7)
-- @
--
-- Every user-facing problem is one 'Diagnostic': a 'Code' from the
-- catalogue in @01-language.md@ §10, a message, a primary span with a
-- label, secondary labelled spans, notes and hints. Later steps create
-- diagnostics; this module only knows how to print them. Rendering is a
-- pure function of the source text, so its output can be compared with
-- golden files.
-- See @docs/11-diagnostics.md@.
module HsGrad.Lang.Diagnostic
  ( -- * Codes
    Code (..)
  , Severity (..)
  , codeText
  , codeSeverity
  , codeSummary

    -- * Diagnostics
  , Label (..)
  , Diagnostic (..)
  , diagnostic
  , withSecondary
  , withNote
  , withHint
  , severity
  , diagSpan

    -- * Collections of diagnostics
  , sortDiagnostics
  , hasErrors
  , summary

    -- * Rendering
  , Sources
  , RenderOptions (..)
  , defaultRenderOptions
  , renderDiagnostic
  , renderDiagnostics
  ) where

import Data.Char (isSpace)
import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.Function (on)
import Data.List (inits, maximumBy, nub, sortOn)
import Data.List.NonEmpty qualified as NonEmpty
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Ord (Down (..), comparing)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..), showPos)

-- ---------------------------------------------------------------------------
-- Codes

-- | Every diagnostic code of the catalogue (@01-language.md@ §10).
--
-- A closed type rather than a string, so later steps can only use codes
-- that are documented: adding a code means adding a constructor here, a
-- line to 'codeSummary' and a row to the catalogue (a test checks that the
-- catalogue and this type agree).
data Code
  = -- E01xx: syntax
    E0101
  | E0102
  | E0103
  | E0104
  | E0105
  | -- E02xx: names and declarations
    E0201
  | E0202
  | E0203
  | E0204
  | E0205
  | E0206
  | E0207
  | -- E03xx: shapes
    E0301
  | E0302
  | E0303
  | E0304
  | E0305
  | E0306
  | E0307
  | E0308
  | E0309
  | E0310
  | E0311
  | E0312
  | E0313
  | -- E04xx: data
    E0401
  | E0402
  | E0403
  | E0404
  | E0405
  | -- E05xx: runtime
    E0501
  | E0502
  | E0503
  | E0504
  | -- W02xx: warnings
    W0201
  | W0202
  | W0203
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | How bad a diagnostic is. Errors stop the pipeline (after all of them
-- have been collected); warnings are printed and the run goes on.
data Severity = Error | Warning
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | The code as printed: @E0302@.
codeText :: Code -> Text
codeText = Text.pack . show

-- | The severity is part of the code: @E…@ codes are errors, @W…@ codes
-- warnings. So a diagnostic cannot be a "warning E0302".
codeSeverity :: Code -> Severity
codeSeverity c = case show c of
  'W' : _ -> Warning
  _ -> Error

-- | What the code means, as one line of the catalogue. Diagnostics carry
-- their own, more specific message; this is the general description (for
-- documentation and a future @hsgrad explain@).
codeSummary :: Code -> Text
codeSummary = \case
  E0101 -> "unexpected token"
  E0102 -> "unterminated block comment or string"
  E0103 -> "malformed number literal"
  E0104 -> "`let` expression without `in`"
  E0105 -> "exponent must be a numeric constant"
  E0201 -> "undefined name"
  E0202 -> "duplicate declaration"
  E0203 -> "used before declaration"
  E0204 -> "kind mismatch: calling a value, or using a function as a value"
  E0205 -> "wrong number of arguments"
  E0206 -> "more than one `model` or `loss`"
  E0207 -> "comprehension pattern doesn't match the dataset schema"
  E0301 -> "elementwise operands with incompatible shapes"
  E0302 -> "matrix product with mismatched dimensions"
  E0303 -> "shape variable bound inconsistently"
  E0304 -> "argument doesn't match the parameter annotation"
  E0305 -> "index out of bounds"
  E0306 -> "indexing a `Real`"
  E0307 -> "`loss` isn't `Real`"
  E0308 -> "body doesn't match the declared type"
  E0309 -> "`model` signature isn't fully concrete"
  E0310 -> "comprehension body has different shapes across iterations"
  E0311 -> "comprehension used outside `sum` / `mean`"
  E0312 -> "matrix literal rows of different lengths"
  E0313 -> "dataset used outside a comprehension source"
  E0401 -> "CSV file not found or unreadable"
  E0402 -> "column missing from the CSV header"
  E0403 -> "non-numeric cell"
  E0404 -> "inline row doesn't match the schema"
  E0405 -> "empty dataset, or batch larger than the dataset"
  E0501 -> "non-finite value"
  E0502 -> "input has no value"
  E0503 -> "`--set` value has the wrong shape"
  E0504 -> "non-finite gradient"
  W0201 -> "local name shadows an outer one"
  W0202 -> "unused declaration"
  W0203 -> "parameter can't affect `loss`"

-- ---------------------------------------------------------------------------
-- Diagnostics

-- | A span of source text with a short message about it. The message may
-- be empty: then only the span is marked.
data Label = Label
  { labelSpan :: !Span
  , labelText :: !Text
  }
  deriving (Eq, Show)

-- | One error or warning.
--
-- Its severity comes from its code ('severity'). It always points
-- somewhere: the primary label is the place the message is about, and is
-- marked with @^^^@. Secondary labels (marked @---@) are related places,
-- anywhere in the same file or in other files. Notes explain, hints
-- suggest a fix.
data Diagnostic = Diagnostic
  { diagCode :: !Code
  , diagMessage :: !Text
  , diagPrimary :: !Label
  , diagSecondary :: ![Label]
  , diagNotes :: ![Text]
  , diagHints :: ![Text]
  }
  deriving (Eq, Show)

-- | A diagnostic with a code, a message, and the primary span and its
-- label; no secondary labels, notes or hints yet. Add those with
-- 'withSecondary', 'withNote' and 'withHint':
--
-- @
-- diagnostic E0202 "duplicate declaration of `w`" sp2 "declared again here"
--   & withSecondary sp1 "first declared here"
--   & withHint "rename one of them"
-- @
diagnostic :: Code -> Text -> Span -> Text -> Diagnostic
diagnostic code msg sp lbl = Diagnostic code msg (Label sp lbl) [] [] []

-- | Add a secondary label, after the ones already there.
withSecondary :: Span -> Text -> Diagnostic -> Diagnostic
withSecondary sp lbl d = d {diagSecondary = diagSecondary d <> [Label sp lbl]}

-- | Add a note, after the ones already there. A note may have several
-- lines.
withNote :: Text -> Diagnostic -> Diagnostic
withNote n d = d {diagNotes = diagNotes d <> [n]}

-- | Add a hint, after the ones already there.
withHint :: Text -> Diagnostic -> Diagnostic
withHint h d = d {diagHints = diagHints d <> [h]}

-- | Error or warning, from the code.
severity :: Diagnostic -> Severity
severity = codeSeverity . diagCode

-- | The primary span: where the diagnostic is.
diagSpan :: Diagnostic -> Span
diagSpan = labelSpan . diagPrimary

-- ---------------------------------------------------------------------------
-- Collections of diagnostics

-- | Sort by the position of the primary span (file, then start, then end),
-- so the output reads top to bottom. The sort is stable: diagnostics at the
-- same span keep the order in which they were reported.
sortDiagnostics :: [Diagnostic] -> [Diagnostic]
sortDiagnostics = sortOn diagSpan

-- | Is any of them an error?
hasErrors :: [Diagnostic] -> Bool
hasErrors = any ((== Error) . severity)

-- | The counts, as printed at the end: @2 errors, 1 warning@, @1 error@,
-- @3 warnings@, or @no errors@.
summary :: [Diagnostic] -> Text
summary ds = case (count Error, count Warning) of
  (0, 0) -> "no errors"
  (e, 0) -> plural e "error"
  (0, w) -> plural w "warning"
  (e, w) -> plural e "error" <> ", " <> plural w "warning"
  where
    count s = length (filter ((== s) . severity) ds)
    plural :: Int -> Text -> Text
    plural 1 noun = "1 " <> noun
    plural n noun = Text.pack (show n) <> " " <> noun <> "s"

-- ---------------------------------------------------------------------------
-- Rendering: options and entry points

-- | The text of every source file a diagnostic may point into, by the file
-- name used in its spans. Usually one @.hsg@ file; a data error may also
-- point into a CSV file.
type Sources = Map FilePath Text

-- | How to render.
data RenderOptions = RenderOptions
  { renderColour :: !Bool
  -- ^ Colour the output with ANSI escape codes (for terminals). Off in
  -- tests and when the output is not a terminal.
  , renderTabWidth :: !Int
  -- ^ Distance between tab stops when a tab in the source is shown.
  }
  deriving (Eq, Show)

-- | No colour, tab stops every 4 columns.
defaultRenderOptions :: RenderOptions
defaultRenderOptions = RenderOptions {renderColour = False, renderTabWidth = 4}

-- | Render one diagnostic. The result ends with a newline.
--
-- Spans in files that are not in the 'Sources' are still reported, by
-- position, but without a snippet. Rendering never fails: spans beyond the
-- end of the text show empty lines.
renderDiagnostic :: RenderOptions -> Sources -> Diagnostic -> Text
renderDiagnostic opts sources d = renderLines opts (diagnosticLines opts sources d)

-- | Render diagnostics sorted by position ('sortDiagnostics'), each followed
-- by a blank line, then the 'summary' line. No diagnostics, no output.
renderDiagnostics :: RenderOptions -> Sources -> [Diagnostic] -> Text
renderDiagnostics _ _ [] = ""
renderDiagnostics opts sources ds =
  Text.concat [renderDiagnostic opts sources d <> "\n" | d <- sortDiagnostics ds]
    <> renderLines opts [[(Title, summary ds)]]

-- ---------------------------------------------------------------------------
-- Output lines and colour
--
-- The renderer first builds a list of lines, each a list of segments that
-- say what they are ('Paint'). Only the last step decides whether that
-- becomes colour or nothing, so the layout code never sees escape codes,
-- and coloured output is plain output with codes around some segments.

-- | What a piece of output is, which decides its colour.
data Paint
  = Plain
  | -- | The message, and the summary line.
    Title
  | -- | @error[E0302]@ / @warning[W0201]@, and the primary label's marks
    -- and text.
    Accent !Severity
  | -- | Secondary labels' marks and text.
    Secondary
  | -- | Line numbers, @|@, @-->@, @=@, @...@.
    Gutter
  | -- | The words @note@ and @hint@.
    NoteKind
  deriving (Eq)

type Segment = (Paint, Text)

type Line = [Segment]

-- | The ANSI SGR parameters of a paint: bold, plus a colour.
ansiCode :: Paint -> Maybe Text
ansiCode = \case
  Plain -> Nothing
  Title -> Just "1"
  Accent Error -> Just "1;31"
  Accent Warning -> Just "1;33"
  Secondary -> Just "1;34"
  Gutter -> Just "1;34"
  NoteKind -> Just "1"

-- | Turn lines into text: drop trailing whitespace (golden files should
-- not depend on it), add colour if asked, end every line with a newline.
renderLines :: RenderOptions -> [Line] -> Text
renderLines opts ls = Text.concat [Text.concat (map paint (trimEnd l)) <> "\n" | l <- ls]
  where
    paint (p, t) = case ansiCode p of
      Just code | renderColour opts -> "\ESC[" <> code <> "m" <> t <> "\ESC[0m"
      _ -> t
    trimEnd = reverse . dropBlank . reverse
    dropBlank [] = []
    dropBlank ((p, t) : rest)
      | Text.null t' = dropBlank rest
      | otherwise = (p, t') : rest
      where
        t' = Text.stripEnd t

-- ---------------------------------------------------------------------------
-- Rendering a diagnostic

-- | A label on its way to the screen: in which role, and its span with
-- the positions put in order and clamped to line and column 1 or more.
data Anno = Anno
  { annoPrimary :: !Bool
  , annoText :: !Text
  , annoStart :: !Pos
  , annoEnd :: !Pos
  }

-- | The labels of one file, and how to show them.
data Group = Group
  { groupFile :: !FilePath
  , groupLocation :: !Pos
  -- ^ Shown in the @-->@ line: the primary span's start in the primary
  -- file, the first label's start in other files.
  , groupBody :: !Body
  }

data Body
  = -- | The source is known: a snippet.
    Snippet !(Int -> Text) ![Anno] ![Int]
  -- ^ Source line by number, the labels (normalised), the line numbers
  -- to show.
  | -- | The source is not known: positions only.
    NoSource ![Anno]

diagnosticLines :: RenderOptions -> Sources -> Diagnostic -> [Line]
diagnosticLines opts sources d =
  header : concat (zipWith groupLines [0 :: Int ..] groups) <> noteLines
  where
    sev = severity d
    header =
      [ (Accent sev, severityWord sev <> "[" <> codeText (diagCode d) <> "]")
      , (Title, ": " <> diagMessage d)
      ]

    labels = (True, diagPrimary d) : [(False, l) | l <- diagSecondary d]
    files = nub [spanFile (labelSpan l) | (_, l) <- labels]
    groups =
      [ prepareGroup (Map.lookup file sources) file [a | (isP, l) <- labels, spanFile (labelSpan l) == file, let a = toAnno isP l]
      | file <- files
      ]

    -- The gutter fits the largest line number of any snippet.
    width = length (show (maximum (1 : concatMap shownLines groups)))
    shownLines g = case groupBody g of
      Snippet _ _ shown -> shown
      NoSource _ -> []

    pad = Text.replicate width " "
    blankGutter = [(Gutter, pad <> " |")]

    groupLines i g =
      [blankGutter | i > 0]
        <> [ [ (Plain, pad)
             , (Gutter, (if i == 0 then "-->" else ":::") <> " ")
             , (Plain, Text.pack (groupFile g) <> ":" <> showPos (groupLocation g))
             ]
           ]
        <> case groupBody g of
          Snippet line annos shown ->
            blankGutter : map (rowLine width) (snippetRows opts sev line annos shown)
          NoSource annos ->
            [ [(Plain, pad <> " "), (Gutter, "= "), (annoPaint sev a, showPos (annoStart a))]
              <> [(Plain, ": " <> annoText a) | not (Text.null (annoText a))]
            | a <- sortOn annoStart annos
            ]

    noteItems = [("note", n) | n <- diagNotes d] <> [("hint", h) | h <- diagHints d]
    noteLines
      | null noteItems = []
      | otherwise = blankGutter : concatMap noteItem noteItems
    noteItem (kind, text) = case Text.splitOn "\n" text of
      [] -> []
      firstLine : rest ->
        [(Plain, pad <> " "), (Gutter, "= "), (NoteKind, kind <> ":"), (Plain, " " <> firstLine)]
          : [[(Plain, Text.replicate (width + 3 + Text.length kind + 2) " " <> r)] | r <- rest]

severityWord :: Severity -> Text
severityWord Error = "error"
severityWord Warning = "warning"

-- | A label with its positions in order and at least line 1, column 1, so
-- that nonsense spans still render.
toAnno :: Bool -> Label -> Anno
toAnno isPrimary (Label sp txt) = Anno isPrimary (oneLine txt) (min s e) (max s e)
  where
    s = clamp (spanStart sp)
    e = clamp (spanEnd sp)
    clamp (Pos l c) = Pos (max 1 l) (max 1 c)
    oneLine = Text.map (\ch -> if ch == '\n' then ' ' else ch)

-- | Gather what a group needs: split the source into lines, normalise the
-- labels' ends and pick the lines to show.
prepareGroup :: Maybe Text -> FilePath -> [Anno] -> Group
prepareGroup source file annos = Group file location body
  where
    -- The primary label comes first in its file's group; other groups
    -- start at their first label.
    location = case [annoStart a | a <- annos, annoPrimary a] of
      p : _ -> p
      [] -> minimum (map annoStart annos)
    body = case source of
      Nothing -> NoSource annos
      Just text ->
        let ls = sourceLines text
            line l = IntMap.findWithDefault "" l ls
            annos' = map (normaliseEnd line) annos
         in Snippet line annos' (linesToShow annos')

-- | Source lines by number. Lines are separated by @\\n@; a @\\r@ before it
-- (Windows line ends) is not part of the line. A file that ends with a
-- newline has an empty last line, where an "end of file" span points.
sourceLines :: Text -> IntMap Text
sourceLines = IntMap.fromList . zip [1 ..] . map dropCR . Text.splitOn "\n"
  where
    dropCR l = fromMaybe l (Text.stripSuffix "\r" l)

-- | A span that ends at column 1 of a line really ends with the newline of
-- the line before (the end is exclusive). Draw it there, rather than as a
-- span reaching into a line it does not touch.
normaliseEnd :: (Int -> Text) -> Anno -> Anno
normaliseEnd line a
  | el > posLine (annoStart a) && ec == 1 =
      a {annoEnd = Pos (el - 1) (Text.length (line (el - 1)) + 1)}
  | otherwise = a
  where
    Pos el ec = annoEnd a

isMultiLine :: Anno -> Bool
isMultiLine a = posLine (annoEnd a) > posLine (annoStart a)

-- | The lines a snippet shows: every line where a label starts or ends.
-- A single line between two shown lines is shown too, since the @...@
-- that would replace it takes as much room.
linesToShow :: [Anno] -> [Int]
linesToShow annos = fill (Set.toAscList (Set.fromList ends))
  where
    ends = concat [[posLine (annoStart a), posLine (annoEnd a)] | a <- annos]
    fill (a : b : rest)
      | b - a == 2 = a : (a + 1) : fill (b : rest)
      | otherwise = a : fill (b : rest)
    fill xs = xs

annoPaint :: Severity -> Anno -> Paint
annoPaint sev a
  | annoPrimary a = Accent sev
  | otherwise = Secondary

markChar :: Anno -> Char
markChar a
  | annoPrimary a = '^'
  | otherwise = '-'

-- ---------------------------------------------------------------------------
-- Snippets

-- | One row of a snippet, without its gutter.
data Row
  = -- | A source line, with its number.
    SourceRow !Int !Line
  | -- | Marks and labels under a source line.
    MarkRow !Line
  | -- | Skipped lines.
    GapRow !Line

-- | Add the gutter: @14 | @, @   | @ or @... @, all @width + 3@ wide.
rowLine :: Int -> Row -> Line
rowLine width = \case
  SourceRow n content -> (Gutter, padRight (Text.pack (show n)) <> " |") : (Plain, " ") : content
  MarkRow content -> (Gutter, Text.replicate width " " <> " |") : (Plain, " ") : content
  GapRow content -> (Gutter, "...") : (Plain, Text.replicate width " ") : content
  where
    padRight t = t <> Text.replicate (width - Text.length t) " "

-- | A row under construction: display column to character. Columns that
-- were never written are spaces.
type Canvas = IntMap (Char, Paint)

-- | The canvas as segments, neighbouring characters of the same paint
-- in one segment.
canvasLine :: Canvas -> Line
canvasLine canvas = case IntMap.lookupMax canvas of
  Nothing -> []
  Just (end, _) ->
    [ (fst (NonEmpty.head run), Text.pack (map snd (NonEmpty.toList run)))
    | run <- NonEmpty.groupBy ((==) `on` fst) cells
    ]
    where
      cells =
        [ (p, ch)
        | col <- [0 .. end]
        , let (ch, p) = IntMap.findWithDefault (' ', Plain) col canvas
        ]

-- | Write text at a column, one character per column.
putText :: Int -> Paint -> Text -> Canvas -> Canvas
putText col p t canvas =
  foldl (\cv (i, ch) -> IntMap.insert i (ch, p) cv) canvas (zip [col ..] (Text.unpack t))

-- | Where a row is, relative to the other rows of the same source line.
-- Rows are emitted in this order: the marks of single-line labels, then
-- the start connectors of multi-line labels in lane order (outer labels
-- first), then their ends in reverse lane order (inner labels close
-- first). The derived 'Ord' follows the constructor order.
data Phase = PSingle | PConnect !Int | PEnd !(Down Int)
  deriving (Eq, Ord)

-- | The rows of one group's snippet.
--
-- Single-line labels are marked under their line: @^^^@ for the primary,
-- @---@ for secondary ones. The label whose marks end furthest right gets
-- its text on the mark row; the others hang below on @|@ lines, right to
-- left, as in rustc.
--
-- Each multi-line label gets a lane in a margin left of the source. Its
-- lane shows @/@ at its first line if only whitespace precedes the start
-- there, or else a @__^@ connector under that line; then @|@ down to its
-- last line, closed by @|__^ text@ under that line. Labels whose lines do
-- not overlap share a lane, so the margin is as narrow as it can be.
snippetRows :: RenderOptions -> Severity -> (Int -> Text) -> [Anno] -> [Int] -> [Row]
snippetRows opts sev line annos shown = go shown
  where
    tabWidth = max 1 (renderTabWidth opts)
    singles = filter (not . isMultiLine) annos
    multis = assignLanes (sortOn (\a -> (annoStart a, annoEnd a)) (filter isMultiLine annos))
    lanes = case multis of
      [] -> 0
      _ -> 1 + maximum (map fst multis)

    -- Greedy, in order of start: the lowest lane whose last label ended on
    -- an earlier line, or a new lane. @ends@ holds each lane's last line.
    assignLanes = assign []
      where
        assign _ [] = []
        assign ends (a : as) = case [k | (k, e) <- zip [0 ..] ends, e < sLine a] of
          k : _ -> (k, a) : assign (take k ends <> [eLine a] <> drop (k + 1) ends) as
          [] -> (length ends, a) : assign (ends <> [eLine a]) as

    -- The label in lane j that covers line l, if any. Labels of one lane
    -- never share a line.
    occupant l j = case [x | x@(k, a) <- multis, k == j, sLine a <= l, l <= eLine a] of
      x : _ -> Just x
      [] -> Nothing
    laneList = [0 .. lanes - 1]

    go (l : rest@(l' : _)) = lineRows l <> [gapRow l l' | l' - l > 1] <> go rest
    go [l] = lineRows l
    go [] = []

    -- Display column (0-based) of a character column of a line. Tabs jump
    -- to the next tab stop; columns past the end of the line continue
    -- after it.
    dispCol l col = width (Text.take (col - 1) txt) + max 0 (col - 1 - Text.length txt)
      where
        txt = line l
        width = Text.foldl' (\acc ch -> if ch == '\t' then tabStop acc else acc + 1) 0
    tabStop acc = (acc `div` tabWidth + 1) * tabWidth
    expandTabs = Text.pack . expand 0 . Text.unpack
      where
        expand _ [] = []
        expand acc ('\t' : cs) = let n = tabStop acc - acc in replicate n ' ' <> expand (acc + n) cs
        expand acc (c : cs) = c : expand (acc + 1) cs

    paintOf = annoPaint sev
    sLine = posLine . annoStart
    eLine = posLine . annoEnd

    -- Only whitespace before the start: the lane starts with '/' on the
    -- source line itself, and no connector row is needed.
    slashed a = Text.all isSpace (Text.take (posCol (annoStart a) - 1) (line (sLine a)))

    -- The margin, then one blank column, then the source area. Without
    -- multi-line labels there is no margin and no blank column.
    margin cells sepCell
      | lanes == 0 = []
      | otherwise = [(p, Text.singleton ch) | (ch, p) <- cells <> [sepCell]]
    blank = (' ', Plain)
    bar a = ('|', paintOf a)

    -- A lane on a source line.
    sourceLane l a
      | sLine a < l && l <= eLine a = bar a
      | l == sLine a && slashed a = ('/', paintOf a)
      | otherwise = blank

    -- A lane on a mark row of line l in the given phase.
    markLane l ph (k, a)
      | sLine a < l && l < eLine a = bar a
      | l == sLine a = if slashed a || ph > PConnect k then bar a else blank
      | l == eLine a = if ph <= PEnd (Down k) then bar a else blank
      | otherwise = blank

    -- Lanes of rows that are not a label's own connector or end row.
    sourceCells l = [maybe blank (sourceLane l . snd) (occupant l j) | j <- laneList]
    markCells l ph = [maybe blank (markLane l ph) (occupant l j) | j <- laneList]

    gapRow l l' =
      GapRow (margin [maybe blank (\(_, a) -> if eLine a >= l' then bar a else blank) (occupant l j) | j <- laneList] blank)

    lineRows l =
      SourceRow l (margin (sourceCells l) blank <> [(Plain, expandTabs (line l))])
        : singleRows l
          <> [connectRow l x | x@(_, a) <- sortOn fst multis, sLine a == l, not (slashed a)]
          <> [endRow l x | x@(_, a) <- sortOn (Down . fst) multis, eLine a == l]

    -- Single-line labels of line l.
    singleRows l = case [(i, a) | (i, a) <- zip [0 :: Int ..] singles, sLine a == l] of
      [] -> []
      here ->
        let range a =
              let f = dispCol l (posCol (annoStart a))
               in (f, max (f + 1) (dispCol l (posCol (annoEnd a))))
            from = fst . range
            cells = markCells l PSingle
            row canvas extra = MarkRow (margin cells blank <> canvasLine canvas <> extra)
            -- Secondary marks first, so primary marks win where they
            -- overlap.
            marks =
              foldl
                (\cv a -> let (f, t) = range a in putText f (paintOf a) (Text.replicate (t - f) (Text.singleton (markChar a))) cv)
                IntMap.empty
                (sortOn annoPrimary (map snd here))
            (inlineIx, inline) = maximumBy (comparing (\(_, a) -> (snd (range a), fst (range a)))) here
            inlineText = [(paintOf inline, " " <> annoText inline) | not (Text.null (annoText inline))]
            hanging = sortOn from [a | (i, a) <- here, i /= inlineIx, not (Text.null (annoText a))]
            bars as = IntMap.fromList [(from a, bar a) | a <- as]
         in row marks inlineText
              : [row (bars hanging) [] | not (null hanging)]
              <> [ row (putText (from a) (paintOf a) (annoText a) (bars before)) []
                 | (before, a) <- reverse (zip (inits hanging) hanging)
                 ]

    -- The @__^@ row under the first line of a multi-line label.
    connectRow l (k, a) =
      MarkRow (margin (laneCells l (PConnect k) (k, a) blank) under <> canvasLine area)
      where
        col = dispCol l (posCol (annoStart a))
        under = ('_', paintOf a)
        area = IntMap.insert col (markChar a, paintOf a) (IntMap.fromList [(i, under) | i <- [0 .. col - 1]])

    -- The @|__^ text@ row under the last line of a multi-line label.
    endRow l (k, a) =
      MarkRow
        ( margin (laneCells l (PEnd (Down k)) (k, a) (bar a)) under
            <> canvasLine area
            <> [(paintOf a, " " <> annoText a) | not (Text.null (annoText a))]
        )
      where
        col = dispCol l (max 1 (posCol (annoEnd a) - 1))
        under = ('_', paintOf a)
        area = IntMap.insert col (markChar a, paintOf a) (IntMap.fromList [(i, under) | i <- [0 .. col - 1]])

    -- Lanes of a connector or end row of lane k: lanes left of it as
    -- usual, its own lane as given, lanes right of it crossed by the
    -- underscores (unless they are active there).
    laneCells l ph (k, a) own =
      [ if j < k
          then cell
          else
            if j == k
              then own
              else case cell of
                ('|', _) -> cell
                _ -> ('_', paintOf a)
      | (j, cell) <- zip laneList (markCells l ph)
      ]
