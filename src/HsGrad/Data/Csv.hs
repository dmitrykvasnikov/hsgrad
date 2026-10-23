{-# LANGUAGE OverloadedStrings #-}

-- | A small reader for CSV files of numbers.
--
-- A dataset declared with @csv "data/moons.csv" { x = [x1, x2], y = label }@
-- (@01-language.md@ §3) reads a file like this one:
--
-- > x1,x2,label
-- > 0.912,0.334,-1
-- > 1.587,-0.441,1
--
-- The format is the common subset that spreadsheets, pandas and numpy
-- write:
--
-- * the first line is the __header__: one name per column;
-- * every further line is a __record__ with exactly one cell per column;
-- * cells are separated by commas; spaces and tabs around a cell are not
--   part of it;
-- * a cell may be __quoted__, @"x1"@, which allows commas in it and writes
--   a quote as two quotes (@"say ""hi"""@); a quoted cell ends on the line
--   it starts on;
-- * lines end with @\\n@ or @\\r\\n@; a UTF-8 byte-order mark at the start is
--   skipped; blank lines are allowed at the end of the file only.
--
-- Reading happens in two stages. 'parseCsv' checks the structure and gives
-- a 'Table' of text cells; 'numericColumns' then reads the columns a
-- dataset actually uses as numbers. So a file may have other columns (an
-- id, a comment) with anything in them.
--
-- Every cell keeps its 'Span' in the file, so a problem can be shown in a
-- snippet of the CSV file itself, like any error in a @.hsg@ file. This
-- module only reports /what/ is wrong ('CsvProblem', 'BadCell'); turning
-- that into a diagnostic with a code is "HsGrad.Data.Dataset"'s job.
-- See @docs/14-data.md@.
module HsGrad.Data.Csv
  ( -- * Tables
    Table (..)
  , Record (..)
  , Cell (..)
  , columnNames
  , columnIndices

    -- * Parsing
  , CsvProblem (..)
  , parseCsv

    -- * Numbers
  , NumberProblem (..)
  , readNumber
  , BadCell (..)
  , numericColumns
  ) where

import Control.Monad (guard)
import Data.Char (isDigit, isSpace, toLower)
import Data.Either (partitionEithers)
import Data.List (dropWhileEnd)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..))

-- ---------------------------------------------------------------------------
-- Tables

-- | One cell: its text (without surrounding whitespace, and without the
-- quotes of a quoted cell) and where it is. The span of a quoted cell
-- includes its quotes.
data Cell = Cell
  { cellText :: !Text
  , cellSpan :: !Span
  }
  deriving (Eq, Show)

-- | One line of the file with its cells.
data Record = Record
  { recordLine :: !Int
  -- ^ 1-based line number; the header is line 1.
  , recordSpan :: !Span
  -- ^ The whole line, without its line end.
  , recordCells :: ![Cell]
  }
  deriving (Eq, Show)

-- | A parsed file: the header and the records, in file order. Every
-- record has as many cells as the header.
data Table = Table
  { tableHeader :: !Record
  , tableRecords :: ![Record]
  }
  deriving (Eq, Show)

-- | The names in the header, in column order.
columnNames :: Table -> [Text]
columnNames = map cellText . recordCells . tableHeader

-- | The 0-based positions of the columns with this name. Usually one or
-- none; a header may repeat a name, and the caller decides whether that
-- matters.
columnIndices :: Text -> Table -> [Int]
columnIndices name table = [i | (i, n) <- zip [0 ..] (columnNames table), n == name]

-- ---------------------------------------------------------------------------
-- Parsing

-- | What can be wrong with the structure of a file. Each problem except
-- 'NoHeader' says where it is.
data CsvProblem
  = -- | The file is empty, or has only blank lines: there is no header.
    NoHeader
  | -- | A blank line that is not at the end of the file.
    BlankLine !Span
  | -- | A quoted cell whose closing quote is missing; the span runs from
    -- the opening quote to the end of the line.
    UnterminatedQuote !Span
  | -- | Something other than @,@ after the closing quote of a cell, such as
    -- @"a"b@; the span covers it.
    TextAfterQuote !Span
  | -- | A record with a different number of cells than the header; the
    -- record, then the header.
    WrongCellCount !Record !Record
  deriving (Eq, Show)

-- | Parse a file's text into a table, or report every structural problem
-- (in line order). The file name is only used in the spans.
parseCsv :: FilePath -> Text -> Either [CsvProblem] Table
parseCsv file text = case dropWhileEnd (isBlank . snd) numbered of
  [] -> Left [NoHeader]
  content -> case partitionEithers (map line content) of
    ([], header : records) ->
      let width = length (recordCells header)
       in case [WrongCellCount r header | r <- records, length (recordCells r) /= width] of
            [] -> Right (Table header records)
            ragged -> Left ragged
    ([], []) -> Left [NoHeader] -- impossible: the last line of content is not blank
    (problems, _) -> Left problems
  where
    numbered = zip [1 ..] (map dropCR (Text.splitOn "\n" (dropBom text)))
    line (l, t)
      | isBlank t = Left (BlankLine (lineSpan file l t))
      | otherwise = lineRecord file l t
    isBlank = Text.all isSpace
    dropCR l = fromMaybe l (Text.stripSuffix "\r" l)
    -- A byte-order mark becomes a space: it is still column 1 for the
    -- renderer (which shows the original text), and like a space it is
    -- not part of the first cell.
    dropBom t = maybe t (Text.cons ' ') (Text.stripPrefix "\xFEFF" t)

-- | The span of a whole line.
lineSpan :: FilePath -> Int -> Text -> Span
lineSpan file l t = SrcSpan file (Pos l 1) (Pos l (1 + Text.length t))

-- | One non-blank line as a record.
lineRecord :: FilePath -> Int -> Text -> Either CsvProblem Record
lineRecord file l t = Record l (lineSpan file l t) <$> splitCells file l t

-- | Split a line into cells.
--
-- The scan goes left to right and keeps the column it is at. A cell starts
-- after optional blanks: a @"@ there makes it a quoted cell, and anything
-- else is an unquoted cell that runs to the next comma. After a cell comes
-- the end of the line, or a comma and the next cell, so @1,2,@ has three
-- cells, the last one empty.
splitCells :: FilePath -> Int -> Text -> Either CsvProblem [Cell]
splitCells file l = cellsFrom 1
  where
    sp c1 c2 = SrcSpan file (Pos l c1) (Pos l c2)
    isBlank ch = ch == ' ' || ch == '\t'

    -- The rest of the line, starting at column c, is a cell and what follows.
    cellsFrom c t =
      let (ws, t1) = Text.span isBlank t
          c1 = c + Text.length ws
       in case Text.uncons t1 of
            Just ('"', t2) -> quoted c1 t2
            _ ->
              let (raw, t2) = Text.break (== ',') t1
                  content = Text.dropWhileEnd isBlank raw
               in (Cell content (sp c1 (c1 + Text.length content)) :) <$> afterCell (c1 + Text.length raw) t2

    -- After a cell, at column c: the end of the line or a comma.
    afterCell c t = case Text.uncons t of
      Nothing -> Right []
      Just (_comma, rest) -> cellsFrom (c + 1) rest

    -- A quoted cell whose opening quote is at column c1; t is the text
    -- after that quote.
    quoted c1 t = case closingQuote t of
      Nothing -> Left (UnterminatedQuote (sp c1 (c1 + 1 + Text.length t)))
      Just (content, used, rest) ->
        let c2 = c1 + 1 + used
            (ws, rest') = Text.span isBlank rest
            cell = Cell content (sp c1 c2)
         in case Text.uncons rest' of
              Nothing -> Right [cell]
              Just (',', _) -> (cell :) <$> afterCell (c2 + Text.length ws) rest'
              Just _ ->
                let junk = Text.dropWhileEnd isBlank (Text.takeWhile (/= ',') rest')
                    c3 = c2 + Text.length ws
                 in Left (TextAfterQuote (sp c3 (c3 + Text.length junk)))

-- | The text of a quoted cell, from just after its opening quote: the
-- content (with @""@ turned into @"@), how many characters it used up
-- including the closing quote, and the rest of the line. 'Nothing' if the
-- quote is never closed.
closingQuote :: Text -> Maybe (Text, Int, Text)
closingQuote = go [] 0
  where
    go acc n t =
      let (piece, rest) = Text.break (== '"') t
          after = Text.drop 1 rest
          n' = n + Text.length piece
       in if Text.null rest
            then Nothing
            else case Text.uncons after of
              Just ('"', more) -> go (acc <> [piece, "\""]) (n' + 2) more
              _ -> Just (Text.concat (acc <> [piece]), n' + 1, after)

-- ---------------------------------------------------------------------------
-- Numbers

-- | Why a cell is not a number.
data NumberProblem
  = -- | Nothing in the cell.
    EmptyCell
  | -- | Not a decimal number at all: @abc@, @1,5@ (quoted), @0x10@.
    NotANumber
  | -- | A number, but not a finite 'Double': @nan@, @inf@, @1e999@.
    NotFinite
  deriving (Eq, Show)

-- | Read a decimal number, the way CSV writers print them:
--
-- > sign? ( digits ( "." digits? )? | "." digits ) ( ("e" | "E") sign? digits )?
--
-- where @sign@ is @+@ or @-@. So @2@, @-0.5@, @+3@, @.5@, @1.@, @1e-4@ and
-- @3.0E+2@ are numbers. The text is read exactly (as a fraction) and then
-- rounded once to the nearest 'Double', so @0.1@ is the same 'Double' as
-- the Haskell literal @0.1@. A number too small for a 'Double' becomes 0;
-- one too large is 'NotFinite', and so are @nan@ and @inf@: the values of a
-- dataset must be finite (as every value in the graph, @05-core-eval.md@).
readNumber :: Text -> Either NumberProblem Double
readNumber t
  | Text.null t = Left EmptyCell
  | Text.map toLower (Text.dropWhile (`elem` ['+', '-']) t) `elem` ["nan", "inf", "infinity"] = Left NotFinite
  | otherwise = maybe (Left NotANumber) toDouble (scanDecimal t)

-- | A decimal number as read: @sign · digits · 10^exponent@.
data Decimal = Decimal !Bool !Integer !Integer

-- | Scan the grammar of 'readNumber'; the whole text must match.
scanDecimal :: Text -> Maybe Decimal
scanDecimal t0 = do
  let (negative, t1) = sign t0
      (whole, t2) = Text.span isDigit t1
      (fraction, t3) = case Text.uncons t2 of
        Just ('.', rest) -> Text.span isDigit rest
        _ -> ("", t2)
  guard (not (Text.null whole && Text.null fraction))
  e <- exponentPart t3
  pure (Decimal negative (digitsValue (whole <> fraction)) (e - fromIntegral (Text.length fraction)))
  where
    sign t = case Text.uncons t of
      Just ('-', rest) -> (True, rest)
      Just ('+', rest) -> (False, rest)
      _ -> (False, t)
    exponentPart t = case Text.uncons t of
      Nothing -> Just 0
      Just (e, rest) | e == 'e' || e == 'E' -> do
        let (negative, rest') = sign rest
        guard (not (Text.null rest') && Text.all isDigit rest')
        pure (if negative then negate (digitsValue rest') else digitsValue rest')
      Just _ -> Nothing
    digitsValue = Text.foldl' (\acc d -> 10 * acc + fromIntegral (fromEnum d - fromEnum '0')) 0

-- | Round a decimal to the nearest 'Double'.
--
-- @fromRational@ rounds correctly, but building the fraction for @1e999999@
-- would take a long time, so exponents far outside the range of 'Double'
-- (about 1e-324 to 1.8e308) are decided before: too small is 0, too large
-- is 'NotFinite'.
toDouble :: Decimal -> Either NumberProblem Double
toDouble (Decimal negative m e)
  | m == 0 = Right (signed 0)
  | magnitude > 310 = Left NotFinite
  | magnitude < -330 = Right (signed 0)
  | isInfinite value = Left NotFinite
  | otherwise = Right (signed value)
  where
    -- The number is between 10^(magnitude - 1) and 10^magnitude.
    magnitude = e + fromIntegral (length (show m))
    value = fromRational (fromIntegral m * 10 ^^ e) :: Double
    signed x = if negative then negate x else x

-- | A cell of a numeric column that is not a number.
data BadCell = BadCell
  { badColumn :: !Int
  -- ^ The 0-based column.
  , badCell :: !Cell
  , badProblem :: !NumberProblem
  }
  deriving (Eq, Show)

-- | Read the given columns of every record as numbers: one list per
-- record, with the values in the order of the column list (a column may
-- be listed more than once). Every cell that is not a number is reported,
-- record by record.
--
-- The columns must exist ('columnIndices' finds them); a column the table
-- doesn't have is skipped.
numericColumns :: [Int] -> Table -> Either [BadCell] [[Double]]
numericColumns cols table = case concat problems of
  [] -> Right rows
  bad -> Left bad
  where
    (problems, rows) = unzip (map record (tableRecords table))
    record r = partitionEithers [number c cell | c <- cols, cell <- take 1 (drop c (recordCells r))]
    number c cell = either (Left . BadCell c cell) Right (readNumber (cellText cell))
