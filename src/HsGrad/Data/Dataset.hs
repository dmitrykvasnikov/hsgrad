{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Datasets: the rows of a @data@ declaration, in memory, checked against
-- its schema.
--
-- @
-- data xor   : { x : Vec 2, y : Real } = rows [ ([0, 0], -1), … ]
-- data moons : { x : Vec 2, y : Real } = csv "data/moons.csv" { x = [x1, x2], y = label }
-- @
--
-- The checker (step 12) has checked the schema, but not the source. This
-- module reads the source and makes sure it fits: inline rows must have one
-- literal of the right shape per field (E0404); a CSV mapping must give
-- every field the right number of columns (E0404), the file must be
-- readable (E0401), have those columns (E0402), numbers in them (E0403)
-- and at least one row (E0405).
--
-- The result is a 'Dataset': one list of numbers per row, in /column/
-- order, where a @Vec n@ field gives @n@ columns. For the schema above the
-- columns are @x[0]@, @x[1]@, @y@. That is exactly the row format of
-- "HsGrad.Train.Loop"'s 'Feed', and 'toFeed' makes the feed.
--
-- IO stays at the edge. 'buildDataset' is pure: it gets the contents of the
-- CSV files as a 'Files' map. 'loadDatasets' reads the files ('readFiles')
-- and then calls it. Since an error inside a CSV file points into that
-- file, 'loadDatasets' also returns the texts it read as 'Sources' for the
-- diagnostics renderer.
-- See @docs/14-data.md@.
module HsGrad.Data.Dataset
  ( -- * Datasets
    Dataset (..)
  , datasetSize
  , datasetColumns
  , fieldColumns

    -- * Building (pure)
  , ReadProblem (..)
  , Files
  , csvPath
  , buildDataset
  , buildDatasets
  , fileSources

    -- * Reading files
  , readCsvFile
  , readFiles
  , loadDatasets

    -- * Training
  , checkBatch
  , slotNames
  , toFeed
  ) where

import Control.Exception (Handler (..), catches, evaluate)
import Data.Bifunctor (first)
import Data.Either (partitionEithers)
import Data.Function ((&))
import Data.List (nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, isJust, mapMaybe)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.Encoding.Error (UnicodeException)
import Data.Text.IO.Utf8 qualified as Utf8
import HsGrad.Core.Op (Name (..))
import HsGrad.Data.Csv
import HsGrad.Lang.Check (DatasetInfo (..), suggestName)
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Shape (Shape (..), prettyShape)
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..), mergeSpan, pointSpan)
import HsGrad.Lang.Syntax
import HsGrad.Train.Loop (Feed (..))
import System.FilePath (isRelative, normalise, takeDirectory, (</>))
import System.IO.Error (ioeGetErrorString, isDoesNotExistError, isPermissionError)

-- ---------------------------------------------------------------------------
-- Datasets

-- | A dataset in memory.
data Dataset = Dataset
  { dataName :: !Text
  , dataNameSpan :: !Span
  -- ^ The name in the declaration, where dataset-wide problems point.
  , dataFields :: ![(Text, Shape Int)]
  -- ^ The schema: fields in order, each @Real@ or @Vec n@.
  , dataRows :: ![[Double]]
  -- ^ At least one row; each row has one value per column
  -- ('datasetColumns'), all finite.
  }
  deriving (Eq, Show)

-- | The number of rows.
datasetSize :: Dataset -> Int
datasetSize = length . dataRows

-- | The names of the columns, in row order: @x[0]@, @x[1]@, @y@.
datasetColumns :: Dataset -> [Text]
datasetColumns = fieldColumns . dataFields

-- | The columns of a schema: a @Real@ field is one column named like the
-- field, a @Vec n@ field @n@ columns named like its elements, the same
-- way parameter elements are named (@10-train.md@ §3.1).
fieldColumns :: [(Text, Shape Int)] -> [Text]
fieldColumns fields = concat [columns f s | (f, s) <- fields]
  where
    columns f = \case
      Real -> [f]
      Vec n -> [f <> "[" <> showT i <> "]" | i <- [0 .. n - 1]]
      -- Not in a checked schema (E0309), but the names would be these.
      Mat r c -> [f <> "[" <> showT i <> "," <> showT j <> "]" | i <- [0 .. r - 1], j <- [0 .. c - 1]]

-- ---------------------------------------------------------------------------
-- Building

-- | Why a file could not be read.
data ReadProblem
  = FileNotFound
  | PermissionDenied
  | -- | The bytes are not UTF-8 text.
    NotUtf8
  | -- | Anything else, with the system's description (@is a directory@).
    OtherReadError !Text
  deriving (Eq, Show)

-- | The files read for the datasets, by their resolved path ('csvPath'):
-- the text, or why there is none.
type Files = Map FilePath (Either ReadProblem Text)

-- | The file a @csv@ dataset reads, given the path of the @.hsg@ file
-- (as the user wrote it on the command line). A relative path in the
-- declaration is relative to the directory of the @.hsg@ file
-- (@01-language.md@ §3), so @examples/moons.hsg@ reading
-- @"data/moons.csv"@ gives @examples/data/moons.csv@. 'Nothing' for inline
-- rows.
csvPath :: FilePath -> DatasetInfo -> Maybe FilePath
csvPath hsg info = case datasetSource info of
  Csv _ path _ -> Just (normalise (takeDirectory hsg </> Text.unpack (stringValue path)))
  Rows _ _ -> Nothing

-- | The dataset of a checked @data@ declaration, or every problem with it.
-- The first argument is the path of the @.hsg@ file, to find CSV files in
-- the 'Files' map.
buildDataset :: FilePath -> Files -> DatasetInfo -> Either [Diagnostic] Dataset
buildDataset hsg files info =
  Dataset (datasetName info) (datasetNameSpan info) (datasetFields info) <$> case datasetSource info of
    Rows sp rows -> inlineRows info sp rows
    Csv sp path maps ->
      let file = fromMaybe "" (csvPath hsg info)
          contents = Map.findWithDefault (Left (OtherReadError "the file was not read")) file files
       in csvRows (CsvContext info file path) sp maps contents

-- | All datasets, or the problems of all of them.
buildDatasets :: FilePath -> Files -> [DatasetInfo] -> Either [Diagnostic] [Dataset]
buildDatasets hsg files infos = case partitionEithers (map (buildDataset hsg files) infos) of
  ([], datasets) -> Right datasets
  (problems, _) -> Left (concat problems)

-- | The texts of the files that could be read, for the renderer.
fileSources :: Files -> Sources
fileSources = Map.mapMaybe (either (const Nothing) Just)

-- | Fail with the diagnostics of a stage, if it has any.
stage :: [Diagnostic] -> a -> Either [Diagnostic] a
stage [] a = Right a
stage ds _ = Left ds

-- ---------------------------------------------------------------------------
-- Inline rows

-- | The rows of @rows [ … ]@: each row must have one literal per field, a
-- number for a @Real@ field and a vector of @n@ numbers for a @Vec n@
-- field. Every row is checked; all problems are E0404.
inlineRows :: DatasetInfo -> Span -> [Row Span] -> Either [Diagnostic] [[Double]]
inlineRows info sp rows = case partitionEithers (map row rows) of
  ([], values) -> stage [emptyRows | null values] values
  (problems, _) -> Left (concat problems)
  where
    fields = datasetFields info
    name = datasetName info
    schemaLabel = "schema " <> code (schemaText fields)

    emptyRows = diagnostic E0405 ("dataset " <> code name <> " is empty") sp "no rows"

    row (Row rsp lits)
      | length lits /= length fields =
          Left
            [ diagnostic
                E0404
                ("row has " <> count (length lits) "value" <> ", but dataset " <> code name <> " has " <> count (length fields) "field")
                rsp
                (count (length lits) "value")
                & withSecondary (datasetNameSpan info) schemaLabel
                & maybe id withHint (regroup lits)
            ]
      | otherwise = case partitionEithers (zipWith value fields lits) of
          ([], vs) -> Right (concat vs)
          (problems, _) -> Left problems

    value (f, shape) lit = case (shape, lit) of
      (Real, LitNumber n) -> (: []) <$> number n
      (Vec k, LitVector _ ls)
        | Just ns <- mapM asNumber ls, length ns == k -> mapM number ns
      _ ->
        Left $
          diagnostic
            E0404
            ("field " <> code f <> " is " <> article (code (prettyShape shape)) <> ", but this value is " <> describeLiteral lit)
            (literalAnn lit)
            (describeLiteral lit)
            & withSecondary (datasetNameSpan info) schemaLabel

    -- The parser already rejects literals too large for a Double (E0103);
    -- checking again keeps "every value is finite" a property of this
    -- module rather than of its input.
    number n
      | isInfinite d =
          Left (diagnostic E0404 ("number " <> code (showNumber (numberValue n)) <> " is too large") (numberAnn n) "not a finite number")
      | otherwise = Right d
      where
        d = fromRational (numberValue n) :: Double

    -- A row with exactly as many numbers as the schema has columns has
    -- probably lost the brackets of its vector fields: show the row with
    -- them.
    regroup lits = do
      ns <- mapM asNumber lits
      let widths = [fromMaybe 1 (vecWidth s) | (_, s) <- fields]
      if length ns == sum widths && any (/= 1) widths
        then Just ("vector fields are written in brackets: " <> code ("(" <> commas (groupInto widths ns) <> ")"))
        else Nothing
    vecWidth = \case
      Vec k -> Just k
      _ -> Nothing
    groupInto (w : ws) ns =
      let (here, rest) = splitAt w ns
          shown = map (showNumber . numberValue) here
       in (if w == 1 then commas shown else "[" <> commas shown <> "]") : groupInto ws rest
    groupInto [] _ = []

-- | A number literal: a positive number, or a negative one (the parser
-- reads @-1@ in a literal as one signed number).
asNumber :: Literal a -> Maybe (Number a)
asNumber = \case
  LitNumber n -> Just n
  LitVector _ _ -> Nothing

-- | What a literal is, for messages: @a `Real`@, @a `Vec 3`@, @a nested
-- vector@.
describeLiteral :: Literal a -> Text
describeLiteral = \case
  LitNumber _ -> "a " <> code "Real"
  LitVector _ ls
    | all (isJust . asNumber) ls -> "a " <> code ("Vec " <> showT (length ls))
    | otherwise -> "a nested vector"

-- ---------------------------------------------------------------------------
-- CSV sources

-- | What every CSV diagnostic needs to know.
data CsvContext = CsvContext
  { ctxInfo :: !DatasetInfo
  , ctxFile :: !FilePath
  -- ^ The resolved path: the file name in the CSV file's spans.
  , ctxPath :: !(StringLit Span)
  -- ^ The path as written in the declaration.
  }

-- | A mapped column: the field it feeds and its name in the mapping.
type Mapped = (Text, Ident Span)

-- | The rows of a @csv@ source, in stages. Each stage reports all its
-- problems, and a stage only runs if the ones before it succeeded, since
-- it needs their results:
--
-- 1. the mapping against the schema (E0404);
-- 2. the file's text (E0401);
-- 3. its structure ('parseCsv': E0405 for an empty file, E0403);
-- 4. the mapped columns in the header (E0402);
-- 5. at least one record (E0405);
-- 6. numbers in the mapped columns (E0403).
csvRows :: CsvContext -> Span -> [ColumnMap Span] -> Either ReadProblem Text -> Either [Diagnostic] [[Double]]
csvRows ctx sp maps contents = do
  mapped <- checkMapping (ctxInfo ctx) sp maps
  text <- first (\p -> [readError ctx p]) contents
  table <- first (structureDiagnostics ctx mapped) (parseCsv (ctxFile ctx) text)
  columns <- resolveColumns ctx table mapped
  stage [noRecords ctx table | null (tableRecords table)] ()
  first (cellDiagnostics table columns) (numericColumns [i | (i, _) <- columns] table)

-- | Stage 1: the columns of every field, in schema order, or E0404 for
-- every way the mapping doesn't fit the schema: a field mapped twice or
-- not at all, a name that is not a field, a @Real@ field with a list of
-- columns, a @Vec n@ field without a list of @n@ columns.
checkMapping :: DatasetInfo -> Span -> [ColumnMap Span] -> Either [Diagnostic] [Mapped]
checkMapping info sp maps = stage problems [(f, c) | (f, _) <- fields, ColumnMap _ cols <- take 1 (mapsOf f), c <- columnIdents cols]
  where
    fields = datasetFields info
    name = datasetName info
    mapsOf f = [m | m@(ColumnMap f' _) <- maps, identName f' == f]
    schema d = d & withSecondary (datasetNameSpan info) ("schema " <> code (schemaText fields))

    problems = concat (zipWith mapProblems [0 :: Int ..] maps) <> missing

    missing =
      [ diagnostic E0404 ("field " <> code f <> " has no column") sp ("no column for " <> code f)
          & schema
          & withHint ("add " <> code (f <> " = " <> example s) <> " to the mapping")
      | (f, s) <- fields
      , null (mapsOf f)
      ]
    example = \case
      Vec n -> "[" <> commas (replicate n "…") <> "]"
      _ -> "…"

    mapProblems i (ColumnMap f cols) = case lookup (identName f) fields of
      Nothing ->
        [ diagnostic E0404 (code (identName f) <> " is not a field of dataset " <> code name) (identAnn f) "not a field"
            & schema
            & maybe id (\s -> withHint ("did you mean " <> code s <> "?")) (suggestName (identName f) (map fst fields))
        ]
      Just shape -> case [f' | ColumnMap f' _ <- take i maps, identName f' == identName f] of
        earlier : _ ->
          [ diagnostic E0404 ("field " <> code (identName f) <> " is mapped twice") (identAnn f) "mapped again here"
              & withSecondary (identAnn earlier) "first mapped here"
          ]
        [] -> columnProblems (identName f) shape cols

    columnProblems f shape cols = case (shape, cols) of
      (Real, Column _) -> []
      (Real, Columns csp cs) ->
        [ diagnostic E0404 ("field " <> code f <> " is a " <> code "Real" <> ": it takes one column, without brackets") csp "a list of columns"
            & schema
            & case cs of
              [c] -> withHint ("write " <> code (f <> " = " <> identName c))
              _ -> id
        ]
      (Vec n, Columns csp cs)
        | length cs == n -> []
        | otherwise ->
            [ diagnostic
                E0404
                ("field " <> code f <> " is a " <> code (prettyShape shape) <> ", but " <> count (length cs) "column" <> (if length cs == 1 then " is" else " are") <> " mapped to it")
                csp
                (count (length cs) "column")
                & schema
            ]
      (Vec n, Column c) ->
        [ diagnostic
            E0404
            ("field " <> code f <> " is a " <> code (prettyShape shape) <> ": it takes a list of " <> count n "column")
            (identAnn c)
            "a single column"
            & schema
            & if n == 1 then withHint ("write " <> code (f <> " = [" <> identName c <> "]")) else id
        ]
      (Mat _ _, _) ->
        [ diagnostic E0404 ("field " <> code f <> " is a " <> code (prettyShape shape) <> ", which can't be read from CSV") (datasetNameSpan info) "a matrix field"
        ]

    columnIdents = \case
      Column c -> [c]
      Columns _ cs -> cs

-- | Stage 2: E0401, at the path in the declaration.
readError :: CsvContext -> ReadProblem -> Diagnostic
readError ctx problem =
  diagnostic E0401 ("cannot read " <> code file) (stringAnn (ctxPath ctx)) label
    & relativeNote
    & case problem of
      NotUtf8 -> withNote "CSV files are read as UTF-8 text"
      _ -> id
  where
    file = Text.pack (ctxFile ctx)
    written = stringValue (ctxPath ctx)
    label = case problem of
      FileNotFound -> "file not found"
      PermissionDenied -> "permission denied"
      NotUtf8 -> "not UTF-8 text"
      OtherReadError msg -> msg
    -- Explain the resolved path when it is not what the user wrote.
    relativeNote
      | problem == FileNotFound && isRelative (Text.unpack written) && file /= written =
          withNote ("a path in " <> code "csv" <> " is relative to the directory of the " <> code ".hsg" <> " file: here " <> code file)
      | otherwise = id

-- | Stage 3: the structural problems of a file. Problems of the same kind
-- are reported once, at the first; a note lists the lines of the others,
-- so a file with a hundred ragged rows gives one error, not a hundred.
structureDiagnostics :: CsvContext -> [Mapped] -> [CsvProblem] -> [Diagnostic]
structureDiagnostics ctx mapped problems =
  [ structureDiagnostic p & alsoOn [l | Just l <- map problemLine rest]
  | p : rest <- groupsBy kind problems
  ]
  where
    file = Text.pack (ctxFile ctx)
    inFile d = d & withSecondary (stringAnn (ctxPath ctx)) ("dataset " <> code (datasetName (ctxInfo ctx)) <> " reads this file")
    alsoOn [] = id
    alsoOn ls = withNote ("the same problem on " <> listLines ls)

    kind :: CsvProblem -> Int
    kind = \case
      NoHeader -> 0
      BlankLine _ -> 1
      UnterminatedQuote _ -> 2
      TextAfterQuote _ -> 3
      WrongCellCount _ _ -> 4

    problemLine = \case
      NoHeader -> Nothing
      BlankLine s -> Just (posLine (spanStart s))
      UnterminatedQuote s -> Just (posLine (spanStart s))
      TextAfterQuote s -> Just (posLine (spanStart s))
      WrongCellCount r _ -> Just (recordLine r)

    structureDiagnostic = \case
      NoHeader ->
        diagnostic E0405 ("dataset " <> code (datasetName (ctxInfo ctx)) <> " is empty") (stringAnn (ctxPath ctx)) (code file <> " is empty")
          & withNote ("a CSV file starts with a header line naming its columns, such as " <> code (Text.intercalate "," (nub [identName c | (_, c) <- mapped])))
      BlankLine s ->
        diagnostic E0403 ("blank line in " <> code file) s "blank line"
          & inFile
          & withHint "remove it: blank lines are only allowed at the end of the file"
      UnterminatedQuote s ->
        diagnostic E0403 ("unterminated quoted cell in " <> code file) s "this quote is not closed"
          & inFile
          & withNote ("a quoted cell ends on its line; a quote inside it is written " <> code "\"\"")
      TextAfterQuote s ->
        diagnostic E0403 ("text after a quoted cell in " <> code file) s ("expected " <> code "," <> " or the end of the line")
          & inFile
      WrongCellCount r header ->
        let cells = recordCells r
            k = length cells
            width = length (recordCells header)
            lastCell = lastOr (Cell "" (recordSpan r)) cells
            (at, label) = case drop width cells of
              extra@(e : _) -> (mergeSpan (cellSpan e) (cellSpan (lastOr e extra)), count (k - width) "cell" <> " too many")
              [] -> (pointSpan (ctxFile ctx) (spanEnd (cellSpan lastCell)), count (width - k) "cell" <> " missing")
         in diagnostic E0403 ("row has " <> count k "cell" <> ", but the header has " <> count width "column") at label
              & withSecondary (recordSpan header) "the header"
              & inFile

-- | Stage 4: the header position of every mapped column, or E0402 for a
-- column that is not in the header (with a suggestion), or that is in it
-- more than once.
resolveColumns :: CsvContext -> Table -> [Mapped] -> Either [Diagnostic] [(Int, Mapped)]
resolveColumns ctx table mapped = case partitionEithers (map resolve mapped) of
  ([], columns) -> Right columns
  (problems, _) -> Left problems
  where
    file = Text.pack (ctxFile ctx)
    header = tableHeader table
    names = columnNames table
    resolve m@(_, c) = case columnIndices (identName c) table of
      [i] -> Right (i, m)
      [] ->
        Left $
          diagnostic E0402 ("no column " <> code (identName c) <> " in " <> code file) (identAnn c) "not in the header"
            & withSecondary (recordSpan header) "the header"
            & maybe id (\s -> withHint ("did you mean " <> code s <> "?")) (suggestName (identName c) (filter isIdentifier names))
            & separatorHint
      is ->
        Left $
          foldl
            (\d sp -> d & withSecondary sp "")
            (diagnostic E0402 ("column " <> code (identName c) <> " appears " <> showT (length is) <> " times in the header of " <> code file) (identAnn c) "ambiguous")
            [cellSpan cell | (j, cell) <- zip [0 ..] (recordCells header), j `elem` is]
    -- A header that is one column with a ';' or a tab in it is a file with
    -- another separator.
    separatorHint = case names of
      [single]
        | Just sep <- lookup True [(Text.any (== ch) single, s) | (ch, s) <- [(';', "`;`"), ('\t', "tabs")]] ->
            withHint ("the header is a single column " <> code single <> ": the file seems to be separated by " <> sep <> ", but hsgrad reads comma-separated files")
      _ -> id
    isIdentifier t = case Text.uncons t of
      Just (ch, rest) -> isIdentStart ch && Text.all isIdentChar rest && not (isReserved t)
      Nothing -> False

-- | Stage 5: a header but no records.
noRecords :: CsvContext -> Table -> Diagnostic
noRecords ctx table =
  diagnostic E0405 ("dataset " <> code (datasetName (ctxInfo ctx)) <> " is empty") (stringAnn (ctxPath ctx)) (code (Text.pack (ctxFile ctx)) <> " has no rows")
    & withSecondary (recordSpan (tableHeader table)) "only a header"

-- | Stage 6: cells that are not numbers, E0403 at the cell. As for the
-- structure, one diagnostic per column, with a note for the others.
cellDiagnostics :: Table -> [(Int, Mapped)] -> [BadCell] -> [Diagnostic]
cellDiagnostics table columns bad =
  [ cellDiagnostic b & alsoOn b rest
  | b : rest <- groupsBy badColumn (nub bad)
  ]
  where
    column b = fromMaybe "" (lookup (badColumn b) (zip [0 ..] (columnNames table)))
    alsoOn _ [] = id
    alsoOn b rest =
      withNote
        ( "column " <> code (column b) <> " has " <> count (length rest) "more bad cell" <> ", on "
            <> listLines (map (posLine . spanStart . cellSpan . badCell) rest)
        )
    cellDiagnostic b =
      let text = cellText (badCell b)
          (message, label) = case badProblem b of
            EmptyCell -> ("empty cell in column " <> code (column b), "empty")
            NotANumber -> ("cell " <> code text <> " of column " <> code (column b) <> " is not a number", "not a number")
            NotFinite -> ("cell " <> code text <> " of column " <> code (column b) <> " is not a finite number", "not finite")
       in diagnostic E0403 message (cellSpan (badCell b)) label
            & case lookup (badColumn b) columns of
              Just (f, c) -> withSecondary (identAnn c) ("read for field " <> code f)
              Nothing -> id

-- ---------------------------------------------------------------------------
-- Reading files

-- | Read a CSV file as UTF-8 text, whatever the locale.
readCsvFile :: FilePath -> IO (Either ReadProblem Text)
readCsvFile path =
  (Right <$> (Utf8.readFile path >>= evaluate))
    `catches` [ Handler (pure . Left . ioProblem)
              , Handler (\(_ :: UnicodeException) -> pure (Left NotUtf8))
              ]
  where
    ioProblem e
      | isDoesNotExistError e = FileNotFound
      | isPermissionError e = PermissionDenied
      | otherwise = OtherReadError (Text.pack (ioeGetErrorString e))

-- | Read every file once.
readFiles :: [FilePath] -> IO Files
readFiles paths = Map.fromList <$> mapM (\p -> (p,) <$> readCsvFile p) (nub paths)

-- | Read the CSV files of the datasets of a @.hsg@ file and build them.
-- Also returns the texts of the files that were read, to render
-- diagnostics that point into them (merge them with the @.hsg@ file's
-- text).
loadDatasets :: FilePath -> [DatasetInfo] -> IO (Sources, Either [Diagnostic] [Dataset])
loadDatasets hsg infos = do
  files <- readFiles (mapMaybe (csvPath hsg) infos)
  pure (fileSources files, buildDatasets hsg files infos)

-- ---------------------------------------------------------------------------
-- Training

-- | E0405 if a batch size (@--batch@) is larger than the dataset. Smaller
-- or equal is fine: the training loop samples that many distinct rows per
-- step. (A batch size below 1 is the command line's to reject.)
checkBatch :: Int -> Dataset -> [Diagnostic]
checkBatch b d
  | b > n =
      [ diagnostic E0405 (code ("--batch " <> showT b) <> " is larger than dataset " <> code (dataName d)) (dataNameSpan d) (code (dataName d) <> " has " <> count n "row")
          & withHint ("use " <> code ("--batch " <> showT n) <> " or less, or leave " <> code "--batch" <> " out to use every row in every step")
      ]
  | otherwise = []
  where
    n = datasetSize d

-- | The slot leaves of a dataset with @b@ slots (D10, @01-language.md@ §8):
-- for every slot @i@, the leaf name of each column, @moons#i.x[0]@,
-- @moons#i.x[1]@, @moons#i.y@. Depends only on the name and the schema, so
-- the compiler can name its leaves with it before any data is read.
slotNames :: Text -> [(Text, Shape Int)] -> Int -> [[Name]]
slotNames name fields b =
  [[Name (name <> "#" <> showT i <> "." <> c) | c <- fieldColumns fields] | i <- [0 .. b - 1]]

-- | The training loop's view of a dataset: its rows, bound to the given
-- slots (one list of leaf names per slot, in column order; see
-- 'slotNames').
toFeed :: Dataset -> [[Name]] -> Feed
toFeed d = Feed (dataName d) (dataRows d)

-- ---------------------------------------------------------------------------
-- Text helpers

showT :: Int -> Text
showT = Text.pack . show

code :: Text -> Text
code t = "`" <> t <> "`"

commas :: [Text] -> Text
commas = Text.intercalate ", "

-- | @1 row@, @3 rows@.
count :: Int -> Text -> Text
count 1 noun = "1 " <> noun
count n noun = showT n <> " " <> noun <> "s"

-- | @a `Real`@, @a `Vec 2`@: the article for a shape in backticks.
article :: Text -> Text
article = ("a " <>)

-- | The schema as written: @{ x : Vec 2, y : Real }@.
schemaText :: [(Text, Shape Int)] -> Text
schemaText fields = "{ " <> commas [f <> " : " <> prettyShape s | (f, s) <- fields] <> " }"

-- | @line 7@, @lines 7 and 9@, @lines 7, 9 and 12@; after five numbers the
-- rest is counted: @lines 2, 3, 4, 5, 6 and 4 more@.
listLines :: [Int] -> Text
listLines = \case
  [l] -> "line " <> showT l
  ls ->
    let (shown, more) = splitAt 5 ls
        items = map showT shown <> [showT (length more) <> " more" | not (null more)]
     in "lines " <> commas (take (length items - 1) items) <> " and " <> lastOr "" items

-- | The last element, or the default for an empty list.
lastOr :: a -> [a] -> a
lastOr = foldl (\_ x -> x)

-- | Groups of the elements with the same key, in the order in which the
-- keys first appear; each group in list order.
groupsBy :: Eq k => (a -> k) -> [a] -> [[a]]
groupsBy key xs = [[x | x <- xs, key x == k] | k <- nub (map key xs)]
