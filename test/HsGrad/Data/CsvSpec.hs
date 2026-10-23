{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Data.Csv".
--
-- * Structure: header and records, cell positions, whitespace, quotes,
--   line ends, the byte-order mark, blank lines, and every structural
--   problem.
-- * Numbers: the accepted forms, exact rounding, what is rejected, and
--   'show' of any finite 'Double' read back.
-- * Columns: selecting columns as numbers, reporting bad cells.
-- * A round trip: random numeric tables written in random (valid) styles
--   are read back exactly.
module HsGrad.Data.CsvSpec
  ( spec
  ) where

import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Data.Csv
import HsGrad.Lang.Span (Pos (..), SrcSpan (..))
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Gen, arbitrary, chooseInt, elements, forAll, listOf1, oneof, suchThat, vectorOf, (===), (==>))

-- | A table that must parse.
table :: Text -> Table
table t = case parseCsv "t.csv" t of
  Right tbl -> tbl
  Left ps -> error ("does not parse: " <> show ps)

-- | The problems of a file that must not parse.
problems :: Text -> [CsvProblem]
problems t = either id (const []) (parseCsv "t.csv" t)

-- | The cell texts: the header, then every record.
texts :: Table -> [[Text]]
texts tbl = map (map cellText . recordCells) (tableHeader tbl : tableRecords tbl)

-- | Spans of cells of the header (line 1) and records.
spans :: Table -> [[(Int, Int, Int)]]
spans tbl = map (map (pos . cellSpan) . recordCells) (tableHeader tbl : tableRecords tbl)
  where
    pos (SrcSpan _ (Pos l c1) (Pos _ c2)) = (l, c1, c2)

at :: Int -> Int -> Int -> SrcSpan
at l c1 c2 = SrcSpan "t.csv" (Pos l c1) (Pos l c2)

spec :: Spec
spec = do
  structureSpec
  problemSpec
  numberSpec
  columnSpec
  roundTripSpec

-- ---------------------------------------------------------------------------
-- Structure

structureSpec :: Spec
structureSpec = describe "parseCsv" $ do
  it "reads a header and records" $ do
    let tbl = table "a,b\n1,2\n3,4\n"
    columnNames tbl `shouldBe` ["a", "b"]
    texts tbl `shouldBe` [["a", "b"], ["1", "2"], ["3", "4"]]
    map recordLine (tableRecords tbl) `shouldBe` [2, 3]

  it "keeps the position of every cell, without the whitespace around it" $ do
    let tbl = table " a ,\tbb\n1,  22 \n"
    texts tbl `shouldBe` [["a", "bb"], ["1", "22"]]
    spans tbl `shouldBe` [[(1, 2, 3), (1, 6, 8)], [(2, 1, 2), (2, 5, 7)]]

  it "gives a record the span of its whole line" $
    map recordSpan (tableRecords (table "a\n 12 \n")) `shouldBe` [at 2 1 5]

  it "reads quoted cells; their spans include the quotes" $ do
    let tbl = table "\"x1\", \"x 2\" ,label\n1,2,3\n"
    columnNames tbl `shouldBe` ["x1", "x 2", "label"]
    take 1 (spans tbl) `shouldBe` [[(1, 1, 5), (1, 7, 12), (1, 14, 19)]]

  it "allows commas and doubled quotes inside quotes" $
    texts (table "\"a,b\",\"say \"\"hi\"\"\",\"\"\n1,2,3")
      `shouldBe` [["a,b", "say \"hi\"", ""], ["1", "2", "3"]]

  it "accepts \\r\\n line ends" $
    texts (table "a,b\r\n1,2\r\n") `shouldBe` [["a", "b"], ["1", "2"]]

  it "skips a byte-order mark, which still counts as a column" $ do
    let tbl = table "\xFEFFx,y\n1,2\n"
    columnNames tbl `shouldBe` ["x", "y"]
    take 1 (spans tbl) `shouldBe` [[(1, 2, 3), (1, 4, 5)]]

  it "tolerates blank lines at the end, and a missing final newline" $ do
    let expected = [["a"], ["1"]]
    mapM_
      (\t -> texts (table t) `shouldBe` expected)
      ["a\n1", "a\n1\n", "a\n1\n\n\n", "a\n1\n  \n\t\n", "a\r\n1\r\n\r\n"]

  it "reads a header alone as a table without records" $
    tableRecords (table "a,b\n") `shouldBe` []

  it "reads a trailing comma as an empty last cell" $
    texts (table "a,b,\n1,2,\n") `shouldBe` [["a", "b", ""], ["1", "2", ""]]

  it "reads an empty cell between commas" $ do
    let tbl = table "a,b,c\n1,,3\n"
    texts tbl `shouldBe` [["a", "b", "c"], ["1", "", "3"]]
    drop 1 (spans tbl) `shouldBe` [[(2, 1, 2), (2, 3, 3), (2, 4, 5)]]

  it "finds columns by name" $ do
    let tbl = table "a,b,a\n1,2,3\n"
    columnIndices "a" tbl `shouldBe` [0, 2]
    columnIndices "b" tbl `shouldBe` [1]
    columnIndices "c" tbl `shouldBe` []

-- ---------------------------------------------------------------------------
-- Problems

problemSpec :: Spec
problemSpec = describe "parseCsv problems" $ do
  it "has no header in an empty file" $
    mapM_ (\t -> problems t `shouldBe` [NoHeader]) ["", "\n", "\n\n", "  \n\t\n"]

  it "reports a blank line that is not at the end" $ do
    problems "a\n1\n\n2\n" `shouldBe` [BlankLine (at 3 1 1)]
    problems "\na\n1\n" `shouldBe` [BlankLine (at 1 1 1)]
    problems "a\n  \n1\n" `shouldBe` [BlankLine (at 2 1 3)]

  it "reports records with too few or too many cells" $ do
    let ps = problems "a,b\n1\n1,2\n1,2,3\n"
    [(recordLine r, length (recordCells h)) | WrongCellCount r h <- ps] `shouldBe` [(2, 2), (4, 2)]
    length ps `shouldBe` 2

  it "reports an unterminated quote, from the quote to the end of the line" $
    problems "a,b\n1, \"2\n" `shouldBe` [UnterminatedQuote (at 2 4 6)]

  it "reports text after a quoted cell" $
    problems "\"a\" b ,c\n1,2\n" `shouldBe` [TextAfterQuote (at 1 5 6)]

  it "reports every problem, in line order" $
    problems "a\n\"1\n\n\"2\"x\n"
      `shouldBe` [UnterminatedQuote (at 2 1 3), BlankLine (at 3 1 1), TextAfterQuote (at 4 4 5)]

  it "checks cell counts only when the lines themselves are fine" $
    problems "a,b\n1\n\"2\n" `shouldBe` [UnterminatedQuote (at 3 1 3)]

-- ---------------------------------------------------------------------------
-- Numbers

numberSpec :: Spec
numberSpec = describe "readNumber" $ do
  it "reads the forms CSV writers print" $
    mapM_
      (\(t, d) -> readNumber t `shouldBe` Right d)
      [ ("2", 2)
      , ("-0.5", -0.5)
      , ("+3", 3)
      , (".5", 0.5)
      , ("1.", 1)
      , ("007", 7)
      , ("1e-4", 1e-4)
      , ("3.0E+2", 300)
      , ("1E3", 1000)
      , ("-1.5e-3", -1.5e-3)
      , ("0", 0)
      ]

  it "rounds exactly once, like Haskell literals" $
    mapM_
      (\(t, d) -> readNumber t `shouldBe` Right d)
      [ ("0.1", 0.1)
      , ("0.30000000000000004", 0.30000000000000004)
      , ("1.7976931348623157e308", 1.7976931348623157e308)
      , ("4.9e-324", 5.0e-324)
      , ("123456789012345678901234567890", 123456789012345678901234567890)
      ]

  it "reads numbers too small for a Double as 0, quickly" $ do
    readNumber "1e-400" `shouldBe` Right 0
    readNumber "1e-99999999999999" `shouldBe` Right 0

  it "rejects what is not a decimal number" $
    mapM_
      (\t -> readNumber t `shouldBe` Left NotANumber)
      ["abc", "1.2.3", "1e", "e5", ".", "-", "--1", "+-1", "0x10", "1,5", "1 5", "1e+", "1.5e3.0", "١", "½", "1_000"]

  it "reports an empty cell" $
    readNumber "" `shouldBe` Left EmptyCell

  it "rejects non-finite numbers, quickly" $
    mapM_
      (\t -> readNumber t `shouldBe` Left NotFinite)
      ["nan", "NaN", "-inf", "+Inf", "Infinity", "1e999", "-1.8e308", "1e99999999999999"]

  prop "reads back `show` of every finite Double" $ \d ->
    not (isNaN d || isInfinite d) ==> readNumber (Text.pack (show (d :: Double))) === Right d

-- ---------------------------------------------------------------------------
-- Columns

columnSpec :: Spec
columnSpec = describe "numericColumns" $ do
  it "reads the given columns, in the given order, as often as listed" $
    numericColumns [2, 0, 0] (table "a,b,c\n1,2,3\n4,5,6\n") `shouldBe` Right [[3, 1, 1], [6, 4, 4]]

  it "ignores the other columns, whatever is in them" $
    numericColumns [1] (table "id,x\nbob,1.5\n\"x, y\",2\n") `shouldBe` Right [[1.5], [2]]

  it "reports every bad cell with its column, record by record" $
    numericColumns [0, 1] (table "a,b\n1,x\n,2\nnan,3\n")
      `shouldBe` Left
        [ BadCell 1 (Cell "x" (at 2 3 4)) NotANumber
        , BadCell 0 (Cell "" (at 3 1 1)) EmptyCell
        , BadCell 0 (Cell "nan" (at 4 1 4)) NotFinite
        ]

  it "gives no rows for a header alone" $
    numericColumns [0] (table "a\n") `shouldBe` Right []

-- ---------------------------------------------------------------------------
-- Round trip

-- | A numeric table and a way of writing it.
data Style = Style
  { quoteHeader :: Bool
  , padding :: [Text]
  , lineEnd :: Text
  , trailing :: Text
  }
  deriving (Show)

genStyle :: Gen Style
genStyle =
  Style
    <$> elements [False, True]
    <*> listOf1 (elements ["", " ", "  ", "\t"])
    <*> elements ["\n", "\r\n"]
    <*> elements ["", "\n", "\n\n", "\n \n"]

genName :: Gen Text
genName = Text.pack <$> ((:) <$> elements ['a' .. 'z'] <*> (chooseInt (0, 4) >>= \k -> vectorOf k (elements (['a' .. 'z'] <> ['0' .. '9']))))

-- | Write a table in a style: padding is taken in turn around the cells.
render :: Style -> [Text] -> [[Double]] -> Text
render st names rows = Text.intercalate (lineEnd st) (line (map name names) : map (line . map number) rows) <> trailing st
  where
    name n = if quoteHeader st then "\"" <> n <> "\"" else n
    number = Text.pack . show
    line cells = Text.intercalate "," [p <> c <> p | (c, p) <- zip cells (cycle (padding st))]

roundTripSpec :: Spec
roundTripSpec = describe "round trip" $
  it "reads random numeric tables back exactly" $
    forAll genCase $ \(st, names, rows) ->
      case parseCsv "t.csv" (render st names rows) of
        Left ps -> Left (show ps) === Right ()
        Right tbl ->
          (columnNames tbl, numericColumns [0 .. length names - 1] tbl) === (names, Right rows)
  where
    genCase = do
      st <- genStyle
      width <- chooseInt (1, 5)
      names <- vectorOf width genName
      height <- chooseInt (0, 8)
      rows <- vectorOf height (vectorOf width genNumber)
      pure (st, names, rows)
    genNumber :: Gen Double
    genNumber =
      oneof
        [ arbitrary `suchThat` (\d -> not (isNaN d || isInfinite d))
        , elements [0, -1, 0.1, 1e-300, 5.0e-324, 1.5e300, 1.7976931348623157e308, -2.5e-7]
        ]
