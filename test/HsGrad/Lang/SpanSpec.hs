{-# LANGUAGE OverloadedStrings #-}

-- | Tests for "HsGrad.Lang.Span".
--
-- * Positions: advancing over characters, newlines, tabs and Unicode.
-- * Spans: building, merging (with algebraic properties), containment,
--   emptiness, multi-line detection.
-- * Showing locations, and the conversion into Core provenance: a Lang span
--   given to the builder ends up on the nodes.
module HsGrad.Lang.SpanSpec
  ( spec
  ) where

import Data.List (sort)
import Data.List.NonEmpty (NonEmpty (..))
import Data.Text (Text)
import HsGrad.Core.Graph (Node (..), Provenance (..), build, input, lookupNode, withSpan)
import HsGrad.Lang.Span
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck (Arbitrary (..), Gen, choose, (===))

spec :: Spec
spec = do
  describe "positions" positionSpec
  describe "spans" spanSpec
  describe "merging" mergeSpec
  describe "showing and Core" coreSpec

-- ---------------------------------------------------------------------------
-- Generators

-- | Positions in a small file, so that random spans often overlap.
newtype APos = APos Pos
  deriving (Show)

instance Arbitrary APos where
  arbitrary = APos <$> genPos

genPos :: Gen Pos
genPos = Pos <$> choose (1, 6) <*> choose (1, 12)

-- | Spans of one file.
newtype ASpan = ASpan Span
  deriving (Show)

instance Arbitrary ASpan where
  arbitrary = ASpan <$> (mkSpan "a.hsg" <$> genPos <*> genPos)

-- ---------------------------------------------------------------------------
-- Positions

positionSpec :: Spec
positionSpec = do
  it "starts at line 1, column 1" $
    startPos `shouldBe` Pos 1 1

  it "moves one column per character" $
    advanceText startPos "let" `shouldBe` Pos 1 4

  it "moves to column 1 of the next line after a newline" $
    advanceText startPos "ab\ncd" `shouldBe` Pos 2 3

  it "counts a tab as one column (the renderer expands it)" $
    advanceText startPos "\tx" `shouldBe` Pos 1 3

  it "counts a Unicode character as one column" $
    advanceText startPos "\"møø\"" `shouldBe` Pos 1 6

  it "counts \\r as a column, so CRLF files have the same line numbers" $
    advanceText startPos "a\r\nb" `shouldBe` Pos 2 2

  prop "advancing over a concatenation is advancing twice" $ \(APos p) s t ->
    let (a, b) = (text s, text t)
     in advanceText p (a <> b) === advanceText (advanceText p a) b
  where
    -- Random strings as text of letters and newlines.
    text :: String -> Text
    text = foldMap (\c -> if c == '\n' then "\n" else "x")

-- ---------------------------------------------------------------------------
-- Spans

spanSpec :: Spec
spanSpec = do
  it "mkSpan puts the positions in order" $
    mkSpan "f" (Pos 3 1) (Pos 1 5) `shouldBe` SrcSpan "f" (Pos 1 5) (Pos 3 1)

  it "textSpan covers the text" $
    textSpan "f" (Pos 2 3) "W2 @ h1" `shouldBe` SrcSpan "f" (Pos 2 3) (Pos 2 10)

  it "pointSpan is empty" $ do
    spanIsEmpty (pointSpan "f" (Pos 4 1)) `shouldBe` True
    spanIsEmpty (textSpan "f" (Pos 4 1) "x") `shouldBe` False

  it "knows multi-line spans, but not one that ends at column 1 of the next line" $ do
    spanIsMultiLine (textSpan "f" startPos "ab") `shouldBe` False
    spanIsMultiLine (textSpan "f" startPos "ab\ncd") `shouldBe` True
    spanIsMultiLine (textSpan "f" startPos "ab\n") `shouldBe` False
    spanIsMultiLine (textSpan "f" startPos "ab\n\n") `shouldBe` True

  prop "every span contains itself" $ \(ASpan s) ->
    spanContains s s

  prop "contains exactly the spans between its ends" $ \(APos p1) (APos p2) (APos p3) (APos p4) ->
    case sort [p1, p2, p3, p4] of
      [a, b, c, d] ->
        let outer = mkSpan "a.hsg" a d
            inner = mkSpan "a.hsg" b c
         in spanContains outer inner
              && spanContains inner outer == (a == b && c == d)
      _ -> False

  it "does not contain a span of another file" $
    spanContains (mkSpan "a" (Pos 1 1) (Pos 9 9)) (mkSpan "b" (Pos 2 2) (Pos 2 3))
      `shouldBe` False

-- ---------------------------------------------------------------------------
-- Merging

mergeSpec :: Spec
mergeSpec = do
  it "covers a binary expression from its operands" $
    mergeSpan (textSpan "f" (Pos 1 9) "a") (textSpan "f" (Pos 1 13) "b")
      `shouldBe` SrcSpan "f" (Pos 1 9) (Pos 1 14)

  it "keeps the file of the left span" $
    spanFile (mergeSpan (pointSpan "left" startPos) (pointSpan "right" startPos))
      `shouldBe` "left"

  prop "contains both spans" $ \(ASpan a) (ASpan b) ->
    let m = mergeSpan a b in spanContains m a && spanContains m b

  prop "is the smallest such span: from the earlier start to the later end" $ \(ASpan a) (ASpan b) ->
    mergeSpan a b === mkSpan "a.hsg" (min (spanStart a) (spanStart b)) (max (spanEnd a) (spanEnd b))

  prop "is commutative" $ \(ASpan a) (ASpan b) ->
    mergeSpan a b === mergeSpan b a

  prop "is associative" $ \(ASpan a) (ASpan b) (ASpan c) ->
    mergeSpan a (mergeSpan b c) === mergeSpan (mergeSpan a b) c

  prop "is idempotent" $ \(ASpan a) ->
    mergeSpan a a === a

  prop "mergeSpans folds mergeSpan" $ \(ASpan a) as ->
    let ss = [s | ASpan s <- as]
     in mergeSpans (a :| ss) === foldr mergeSpan a ss

-- ---------------------------------------------------------------------------
-- Showing and Core

coreSpec :: Spec
coreSpec = do
  it "shows positions and locations like compilers do" $ do
    showPos (Pos 14 17) `shouldBe` "14:17"
    showLocation (textSpan "examples/moons.hsg" (Pos 14 17) "W2 @ h1")
      `shouldBe` "examples/moons.hsg:14:17"

  it "converts to a provenance with the span and no label" $ do
    let sp = textSpan "f.hsg" (Pos 2 7) "W1"
    spanProvenance sp `shouldBe` Provenance (Just sp) Nothing

  it "is the span the builder records on nodes" $ do
    let sp = textSpan "f.hsg" (Pos 1 7) "a"
    case build (withSpan sp (input "a")) of
      Left err -> fail (show err)
      Right (i, g) ->
        fmap nodeProv (lookupNode i g) `shouldBe` Just (spanProvenance sp)

  it "has orderings for sorting: file, then start, then end" $ do
    let s1 = mkSpan "a" (Pos 2 1) (Pos 2 5)
        s2 = mkSpan "a" (Pos 2 1) (Pos 3 1)
        s3 = mkSpan "a" (Pos 10 1) (Pos 10 2)
        s4 = mkSpan "b" (Pos 1 1) (Pos 1 2)
    [s1, s2, s3, s4] `shouldBe` foldr insertSorted [] [s3, s1, s4, s2]
  where
    insertSorted x xs = filter (< x) xs <> [x] <> filter (>= x) xs
