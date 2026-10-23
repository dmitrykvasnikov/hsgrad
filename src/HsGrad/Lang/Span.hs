{-# LANGUAGE OverloadedStrings #-}

-- | Source positions and spans for the @.hsg@ language.
--
-- A 'Span' is a region of one source file: a start position (inclusive)
-- and an end position (exclusive), each a 1-based line and column.
-- Columns count characters (Unicode code points), so a tab is one column
-- and so is @∂@; only the diagnostics renderer cares how wide a character
-- is on screen.
--
-- The representation is Core's own 'SrcSpan' (step 3): Core records spans
-- in node provenance (D4) and cannot import Lang (layer rule,
-- @00-architecture.md@ §2), so the type lives there and this module gives
-- it the language-facing API: building, merging, positions over text and
-- the conversion into node 'Provenance'.
-- See @docs/11-diagnostics.md@.
module HsGrad.Lang.Span
  ( -- * Positions
    Pos (..)
  , startPos
  , advance
  , advanceText

    -- * Spans
  , Span
  , SrcSpan (..)
  , mkSpan
  , pointSpan
  , textSpan
  , mergeSpan
  , mergeSpans
  , spanContains
  , spanIsEmpty
  , spanIsMultiLine

    -- * Showing
  , showPos
  , showLocation

    -- * Into Core
  , spanProvenance
  ) where

import Data.List.NonEmpty (NonEmpty (..))
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Graph (Pos (..), Provenance (..), SrcSpan (..))

-- ---------------------------------------------------------------------------
-- Positions

-- | The first position of every file: line 1, column 1.
startPos :: Pos
startPos = Pos 1 1

-- | The position just after a character.
--
-- A newline moves to column 1 of the next line; every other character,
-- tabs included, moves one column to the right.
advance :: Pos -> Char -> Pos
advance (Pos l _) '\n' = Pos (l + 1) 1
advance (Pos l c) _ = Pos l (c + 1)

-- | The position just after a piece of text that starts at the given
-- position. @advanceText startPos prefix@ is where the text after @prefix@
-- begins.
advanceText :: Pos -> Text -> Pos
advanceText = Text.foldl' advance

-- ---------------------------------------------------------------------------
-- Spans

-- | A region of a source file: 'spanStart' inclusive, 'spanEnd' exclusive.
--
-- The same type as Core's 'SrcSpan', so a span from the parser can be
-- handed to the 'HsGrad.Core.Graph.withSpan' of the builder as it is, and a
-- span found in a node's provenance can be put in a diagnostic as it is.
-- The derived 'Ord' compares file, then start, then end: the order in
-- which diagnostics are listed.
type Span = SrcSpan

-- | A span between two positions of a file, in either order.
mkSpan :: FilePath -> Pos -> Pos -> Span
mkSpan file a b = SrcSpan file (min a b) (max a b)

-- | The empty span at one position: "here, between two characters". The
-- diagnostics renderer still draws one caret for it.
pointSpan :: FilePath -> Pos -> Span
pointSpan file p = SrcSpan file p p

-- | The span of a piece of text that starts at the given position.
textSpan :: FilePath -> Pos -> Text -> Span
textSpan file p t = SrcSpan file p (advanceText p t)

-- | The smallest span that covers both spans: from the earlier start to the
-- later end. A binary expression's span is the merge of its operands'.
--
-- Merging only makes sense within one file; the result keeps the file of
-- the left span.
mergeSpan :: Span -> Span -> Span
mergeSpan (SrcSpan file s1 e1) (SrcSpan _ s2 e2) =
  SrcSpan file (min s1 s2) (max e1 e2)

-- | The smallest span that covers all of them.
mergeSpans :: NonEmpty Span -> Span
mergeSpans (s :| ss) = foldl mergeSpan s ss

-- | Does the first span cover the second? Every span covers itself, and an
-- empty span covers only empty spans at its position.
spanContains :: Span -> Span -> Bool
spanContains (SrcSpan f1 s1 e1) (SrcSpan f2 s2 e2) =
  f1 == f2 && s1 <= s2 && e2 <= e1

-- | Does the span cover no characters at all?
spanIsEmpty :: Span -> Bool
spanIsEmpty sp = spanStart sp == spanEnd sp

-- | Does the span cover characters on more than one line?
--
-- A span that ends at column 1 of a line ends with the newline before it,
-- so it does not reach that line.
spanIsMultiLine :: Span -> Bool
spanIsMultiLine (SrcSpan _ (Pos l1 _) (Pos l2 c2))
  | c2 == 1 = l2 - 1 > l1
  | otherwise = l2 > l1

-- ---------------------------------------------------------------------------
-- Showing

-- | @line:column@, as in @14:17@.
showPos :: Pos -> Text
showPos (Pos l c) = Text.pack (show l <> ":" <> show c)

-- | Where a span starts, as compilers print it: @file:line:column@, as in
-- @examples/moons.hsg:14:17@. Editors and terminals recognise this form
-- and jump to the position.
showLocation :: Span -> Text
showLocation sp = Text.pack (spanFile sp) <> ":" <> showPos (spanStart sp)

-- ---------------------------------------------------------------------------
-- Into Core

-- | The provenance of a node created by the source text in the span: the
-- span and no label.
--
-- The compiler (step 14) normally uses 'HsGrad.Core.Graph.withSpan', which
-- gives every node created inside it this provenance; 'spanProvenance' is
-- for the places that build a 'Provenance' value directly.
spanProvenance :: Span -> Provenance
spanProvenance sp = Provenance (Just sp) Nothing
