{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | The parser for @.hsg@ files: text in, a 'Program' (every node
-- annotated with its 'Span') or syntax errors out.
--
-- It follows the grammar of @01-language.md@ §9, built from megaparsec
-- combinators, with the operator levels of §5.1 handed to
-- @makeExprParser@. There is no separate lexer: each token parser
-- ('lexeme') reads its token and then the whitespace and comments after
-- it, so every parser starts at the beginning of a token.
--
-- Syntax errors are 'Diagnostic's with the codes E0101–E0105, never
-- megaparsec's own messages. The parser does not stop at the first one:
--
-- * Problems inside a token that still leave a usable token (a malformed
--   number, an unterminated comment or string, a variable exponent) are
--   /registered/ and parsing goes on.
-- * Any other error abandons the current declaration; the parser skips to
--   the next point where a declaration can start and continues there.
--
-- So one run reports every independent syntax error of a file.
-- See @docs/12-parser.md@.
module HsGrad.Lang.Parser
  ( -- * Parsing
    parseProgram
  , parseExpr
  , parseLiteral
  , parseType
  ) where

import Control.Monad (unless, void, when)
import Control.Monad.Combinators.Expr qualified as E
import Data.Char (isDigit, isSpace)
import Data.Function ((&))
import Data.List.NonEmpty (NonEmpty (..))
import Data.List.NonEmpty qualified as NonEmpty
import Data.Maybe (catMaybes)
import Data.Ord (comparing)
import Data.Ratio (numerator, (%))
import Data.Semigroup (sconcat)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Lang.Diagnostic
  ( Code (..)
  , Diagnostic (..)
  , diagSpan
  , diagnostic
  , withHint
  , withNote
  , withSecondary
  )
import HsGrad.Lang.Span (Pos (..), Span, SrcSpan (..), advanceText, mergeSpan, pointSpan, startPos, textSpan)
import HsGrad.Lang.Syntax
import Text.Megaparsec hiding (Pos)
import Text.Megaparsec.Char (char, space1, string)

-- ---------------------------------------------------------------------------
-- Entry points

-- | Parse a whole file. The file name is only used in spans; use the name
-- the diagnostics should show.
--
-- On failure, every syntax error found, in source order.
parseProgram :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Program Span)
parseProgram = runHsg program

-- | Parse one expression, the whole text (for tests and the REPL-like
-- uses of later steps).
parseExpr :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Expr Span)
parseExpr = runHsg (expr <* eof)

-- | Parse one literal, the whole text: the value of @--set x=[1,2,3]@.
parseLiteral :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Literal Span)
parseLiteral = runHsg (literal <* eof)

-- | Parse one type, the whole text.
parseType :: FilePath -> Text -> Either (NonEmpty Diagnostic) (Type Span)
parseType = runHsg (typeP <* eof)

-- ---------------------------------------------------------------------------
-- The parser monad

-- | Our parsers: megaparsec over 'Text', with our own errors.
type Parser = Parsec Problem Text

-- | A syntax error that the parser itself describes, ready to report.
--
-- Megaparsec has two kinds of errors. A /trivial/ error is "unexpected
-- this, expected one of those"; megaparsec assembles it from the labels of
-- the parsers that failed, and 'unexpectedToken' turns it into an E0101
-- diagnostic at the end. A /fancy/ error carries a value of a type we
-- choose. Ours is the finished diagnostic, for errors whose message only
-- the parser knows how to write (E0102–E0105, and some E0101 with hints).
newtype Problem = Problem Diagnostic
  deriving (Eq, Ord, Show)

instance ShowErrorComponent Problem where
  showErrorComponent (Problem d) = Text.unpack (diagMessage d)

-- | Run a parser on a whole text, after skipping leading whitespace.
--
-- Megaparsec counts a tab as 8 columns by default; we set it to 1 so that
-- columns count characters, as 'Span's do (@docs/11-diagnostics.md@ §4.3).
runHsg :: Parser a -> FilePath -> Text -> Either (NonEmpty Diagnostic) a
runHsg p file src = case snd (runParser' (whitespace *> p) initial) of
  Right x -> Right x
  Left bundle -> Left (bundleDiagnostics file src bundle)
  where
    initial =
      State
        { stateInput = src
        , stateOffset = 0
        , statePosState =
            PosState
              { pstateInput = src
              , pstateOffset = 0
              , pstateSourcePos = initialPos file
              , pstateTabWidth = mkPos 1
              , pstateLinePrefix = ""
              }
        , stateParseErrors = []
        }

-- | Report an error and go on parsing.
report :: Diagnostic -> Parser ()
report d = registerFancyFailure (Set.singleton (ErrorCustom (Problem d)))

-- | Fail with an error.
failWith :: Diagnostic -> Parser a
failWith = customFailure . Problem

-- ---------------------------------------------------------------------------
-- Positions and tokens

toPos :: SourcePos -> Pos
toPos p = Pos (unPos (sourceLine p)) (unPos (sourceColumn p))

-- | Run a parser and return the span of the text it consumed.
spanned :: Parser a -> Parser (Span, a)
spanned p = do
  start <- getSourcePos
  x <- p
  end <- getSourcePos
  pure (SrcSpan (sourceName start) (toPos start) (toPos end), x)

-- | The empty span at the current position.
here :: Parser Span
here = do
  p <- getSourcePos
  pure (pointSpan (sourceName p) (toPos p))

-- | The empty span right after a span: where something was expected.
after :: Span -> Span
after sp = sp {spanStart = spanEnd sp}

-- | Whitespace and comments: line comments @-- …@ and nestable block
-- comments @{- … -}@. Hidden, so it never shows up in "expected …".
whitespace :: Parser ()
whitespace = hidden (skipMany (void space1 <|> lineComment <|> blockComment))
  where
    lineComment = void (string "--" *> takeWhileP Nothing (/= '\n'))

-- | A block comment. Comments nest, so we count: @{-@ goes one level
-- deeper, @-}@ one level up, and the comment ends at level 0.
--
-- The end of the file inside a comment is reported (E0102) but not
-- fatal: the comment simply ends there, and parsing goes on at the end of
-- the file.
blockComment :: Parser ()
blockComment = do
  (open, _) <- spanned (string "{-")
  nested open (1 :: Int) 1
  where
    nested open depth deepest = do
      void (takeWhileP Nothing (\c -> c /= '-' && c /= '{'))
      choice
        [ string "-}" *> unless (depth == 1) (nested open (depth - 1) deepest)
        , string "{-" *> nested open (depth + 1) (max deepest (depth + 1))
        , anySingle *> nested open depth deepest
        , eof *> unterminated open deepest
        ]
    unterminated open deepest = do
      end <- here
      report $
        diagnostic E0102 "unterminated block comment" open "this comment is never closed"
          & withSecondary end "the file ends here"
          & (if deepest > 1 then withNote "comments nest: each `{-` inside this comment needs its own `-}`" else id)
          & withHint "close it with `-}`"

-- | A token: run the parser, note its span, skip the whitespace after it.
-- The span ends where the token ends, before the whitespace.
lexeme :: Parser a -> Parser (Span, a)
lexeme p = spanned p <* whitespace

-- | A fixed piece of punctuation or an operator, such as @(@ or @<-@.
symbol :: Text -> Parser Span
symbol t = fst <$> lexeme (string t)

-- | The word (keyword or name) that starts here, if one does. It only
-- looks; nothing is consumed.
--
-- Keywords and names are told apart by looking at the whole word first
-- and only then deciding. A parser that read @in@ and then checked that no
-- letter follows would, on @input@, fail /after/ @in@, and megaparsec
-- would report the error there, in the middle of the word.
peekWord :: Parser (Maybe Text)
peekWord = wordAt <$> getInput

wordAt :: Text -> Maybe Text
wordAt t = case Text.uncons t of
  Just (c, _) | isIdentStart c -> Just (Text.takeWhile isIdentChar t)
  _ -> Nothing

-- | Consume the word that 'peekWord' saw.
word :: Parser Text
word = takeWhile1P Nothing isIdentChar

-- | A keyword (or type name), as a whole word: @in@, but not the start of
-- @input@.
keyword :: Text -> Parser Span
keyword w = label (Text.unpack (quote w)) $ do
  next <- peekWord
  if next == Just w then fst <$> lexeme word else empty

-- | A name: a word that is not reserved.
identifier :: Parser (Ident Span)
identifier = label "a name" $ do
  next <- peekWord
  case next of
    Just w | not (isReserved w) -> do
      (sp, _) <- lexeme word
      pure (Ident sp w)
    _ -> empty

-- | Is the text at the current position the start of the given text?
startsWith :: Text -> Parser Bool
startsWith t = Text.isPrefixOf t <$> getInput

quote :: Text -> Text
quote t = "`" <> t <> "`"

-- ---------------------------------------------------------------------------
-- Numbers

-- | A number token as read, before it becomes a 'Num', a 'Number' or a
-- 'Nat'.
data NumberToken = NumberToken
  { ntSpan :: !Span
  , ntText :: !Text
  , ntValue :: !Rational
  , ntNatural :: !Bool
  -- ^ Written as digits only: usable as an index or dimension.
  }

-- | Does a number start here? A digit, or a @.@ followed by a digit (not
-- valid, but clearly meant as a number: @.5@).
startsNumber :: Text -> Bool
startsNumber t = case Text.uncons t of
  Just (c, rest) -> isDigit c || (c == '.' && maybe False (isDigit . fst) (Text.uncons rest))
  Nothing -> False

-- | How long the number-like run of characters at the start of the text
-- is: digits, letters, @_@, @'@, @.@, and a sign right after the @e@ of an
-- exponent. Reading the whole run, not just the valid prefix, means that
-- @2x@ or @1.2.3@ is one malformed number (E0103) instead of a number
-- followed by a puzzling second token.
numberLength :: Text -> Int
numberLength = go ' ' ' ' 0 . Text.unpack
  where
    go beforePrev prev n (c : cs)
      | isIdentChar c || c == '.' || (c `elem` ("+-" :: String) && prev `elem` ("eE" :: String) && (isDigit beforePrev || beforePrev == '.')) =
          go prev c (n + 1) cs
    go _ _ n _ = n

-- | A number token (@2@, @2.5@, @1e-4@, @3.0E2@). A malformed one is
-- reported (E0103) and read as 0, and parsing goes on.
numberToken :: Parser NumberToken
numberToken = label "a number" $ do
  input <- getInput
  unless (startsNumber input) empty
  (sp, txt) <- lexeme (takeP Nothing (numberLength input))
  case readNumber txt of
    Right (v, isNat) -> pure (NumberToken sp txt v isNat)
    Left problem -> do
      report (numberDiagnostic sp txt problem)
      pure (NumberToken sp txt 0 True)

-- | What can be wrong with a number token.
data NumberProblem
  = LeadingDot
  | MissingFraction Text
  | MissingExponentDigits
  | TrailingText Text Text
  | TooLarge
  | TooSmall

-- | Read a number token exactly:
-- @digits ( "." digits )? ( ("e" | "E") ("+" | "-")? digits )?@.
--
-- The value is computed as a 'Rational' from the digits, never through a
-- 'Double', so @0.1@ is exactly @1/10@. The second component says whether
-- the token is a plain natural number.
readNumber :: Text -> Either NumberProblem (Rational, Bool)
readNumber txt = do
  let (intDigits, afterInt) = Text.span isDigit txt
  when (Text.null intDigits) (Left LeadingDot)
  (fracDigits, afterFrac) <- case Text.uncons afterInt of
    Just ('.', rest) ->
      let (f, rest') = Text.span isDigit rest
       in if Text.null f then Left (MissingFraction intDigits) else Right (f, rest')
    _ -> Right ("", afterInt)
  (expo, afterExp) <- case Text.uncons afterFrac of
    Just (e, rest) | e == 'e' || e == 'E' -> do
      let (sign, rest') = case Text.uncons rest of
            Just ('-', r) -> (-1, r)
            Just ('+', r) -> (1, r)
            _ -> (1, rest)
          (ds, rest'') = Text.span isDigit rest'
      when (Text.null ds) (Left MissingExponentDigits)
      Right (sign * readInteger ds, rest'')
    _ -> Right (0, afterFrac)
  unless (Text.null afterExp) $
    Left (TrailingText (Text.take (Text.length txt - Text.length afterExp) txt) afterExp)
  let mantissa = readInteger (intDigits <> fracDigits)
      scale = expo - toInteger (Text.length fracDigits)
      isNat = Text.null fracDigits && Text.length afterInt == Text.length afterFrac && Text.null afterFrac
  -- Huge exponents are out of range anyway; refuse them before computing
  -- 10^expo, which could take a very long time.
  when (mantissa /= 0 && abs expo > 1000) $ Left (if expo > 0 then TooLarge else TooSmall)
  let value = (mantissa % 1) * (10 ^^ scale)
      asDouble = fromRational value :: Double
  when (isInfinite asDouble) (Left TooLarge)
  when (value /= 0 && asDouble == 0) (Left TooSmall)
  Right (value, isNat)
  where
    readInteger = Text.foldl' (\acc d -> acc * 10 + toInteger (fromEnum d - fromEnum '0')) 0

-- | The E0103 diagnostic for a malformed number.
numberDiagnostic :: Span -> Text -> NumberProblem -> Diagnostic
numberDiagnostic sp txt = \case
  LeadingDot ->
    malformed "a number can't start with `.`" & withHint ("write " <> quote ("0" <> txt))
  MissingFraction ds ->
    malformed "digits must follow the decimal point"
      & withHint ("write " <> quote (ds <> ".0") <> " or " <> quote ds)
  MissingExponentDigits ->
    malformed "the exponent has no digits" & withHint "an exponent looks like `1e-4` or `3.0E2`"
  TrailingText number rest
    | Just ('.', _) <- Text.uncons rest ->
        malformed "a number has at most one decimal point"
    | otherwise ->
        malformed (quote rest <> " can't follow a number")
          & withHint ("to multiply, write " <> quote (number <> " * " <> rest))
  TooLarge ->
    diagnostic E0103 ("number " <> quote txt <> " is too large") sp "the largest number is about 1.8e308"
  TooSmall ->
    diagnostic E0103 ("number " <> quote txt <> " is too small") sp "it would be rounded to 0"
  where
    malformed = diagnostic E0103 ("malformed number literal " <> quote txt) sp

-- | A natural number: an index or a dimension. Other numbers are reported
-- and read as 0.
natural :: Text -> Parser (Nat Span)
natural what = label "a natural number" $ do
  t <- numberToken
  let v = ntValue t
  if
    | not (ntNatural t) -> do
        report $
          diagnostic E0101 ("expected a natural number, found " <> quote (ntText t)) (ntSpan t) (what <> " must be a whole number, like `0` or `16`")
        pure (Nat (ntSpan t) 0)
    | v > 10 ^ (9 :: Int) -> do
        report (diagnostic E0103 ("number " <> quote (ntText t) <> " is too large for " <> what) (ntSpan t) "")
        pure (Nat (ntSpan t) 0)
    | otherwise -> pure (Nat (ntSpan t) (fromInteger (numerator v)))

-- | A number with an optional minus sign: @-1@, @0.5@.
signedNumber :: Parser (Number Span)
signedNumber = do
  minus <- optional (symbol "-")
  t <- numberToken
  pure $ case minus of
    Nothing -> Number (ntSpan t) (ntValue t)
    Just m -> Number (mergeSpan m (ntSpan t)) (negate (ntValue t))

-- ---------------------------------------------------------------------------
-- Strings

-- | A string literal: @"…"@ with the escapes @\\"@ and @\\\\@. A string
-- ends on its line; reaching the end of the line or file first is
-- reported (E0102), and the string ends there.
stringLit :: Parser (StringLit Span)
stringLit = label "a string" $ do
  start <- getSourcePos
  (open, _) <- spanned (char '"')
  value <- contents open []
  end <- getSourcePos
  whitespace
  pure (StringLit (SrcSpan (sourceName start) (toPos start) (toPos end)) (Text.concat (reverse value)))
  where
    contents open acc = do
      piece <- takeWhileP Nothing (\c -> c /= '"' && c /= '\\' && c /= '\n')
      let acc' = piece : acc
      choice
        [ acc' <$ char '"'
        , do
            (sp, c) <- spanned (char '\\' *> optional (satisfy (/= '\n')))
            case c of
              Just e | e == '"' || e == '\\' -> contents open (Text.singleton e : acc')
              Just e -> do
                report $
                  diagnostic E0101 ("unknown escape " <> quote (Text.pack ['\\', e]) <> " in a string") sp "only `\\\"` and `\\\\` can be escaped"
                contents open (Text.singleton e : acc')
              Nothing -> acc' <$ unterminated open
        , acc' <$ unterminated open
        ]
    unterminated open = do
      end <- here
      fileEnds <- Text.null <$> getInput
      report $
        diagnostic E0102 "unterminated string" open "this string is never closed"
          & withSecondary end (if fileEnds then "the file ends here" else "the line ends here")
          & withHint "close it with `\"`; a string can't continue on the next line"

-- ---------------------------------------------------------------------------
-- Types

-- | @Real@, @Vec d@ or @Mat r c@.
typeP :: Parser (Type Span)
typeP =
  label "a type" $
    choice
      [ TReal <$> keyword "Real"
      , do
          kw <- keyword "Vec"
          d <- dimension
          pure (TVec (mergeSpan kw (dimSpan d)) d)
      , do
          kw <- keyword "Mat"
          r <- dimension
          c <- dimension
          pure (TMat (mergeSpan kw (dimSpan c)) r c)
      ]
  where
    dimension = label "a dimension" (DimNat <$> natural "a dimension" <|> DimVar <$> identifier)
    dimSpan = \case
      DimNat n -> natAnn n
      DimVar v -> identAnn v

-- | An optional annotation @: t@.
annotation :: Parser (Maybe (Type Span))
annotation = optional (symbol ":" *> typeP)

-- ---------------------------------------------------------------------------
-- Literals

-- | A literal: a signed number, or @[literal, …]@.
literal :: Parser (Literal Span)
literal = label "a literal" (LitNumber <$> signedNumber <|> vectorLiteral)
  where
    vectorLiteral = do
      open <- symbol "["
      ls <- sepBy1 literal (symbol ",")
      close <- symbol "]"
      pure (LitVector (mergeSpan open close) ls)

-- ---------------------------------------------------------------------------
-- Expressions

-- | An expression: @let x = e in body@, or an operator expression.
--
-- A @let@ inside an expression must have its @in@. When it is missing the
-- error (E0104) points just after the bound expression, where @in@ should
-- be, rather than at whatever token comes next.
expr :: Parser (Expr Span)
expr = letExpr <|> operatorExpr
  where
    letExpr = do
      kw <- hidden (keyword "let")
      x <- identifier
      t <- annotation
      _ <- symbol "="
      bound <- expr
      hasIn <- optional (keyword "in")
      case hasIn of
        Just _ -> do
          body <- expr
          pure (Let (mergeSpan kw (exprAnn body)) x t bound body)
        Nothing ->
          failWith $
            diagnostic E0104 "`let` expression without `in`" (after (exprAnn bound)) "expected `in` here"
              & withSecondary kw "this `let` is inside an expression"
              & withHint
                ( "inside an expression write `let " <> identName x <> " = … in body`;\n"
                    <> "only a top-level `let` needs no `in`"
                )

-- | Levels 3–6 of the precedence table: unary minus, then @\@@, then
-- @* /@, then @+ -@, all binary operators left-associative. Levels 1–2
-- (atoms, indexing, powers) are 'powerExpr'.
operatorExpr :: Parser (Expr Span)
operatorExpr =
  E.makeExprParser
    powerExpr
    [ [E.Prefix (foldr1 (.) <$> some negation)]
    , [E.InfixL (binary MatMul)]
    , [E.InfixL (binary Times), E.InfixL (binary Divide)]
    , [E.InfixL (binary Plus), E.InfixL (binary Minus)]
    ]
  where
    -- `some`, so that `- -x` works: makeExprParser applies one prefix
    -- operator per level.
    negation = do
      sp <- hidden (symbol "-")
      pure (\e -> Negate (mergeSpan sp (exprAnn e)) e)
    binary op = do
      _ <- label "an operator" (symbol (prettyOperator op))
      pure (\l r -> Binary (mergeSpan (exprAnn l) (exprAnn r)) op l r)

-- | Level 2: @e ^ k@ with a constant exponent, not associative.
--
-- A non-constant exponent (E0105) and a second @^@ are reported and
-- parsing goes on, so the rest of the expression is still checked.
powerExpr :: Parser (Expr Span)
powerExpr = postfixExpr >>= raise
  where
    raise base = do
      caret <- optional (label "an operator" (symbol "^"))
      case caret of
        Nothing -> pure base
        Just _ -> do
          k <- exponentOf base
          let e = Power (mergeSpan (exprAnn base) (numberAnn k)) base k
          again <- startsWith "^"
          when again $ do
            sp <- fst <$> spanned (lookAhead (string "^"))
            report $
              diagnostic E0101 "unexpected `^` after a power" sp "a power of a power needs parentheses"
                & withHint "`^` is not associative: write `(x ^ 2) ^ 3`"
          raise e

    exponentOf base = label "a number" $ do
      minus <- optional (symbol "-")
      constant <- startsNumber <$> getInput
      if constant
        then do
          t <- numberToken
          pure $ case minus of
            Nothing -> Number (ntSpan t) (ntValue t)
            Just m -> Number (mergeSpan m (ntSpan t)) (negate (ntValue t))
        else do
          e <- postfixExpr
          let k = maybe e (\m -> Negate (mergeSpan m (exprAnn e)) e) minus
              suggestion = Call (exprAnn k) (Ident (exprAnn k) "exp") [Binary (exprAnn k) Times k (Call (exprAnn base) (Ident (exprAnn base) "log") [base])]
          report $
            diagnostic E0105 "exponent must be a numeric constant" (exprAnn k) "this exponent is not a number"
              & withHint "`^` takes a number: `x ^ 2`, `x ^ -1`, `x ^ 0.5`"
              & withHint ("for a variable exponent write " <> quote (prettyExpr suggestion) <> " (needs a positive base)")
          pure (Number (exprAnn k) 1)

-- | Level 1: an atom followed by any number of indexes @[i]@ / @[i, j]@.
postfixExpr :: Parser (Expr Span)
postfixExpr = atom >>= indexes
  where
    indexes e = do
      more <- startsWith "["
      if more then index e >>= indexes else pure e
    index e = do
      _ <- symbol "["
      i <- natural "an index"
      j <- optional (symbol "," *> natural "an index")
      close <- symbol "]"
      pure (Index (mergeSpan (exprAnn e) close) e i j)

-- | An atom: a number, a name or call, @( e )@, a vector literal or a
-- comprehension.
atom :: Parser (Expr Span)
atom = label "an expression" (choice [numberAtom, nameAtom, parenAtom, bracketAtom])
  where
    numberAtom = do
      t <- numberToken
      pure (Num (ntSpan t) (ntValue t))

    -- The span of a parenthesised expression includes the parentheses, so
    -- that the span of `(x - y) ^ 2` starts at `(`.
    parenAtom = do
      open <- symbol "("
      e <- expr
      close <- symbol ")"
      pure (setExprAnn (mergeSpan open close) e)

    -- `[` e: a vector literal if `,` or `]` follows, a comprehension if `|`.
    bracketAtom = do
      open <- symbol "["
      first <- expr
      bar <- optional (symbol "|")
      case bar of
        Just _ -> comprehension open first
        Nothing -> do
          rest <- many (symbol "," *> expr)
          close <- symbol "]"
          pure (Vector (mergeSpan open close) (first : rest))

-- | A name, possibly called: @x@, @f(a, b)@, or @mean [ e | x <- d ]@.
--
-- A @[@ right after a name is either an index (@v[0]@) or a comprehension
-- that is the name's one argument (@sum [ … | … ]@). It is an index when
-- a number follows the @[@, and after it @]@ or @,@.
nameAtom :: Parser (Expr Span)
nameAtom = do
  f <- identifier
  input <- getInput
  case Text.uncons input of
    Just ('(', _) -> call f
    Just ('[', _) -> do
      isIndex <- indexAhead
      if isIndex then pure (Var (identAnn f) f) else comprehensionCall f
    _ -> do
      juxtaposed f
      pure (Var (identAnn f) f)
  where
    call f = do
      _ <- symbol "("
      args <- sepBy expr (symbol ",")
      close <- symbol ")"
      pure (Call (mergeSpan (identAnn f) close) f args)

    -- Only looks ahead. Whatever this sees (even an unterminated comment)
    -- is parsed again for real afterwards.
    indexAhead = option False (True <$ try (lookAhead (symbol "[" *> numberRun *> whitespace *> (char ']' <|> char ','))))
    numberRun = do
      input <- getInput
      unless (startsNumber input) empty
      takeP Nothing (numberLength input)

    comprehensionCall f = do
      open <- symbol "["
      body <- expr
      closes <- startsWith "]"
      continues <- startsWith ","
      let fn = identName f
          neither = "`[` after " <> quote fn <> " starts neither an index nor a comprehension"
      -- `v[i]`: most likely an index that isn't a number.
      when closes $
        failWith $
          diagnostic E0101 neither (exprAnn body) "an index must be a natural number"
            & withHint ("indices are number literals, like " <> quote (fn <> "[0]") <> "; they can't be computed")
            & withHint ("a call with a vector argument needs parentheses: " <> quote (fn <> "([…])"))
      -- `sum [a, b]`: most likely a vector argument.
      when continues $
        failWith $
          diagnostic E0101 neither (exprAnn body) "a comprehension needs `|` here"
            & withHint ("a call with a vector argument needs parentheses: " <> quote (fn <> "([…])"))
      _ <- symbol "|"
      comp <- comprehension open body
      pure (Call (mergeSpan (identAnn f) (exprAnn comp)) f [comp])

    -- `f x` on one line: a call without parentheses, or a forgotten `*`.
    -- After a name, a valid program never has another name or a number,
    -- so this cannot reject anything valid.
    juxtaposed f = do
      input <- getInput
      pos <- toPos <$> getSourcePos
      let sameLine = posLine pos == posLine (spanEnd (identAnn f))
      case nextToken input of
        Just (tok, desc)
          | sameLine && startsArgument tok -> do
              let sp = textSpan (spanFile (identAnn f)) pos tok
                  fn = identName f
              failWith $
                diagnostic E0101 ("unexpected " <> desc <> " after " <> quote fn) sp "a call needs parentheses"
                  & withHint ("to call " <> quote fn <> ", write " <> quote (fn <> "(" <> tok <> ")"))
                  & withHint ("to multiply, write " <> quote (fn <> " * " <> tok))
        _ -> pure ()
    startsArgument tok = startsNumber tok || maybe False (not . isReserved) (wordAt tok)

-- | The rest of a comprehension, after @[ body |@: @pattern <- source ]@.
comprehension :: Span -> Expr Span -> Parser (Expr Span)
comprehension open body = do
  p <- pattern
  _ <- symbol "<-"
  g <- generator
  close <- symbol "]"
  pure (Comprehension (mergeSpan open close) body p g)
  where
    pattern = label "a pattern" (PVar <$> identifier <|> tuple)
    tuple = do
      open' <- symbol "("
      xs <- sepBy1 identifier (symbol ",")
      close <- symbol ")"
      pure (PTuple (mergeSpan open' close) xs)
    generator = label "a dataset or `params`" (FromParams <$> keyword "params" <|> FromData <$> identifier)

-- ---------------------------------------------------------------------------
-- Declarations

-- | A whole file: declarations up to the end of the file, each one
-- 'recovering' from its errors.
--
-- Each declaration must be followed by the start of the next one or the
-- end of the file, and this check is part of the declaration. That way an
-- error after a complete declaration, as in @let y = x # 2@, is reported
-- together with what could have continued it: "expected an operator or a
-- declaration", not just "expected a declaration".
program :: Parser (Program Span)
program = Program . catMaybes <$> declarations
  where
    declarations = ([] <$ eof) <|> ((:) <$> recovering (declaration <* nextDeclaration) <*> declarations)
    nextDeclaration =
      label "a declaration" (lookAhead (eof <|> void (choice (map keyword declarationKeywords))))

-- | Error recovery: if the declaration fails, register its error, skip to
-- where the next declaration can start, and go on without it.
--
-- A declaration keyword on the line where the broken declaration started
-- is skipped too: in @let input = 1@ the error is at @input@, and parsing
-- @input = 1@ as the next declaration would only report a second,
-- confusing error. Declarations normally start on lines of their own.
recovering :: Parser a -> Parser (Maybe a)
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
    lineNumber = unPos . sourceLine <$> getSourcePos
    atEof = Text.null <$> getInput

-- | Skip tokens up to a point where a declaration can start: one of the
-- keywords @input param data fn model loss@, which only ever start
-- declarations, or @let@ at the start of a line, or the end of the file.
--
-- @let@ also starts expressions, which are normally indented, so only a
-- @let@ in column 1 counts. A file indented differently only loses
-- recovery (fewer errors reported), never correctness.
--
-- Skipping goes token by token, so a keyword inside a comment or a string,
-- or in the middle of a word like @inputs@, is never mistaken for one.
skipToDeclaration :: Parser ()
skipToDeclaration = do
  input <- getInput
  column <- unPos . sourceColumn <$> getSourcePos
  let atStart = case wordAt input of
        Just w -> (w /= "let" && w `elem` declarationKeywords) || (w == "let" && column == 1)
        Nothing -> Text.null input
  unless atStart (skipToken *> skipToDeclaration)

-- | Skip one token: a string, a word, a number or a single character.
skipToken :: Parser ()
skipToken = do
  input <- getInput
  case Text.uncons input of
    Just ('"', _) -> void stringLit
    Just (c, _)
      | isIdentStart c -> void (lexeme word)
      | startsNumber input -> void (lexeme (takeP Nothing (numberLength input)))
    _ -> void (lexeme anySingle)

-- | The keywords that start declarations.
declarationKeywords :: [Text]
declarationKeywords = ["input", "param", "data", "fn", "model", "let", "loss"]

-- | One declaration. Each starts with its own keyword.
declaration :: Parser (Decl Span)
declaration =
  label "a declaration" $
    choice [inputDecl, paramDecl, dataDecl, fnDecl, modelDecl, letDecl, lossDecl]

inputDecl :: Parser (Decl Span)
inputDecl = do
  kw <- keyword "input"
  x <- identifier
  _ <- symbol ":"
  t <- typeP
  def <- optional (symbol "=" *> literal)
  pure (InputDecl (mergeSpan kw (maybe (typeAnn t) literalAnn def)) x t def)

paramDecl :: Parser (Decl Span)
paramDecl = do
  kw <- keyword "param"
  x <- identifier
  _ <- symbol ":"
  t <- typeP
  ini <- choice [InitConst <$> (symbol "=" *> literal), randomInit, missing x t]
  let end = case ini of
        InitConst l -> literalAnn l
        InitRandom sp _ _ _ -> sp
  pure (ParamDecl (mergeSpan kw end) x t ini)
  where
    randomInit = do
      _ <- symbol "~"
      f <- identifier
      _ <- symbol "("
      lo <- signedNumber
      _ <- symbol ","
      hi <- signedNumber
      close <- symbol ")"
      pure (InitRandom (mergeSpan (identAnn f) close) f lo hi)
    missing x t =
      failWith $
        diagnostic E0101 ("parameter " <> quote (identName x) <> " has no initial value") (after (typeAnn t)) "expected `=` or `~` here"
          & withHint "write `= 0` for a constant, or `~ uniform(-1, 1)` for random values"

dataDecl :: Parser (Decl Span)
dataDecl = do
  kw <- keyword "data"
  x <- identifier
  _ <- symbol ":"
  _ <- symbol "{"
  fields <- sepBy1 (Field <$> identifier <*> (symbol ":" *> typeP)) (symbol ",")
  _ <- symbol "}"
  _ <- symbol "="
  src <- label "`rows` or `csv`" (rows <|> csv)
  pure (DataDecl (mergeSpan kw (dataSourceAnn src)) x fields src)
  where
    rows = do
      kw <- keyword "rows"
      _ <- symbol "["
      rs <- sepBy1 row (symbol ",")
      close <- symbol "]"
      pure (Rows (mergeSpan kw close) rs)
    row = label "a row" (tuple <|> bare)
    tuple = do
      open <- symbol "("
      ls <- sepBy1 literal (symbol ",")
      close <- symbol ")"
      pure (Row (mergeSpan open close) ls)
    bare = do
      l <- literal
      pure (Row (literalAnn l) [l])
    csv = do
      kw <- keyword "csv"
      path <- stringLit
      _ <- symbol "{"
      ms <- sepBy1 mapping (symbol ",")
      close <- symbol "}"
      pure (Csv (mergeSpan kw close) path ms)
    mapping = do
      f <- identifier
      _ <- symbol "="
      cols <- (Column <$> identifier) <|> columns
      pure (ColumnMap f cols)
    columns = do
      open <- symbol "["
      cs <- sepBy1 identifier (symbol ",")
      close <- symbol "]"
      pure (Columns (mergeSpan open close) cs)

-- | The parameter list of a @fn@ or @model@: @(x : Vec n, b)@.
binders :: Parser [Binder Span]
binders = do
  _ <- symbol "("
  bs <- sepBy (Binder <$> identifier <*> annotation) (symbol ",")
  _ <- symbol ")"
  pure bs

fnDecl :: Parser (Decl Span)
fnDecl = do
  kw <- keyword "fn"
  f <- identifier
  bs <- binders
  t <- annotation
  _ <- symbol "="
  body <- expr
  pure (FnDecl (mergeSpan kw (exprAnn body)) f bs t body)

modelDecl :: Parser (Decl Span)
modelDecl = do
  kw <- keyword "model"
  f <- identifier
  bs <- binders
  _ <- symbol ":"
  t <- typeP
  _ <- symbol "="
  body <- expr
  pure (ModelDecl (mergeSpan kw (exprAnn body)) f bs t body)

letDecl :: Parser (Decl Span)
letDecl = do
  kw <- keyword "let"
  x <- identifier
  t <- annotation
  _ <- symbol "="
  e <- expr
  pure (LetDecl (mergeSpan kw (exprAnn e)) x t e)

lossDecl :: Parser (Decl Span)
lossDecl = do
  kw <- keyword "loss"
  _ <- symbol "="
  e <- expr
  pure (LossDecl (mergeSpan kw (exprAnn e)) kw e)

-- ---------------------------------------------------------------------------
-- From megaparsec's errors to diagnostics

-- | All errors of a failed parse, as diagnostics in source order.
--
-- One cascade is removed: an unterminated comment or string swallows the
-- rest of the file, so whatever the parser expected at the end of the file
-- is a consequence, not a second mistake.
bundleDiagnostics :: FilePath -> Text -> ParseErrorBundle Text Problem -> NonEmpty Diagnostic
bundleDiagnostics file src bundle = NonEmpty.sortBy (comparing diagSpan) (NonEmpty.nub chosen)
  where
    converted = sconcat (NonEmpty.map (convert file src) (bundleErrors bundle))
    unterminated = any ((== E0102) . diagCode . snd) converted
    kept = [d | (trailing, d) <- NonEmpty.toList converted, not (unterminated && trailing)]
    chosen = maybe (NonEmpty.map snd converted) id (NonEmpty.nonEmpty kept)

-- | One megaparsec error as diagnostics, each marked with whether it is a
-- trivial error at the end of the file.
convert :: FilePath -> Text -> ParseError Text Problem -> NonEmpty (Bool, Diagnostic)
convert file src = \case
  TrivialError off _ expected -> pure (trailingError off, unexpectedToken file src off expected)
  FancyError off fancies -> case [d | ErrorCustom (Problem d) <- Set.toList fancies] of
    d : ds -> (False,) <$> (d :| ds)
    [] -> pure (trailingError off, unexpectedToken file src off Set.empty)
  where
    trailingError off = Text.all isSpace (Text.drop off src)

-- | E0101 from a trivial error: "unexpected @token@", labelled with what
-- was expected there.
--
-- Megaparsec only knows the offset and, as the unexpected item, the first
-- character. We look at the source again to find the whole token, so that
-- the message says "unexpected keyword `in`" and the marks cover the word.
unexpectedToken :: FilePath -> Text -> Int -> Set.Set (ErrorItem Char) -> Diagnostic
unexpectedToken file src off expected = case nextToken rest of
  Nothing ->
    diagnostic E0101 "unexpected end of file" (pointSpan file (advanceText startPos (Text.stripEnd src))) expectation
      & hints Nothing
  Just (tok, desc) ->
    diagnostic E0101 ("unexpected " <> desc) (textSpan file pos tok) expectation
      & hints (Just tok)
  where
    (spaces, rest) = Text.span isSpace (Text.drop off src)
    pos = advanceText startPos (Text.take off src <> spaces)

    -- Tokens first, then descriptions; "a declaration", the widest, last.
    items = Set.toList expected
    labels = [Text.pack (NonEmpty.toList l) | Label l <- items]
    names =
      [quote (Text.pack (NonEmpty.toList ts)) | Tokens ts <- items]
        <> filter (/= "a declaration") labels
        <> filter (== "a declaration") labels
    expectedNames
      | null names && EndOfInput `elem` items = ["end of file"]
      | otherwise = names
    expectation = case expectedNames of
      [] -> ""
      xs -> "expected " <> orList xs
    orList = \case
      [x] -> x
      [x, y] -> x <> " or " <> y
      x : xs -> x <> ", " <> orList xs
      [] -> ""

    expects n = n `elem` expectedNames
    hints tok d = foldl (flip withHint) d (concat (hintsFor tok))
    hintsFor tok =
      [ [quote w <> " is a keyword, so it can't be a name" | Just w <- [tok], isReserved w, expects "a name"]
      , ["a top-level `let` has no `in`" | tok == Just "in", expects "a declaration"]
      , [ "indices and dimensions are number literals, like `v[0]` or `Vec 3`"
        | Just w <- [tok >>= wordAt]
        , not (isReserved w)
        , expects "a natural number" || expects "a dimension"
        ]
      ]

-- | The token at the start of a text and how messages name it: (@in@,
-- "keyword `in`"), (@2.5@, "number `2.5`"), (@#@, "character `#`").
-- Nothing at the end of the text.
nextToken :: Text -> Maybe (Text, Text)
nextToken t = case Text.uncons t of
  Nothing -> Nothing
  Just (c, rest)
    | Just w <- wordAt t -> Just (w, if w `elem` keywords then "keyword " <> quote w else quote w)
    | startsNumber t -> let n = Text.take (numberLength t) t in Just (n, "number " <> quote n)
    | c == '"' ->
        let body = Text.takeWhile (\x -> x /= '"' && x /= '\n') rest
            closing = if "\"" `Text.isPrefixOf` Text.drop (Text.length body) rest then "\"" else ""
         in Just ("\"" <> body <> closing, "string")
    | "<-" `Text.isPrefixOf` t -> Just ("<-", "`<-`")
    | c `elem` ("+-*/^@=:~|()[]{},<" :: String) -> Just (Text.singleton c, quote (Text.singleton c))
    | otherwise -> Just (Text.singleton c, "character " <> quote (Text.singleton c))
