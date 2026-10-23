{-# LANGUAGE OverloadedStrings #-}

-- | Rendering expressions as text: @-2·b@, @1 - tanh(s)²@, @x/(y + z)@.
--
-- The expression rooted at a node is printed in ordinary mathematical
-- notation, with the precedence of the @.hsg@ language
-- (@docs/01-language.md@ §5.1) and only the parentheses the structure
-- needs. The printer is faithful: it shows the nodes as they are, so
-- @x·(y·z)@ keeps its parentheses, and simplifying is left to
-- "HsGrad.Core.Simplify".
--
-- A graph is a DAG, and printing it as a tree can be exponentially long
-- (a chain of 30 squarings is two billion nodes written out). Shared
-- subexpressions that are not tiny are therefore printed once, as named
-- bindings:
--
-- > let t1 = 1 - tanh(x[0]·w[0] + x[1]·w[1] + x[2]·w[2] + b)²
-- > in x[0]·t1
--
-- so the text is never more than a constant factor longer than the graph.
-- Used by the tests and by @hsgrad derive --format text@ (step 15).
-- See @docs/09-core-simplify.md@.
module HsGrad.Core.Pretty
  ( -- * Options
    PrettyOptions (..)
  , Style (..)
  , defaultPrettyOptions

    -- * Rendering
  , Rendered (..)
  , renderNodes
  , prettyNode
  , prettyNodeWith

    -- * Numbers
  , showConstant
  , showExponent
  ) where

import Data.Char (isAlpha, isAlphaNum)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Ratio (denominator, numerator)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Graph (Graph, Node (..), NodeId, Provenance (..), nodeAt, reachable)
import HsGrad.Core.Op (BinOp (..), Leaf (..), Name (..), Op (..), UnOp (..))
import Numeric (floatToDigits)

-- ---------------------------------------------------------------------------
-- Options

-- | Which characters to print with.
data Style
  = -- | @·@ for products and superscripts for integer powers: @-2·x²@.
    -- Reads like the docs; the default.
    Unicode
  | -- | @*@ and @^@ only: @-2*x^2@. For terminals without Unicode, and
    -- close to what a @.hsg@ file would say.
    Ascii
  deriving (Eq, Show)

-- | How to print.
data PrettyOptions = PrettyOptions
  { prettyStyle :: !Style
  , prettyInlineSize :: !Int
  -- ^ A subexpression used more than once is printed in place, every
  -- time, if it has at most this many nodes (counting a named
  -- subexpression as one), and gets a binding otherwise. With the default
  -- @3@, @x²@ and @a·b@ are repeated, @a·b + c@ is named.
  }
  deriving (Eq, Show)

-- | Unicode, and bindings for shared subexpressions of more than 3 nodes.
defaultPrettyOptions :: PrettyOptions
defaultPrettyOptions = PrettyOptions {prettyStyle = Unicode, prettyInlineSize = 3}

-- ---------------------------------------------------------------------------
-- Rendering

-- | Several expressions, rendered together so that they can share
-- bindings.
data Rendered = Rendered
  { renderedBindings :: ![(Text, Text)]
  -- ^ Name and expression of each shared subexpression, in an order where
  -- every binding only uses the ones before it.
  , renderedRoots :: ![Text]
  -- ^ One expression per root, in the order given. A root that is itself
  -- shared (by another root) is just its name.
  }
  deriving (Eq, Show)

-- | Render the expressions rooted at some nodes.
--
-- Partial, like 'nodeAt': every root must be a node of the graph.
renderNodes :: PrettyOptions -> Graph -> [NodeId] -> Rendered
renderNodes opts g roots =
  Rendered
    { renderedBindings = [(names Map.! i, fst (expand i)) | i <- order, i `Set.member` shared]
    , renderedRoots = map root roots
    }
  where
    order = Set.toAscList (reachable g roots)
    shared = sharedNodes opts g roots order
    names = bindingNames g order shared
    -- A node's text, with named arguments printed as their names.
    expand i = nodeDoc (prettyStyle opts) argDoc (nodeAt g i)
    argDoc a = case Map.lookup a names of
      Just t -> (t, Atomic)
      Nothing -> expand a
    root r = fromMaybe (fst (expand r)) (Map.lookup r names)

-- | Render one expression with the default options: a single line, or
-- @let … in …@ over several lines when it has shared subexpressions.
prettyNode :: Graph -> NodeId -> Text
prettyNode = prettyNodeWith defaultPrettyOptions

-- | 'prettyNode' with options.
prettyNodeWith :: PrettyOptions -> Graph -> NodeId -> Text
prettyNodeWith opts g i = case renderNodes opts g [i] of
  Rendered [] rs -> Text.concat rs
  Rendered bs rs ->
    Text.intercalate "\n" (zipWith binding ("let " : repeat "    ") bs)
      <> "\nin "
      <> Text.concat rs
  where
    binding prefix (name, e) = prefix <> name <> " = " <> e

-- ---------------------------------------------------------------------------
-- Sharing

-- | The nodes that get a binding: interior nodes used at least twice (as
-- arguments, or as a root) whose text would have more than
-- 'prettyInlineSize' nodes.
--
-- One pass in ascending order (arguments before users, D2) computes each
-- node's /inline size/: the number of nodes printed for it, where a named
-- argument counts as one. Nodes used once are printed once, and shared
-- nodes printed in place are small, so the whole text is linear in the
-- size of the graph.
sharedNodes :: PrettyOptions -> Graph -> [NodeId] -> [NodeId] -> Set NodeId
sharedNodes opts g roots order = snd (foldl' visit (Map.empty, Set.empty) order)
  where
    uses :: Map NodeId Int
    uses =
      Map.fromListWith (+) ([(r, 1) | r <- roots] ++ [(a, 1) | i <- order, a <- nodeArgs (nodeAt g i)])
    visit (sizes, named) i =
      let n = nodeAt g i
          argSize a = if a `Set.member` named then 1 else Map.findWithDefault 1 a sizes
          size = 1 + sum (map argSize (nodeArgs n))
          isShared =
            not (isLeaf n)
              && Map.findWithDefault 0 i uses >= 2
              && size > prettyInlineSize opts
       in (Map.insert i size sizes, if isShared then Set.insert i named else named)

-- | Names for the bindings, in ascending order: the node's label if it is
-- a plain name (@h@, @h1@, @W1[3,0]@, but not @∂L/∂a@) that is not
-- already taken, otherwise the first free one of @t1@, @t2@, …. Names of
-- the leaves in the expression are taken from the start, so a binding can
-- never be confused with a leaf.
bindingNames :: Graph -> [NodeId] -> Set NodeId -> Map NodeId Text
bindingNames g order shared = snd (foldl' assign (leafNames, Map.empty) [i | i <- order, i `Set.member` shared])
  where
    leafNames = Set.fromList [t | i <- order, Just t <- [leafName (nodeAt g i)]]
    assign (taken, names) i =
      let name = case provLabel (nodeProv (nodeAt g i)) of
            Just l | plainName l && not (l `Set.member` taken) -> l
            _ -> fresh taken 1
       in (Set.insert name taken, Map.insert i name names)
    fresh taken k =
      let t = "t" <> Text.pack (show (k :: Int))
       in if t `Set.member` taken then fresh taken (k + 1) else t
    plainName l = case Text.uncons l of
      Just (c, _) -> isAlpha c && Text.all (\x -> isAlphaNum x || x `elem` ("_'[],#." :: String)) l
      Nothing -> False

-- | The name of an input or parameter leaf.
leafName :: Node -> Maybe Text
leafName n = case nodeOp n of
  Leaf (Input (Name t)) -> Just t
  Leaf (Param (Name t)) -> Just t
  _ -> Nothing

isLeaf :: Node -> Bool
isLeaf n = case nodeOp n of
  Leaf _ -> True
  _ -> False

-- ---------------------------------------------------------------------------
-- Precedence

-- | How tightly a piece of text binds, following the language's
-- precedence table (@docs/01-language.md@ §5.1), tightest first.
data Level
  = -- | Leaves, positive constants, @f(…)@, anything in parentheses.
    Atomic
  | -- | @x²@, @x^0.5@.
    Power
  | -- | @-x@ and negative constants.
    Prefix
  | -- | @a·b@, @a/b@.
    Product
  | -- | @a + b@, @a - b@.
    Sum
  deriving (Eq)

-- | Text and how tightly it binds.
type Doc = (Text, Level)

-- | The text, in parentheses unless its level is one of those allowed in
-- this position.
inside :: [Level] -> Doc -> Text
inside allowed (t, l)
  | l `elem` allowed = t
  | otherwise = "(" <> t <> ")"

-- | Print one node, given how to print its arguments.
--
-- The rules, per position:
--
-- * @+@ and @-@ are left-associative: the left operand can be anything,
--   the right one must bind tighter (@a - (b - c)@, but @a - b - c@).
-- * @·@ and @/@ likewise: @a·b/c@, but @a/(b·c)@ and @a·(b/c)@.
-- * the base of a power must be atomic: @(-x)²@, @(x²)³@, @(a + b)²@;
--   @-x²@ is @-(x²)@, as in the language.
-- * the operand of @-@ must be a power or atomic: @-(a·b)@, @-(-x)@.
-- * a negative number or a negation as a /right/ operand gets parentheses
--   although the precedence does not need them: @x·(-2)@, @a - (-b)@
--   instead of @x·-2@, @a - -b@.
nodeDoc :: Style -> (NodeId -> Doc) -> Node -> Doc
nodeDoc style arg n = case (nodeOp n, map arg (nodeArgs n)) of
  (Leaf (Const c), _) -> (showConstant c, if c < 0 then Prefix else Atomic)
  (Leaf (Input (Name t)), _) -> (t, Atomic)
  (Leaf (Param (Name t)), _) -> (t, Atomic)
  (Un Neg, [x]) -> ("-" <> inside [Atomic, Power] x, Prefix)
  (Un (PowC c), [x]) -> (inside [Atomic] x <> powerSuffix style c, Power)
  (Un op, [x]) -> (functionName op <> "(" <> fst x <> ")", Atomic)
  (Bin op, [x, y]) -> case op of
    Add -> sumDoc " + " x y
    Sub -> sumDoc " - " x y
    Mul -> productDoc (case style of Unicode -> "·"; Ascii -> "*") x y
    Div -> productDoc "/" x y
  -- Not reached: the builder guarantees that a node has 'arity' arguments.
  (op, _) -> (Text.pack (show op), Atomic)
  where
    sumDoc sym x y = (fst x <> sym <> inside [Atomic, Power, Product] y, Sum)
    productDoc sym x y = (inside [Atomic, Power, Prefix, Product] x <> sym <> inside [Atomic, Power] y, Product)

-- | The name of a unary op printed as a function, as in the language.
functionName :: UnOp -> Text
functionName op = case op of
  Exp -> "exp"
  Log -> "log"
  Tanh -> "tanh"
  Relu -> "relu"
  Step -> "step"
  Neg -> "neg" -- not reached: printed as a prefix
  PowC _ -> "pow" -- not reached: printed as a power

-- | What follows the base of a power: superscript digits for an integer
-- exponent in the Unicode style (@²@, @⁻¹@), @^@ and the exponent
-- otherwise (@^0.5@, @^(1/3)@, @^2@ in ASCII).
powerSuffix :: Style -> Rational -> Text
powerSuffix Unicode c
  | denominator c == 1 = Text.map superscript (Text.pack (show (numerator c)))
powerSuffix _ c = "^" <> showExponent c

-- | The superscript version of a digit or minus sign.
superscript :: Char -> Char
superscript ch = case ch of
  '0' -> '⁰'
  '1' -> '¹'
  '2' -> '²'
  '3' -> '³'
  '4' -> '⁴'
  '5' -> '⁵'
  '6' -> '⁶'
  '7' -> '⁷'
  '8' -> '⁸'
  '9' -> '⁹'
  '-' -> '⁻'
  _ -> ch

-- ---------------------------------------------------------------------------
-- Numbers

-- | A constant, as short as possible while reading back as exactly the
-- same 'Double': @-2@ (not @-2.0@), @0.5@, @0.1@, @0.0001@, @1.5e-7@,
-- @1e20@.
--
-- The digits are the shortest that identify the number
-- ('floatToDigits'), so @0.1@ prints as @0.1@ although the 'Double' is not
-- exactly one tenth. Numbers from @0.0001@ up to (not including) @10¹⁵@ are
-- written out; smaller and larger ones use scientific notation.
showConstant :: Double -> Text
showConstant x
  | x == 0 = "0" -- also for -0.0
  | x < 0 = "-" <> showConstant (negate x)
  | e > -4 && e <= 15 = Text.pack positional
  | otherwise = Text.pack scientific
  where
    -- x = 0.d₁d₂…dₙ · 10^e
    (ds, e) = floatToDigits 10 x
    digits = concatMap show ds
    n = length ds
    positional
      | e >= n = digits <> replicate (e - n) '0'
      | e > 0 = take e digits <> "." <> drop e digits
      | otherwise = "0." <> replicate (negate e) '0' <> digits
    scientific = mantissa <> "e" <> show (e - 1)
    mantissa = case digits of
      d : rest@(_ : _) -> d : '.' : rest
      _ -> digits

-- | An exponent, exactly: @2@, @-1@, @0.5@, @-1.5@, @0.125@ for numbers
-- that have a finite decimal expansion, @(1/3)@ and @(-2/3)@ for the
-- others. A 'Rational' has a finite decimal expansion exactly when its
-- denominator has no prime factors but 2 and 5.
showExponent :: Rational -> Text
showExponent c
  | q == 1 = Text.pack (show p)
  | finiteDecimal = Text.pack (sign <> whole <> "." <> frac)
  | otherwise = Text.pack ("(" <> show p <> "/" <> show q <> ")")
  where
    p = numerator c
    q = denominator c
    (twos, rest) = strip 2 q
    (fives, rest') = strip 5 rest
    finiteDecimal = rest' == 1
    -- |c| · 10^k is an integer for k = max twos fives.
    k = max twos fives
    scaled = show ((abs p * 10 ^ k) `div` q)
    padded = replicate (k + 1 - length scaled) '0' <> scaled
    (whole, frac) = splitAt (length padded - k) padded
    sign = if p < 0 then "-" else ""
    -- How often f divides m, and what is left.
    strip :: Integer -> Integer -> (Int, Integer)
    strip f m
      | m `mod` f == 0 = let (t, r) = strip f (m `div` f) in (t + 1, r)
      | otherwise = (0, m)
