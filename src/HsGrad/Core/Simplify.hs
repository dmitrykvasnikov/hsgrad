-- | The simplifier: rewrite expressions into a canonical sum of products
-- (decision D7) and build the result as a new graph.
--
-- Every node is turned into a 'Poly': a constant plus a sum of terms
-- @k · m@, where @k@ is a coefficient and @m@ a 'Monomial', a product of
-- 'Atom's raised to integer powers. Atoms are what the canonical form
-- cannot look into: leaves, the non-polynomial ops (@exp@, @log@, @tanh@,
-- @relu@, @step@, fractional powers) with simplified arguments, and sums
-- that are used as a factor. Computing the canonical forms bottom-up gives,
-- with no rewrite rules at all:
--
-- * like terms gathered: @x + x → 2·x@, @x·y - y·x → 0@;
-- * powers merged: @x·x·x → x³@, @x·x⁻¹ → 1@;
-- * coefficients and constants folded: @2·x·3 → 6·x@, @exp(0) → 1@;
-- * the identities of @0@ and @1@: @x + 0 → x@, @1·x → x@, @0·x → 0@.
--
-- Products are /not/ distributed over sums: @x·(y + z)@ stays a product
-- with the atom @y + z@, because expanding products of sums can blow up
-- exponentially (D7). Only constant factors are distributed,
-- @2·(x + y) → 2·x + 2·y@, which never adds terms.
--
-- Canonical forms are memoised per 'NodeId', so a shared node is
-- simplified once (no blow-up on DAGs), and they are built back into
-- nodes with the hash-consing 'Builder' (D3): expressions that are equal
-- as canonical forms become the very same node.
--
-- Some identities make an expression defined in more places than before
-- (@x·x⁻¹ → 1@ at @x = 0@, @0·log x → 0@ at @x ≤ 0@); see
-- @docs/09-core-simplify.md@ §1.5.
module HsGrad.Core.Simplify
  ( simplify
  , simplifyNodes
  ) where

import Control.Monad (foldM)
import Control.Monad.State.Strict (StateT, evalStateT, gets, lift, mapStateT, modify')
import Data.IntMap.Strict (IntMap)
import Data.IntMap.Strict qualified as IntMap
import Data.Map.Merge.Strict qualified as Merge
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Ratio (denominator, numerator)
import Data.Set qualified as Set
import HsGrad.Core.Graph
  ( Builder
  , Graph
  , GraphError
  , Node (..)
  , NodeId
  , Provenance (..)
  , binary
  , build
  , constant
  , currentGraph
  , label
  , lookupNode
  , node
  , nodeAt
  , nodeIndex
  , reachable
  , unary
  , withSpan
  )
import HsGrad.Core.Op (BinOp (..), Leaf (..), Op (..), UnOp (..), forwardBin, forwardUn)

-- ---------------------------------------------------------------------------
-- Atoms

-- | An atom: a node of the graph being built, which the canonical form
-- treats as an indivisible variable.
--
-- An atom is identified by its node: hash-consing guarantees that two
-- nodes of one graph are structurally equal exactly when they are the same
-- node. The op and the arguments (as atoms of their nodes) are kept only to
-- /order/ atoms by their structure rather than by their ids, which depend
-- on the order nodes happened to be created in (§3.2 of the doc).
data Atom = Atom !NodeId !Op [Atom]

-- | The node an atom stands for.
atomNode :: Atom -> NodeId
atomNode (Atom i _ _) = i

instance Eq Atom where
  a == b = atomNode a == atomNode b

-- | Structural order: by op, then by arguments, left to right. Equal ids
-- mean equal structure, so the comparison stops there; two different
-- nodes with the same op differ in some argument, so it follows exactly
-- one path down the graph and takes time proportional to its depth, never
-- to the size of the expression written out as a tree.
instance Ord Atom where
  compare (Atom i op as) (Atom j op' bs)
    | i == j = EQ
    | otherwise = compare op op' <> compare as bs

-- ---------------------------------------------------------------------------
-- Monomials and canonical forms

-- | A product of atoms raised to non-zero integer powers: @x²·y⁻¹@ is
-- @{x ↦ 2, y ↦ -1}@. The empty product is @1@.
newtype Monomial = Monomial (Map Atom Integer)
  deriving (Eq, Ord)

-- | The empty product.
unitM :: Monomial
unitM = Monomial Map.empty

-- | One atom, to the first power.
atomM :: Atom -> Monomial
atomM a = Monomial (Map.singleton a 1)

-- | Multiply two monomials: add the exponents, drop the ones that become
-- @0@. This is where @x·x⁻¹ → 1@ happens.
mulM :: Monomial -> Monomial -> Monomial
mulM (Monomial a) (Monomial b) = Monomial (addNonZero a b)

-- | Raise a monomial to a non-zero integer power.
powM :: Integer -> Monomial -> Monomial
powM n (Monomial a) = Monomial (Map.map (* n) a)

-- | The canonical form of an expression: a constant plus terms
-- @coefficient · monomial@,
--
-- > c + k₁·m₁ + k₂·m₂ + …
--
-- with distinct monomials, non-zero finite coefficients and a finite
-- constant. Because exponents can be negative this is a Laurent polynomial
-- in the atoms, a "sum of products" (D7).
data Poly = Poly !Double !(Map Monomial Double)

-- | A constant.
constP :: Double -> Poly
constP c = Poly c Map.empty

-- | A single term @k · m@.
termP :: Double -> Monomial -> Poly
termP k m@(Monomial factors)
  | k == 0 = constP 0
  | Map.null factors = constP k
  | otherwise = Poly 0 (Map.singleton m k)

-- | One atom.
atomP :: Atom -> Poly
atomP = termP 1 . atomM

-- | The value of a canonical form that is a constant.
constantOf :: Poly -> Maybe Double
constantOf (Poly c ts)
  | Map.null ts = Just c
  | otherwise = Nothing

-- | Sum: add the constants and the coefficients of equal monomials, drop
-- the terms that cancel. This is where like terms are gathered.
addP :: Poly -> Poly -> Poly
addP (Poly c ts) (Poly d us) = Poly (c + d) (addNonZero ts us)

-- | Negation.
negP :: Poly -> Poly
negP (Poly c ts) = Poly (negate c) (Map.map negate ts)

-- | Multiply by a constant. Multiplying by @0@ gives @0@, whatever the
-- other factor (the domain caveat of §1.5). A coefficient can underflow to
-- @0@, so zero terms are dropped.
scaleP :: Double -> Poly -> Poly
scaleP 0 _ = constP 0
scaleP k (Poly c ts) = Poly (k * c) (Map.filter (/= 0) (Map.map (k *) ts))

-- | Are the constant and every coefficient finite? Folding can overflow
-- (@exp 1000@, @1/0@); such a form is not used (see 'canonicalOf').
finiteP :: Poly -> Bool
finiteP (Poly c ts) = finite c && all finite ts
  where
    finite x = not (isNaN x || isInfinite x)

-- | Merge two maps, adding the values of common keys and dropping sums
-- that are @0@. Keys found in only one map are kept as they are. The merge
-- tactics of "Data.Map.Merge.Strict" make this fast when one map is much
-- smaller than the other (a long chain of @+@).
addNonZero :: (Ord k, Eq a, Num a) => Map k a -> Map k a -> Map k a
addNonZero =
  Merge.merge Merge.preserveMissing Merge.preserveMissing (Merge.zipWithMaybeMatched add)
  where
    add _ x y = let s = x + y in if s == 0 then Nothing else Just s

-- ---------------------------------------------------------------------------
-- The simplifier's state

-- | Everything the simplifier memoises.
data St = St
  { stForms :: !(IntMap Poly)
  -- ^ Canonical form of each source node, keyed by its 'nodeIndex' in the
  -- source graph.
  , stNodes :: !(IntMap NodeId)
  -- ^ The node built for the canonical form of a source node, for the
  -- source nodes that needed one (outputs, arguments of atoms, sums used
  -- as factors).
  , stAtoms :: !(IntMap Atom)
  -- ^ The atom of each node of the /new/ graph that was needed as one,
  -- keyed by its 'nodeIndex' in the new graph.
  , stSums :: !(IntMap Poly)
  -- ^ The canonical form of each node of the new graph that was made an
  -- atom because it is a sum used as a factor (see 'termS').
  }

-- | The simplifier runs in the builder of the new graph, with its memo
-- tables on top.
type Simp = StateT St Builder

-- | Look up a memoised result, or compute and remember it.
memo :: (St -> IntMap a) -> (IntMap a -> St -> St) -> Int -> Simp a -> Simp a
memo get set key compute = do
  known <- gets (IntMap.lookup key . get)
  case known of
    Just x -> pure x
    Nothing -> do
      x <- compute
      modify' (\s -> set (IntMap.insert key x (get s)) s)
      pure x

-- ---------------------------------------------------------------------------
-- Running the simplifier

-- | Simplify the expressions rooted at some nodes of a graph, adding the
-- simplified nodes to the graph being built. Returns the simplified node
-- of each given node.
--
-- The given ids belong to the /source/ graph, the results to the graph
-- being built: usually a fresh one (see 'simplify'). Ids that are not
-- nodes of the source graph are left out of the result.
--
-- Provenance is kept where it means the same thing: the nodes built for a
-- source node get its source span, and the simplified node of a labelled
-- source node (an output such as @L@, a gradient node such as @∂L/∂a@)
-- gets its label, unless it is a leaf or already labelled.
simplifyNodes :: Graph -> [NodeId] -> Builder (Map NodeId NodeId)
simplifyNodes g outs = evalStateT (Map.fromList <$> mapM simplified known) empty
  where
    known = [o | o <- outs, Just _ <- [lookupNode o g]]
    simplified o = (o,) <$> nodeOf g o
    empty = St IntMap.empty IntMap.empty IntMap.empty IntMap.empty

-- | Simplify the expressions rooted at some nodes of a graph, into a new
-- graph that holds only the simplified expressions. Returns the simplified
-- node of each given node (ids that are not nodes of the graph are left
-- out) and the new graph.
--
-- It cannot fail in practice: the simplifier never builds a non-finite
-- constant or an ill-formed node. The 'Either' is the builder's, as for
-- 'HsGrad.Core.Symbolic.symbolicGradients'.
simplify :: Graph -> [NodeId] -> Either GraphError (Map NodeId NodeId, Graph)
simplify g outs = do
  (simplified, scratch) <- build (simplifyNodes g outs)
  -- Building canonical forms can leave nodes that the results do not use
  -- (the atoms of terms that cancelled, say). Copy only what the results
  -- need, so that the new graph is exactly the simplified expressions.
  (copies, g') <- build (copyReachable scratch (Map.elems simplified))
  pure (Map.map (copies Map.!) simplified, g')

-- | Copy the nodes some roots depend on from another graph, with their
-- provenance. Returns the copy of every copied node. Ascending order is a
-- topological order (D2), so a node's arguments are copied before it, and
-- the lookups cannot fail.
copyReachable :: Graph -> [NodeId] -> Builder (Map NodeId NodeId)
copyReachable from roots = foldM copy Map.empty (Set.toAscList (reachable from roots))
  where
    copy acc i = do
      let n = nodeAt from i
          prov = nodeProv n
      j <- maybe id withSpan (provSpan prov) (node (nodeOp n) (map (acc Map.!) (nodeArgs n)))
      mapM_ (`label` j) (provLabel prov)
      pure (Map.insert i j acc)

-- ---------------------------------------------------------------------------
-- Canonical forms of source nodes

-- | The canonical form of a source node, memoised.
--
-- A form whose folding overflowed (a non-finite constant or coefficient)
-- is replaced by the node itself as an atom: @1/0@ stays @1/0@, and
-- evaluating it still gives the error it gave before. The graph never gets
-- a non-finite constant (D3, @docs/03-core-op.md@ §3.5).
canonicalOf :: Graph -> NodeId -> Simp Poly
canonicalOf g i = memo stForms (\m s -> s {stForms = m}) (nodeIndex i) $
  inSpanOf n $ do
    p <- canonical g n
    if finiteP p then pure p else asAtom g n
  where
    n = nodeAt g i

-- | The canonical form of a node, from the forms of its arguments: the
-- forward pass of "HsGrad.Core.Eval", with canonical forms for numbers.
canonical :: Graph -> Node -> Simp Poly
canonical g n = case (nodeOp n, nodeArgs n) of
  (Leaf (Const c), _) -> pure (constP c)
  (Leaf l, _) -> atomP <$> newAtom (Leaf l) []
  (Un op, [a]) -> do
    p <- canonicalOf g a
    unaryForm g n op (a, p)
  (Bin op, [a, b]) -> do
    p <- canonicalOf g a
    q <- canonicalOf g b
    binaryForm g op (a, p) (b, q)
  -- Not reached: the builder guarantees that a node has 'arity' arguments.
  _ -> asAtom g n

-- | The canonical form of a unary op applied to an argument (its source
-- id and form).
unaryForm :: Graph -> Node -> UnOp -> (NodeId, Poly) -> Simp Poly
unaryForm g n op (a, p) = case (op, constantOf p) of
  -- Constant folding, with the forward pass's own arithmetic.
  (_, Just x) -> pure (constP (forwardUn op x))
  (Neg, _) -> pure (negP p)
  (PowC c, _)
    | denominator c == 1 -> intPower (numerator c)
  -- exp, log, tanh, relu, step and fractional powers are atoms.
  _ -> asAtom g n
  where
    -- @p⁰ = 1@ holds even for @p = 0@ (@0 ^^ 0 = 1@); @p¹ = p@; otherwise
    -- @(k·m)ⁿ = kⁿ·mⁿ@, where a sum is a single atom.
    intPower 0 = pure (constP 1)
    intPower 1 = pure p
    intPower e = do
      (k, m) <- factor g a p
      termS (k ^^ e) (powM e m)

-- | The canonical form of a binary op applied to two arguments.
binaryForm :: Graph -> BinOp -> (NodeId, Poly) -> (NodeId, Poly) -> Simp Poly
binaryForm g op (a, p) (b, q) = case (op, constantOf p, constantOf q) of
  (_, Just x, Just y) -> pure (constP (forwardBin op x y))
  (Add, _, _) -> pure (addP p q)
  (Sub, _, _) -> pure (addP p (negP q))
  -- A constant factor is distributed over the terms of the other side.
  (Mul, Just x, _) -> pure (scaleP x q)
  (Mul, _, Just y) -> pure (scaleP y p)
  -- Anything else is a product of two products; sums become atoms.
  (Mul, _, _) -> do
    fp <- factor g a p
    fq <- factor g b q
    productS fp fq
  (Div, _, Just y) -> pure (scaleP (1 / y) p)
  (Div, _, _) -> do
    fp <- factor g a p
    (k, m) <- factor g b q
    productS fp (1 / k, powM (-1) m)
  where
    productS (k1, m1) (k2, m2) = termS (k1 * k2) (mulM m1 m2)

-- | A canonical form as a single product @k · m@, for multiplying it: a
-- constant is @k · 1@, a single term is itself, and anything with more
-- than one part is a sum, which becomes one atom (the node built for it).
-- This is the step that refuses to distribute.
factor :: Graph -> NodeId -> Poly -> Simp (Double, Monomial)
factor g i (Poly c ts) = case Map.toList ts of
  [] -> pure (c, unitM)
  [(m, k)] | c == 0 -> pure (k, m)
  _ -> do
    x <- nodeOf g i
    modify' (\s -> s {stSums = IntMap.insert (nodeIndex x) (Poly c ts) (stSums s)})
    a <- atomOf x
    pure (1, atomM a)

-- | The term @k · m@, as 'termP', except that a sum atom on its own is
-- put back as the sum: @k · (x + y)@ is @k·x + k·y@.
--
-- A product can leave a sum atom on its own (@(x + y)·z/z@), and a form
-- @k·(x + y)@ would not be canonical: built as nodes and simplified again,
-- it is a constant times a sum, which is distributed. Keeping the
-- invariant "no term is a single sum atom to the power 1" makes the form
-- canonical and the simplifier idempotent.
termS :: Double -> Monomial -> Simp Poly
termS k m@(Monomial factors) = case Map.toList factors of
  [(a, 1)] -> do
    known <- gets (IntMap.lookup (nodeIndex (atomNode a)) . stSums)
    pure (maybe (termP k m) (scaleP k) known)
  _ -> pure (termP k m)

-- | A source node as an atom: the same op, applied to the simplified
-- arguments.
asAtom :: Graph -> Node -> Simp Poly
asAtom g n = do
  xs <- mapM (nodeOf g) (nodeArgs n)
  atomP <$> newAtom (nodeOp n) xs

-- | Build a node in the new graph and take it as an atom.
newAtom :: Op -> [NodeId] -> Simp Atom
newAtom op args = atomOf =<< lift (node op args)

-- | The atom of a node of the new graph, memoised.
atomOf :: NodeId -> Simp Atom
atomOf x = memo stAtoms (\m s -> s {stAtoms = m}) (nodeIndex x) $ do
  n <- (`nodeAt` x) <$> lift currentGraph
  args <- mapM atomOf (nodeArgs n)
  pure (Atom x (nodeOp n) args)

-- | The node of the new graph for a source node: its canonical form, built
-- as nodes. Memoised, so that a shared source node is built once.
nodeOf :: Graph -> NodeId -> Simp NodeId
nodeOf g i = memo stNodes (\m s -> s {stNodes = m}) (nodeIndex i) $ do
  p <- canonicalOf g i
  let n = nodeAt g i
  lift $ do
    x <- maybe id withSpan (provSpan (nodeProv n)) (rebuild p)
    keepLabel n x
    pure x

-- | Give the new node the label of the source node, unless the new node is
-- a leaf: @x + 0@ labelled @d@ simplifies to the leaf @x@, which must not be
-- renamed. Labels are "first wins" (step 3).
keepLabel :: Node -> NodeId -> Builder ()
keepLabel n x = case provLabel (nodeProv n) of
  Nothing -> pure ()
  Just l -> do
    g' <- currentGraph
    case nodeOp (nodeAt g' x) of
      Leaf _ -> pure ()
      _ -> label l x

-- | Run a simplifier step with the source span of a node, so that the
-- atoms it builds record where they came from.
inSpanOf :: Node -> Simp a -> Simp a
inSpanOf n = maybe id (mapStateT . withSpan) (provSpan (nodeProv n))

-- ---------------------------------------------------------------------------
-- Building canonical forms as nodes

-- | Build a canonical form as nodes, in a fixed layout:
--
-- * the constant first (if it is not @0@), then the terms in the order of
--   their monomials: @1 - y²@, @a·b + c@;
-- * a negative coefficient after the first part becomes @-@:
--   @x - 2·y@ rather than @x + -2·y@;
-- * a term is its coefficient (left out if @1@, a @-@ if @-1@), then its
--   factors in atom order, then @/@ and the factors with negative
--   exponents: @-2·a·b@, @-x·y@, @0.5·x/y@, @1/x@. If the first factor is
--   a sum, the coefficient multiplies the whole product instead:
--   @2·((x + y)·(x + z))@, never @(2·(x + y))·(x + z)@, whose first
--   product would be distributed when simplified again.
--
-- The layout is a function of the canonical form alone, so equal forms
-- become the same nodes, and simplifying the result again gives the same
-- form back (idempotence).
rebuild :: Poly -> Builder NodeId
rebuild (Poly c ts) = case Map.toAscList ts of
  -- @c == 0@ also matches @-0.0@, which is built as @0@.
  [] -> constant (if c == 0 then 0 else c)
  first : rest
    | c == 0 -> do
        x <- termNode first
        foldM addTerm x rest
    | otherwise -> do
        x <- constant c
        foldM addTerm x (first : rest)
  where
    addTerm acc (m, k)
      | k < 0 = binary Sub acc =<< termNode (m, negate k)
      | otherwise = binary Add acc =<< termNode (m, k)

-- | Build one term @k · m@ (with @m@ not empty).
termNode :: (Monomial, Double) -> Builder NodeId
termNode (Monomial factors, k) = do
  ups <- mapM power upAtoms
  downs <- mapM power [(a, negate e) | (a, e) <- Map.toAscList factors, e < 0]
  let over numer = case downs of
        [] -> pure numer
        d : ds -> binary Div numer =<< foldM (binary Mul) d ds
  case ups of
    [] -> over =<< constant k
    f : fs
      | startsWithSum -> scaled =<< over =<< foldM (binary Mul) f fs
      | otherwise -> do
          f' <- scaled f
          over =<< foldM (binary Mul) f' fs
  where
    upAtoms = [(a, e) | (a, e) <- Map.toAscList factors, e > 0]
    startsWithSum = case upAtoms of
      (Atom _ (Bin op) _, 1) : _ -> op == Add || op == Sub
      _ -> False
    scaled x
      | k == 1 = pure x
      | k == -1 = unary Neg x
      | otherwise = do
          kc <- constant k
          binary Mul kc x
    power (a, e)
      | e == 1 = pure (atomNode a)
      | otherwise = unary (PowC (fromInteger e)) (atomNode a)
