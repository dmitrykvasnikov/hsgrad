{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Parameter files: trained parameters saved by @hsgrad train
-- --save-params@ and read back by @--params@.
--
-- A parameter file is a small @.hsg@ program made only of @param@
-- declarations with explicit values:
--
-- @
-- -- Parameters of examples/xor.hsg, saved by hsgrad train.
-- param W1 : Mat 4 2 = [
--   [1.5240938, -1.6109823],
--   …
-- ]
-- param b1 : Vec 4   = [-0.0215813, 0.6436045, -0.4213497, 0.0026461]
-- param w2 : Vec 4   = [1.5733114, -1.3086364, 1.1025049, -0.4317186]
-- param b2 : Real    = -0.1046372
-- @
--
-- So it is human-readable and can be written by hand (@= 0@ fills a
-- whole shape, comments are allowed), it can be pasted into the program
-- itself, and reading it needs no new parser: the language's own parser
-- and checker read it, with their diagnostics. What is left here is
-- comparing the file with the program (E0505). Numbers are written with
-- 'showConstant', the shortest decimal that reads back as exactly the
-- same 'Double', so saving and loading loses nothing.
-- See @docs/16-cli.md@ §2.5.
module HsGrad.Driver.ParamsFile
  ( renderParams
  , readParams
  , trainShape
  ) where

import Data.Foldable (toList)
import Data.Function ((&))
import Data.List (find)
import Data.Map.Strict qualified as Map
import Data.Maybe (mapMaybe)
import Data.Text (Text)
import Data.Text qualified as Text
import HsGrad.Core.Pretty (showConstant)
import HsGrad.Lang.Check (Init (..), ParamInfo (..), checkProgram, checkedParams, suggestName)
import HsGrad.Lang.Diagnostic
import HsGrad.Lang.Parser (parseProgram)
import HsGrad.Lang.Shape (Shape, prettyShape)
import HsGrad.Lang.Shape qualified as Shape
import HsGrad.Lang.Span (Span)
import HsGrad.Lang.Syntax (Decl (..), Ident (..), ParamInit (..), Program (..), declAnn)
import HsGrad.Train.Param (ParamSpec (..), Params, leafNames, paramValues)
import HsGrad.Train.Param qualified as Train

-- ---------------------------------------------------------------------------
-- Writing

-- | The text of a parameter file: the comment lines (without @--@), then
-- one declaration per parameter, in the order given. Names and types are
-- padded into columns, like the examples; a matrix gets one row per
-- line. 'Nothing' if the store lacks a value.
renderParams :: [Text] -> [ParamSpec] -> Params -> Maybe Text
renderParams comments specs store = do
  values <- mapM (`paramValues` store) specs
  let heads = [(specName s, typeText (specShape s)) | s <- specs]
      nameWidth = maximum (0 : map (Text.length . fst) heads)
      typeWidth = maximum (0 : map (Text.length . snd) heads)
      declaration (n, t) spec vs =
        "param " <> Text.justifyLeft nameWidth ' ' n <> " : " <> Text.justifyLeft typeWidth ' ' t <> " = " <> valueText (specShape spec) vs
  pure . Text.unlines $
    map ("-- " <>) comments
      <> zipWith3 declaration heads specs values
  where
    typeText = prettyShape . langShape

-- | A value as a literal: @0.5@, @[1, 2]@, or a matrix with one row per
-- line.
valueText :: Train.Shape -> [Double] -> Text
valueText sh vs = case sh of
  Train.Scalar -> Text.concat (map showConstant vs)
  Train.Vector _ -> list vs
  Train.Matrix _ c -> "[\n" <> Text.intercalate ",\n" ["  " <> list row | row <- chunks c vs] <> "\n]"
  where
    list xs = "[" <> Text.intercalate ", " (map showConstant xs) <> "]"
    chunks n xs = case splitAt n xs of
      (row, []) -> [row]
      (row, rest) -> row : chunks n rest

-- ---------------------------------------------------------------------------
-- Reading

-- | Read a parameter file for a program, given the program's @param@
-- declarations. Everything that is wrong is reported, in three stages
-- (each only if the one before found nothing):
--
-- 1. the file must parse, and hold nothing but @param@ declarations with
--    explicit values (E01xx, E0505);
-- 2. it must pass the checker: types, literals, duplicates (E02xx, E03xx;
--    warnings such as "never used" mean nothing here and are dropped);
-- 3. it must have every parameter of the program, at the program's
--    shape, and no other (E0505).
readParams :: FilePath -> Text -> [ParamInfo] -> Either [Diagnostic] Params
readParams path src wanted = do
  prog <- either (Left . toList) Right (parseProgram path src)
  stage (mapMaybe onlyValues (programDecls prog))
  let (diags, checked) = checkProgram prog
  stage (filter ((== Error) . severity) diags)
  given <- maybe (Left []) (Right . checkedParams) checked
  stage (mapMaybe (unknownOrReshaped wanted) given <> mapMaybe (missing path given) wanted)
  pure (Map.fromList (concatMap leaves given))
  where
    stage ds = if null ds then Right () else Left ds
    leaves p = zip (leafNames (paramName p) (trainShape (paramShape p))) (initValues p)

-- | Stage 1: a declaration that has no place in a parameter file.
onlyValues :: Decl Span -> Maybe Diagnostic
onlyValues = \case
  ParamDecl _ (Ident _ x) _ (InitRandom sp _ _ _) ->
    Just $
      diagnostic E0505 ("parameter `" <> x <> "` has no values") sp "a random initialiser"
        & withNote "a parameter file holds values, so that predictions don't depend on a seed"
        & withHint "write the values, for example `= [1, 2]`, or `= 0` to fill the whole shape"
  ParamDecl {} -> Nothing
  d ->
    Just $
      diagnostic E0505 "a parameter file holds only `param` declarations" (declAnn d) "not a `param`"
        & withHint "parameter files are written by `hsgrad train --save-params`"

-- | Stage 3, for a parameter of the file: is it one of the program's, at
-- the same shape?
unknownOrReshaped :: [ParamInfo] -> ParamInfo -> Maybe Diagnostic
unknownOrReshaped wanted p = case find ((== paramName p) . paramName) wanted of
  Nothing ->
    Just $
      diagnostic E0505 ("the program has no parameter `" <> paramName p <> "`") (paramNameSpan p) "unknown parameter"
        & maybe id (\s -> withHint ("did you mean `" <> s <> "`?")) (suggestName (paramName p) (map paramName wanted))
  Just w
    | paramShape w /= paramShape p ->
        Just $
          diagnostic
            E0505
            ("parameter `" <> paramName p <> "` has the wrong shape")
            (paramNameSpan p)
            ("a `" <> prettyShape (paramShape p) <> "` here")
            & withSecondary (paramNameSpan w) ("a `" <> prettyShape (paramShape w) <> "` in the program")
    | otherwise -> Nothing

-- | Stage 3, for a parameter of the program: is it in the file? Reported
-- at the program's declaration, since the file has no place for it.
missing :: FilePath -> [ParamInfo] -> ParamInfo -> Maybe Diagnostic
missing path given w
  | any ((== paramName w) . paramName) given = Nothing
  | otherwise =
      Just $
        diagnostic
          E0505
          ("parameter `" <> paramName w <> "` is missing from `" <> Text.pack path <> "`")
          (paramNameSpan w)
          "has no value in the parameter file"
          & withHint "a parameter file needs a value for every parameter of the program"

-- | The values of a checked @param@ of the file, row-major.
initValues :: ParamInfo -> [Double]
initValues p = case paramInit p of
  InitFill c -> replicate (Shape.shapeSize (paramShape p)) (fromRational c)
  InitValues cs -> map fromRational cs
  -- Rejected in stage 1; never reached.
  InitUniform _ _ -> []
  InitNormal _ _ -> []

-- ---------------------------------------------------------------------------
-- Shapes

-- | The language's shape as "HsGrad.Train.Param" describes it.
trainShape :: Shape Int -> Train.Shape
trainShape = \case
  Shape.Real -> Train.Scalar
  Shape.Vec n -> Train.Vector n
  Shape.Mat r c -> Train.Matrix r c

-- | And back.
langShape :: Train.Shape -> Shape Int
langShape = \case
  Train.Scalar -> Shape.Real
  Train.Vector n -> Shape.Vec n
  Train.Matrix r c -> Shape.Mat r c
