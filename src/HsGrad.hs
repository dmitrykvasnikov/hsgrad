-- | Top-level module of the hsgrad library.
--
-- For now it only exposes the package version. Later steps add the
-- @HsGrad.Core.*@, @HsGrad.Train.*@, @HsGrad.Lang.*@ modules described in
-- @docs/00-architecture.md@.
module HsGrad
  ( version
  ) where

import Data.Version (showVersion)
import Paths_hsgrad qualified as Paths

-- | The package version, taken from @hsgrad.cabal@ (e.g. @"0.1.0.0"@).
version :: String
version = showVersion Paths.version
