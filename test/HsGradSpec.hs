-- | Tests for the top-level "HsGrad" module.
--
-- They mostly check that the test tooling works: one hspec example and one
-- QuickCheck property.
module HsGradSpec
  ( spec
  ) where

import HsGrad (version)
import Test.Hspec (Spec, it, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck ((===))

spec :: Spec
spec = do
  it "has a non-empty version string" $
    version `shouldSatisfy` (not . null)

  prop "reversing a list twice gives it back" $ \xs ->
    reverse (reverse xs) === (xs :: [Int])
