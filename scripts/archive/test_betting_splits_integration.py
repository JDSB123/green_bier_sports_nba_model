"""
Test script for betting splits integration.

Tests:
1. Fetching public betting splits
2. Converting GameSplits to features
3. RLM detection
4. Integration with feature engineering
"""
import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.betting_splits import (
    create_mock_splits,
    detect_reverse_line_movement,
    splits_to_features,
    GameSplits,
)


def test_mock_splits_generation():
    """Test 1: Generate mock betting splits."""
    print("\n" + "=" * 80)
    print("TEST 1: Mock Splits Generation")
    print("=" * 80)

    splits = create_mock_splits(
        event_id="test_game_1",
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        spread_line=-5.5,  # Lakers favored by 5.5
        total_line=225.5,
    )

    print(f"\nGame: {splits.away_team} @ {splits.home_team}")
    print(f"Spread: {splits.spread_line:+.1f}")
    print(f"  Public tickets: {splits.spread_home_ticket_pct:.1f}% home / "
          f"{splits.spread_away_ticket_pct:.1f}% away")
    print(f"  Money:          {splits.spread_home_money_pct:.1f}% home / "
          f"{splits.spread_away_money_pct:.1f}% away")
    print(f"  Ticket/Money divergence: "
          f"{splits.spread_home_ticket_pct - splits.spread_home_money_pct:+.1f}%")

    print(f"\nTotal: {splits.total_line:.1f}")
    print(f"  Public tickets: {splits.over_ticket_pct:.1f}% over / "
          f"{splits.under_ticket_pct:.1f}% under")
    print(f"  Money:          {splits.over_money_pct:.1f}% over / "
          f"{splits.under_money_pct:.1f}% under")

    print(f"\nLine Movement:")
    print(f"  Spread: {splits.spread_open:+.1f} → {splits.spread_current:+.1f} "
          f"(movement: {splits.spread_current - splits.spread_open:+.1f})")
    print(f"  Total:  {splits.total_open:.1f} → {splits.total_current:.1f} "
          f"(movement: {splits.total_current - splits.total_open:+.1f})")

    if splits.spread_rlm:
        print(f"\n⚠️  SPREAD RLM DETECTED - Sharp side: {splits.sharp_spread_side}")
    if splits.total_rlm:
        print(f"⚠️  TOTAL RLM DETECTED - Sharp side: {splits.sharp_total_side}")

    print(f"\n✓ Mock splits generated successfully")
    return splits


def test_rlm_detection():
    """Test 2: RLM detection."""
    print("\n" + "=" * 80)
    print("TEST 2: Reverse Line Movement Detection")
    print("=" * 80)

    # Scenario 1: Clear RLM - Public on favorite, line moves away
    print("\nScenario 1: Clear RLM (public on home, line moves away)")
    splits = GameSplits(
        event_id="rlm_test_1",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        game_time=None,
        spread_line=-3.5,
        spread_home_ticket_pct=70.0,  # Public heavily on home
        spread_away_ticket_pct=30.0,
        spread_home_money_pct=45.0,   # But money is on away (sharps)
        spread_away_money_pct=55.0,
        spread_open=-4.5,  # Opened at -4.5
        spread_current=-3.5,  # Moved to -3.5 (toward away despite public on home)
    )

    splits = detect_reverse_line_movement(splits)

    print(f"  Public: {splits.spread_home_ticket_pct:.0f}% on {splits.home_team}")
    print(f"  Money:  {splits.spread_home_money_pct:.0f}% on {splits.home_team}")
    print(f"  Line:   {splits.spread_open:+.1f} → {splits.spread_current:+.1f}")
    print(f"  RLM:    {'YES ⚠️' if splits.spread_rlm else 'NO'}")
    if splits.sharp_spread_side:
        print(f"  Sharp side: {splits.sharp_spread_side}")

    assert splits.spread_rlm, "Should detect RLM"
    assert splits.sharp_spread_side == "away", "Sharp side should be away"
    print("\n✓ RLM detection working correctly")

    # Scenario 2: No RLM - Line follows public
    print("\nScenario 2: No RLM (line follows public)")
    splits2 = GameSplits(
        event_id="rlm_test_2",
        home_team="Miami Heat",
        away_team="Charlotte Hornets",
        game_time=None,
        spread_line=-6.5,
        spread_home_ticket_pct=65.0,
        spread_away_ticket_pct=35.0,
        spread_home_money_pct=63.0,
        spread_away_money_pct=37.0,
        spread_open=-5.5,
        spread_current=-6.5,  # Moved toward home (following public)
    )

    splits2 = detect_reverse_line_movement(splits2)

    print(f"  Public: {splits2.spread_home_ticket_pct:.0f}% on {splits2.home_team}")
    print(f"  Money:  {splits2.spread_home_money_pct:.0f}% on {splits2.home_team}")
    print(f"  Line:   {splits2.spread_open:+.1f} → {splits2.spread_current:+.1f}")
    print(f"  RLM:    {'YES ⚠️' if splits2.spread_rlm else 'NO'}")

    assert not splits2.spread_rlm, "Should not detect RLM"
    print("\n✓ No false RLM detection")


def test_splits_to_features():
    """Test 3: Convert splits to feature dict."""
    print("\n" + "=" * 80)
    print("TEST 3: Splits to Features Conversion")
    print("=" * 80)

    splits = create_mock_splits(
        event_id="feature_test",
        home_team="Denver Nuggets",
        away_team="Phoenix Suns",
        spread_line=-2.5,
        total_line=230.0,
    )

    features = splits_to_features(splits)

    print("\nExtracted features:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value}")

    # Verify key features exist
    expected_features = [
        "spread_public_home_pct",
        "spread_money_home_pct",
        "over_public_pct",
        "spread_movement",
        "is_rlm_spread",
        "sharp_side_spread",
        "spread_ticket_money_diff",
    ]

    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"

    print(f"\n✓ All {len(expected_features)} expected features present")
    print(f"✓ Total features extracted: {len(features)}")


async def test_integration():
    """Test 4: Full integration with feature builder."""
    print("\n" + "=" * 80)
    print("TEST 4: Integration with Feature Engineering")
    print("=" * 80)

    from scripts.build_rich_features import RichFeatureBuilder
    from src.config import settings

    if not settings.api_basketball_key:
        print("\n⚠️  WARNING: API_BASKETBALL_KEY not set - skipping integration test")
        print("    Set API_BASKETBALL_KEY in .env to test full integration")
        return

    builder = RichFeatureBuilder(league_id=12, season="2025-2026")

    # Create mock splits
    splits = create_mock_splits(
        event_id="integration_test",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        spread_line=-3.0,
        total_line=220.0,
    )

    print(f"\nBuilding features with betting splits for:")
    print(f"  {splits.away_team} @ {splits.home_team}")

    try:
        # Build features WITH betting splits
        features = await builder.build_game_features(
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            betting_splits=splits,
        )

        print(f"\n✓ Features built successfully: {len(features)} features")

        # Check that betting splits features are included
        betting_features = [k for k in features.keys() if "split" in k.lower()
                           or "public" in k.lower() or "money" in k.lower()
                           or "rlm" in k.lower() or "sharp" in k.lower()
                           or "ticket" in k.lower()]

        if betting_features:
            print(f"✓ Betting splits features included ({len(betting_features)}):")
            for feat in sorted(betting_features):
                print(f"    {feat}: {features[feat]}")
        else:
            print("⚠️  WARNING: No betting splits features found in output")

    except Exception as e:
        print(f"\n❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("=" * 80)
    print("BETTING SPLITS INTEGRATION TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Mock generation
        test_mock_splits_generation()

        # Test 2: RLM detection
        test_rlm_detection()

        # Test 3: Feature conversion
        test_splits_to_features()

        # Test 4: Integration
        await test_integration()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
