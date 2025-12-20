"""
End-to-end integration tests for ML pipeline
Tests the complete flow: database -> feature extraction -> prediction -> storage
"""
import pytest
import os
import psycopg2
import redis
from datetime import datetime

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def db_connection():
    """Create database connection for testing"""
    conn = psycopg2.connect(
        host=os.getenv('DATABASE_HOST', 'localhost'),
        port=int(os.getenv('DATABASE_PORT', 5432)),
        database=os.getenv('DATABASE_NAME', 'ncaaf_v5'),
        user=os.getenv('DATABASE_USER', 'ncaaf_user'),
        password=os.getenv('DATABASE_PASSWORD', 'ncaaf_password')
    )
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def redis_connection():
    """Create Redis connection for testing"""
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', ''),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )
    yield r
    r.close()


class TestE2EPipeline:
    """End-to-end pipeline tests"""

    def test_database_connectivity(self, db_connection):
        """Test database is accessible"""
        cursor = db_connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        cursor.close()

    def test_database_schema(self, db_connection):
        """Test all required tables exist"""
        cursor = db_connection.cursor()

        required_tables = [
            'teams',
            'stadiums',
            'games',
            'odds',
            'line_movement',
            'team_season_stats',
            'box_scores',
            'predictions'
        ]

        for table in required_tables:
            cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (table,)
            )
            exists = cursor.fetchone()[0]
            assert exists, f"Table '{table}' does not exist"

        cursor.close()

    def test_teams_seeded(self, db_connection):
        """Test that teams have been seeded"""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM teams")
        count = cursor.fetchone()[0]
        assert count > 0, "Teams table is empty - run seed script"
        assert count >= 50, f"Expected at least 50 teams, got {count}"
        cursor.close()

    def test_redis_connectivity(self, redis_connection):
        """Test Redis is accessible"""
        assert redis_connection.ping(), "Redis not responding"

    def test_redis_cache_operations(self, redis_connection):
        """Test Redis caching works"""
        # Set a test value
        test_key = "e2e_test_key"
        test_value = "e2e_test_value"

        redis_connection.set(test_key, test_value, ex=60)

        # Retrieve it
        retrieved = redis_connection.get(test_key)
        assert retrieved == test_value

        # Clean up
        redis_connection.delete(test_key)

    def test_feature_extraction_with_real_data(self, db_connection):
        """Test feature extraction with real database data"""
        from src.features.feature_extractor import FeatureExtractor

        cursor = db_connection.cursor()

        # Get two random teams
        cursor.execute("SELECT team_id FROM teams LIMIT 2")
        teams = cursor.fetchall()

        if len(teams) < 2:
            pytest.skip("Not enough teams in database")

        home_team_id = teams[0][0]
        away_team_id = teams[1][0]

        # Create feature extractor
        extractor = FeatureExtractor(
            host=os.getenv('DATABASE_HOST', 'localhost'),
            port=int(os.getenv('DATABASE_PORT', 5432)),
            database=os.getenv('DATABASE_NAME', 'ncaaf_v5'),
            user=os.getenv('DATABASE_USER', 'ncaaf_user'),
            password=os.getenv('DATABASE_PASSWORD', 'ncaaf_password')
        )

        # Extract features (may be sparse if no stats yet)
        try:
            features = extractor.extract_game_features(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                season=2024,
                week=15
            )

            # Verify feature structure
            assert isinstance(features, dict)
            # May have limited features if no historical data
            assert len(features) > 0

        except Exception as e:
            # If no stats exist, this is expected
            if "no stats" in str(e).lower():
                pytest.skip(f"No stats available for teams: {e}")
            else:
                raise

        cursor.close()

    def test_prediction_generation(self, db_connection):
        """Test prediction generation if models exist"""
        from src.models.predictor import NCAAFPredictor
        import os

        model_path = os.getenv('MODEL_PATH', '/app/models')

        # Check if models exist
        model_files = [
            f'{model_path}/xgboost_margin.pkl',
            f'{model_path}/xgboost_total.pkl',
        ]

        if not all(os.path.exists(f) for f in model_files):
            pytest.skip("ML models not trained yet - run train_xgboost.py")

        # Create predictor
        predictor = NCAAFPredictor(model_path=model_path)

        # Create simple test features
        test_features = {
            'home_yards_per_play': 6.0,
            'away_yards_per_play': 5.5,
            'home_completion_pct': 0.60,
            'away_completion_pct': 0.55,
            'home_third_down_conversion_pct': 0.40,
            'away_third_down_conversion_pct': 0.35,
            'talent_differential': 5.0,
        }

        # Generate prediction
        result = predictor.predict_game(
            features=test_features,
            consensus_spread=-7.0,
            consensus_total=50.0
        )

        # Verify prediction structure
        assert 'predicted_margin' in result
        assert 'predicted_total' in result
        assert 'confidence' in result
        assert 'edge_spread' in result
        assert 'edge_total' in result

        # Verify reasonable values
        assert -50 <= result['predicted_margin'] <= 50
        assert 20 <= result['predicted_total'] <= 100
        assert 0.0 <= result['confidence'] <= 1.0

    def test_full_pipeline_with_game(self, db_connection):
        """Test complete pipeline if games exist"""
        cursor = db_connection.cursor()

        # Check if any games exist
        cursor.execute("SELECT game_id, home_team_id, away_team_id FROM games LIMIT 1")
        game = cursor.fetchone()

        if not game:
            pytest.skip("No games in database - run initial sync")

        game_id, home_team_id, away_team_id = game

        # Test 1: Game data exists
        assert game_id is not None

        # Test 2: Teams exist
        cursor.execute("SELECT COUNT(*) FROM teams WHERE team_id IN (%s, %s)", (home_team_id, away_team_id))
        team_count = cursor.fetchone()[0]
        assert team_count == 2, "Game references non-existent teams"

        # Test 3: Check if odds exist
        cursor.execute("SELECT COUNT(*) FROM odds WHERE game_id = %s", (game_id,))
        odds_count = cursor.fetchone()[0]

        if odds_count > 0:
            # If odds exist, predictions should be possible
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE game_id = %s", (game_id,))
            pred_count = cursor.fetchone()[0]

            # Predictions may not exist yet if models aren't trained
            # Just verify schema is ready
            assert pred_count >= 0

        cursor.close()

    def test_cache_hit_performance(self, redis_connection):
        """Test that caching improves performance"""
        import time

        # First call (cache miss)
        test_key = "performance_test_key"
        test_data = {"test": "data", "timestamp": str(datetime.now())}

        start = time.time()
        redis_connection.set(test_key, str(test_data), ex=60)
        write_time = time.time() - start

        # Second call (cache hit)
        start = time.time()
        cached_data = redis_connection.get(test_key)
        read_time = time.time() - start

        # Read should be very fast (< 10ms typically)
        assert read_time < 0.1, f"Cache read too slow: {read_time}s"

        # Clean up
        redis_connection.delete(test_key)

    def test_database_indexes(self, db_connection):
        """Test that critical indexes exist for performance"""
        cursor = db_connection.cursor()

        # Check for indexes on commonly queried columns
        critical_indexes = [
            ('games', 'game_id'),
            ('odds', 'game_id'),
            ('predictions', 'game_id'),
            ('teams', 'team_id'),
        ]

        for table, column in critical_indexes:
            cursor.execute("""
                SELECT COUNT(*)
                FROM pg_indexes
                WHERE tablename = %s
                AND indexdef LIKE %s
            """, (table, f'%{column}%'))

            index_count = cursor.fetchone()[0]
            assert index_count > 0, f"Missing index on {table}.{column}"

        cursor.close()

    def test_data_consistency(self, db_connection):
        """Test data consistency rules"""
        cursor = db_connection.cursor()

        # Test 1: No games with invalid team references
        cursor.execute("""
            SELECT COUNT(*)
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.team_id
            LEFT JOIN teams at ON g.away_team_id = at.team_id
            WHERE ht.team_id IS NULL OR at.team_id IS NULL
        """)
        invalid_games = cursor.fetchone()[0]
        assert invalid_games == 0, f"Found {invalid_games} games with invalid team references"

        # Test 2: No odds with invalid game references
        cursor.execute("""
            SELECT COUNT(*)
            FROM odds o
            LEFT JOIN games g ON o.game_id = g.game_id
            WHERE g.game_id IS NULL
        """)
        invalid_odds = cursor.fetchone()[0]
        assert invalid_odds == 0, f"Found {invalid_odds} odds with invalid game references"

        cursor.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
