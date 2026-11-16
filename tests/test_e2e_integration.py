"""
End-to-end integration tests for the complete microstructure analysis flow.

Feature: convolutional-order-book
Task: 12.1 Create end-to-end integration test

Tests the full flow: ticker → intraday data → heatmap → CNN → recommendation adjustment
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src import market
from src.microstructure import analyze_liquidity, df_to_heatmap
from src.logic import get_trading_recommendation


class TestEndToEndIntegration:
    """
    End-to-end integration tests using real yfinance data.
    
    These tests verify that all components work together correctly
    in the complete flow from ticker to trading recommendation.
    """
    
    @pytest.mark.integration
    def test_full_flow_with_real_data(self):
        """
        Test full flow: ticker → intraday data → heatmap → CNN → recommendation adjustment.
        
        Uses a local intraday frame. The suite blocks sockets, so this does not
        call Yahoo.
        
        **Validates: All requirements**
        """
        ticker = "AAPL"
        dates = pd.date_range(start="2024-01-02 09:30", periods=120, freq="1min")
        local_df = pd.DataFrame(
            {
                "Open": [150 + i * 0.05 for i in range(120)],
                "High": [150.4 + i * 0.05 for i in range(120)],
                "Low": [149.6 + i * 0.05 for i in range(120)],
                "Close": [150.1 + i * 0.05 for i in range(120)],
                "Volume": [1000 + i * 10 for i in range(120)],
            },
            index=dates,
        )

        try:
            with patch("src.market.get_intraday_data", return_value=local_df):
                df = market.get_intraday_data(ticker)
            
                # Verify data was fetched
                assert df is not None, "Intraday data should not be None"
            
                # Verify data structure
                assert 'High' in df.columns, "Data should have High column"
                assert 'Low' in df.columns, "Data should have Low column"
                assert 'Volume' in df.columns, "Data should have Volume column"
            
                # Step 2: Convert to heatmap
                heatmap = df_to_heatmap(df)
            
                # Verify heatmap was generated
                assert heatmap is not None, "Heatmap should not be None"
                assert heatmap.shape == (64, 64), "Heatmap should be 64x64"
                assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0), \
                    "Heatmap values should be in [0, 1]"
            
                # Step 3: Analyze liquidity (CNN inference)
                result = analyze_liquidity(ticker)
            
                # Verify analysis result
                assert 'anomaly_score' in result, "Result should have anomaly_score"
                assert 'status' in result, "Result should have status"
                assert 'confidence' in result, "Result should have confidence"
            
                # Verify anomaly score is in valid range
                assert 0.0 <= result['anomaly_score'] <= 1.0, \
                    f"Anomaly score should be in [0, 1], got {result['anomaly_score']}"
            
                # Verify status is one of the expected values
                valid_statuses = [
                    'Critical liquidity void detected',
                    'High microstructure volatility',
                    'Normal market conditions',
                    'Strong accumulation zone detected',
                    'Insufficient data for microstructure analysis'
                ]
                assert result['status'] in valid_statuses, \
                    f"Status should be one of {valid_statuses}, got {result['status']}"
            
                # Step 4: Test recommendation adjustment
                # Create mock sentiment data
                sentiment_df = pd.DataFrame({
                    'Compound_Score': [0.5] * 10
                })
            
                # Create mock price data
                price_data = pd.DataFrame({
                    'Close': [150 + i * 0.1 for i in range(100)]
                }, index=pd.date_range(start='2024-01-01', periods=100, freq='1D'))
            
                # Get trading recommendation (should include microstructure adjustment)
                recommendation = get_trading_recommendation(
                    ticker=ticker,
                    final_buy_score=6.0,  # BUY signal
                    sentiment_df=sentiment_df,
                    price_data=price_data,
                    strategy='momentum'
                )
            
                # Verify recommendation structure
                assert 'recommendation' in recommendation, "Should have recommendation"
                assert 'confidence' in recommendation, "Should have confidence"
                assert 'reasoning' in recommendation, "Should have reasoning"
            
                # Verify confidence is in valid range
                assert 0 <= recommendation['confidence'] <= 100, \
                    f"Confidence should be in [0, 100], got {recommendation['confidence']}"
            
                # Verify no exceptions were raised
                print(f"✓ Full flow completed successfully for {ticker}")
                print(f"  - Fetched {len(df)} minutes of intraday data")
                print(f"  - Generated {heatmap.shape} heatmap")
                print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
                print(f"  - Status: {result['status']}")
                print(f"  - Recommendation: {recommendation['recommendation']}")
                print(f"  - Confidence: {recommendation['confidence']:.1f}")
            
        except Exception as e:
            pytest.fail(f"End-to-end test failed with exception: {e}")
    
    @pytest.mark.integration
    def test_full_flow_with_mocked_data(self):
        """
        Test full flow with mocked data to ensure deterministic behavior.
        
        This test uses mocked data to verify the complete flow works
        correctly without depending on external APIs.
        
        **Validates: All requirements**
        """
        ticker = "TEST"
        
        # Create mock intraday data
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(100)],
            'High': [100.5 + i * 0.1 for i in range(100)],
            'Low': [99.5 + i * 0.1 for i in range(100)],
            'Close': [100 + i * 0.1 for i in range(100)],
            'Volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Mock get_intraday_data
        with patch('src.market.get_intraday_data', return_value=mock_df):
            # Step 1: Fetch intraday data through the module the patch replaces.
            df = market.get_intraday_data(ticker)
            assert not df.empty, "Should have data"
            
            # Step 2: Convert to heatmap
            heatmap = df_to_heatmap(df)
            assert heatmap is not None, "Heatmap should be generated"
            assert heatmap.shape == (64, 64), "Heatmap should be 64x64"
            
            # Step 3: Analyze liquidity
            result = analyze_liquidity(ticker)
            assert 'anomaly_score' in result, "Should have anomaly score"
            assert 0.0 <= result['anomaly_score'] <= 1.0, "Score should be in [0, 1]"
            
            # Step 4: Test recommendation adjustment
            sentiment_df = pd.DataFrame({
                'Compound_Score': [0.5] * 10
            })
            
            price_data = pd.DataFrame({
                'Close': [100 + i * 0.1 for i in range(100)]
            }, index=pd.date_range(start='2024-01-01', periods=100, freq='1D'))
            
            recommendation = get_trading_recommendation(
                ticker=ticker,
                final_buy_score=6.0,
                sentiment_df=sentiment_df,
                price_data=price_data,
                strategy='momentum'
            )
            
            # Verify recommendation
            assert recommendation['recommendation'] in ['BUY', 'SELL', 'HOLD'], \
                "Should have valid recommendation"
            assert 0 <= recommendation['confidence'] <= 100, \
                "Confidence should be in valid range"
            
            print(f"✓ Mocked flow completed successfully")
            print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
            print(f"  - Recommendation: {recommendation['recommendation']}")
            print(f"  - Confidence: {recommendation['confidence']:.1f}")
    
    @pytest.mark.integration
    def test_no_exceptions_raised_in_full_flow(self):
        """
        Test that no exceptions are raised during the full flow.
        
        This test verifies that the system handles all edge cases gracefully
        and never crashes, even with problematic data.
        
        **Validates: Requirements 6.1, 6.2**
        """
        ticker = "TEST"
        
        # Test with various edge cases
        test_cases = [
            # Empty data
            pd.DataFrame(),
            # Insufficient data
            pd.DataFrame({
                'High': [100],
                'Low': [99],
                'Volume': [1000]
            }),
            # Zero price range
            pd.DataFrame({
                'High': [100] * 64,
                'Low': [100] * 64,
                'Volume': [1000] * 64
            }, index=pd.date_range(start='2024-01-01', periods=64, freq='1min')),
            # NaN values
            pd.DataFrame({
                'High': [100 + i * 0.1 if i % 2 == 0 else np.nan for i in range(64)],
                'Low': [99 + i * 0.1 if i % 2 == 0 else np.nan for i in range(64)],
                'Volume': [1000 + i * 10 if i % 2 == 0 else np.nan for i in range(64)]
            }, index=pd.date_range(start='2024-01-01', periods=64, freq='1min'))
        ]
        
        for i, test_df in enumerate(test_cases):
            try:
                with patch('src.market.get_intraday_data', return_value=test_df):
                    # Should not raise any exceptions
                    result = analyze_liquidity(ticker)
                    
                    # Should always return a valid result
                    assert 'anomaly_score' in result
                    assert 'status' in result
                    assert 'confidence' in result
                    
                    print(f"✓ Test case {i+1} handled gracefully")
                    
            except Exception as e:
                pytest.fail(f"Test case {i+1} raised exception: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
