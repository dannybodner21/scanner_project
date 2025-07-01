import os
import json
import xgboost as xgb
import pandas as pd

from django.core.management.base import BaseCommand

# Set your path to the saved model and JSON here
MODEL_PATH = "seven_long_xgb_model.bin"
JSON_PATH = "live_predictions_log.jsonl"

INPUT_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'adx_14', 'ma_200', 'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h',
    'momentum', 'volume_ma_20', 'vol_spike', 'rsi_14', 'macd', 'macd_signal',
    'macd_hist', 'bb_upper', 'bb_lower', 'atr_14', 'obv', 'obv_slope',
    'ema_9', 'ema_21', 'ema_diff', 'volatility', 'bull_regime', 'bear_regime',
    'sideways_regime', 'slope_1h', 'dist_from_high_24h', 'dist_from_low_24h',
    'stoch_k', 'stoch_d', 'price_change_5', 'volume_change_5',
    'high_1h', 'low_1h', 'pos_in_range_1h', 'vwap_1h', 'pos_vs_vwap'
]

class Command(BaseCommand):
    help = "Verify predicted probabilities from saved JSONs using long model"

    def handle(self, *args, **kwargs):
        if not os.path.exists(MODEL_PATH):
            self.stderr.write(f"Model file not found at {MODEL_PATH}")
            return

        if not os.path.exists(JSON_PATH):
            self.stderr.write(f"JSON file not found at {JSON_PATH}")
            return

        # Load model
        model = xgb.Booster()
        model.load_model(MODEL_PATH)

        with open(JSON_PATH, 'r') as f:
            lines = f.readlines()

        for line in lines:
            try:
                record = json.loads(line)
                features = record['features']
                orig_prob = record.get('predicted_long_prob')

                # Ensure feature alignment
                df = pd.DataFrame([features])
                df = df[INPUT_COLUMNS]

                dmatrix = xgb.DMatrix(df, feature_names=INPUT_COLUMNS)
                pred = model.predict(dmatrix)[0]

                self.stdout.write(
                    f"{record['timestamp']} - {record['coin']}\n"
                    f"🔹 Original prob: {orig_prob:.6f}\n"
                    f"🔹 Recomputed:    {pred:.6f}\n"
                    f"{'✅ Match' if abs(pred - orig_prob) < 1e-5 else '❌ Mismatch'}\n"
                )

            except Exception as e:
                self.stderr.write(f"Error processing line: {e}")
