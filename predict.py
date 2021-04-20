import random
from typing import List

import pandas as pd
from feast import FeatureStore
from joblib import load
import random

import helpers


class DriverRankingModel:
    def __init__(self):
        # Load model
        self.model = load("driver_model.bin")

        # Set up feature store
        self.fs = FeatureStore(repo_path="driver_ranking/")

    def predict(self, driver_ids):
        # Read features from Feast
        driver_features = self.fs.get_online_features(
            entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
            feature_refs=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
        )
        features_df = pd.DataFrame.from_dict(driver_features.to_dict())

        # Make prediction
        features_df["prediction"] = self.model.predict(features_df)

        # Choose best driver
        best_driver_id = features_df["driver_id"].iloc[
            features_df["prediction"].argmax()
        ]

        # return best driver
        return best_driver_id


if __name__ == "__main__":
    drivers = [1001, 1002, 1003, 1004]
    model = DriverRankingModel()
    best_driver = model.predict(drivers)
    print(best_driver)
