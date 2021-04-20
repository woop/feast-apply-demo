from datetime import datetime, timedelta

import pandas as pd
from feast import FeatureStore
from joblib import dump
from sklearn.linear_model import LinearRegression

import helpers

# Load driver order data
orders = pd.read_csv("driver_orders.csv", sep="\t")
orders["event_timestamp"] = pd.to_datetime(orders["event_timestamp"])

# Set up feature store
fs = FeatureStore(repo_path="driver_ranking/")

# Retrieve training data from BigQuery
training_df = fs.get_historical_features(
    entity_df=orders,
    feature_refs=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ],
).to_df()

# # Print output
# print(training_df)

# Train model
target = "trip_completed"

reg = LinearRegression()
train_X = training_df[training_df.columns.drop(target).drop("event_timestamp")]
train_Y = training_df.loc[:, target]
reg.fit(train_X, train_Y)
dump(reg, "driver_model.bin")
