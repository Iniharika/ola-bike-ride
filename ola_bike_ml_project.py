
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ==============================
# 1. Load Dataset
# ==============================
print("Loading dataset...")

df = pd.read_csv("processed_ola_bike_data.csv", parse_dates=["datetime"])

print("\nDataset preview:")
print(df.head())

# ==============================
# 2. Feature Engineering
# ==============================
df = df.sort_values("datetime")

df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

# Lag features (important for ML forecasting)
df["lag_1"] = df["count"].shift(1)
df["lag_24"] = df["count"].shift(24)
df["rolling_mean_3"] = df["count"].rolling(3).mean()

df = df.dropna()

# ==============================
# 3. Features and Target
# ==============================
features = [
    "temperature",
    "humidity",
    "windspeed",
    "hour",
    "day_of_week",
    "month",
    "lag_1",
    "lag_24",
    "rolling_mean_3"
]

X = df[features]
y = df["count"]

# ==============================
# 4. Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("\nTraining size:", len(X_train))
print("Testing size:", len(X_test))

# ==============================
# 5. ML Pipeline
# ==============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    ))
])

print("\nTraining Machine Learning model...")

pipeline.fit(X_train, y_train)

# ==============================
# 6. Predictions
# ==============================
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE:", mae)
print("R2 Score:", r2)

# ==============================
# 7. Visualization
# ==============================

# Graph 1: Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Actual Demand")
plt.plot(y_pred[:100], label="Predicted Demand")
plt.title("Actual vs Predicted Bike Demand")
plt.xlabel("Samples")
plt.ylabel("Demand")
plt.legend()
plt.show()

# Graph 2: Demand Trend
plt.figure(figsize=(10,5))
plt.plot(df["datetime"][:200], df["count"][:200])
plt.title("Bike Demand Trend Over Time")
plt.xlabel("Datetime")
plt.ylabel("Demand")
plt.show()

# Graph 3: Feature Importance
model = pipeline.named_steps["model"]
importance = model.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()

# ==============================
# 8. Future Prediction (Real ML)
# ==============================

print("\n----- Predict Future Ola Bike Demand -----")

temp = float(input("Temperature: "))
hum = float(input("Humidity: "))
wind = float(input("Windspeed: "))
hour = int(input("Hour (0-23): "))
dow = int(input("Day of week (0=Mon,6=Sun): "))
month = int(input("Month: "))

# Using last known demand values for lag features
lag1 = df["count"].iloc[-1]
lag24 = df["count"].iloc[-24]
roll = df["count"].rolling(3).mean().iloc[-1]

sample = pd.DataFrame([{
    "temperature": temp,
    "humidity": hum,
    "windspeed": wind,
    "hour": hour,
    "day_of_week": dow,
    "month": month,
    "lag_1": lag1,
    "lag_24": lag24,
    "rolling_mean_3": roll
}])
prediction = pipeline.predict(sample)

print("\nPredicted Ola Bike Ride Demand:", int(prediction[0]))