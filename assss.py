import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import networkx as nx


# --- Load dataset ---
data = pd.read_csv(r"C:\Users\ashta\OneDrive\Desktop\ai\routes.csv")
print("📄 Dataset Preview:")
print(data.head(), "\n")

# --- Prepare features (X) and target (y) ---
X = data[["time_of_day", "day_of_week", "distance_km", "avg_speed"]]
y = data["travel_time"]

# --- Train a simple AI model to predict travel time ---
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)
print("✅ Model trained successfully!\n")

# --- Define possible routes to predict ---
routes = {
    "A-B": [8, 1, 5, 40],
    "A-C": [14, 3, 10, 50],
    "B-C": [18, 5, 8, 35],
}

# --- Predict travel time for each route ---
predictions = {}
for route, features in routes.items():
    # Make sure we pass data as a DataFrame to avoid warnings
    features_df = pd.DataFrame([features], columns=X.columns)
    predicted_time = model.predict(features_df)[0]
    predictions[route] = predicted_time
    print(f"Predicted travel time for {route}: {predicted_time:.2f} minutes")

# --- Build a simple graph of routes ---
G = nx.Graph()
G.add_edge("A", "B", weight=predictions["A-B"])
G.add_edge("A", "C", weight=predictions["A-C"])
G.add_edge("B", "C", weight=predictions["B-C"])

# --- Find the best route (shortest predicted time) ---
best_path = nx.shortest_path(G, source="A", target="C", weight="weight")
best_time = nx.shortest_path_length(G, source="A", target="C", weight="weight")

print("\n🚗 Best Route:", best_path)
print(f"🕒 Estimated total travel time: {best_time:.2f} minutes")
