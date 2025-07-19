import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# 1. Load dataset
df = pd.read_csv('./output/features.csv')

# 2. Prepare data
X = df.drop(['wallet', 'score'], axis=1)
y = df['score']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} | R²: {r2:.2f}")

# 6. Save model
os.makedirs('./output', exist_ok=True)
joblib.dump(model, './output/model.pkl')
print("Model saved as model.pkl")

# 7. Plot predictions vs actual
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('True Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Wallet Scores')
plt.grid(True)
plt.savefig('./output/prediction_vs_actual.png')
