import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

print("--- Step 1: Loading California Housing Dataset ---")
housing = fetch_california_housing()
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

print("\n--- Step 2: Dataset Preview (First 5 Rows) ---")
print(df.head())

# Data Preparation
X = df[housing.feature_names]
y = df['PRICE']

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Step 3: Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Step 4: Model Evaluation Results ---")
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Visualizing Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Perfect Fit Line')
plt.xlabel('Actual Price (in $100k)')
plt.ylabel('Predicted Price (in $100k)')
plt.title('Actual vs Predicted House Prices (California Dataset)')
plt.legend()
plt.grid(True)

# Graph-ah image-ah save panrom (Report-la okka)
plt.savefig('actual_vs_predicted.png')
print("\n[SUCCESS] Graph 'actual_vs_predicted.png' nu save aayiruchu!")
plt.show()