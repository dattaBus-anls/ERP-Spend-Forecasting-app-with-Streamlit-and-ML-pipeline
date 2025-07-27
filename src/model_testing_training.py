# ====================================================================================================
# 🧠 Regression Model Training Script
# ----------------------------------------------------------------------------------------------------
# Trains multiple regressors on ERP procurement data to forecast Spend.
# Models: Linear Regression (baseline), Ridge, Lasso, Random Forest, XGBoost
# Metrics: MAE, RMSE, R^2; Best model saved to disk.
# Output Artifacts: best_model.joblib, feature_names.joblib, model_metadata.joblib
# ====================================================================================================

# ============================================================
# 📦 Import Required Libraries
# ============================================================
import pandas as pd
import numpy as np
import os
import joblib
from joblib import dump
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  
from datetime import datetime

print("=" * 100)
print("📂 Loading cleaned dataset...")
data_path = os.path.join("data", "processed", "cleaned_procurement_with_ppi_extended.csv")
df = pd.read_csv(data_path)
print(f"✅ Dataset shape: {df.shape}")

# ============================================================
# 🧠 Define Feature Set and Target Variable
# ============================================================
print("=" * 100)
print("🧪 Selecting features and target for modeling...")
# Removing this line 
# features = ['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'PPI', 'Qty_LeadTime_Interaction']
features = ['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'PPI']
target = 'Spend'

X = df[features]
y = df[target]
print(f"✅ Features used: {features}")

# ============================================================
# ✂️ Train/Test Split (80/20)
# ============================================================
print("=" * 100)
print("✂️ Splitting data into training and testing sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================
# 📉 Feature Scaling (for Linear Models)
# ============================================================
print("📉 Scaling features for linear models only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Feature scaling complete (Linear, Ridge, Lasso will use this)")


# ✅ Save the scaler object for future use in Streamlit
dump(scaler, "models/feature_scaler.joblib")
print("✅ Feature scaling complete and scaler saved to models/feature_scaler.joblib")


# ============================================================
# 🤖 Initialize Models
# ============================================================
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

results = {}
all_metrics = []

# ============================================================
# 🚂 Model Training & Evaluation
# ============================================================
print("=" * 100)
print("🚀 Training models and evaluating performance...")
for name, model in models.items():
    print(f"\n🔍 Model: {name}")

    # ▶️ Use scaled data for linear models only
    if name in ['LinearRegression', 'Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print("⚠️ Scaled data used for linear model.")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # 🧮 Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 📊 Print Results
    print(f"📉 MAE: {mae:.2f}")
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 R²: {r2:.4f}")
    print(f"🔁 CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 🧠 Save results
    results[name] = {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_r2_mean': cv_scores.mean()
    }

    all_metrics.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'CV_R2': cv_scores.mean()
    })

# ============================================================
# 🏆 Select Best Model by R²
# ============================================================
print("=" * 100)
print("🏆 Selecting best model based on R² score...")
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
feature_names = features
model_metadata = results[best_model_name]

# 🔍 Feature Importance Plot (for tree-based models only)
if best_model_name in ['RandomForest', 'XGBoost']:
    print(f"📊 Plotting Feature Importances for: {best_model_name}")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 📉 Bar Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Feature Importance: {best_model_name}')
    plt.tight_layout()

    # 📅 Save plot
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/feature_importance_plot.png")
    print("✅ Feature importance plot saved as: models/feature_importance_plot.png")

    # 📁 Show top 5 in console
    top_features = feature_importance_df.head(5)
    print("📌 Top 5 Important Features:")
    for _, row in top_features.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
print("=" * 100)
# ============================================================
# 📆 Model Versioning and Saving Artifacts
# ============================================================
os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/best_model_{best_model_name}_{timestamp}.joblib"
feature_path = f"models/feature_names_{timestamp}.joblib"
meta_path = f"models/model_metadata_{timestamp}.joblib"

dump(best_model, model_path)
dump(feature_names, feature_path)
dump(model_metadata, meta_path)

dump(best_model, "models/best_model.joblib")
dump(feature_names, "models/feature_names.joblib")
dump(model_metadata, "models/model_metadata.joblib")

print(f"✅ Model saved as: {model_path}")
print(f"📦 Artifacts also versioned for future traceability.")


print("=" * 100)
# 📊 Save evaluation summary CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(f"models/model_evaluation_summary_{timestamp}.csv", index=False)
print("📜 Model evaluation summary saved to CSV.")
print("=" * 100)

# ============================================================
# 📈 Predicted vs Actual Plot
# ============================================================
print("📊 Generating model evaluation plots...")
plt.figure(figsize=(6,6))
plt.scatter(y_test, best_model.predict(X_test), alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Spend")
plt.ylabel("Predicted Spend")
plt.title(f"{best_model_name}: Actual vs Predicted Spend")
plt.tight_layout()
plt.savefig("models/predicted_vs_actual.png")
print("✅ Plot saved to models/predicted_vs_actual.png")
print("=" * 100)

# ✅ Residuals Plot
print("📊 Generating residual plots...")
residuals = y_test - best_model.predict(X_test)
print("=" * 100)

# Residual vs Actual Plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Spend")
plt.ylabel("Residuals")
plt.title("Residuals vs Actual Spend")
plt.tight_layout()
plt.savefig("models/residuals_vs_actual.png")
print("✅ Residuals vs Actual plot saved.")
print("=" * 100)

# Residual Histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("models/residuals_histogram.png")
print("✅ Residuals histogram saved.")
print("=" * 100)

# 📢 Final Summary of Best Model
print("-" * 100)
print(f"✅ Final Selected Model: {best_model_name}")
print(f"🔍 R² Score: {model_metadata['r2']:.4f}")
print(f"📉 MAE: {model_metadata['mae']:.2f}")
print(f"📉 RMSE: {model_metadata['rmse']:.2f}")
print("📌 Justification: Selected based on highest R² on test data.")


print("=" * 100)
print("🌟 Model training complete. Best model ready for deployment!")
