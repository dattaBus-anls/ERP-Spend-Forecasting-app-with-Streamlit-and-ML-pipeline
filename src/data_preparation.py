# ============================================================
# 📦 Import Required Libraries
# ============================================================
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from fredapi import Fred

# ✅ Load environment variables for FRED API key
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key=fred_api_key)

print("=" * 100)
print("📂 Loading raw procurement dataset...")
raw_file_path = os.path.join("data", "raw", "raw_procurement_kpi_dataset.csv")
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"❌ Raw file not found at: {raw_file_path}")
df = pd.read_csv(raw_file_path, encoding='utf-8')
df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]
print("✅ Dataset loaded successfully.")


print("=" * 100)
print("📦 Assigning item categories and departments...")
categories = ['Office Supplies', 'Packaging', 'MRO', 'Raw Materials', 'Electronics',
              'Chemicals', 'Services', 'Metals', 'Manufacturing', 'Food Products']
departments = ['Production', 'Logistics', 'Procurement', 'Finance', 'Maintenance']
np.random.seed(1)
df['Item_Category'] = np.random.choice(categories, size=len(df))
np.random.seed(42)
df['Department'] = np.random.choice(departments, size=len(df))

print("=" * 100)
print("🛠️ Creating derived columns...")
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
df['Order Month'] = df['Order_Date'].dt.to_period('M').astype(str)
df['Spend'] = df['Quantity'] * df['Negotiated_Price']
df['Lead Time (Days)'] = (df['Delivery_Date'] - df['Order_Date']).dt.days

print("=" * 100)
print("📊 Summary BEFORE cleaning:")
print(df[['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'Spend']].describe())


print("➕ Adding interaction feature: Quantity × Lead Time")
df['Qty_LeadTime_Interaction'] = df['Quantity'] * df['Lead Time (Days)']


print("=" * 100)
print("💲 Calculating Cost per Unit, Markup %, and Flagging Outliers...")
df['Cost_per_Unit'] = df['Spend'] / df['Quantity']
df['Markup_%'] = ((df['Cost_per_Unit'] - df['Negotiated_Price']) / df['Negotiated_Price']) * 100
df['High_Markup_Flag'] = df['Markup_%'] > 300
print("✅ Markup analysis complete. Sample:")
print(df[['Quantity', 'Negotiated_Price', 'Cost_per_Unit', 'Markup_%', 'High_Markup_Flag']].head())



print("=" * 100)
print("🌐 Fetching PPI data by category...")
ppi_series_map = {
    'Office Supplies': 'WPU0911',
    'Packaging': 'WPU091',
    'MRO': 'WPU114',
    'Raw Materials': 'WPU061',
    'Electronics': 'WPU117',
    'Chemicals': 'WPU065',
    'Services': 'WPU381',
    'Metals': 'WPU101',
    'Manufacturing': 'WPU114',
    'Food Products': 'WPU012'
}
ppi_df_all = pd.DataFrame()

for category in df['Item_Category'].unique():
    series_id = ppi_series_map.get(category)
    if series_id:
        try:
            print(f"🔎 Fetching: {category} ({series_id})")
            ppi_series = fred.get_series(series_id)
            ppi_series.index = pd.to_datetime(ppi_series.index)
            ppi_temp = ppi_series.resample('MS').mean().reset_index()
            ppi_temp['Order Month'] = ppi_temp['index'].dt.to_period('M').astype(str)
            ppi_temp['Item_Category'] = category
            ppi_temp = ppi_temp[['Order Month', 'Item_Category', 0]]
            ppi_temp.columns = ['Order Month', 'Item_Category', 'PPI']
            ppi_df_all = pd.concat([ppi_df_all, ppi_temp], ignore_index=True)
        except Exception as e:
            print(f"❌ Error for {category}: {e}")

# 🔁 Merge and fill missing PPI
print("🔁 Merging PPI values into main dataset...")
df = pd.merge(df, ppi_df_all, on=['Order Month', 'Item_Category'], how='left')
df['PPI'] = df.groupby('Item_Category')['PPI'].transform(lambda x: x.fillna(x.rolling(12, 1).mean()))


print("=" * 100)
print("🔍 BEFORE Imputation - Missing Summary:")
print(df[['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'PPI', 'Qty_LeadTime_Interaction']].isnull().sum())


print("=" * 100)
print("🧼 Handling missing values...")
for col in ['Unit_Price', 'Negotiated_Price', 'PPI']:
    df[col] = df.groupby('Item_Category')[col].transform(lambda x: x.fillna(x.mean()))
df['Defective_Units'] = df['Defective_Units'].fillna(df['Defective_Units'].mean())
df['Delivery_Date'] = df['Delivery_Date'].fillna(df['Delivery_Date'].mode()[0])
df['Lead Time (Days)'] = df['Lead Time (Days)'].fillna(df['Lead Time (Days)'].mean())

# Recalculate interaction after lead time imputation
df['Qty_LeadTime_Interaction'] = df['Quantity'] * df['Lead Time (Days)']

print("🔍 AFTER Imputation - Missing Summary:")
print(df[['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'PPI', 'Qty_LeadTime_Interaction']].isnull().sum())


print("=" * 100)
print("🚫 Removing zero or negative Quantity and Price...")
before_filter = df.shape[0]
df = df[(df['Quantity'] > 0) & (df['Negotiated_Price'] > 0)]
after_filter = df.shape[0]
print(f"✅ Removed {before_filter - after_filter} invalid rows with zero/negative values")


print("=" * 100)
print("🧹 Removing outliers using Z-score method...")
outlier_cols = ['Unit_Price', 'Negotiated_Price', 'Defective_Units', 'Lead Time (Days)']
before_outliers = df.shape[0]
z_scores = np.abs(zscore(df[outlier_cols]))
df = df[(z_scores < 3).all(axis=1)]
after_outliers = df.shape[0]
print(f"✅ Outliers removed: {before_outliers - after_outliers} rows")

print("=" * 100)
print("🔧 Fixing negative lead times and interactions...")
df['Lead Time (Days)'] = df['Lead Time (Days)'].apply(lambda x: max(x, 0))
df['Qty_LeadTime_Interaction'] = df['Quantity'] * df['Lead Time (Days)']


print("=" * 100)
print("📊 Summary AFTER cleaning:")
print(df[['Quantity', 'Negotiated_Price', 'Lead Time (Days)', 'Spend', 'PPI', 'Markup_%']].describe())


print("=" * 100)
print("📈 Saving cleaned data...")
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/cleaned_procurement_with_ppi_extended.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned file saved at: {output_path}")

print("🎉 Data preparation complete!")
