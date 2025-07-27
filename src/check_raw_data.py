import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/raw_procurement_kpi_dataset.csv")

print("=" * 50)
print("📊 RAW DATA ANALYSIS")
print("=" * 50)
print("Quantity statistics from raw data:")
print(df['Quantity'].describe())
print(f"\n📈 Max quantity in raw data: {df['Quantity'].max():,}")
print(f"📉 Min quantity in raw data: {df['Quantity'].min():,}")
print(f"📊 Total records: {len(df):,}")

# Show distribution
print(f"\n📋 Quantity distribution:")
print(f"  < 1,000: {len(df[df['Quantity'] < 1000]):,} records")
print(f"  1,000-5,000: {len(df[(df['Quantity'] >= 1000) & (df['Quantity'] <= 5000)]):,} records") 
print(f"  5,000-10,000: {len(df[(df['Quantity'] > 5000) & (df['Quantity'] <= 10000)]):,} records")
print(f"  > 10,000: {len(df[df['Quantity'] > 10000]):,} records")