import pandas as pd
import numpy as np

print("Step 1: Reading data...")
df = pd.read_csv('add.csv', header=None)
print(f"Shape: {df.shape}")

print("\nStep 2: Handling missing values...")
df = df.replace('?', np.nan)

print("\nStep 3: Dropping last column...")
df = df.drop(df.columns[-1], axis=1)

print("\nStep 4: Converting to numeric...")
df = df.apply(pd.to_numeric, errors='coerce')

print("\nStep 5: Dropping NaN rows...")
df_clean = df.dropna()
print(f"Cleaned shape: {df_clean.shape}")

print("\nStep 6: Basic stats...")
print(f"曝光量 - min: {df_clean[1].min()}, max: {df_clean[1].max()}, mean: {df_clean[1].mean():.2f}")
print(f"点击量 - min: {df_clean[2].min()}, max: {df_clean[2].max()}, mean: {df_clean[2].mean():.2f}")
print(f"点击率 - min: {df_clean[3].min():.4f}, max: {df_clean[3].max():.4f}, mean: {df_clean[3].mean():.4f}")

print("\nDone!")