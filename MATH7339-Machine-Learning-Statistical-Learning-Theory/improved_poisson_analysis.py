#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved US Accidents Poisson Regression Analysis
With enhanced feature engineering, data cleaning, and imputation
"""

# =============================================================================
# SECTION 1: SETUP - IMPORTS AND CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
import statsmodels.api as sm
from statsmodels.api import Poisson
from patsy import dmatrices
import warnings
warnings.filterwarnings('ignore')

# --- Configuration Constants ---
CSV_PATH = "US_Accidents_March23.csv"

# Weather variables of interest
WEATHER_VARS = [
    "Temperature(F)",
    "Precipitation(in)",
    "Wind_Speed(mph)",
    "Humidity(%)"
]

# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

print("Loading US Accidents data...")
df = pd.read_csv(CSV_PATH, low_memory=False)

print(f"Dataset shape: {df.shape}")

# Convert Start_Time to datetime
print("\nConverting Start_Time to datetime...")
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

# Remove rows with invalid dates
df = df.dropna(subset=['Start_Time'])

print(f"\nDate range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")

# =============================================================================
# SECTION 3: ADVANCED DATA CLEANING AND IMPUTATION
# =============================================================================

print("\nPerforming advanced data cleaning and imputation...")

# Handle missing values in weather variables using grouped imputation
print("Imputing missing weather data...")

# Extract date components for imputation
df['Date'] = df['Start_Time'].dt.date
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['State'] = df['State']  # Keep state for imputation

# Impute missing weather data using state and month medians
for col in WEATHER_VARS:
    if col in df.columns:
        # Calculate medians for different levels of granularity
        sm_medians = df.groupby(['State', 'Month'])[col].transform('median')
        state_medians = df.groupby('State')[col].transform('median')
        global_median = df[col].median()
        
        # Apply imputation in order of preference
        df[col] = df[col].fillna(sm_medians).fillna(state_medians).fillna(global_median)
        print(f"  {col}: {df[col].isna().sum()} missing values after imputation")

# Clip weather data to reasonable ranges (outlier mitigation)
print("\nClipping weather data outliers...")
clip_bounds = {
    "Temperature(F)": (-50, 130), 
    "Wind_Speed(mph)": (0, 120),
    "Precipitation(in)": (0, 10),
    "Humidity(%)": (0, 100),
}

for col, (lo, hi) in clip_bounds.items():
    if col in df.columns:
        before_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f"  Clipped {col}: {before_outliers} values were outside [{lo}, {hi}]")

# =============================================================================
# SECTION 4: ENHANCED FEATURE ENGINEERING
# =============================================================================

print("\nPerforming enhanced feature engineering...")

# Extract more detailed temporal features
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Day'] = df['Start_Time'].dt.day
df['Hour'] = df['Start_Time'].dt.hour

# Create cyclical features for month and hour
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# Create interaction terms
df['Temp_Humidity'] = df['Temperature(F)'] * df['Humidity(%)']
df['Wind_Precip'] = df['Wind_Speed(mph)'] * df['Precipitation(in)']

# Create polynomial terms for temperature
df['Temp_Squared'] = df['Temperature(F)'] ** 2

# Create log-transformed features (handling zeros)
df['Precip_log'] = np.log1p(df['Precipitation(in)'])
df['Wind_log'] = np.log1p(df['Wind_Speed(mph)'])

# Create categorical features
df['Season'] = df['Month'].apply(lambda x: (x%12 + 3)//3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
df['TimeOfDay'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

print("Feature engineering complete.")
print(f"Dataset shape after feature engineering: {df.shape}")

# =============================================================================
# SECTION 5: DAILY AGGREGATION WITH ENHANCED FEATURES
# =============================================================================

print("\nAggregating data to daily level with enhanced features...")

# Count accidents per day
daily_accidents = df.groupby('Date').size().reset_index(name='ACCIDENT_COUNT')

# Calculate average weather conditions and other features per day
daily_features = df.groupby('Date').agg({
    'Temperature(F)': 'mean',
    'Precipitation(in)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Humidity(%)': 'mean',
    'Temp_Humidity': 'mean',
    'Wind_Precip': 'mean',
    'Temp_Squared': 'mean',
    'Precip_log': 'mean',
    'Wind_log': 'mean',
    'DayOfWeek': 'first',  # All rows for a date have the same DayOfWeek
    'Day': 'first',        # All rows for a date have the same Day
    'Month': 'first',      # All rows for a date have the same Month
    'Month_sin': 'mean',
    'Month_cos': 'mean',
    'Season': 'first'      # All rows for a date have the same Season
}).reset_index()

# Merge into a single daily dataset
daily_data = daily_accidents.merge(daily_features, on='Date', how='inner')

# Convert Date to datetime
daily_data['Date'] = pd.to_datetime(daily_data['Date'])

# Add derived features at daily level
daily_data['Year'] = daily_data['Date'].dt.year
daily_data['DayOfYear'] = daily_data['Date'].dt.dayofyear

print(f"Daily dataset shape: {daily_data.shape}")
print(f"Date range: {daily_data['Date'].min()} to {daily_data['Date'].max()}")

# =============================================================================
# SECTION 6: ENHANCED DATA CLEANING
# =============================================================================

print("\nPerforming enhanced data cleaning...")

# Check for missing values
print("Missing values before final cleaning:")
print(daily_data.isnull().mean() * 100)

# Drop rows with missing values
daily_data = daily_data.dropna()

print(f"Dataset shape after removing missing values: {daily_data.shape}")

# Check for outliers in accident counts
print("\nSummary statistics for daily accident counts:")
print(daily_data['ACCIDENT_COUNT'].describe())

# Remove extreme outliers (accident counts > 3 standard deviations from mean)
mean_accidents = daily_data['ACCIDENT_COUNT'].mean()
std_accidents = daily_data['ACCIDENT_COUNT'].std()
upper_bound = mean_accidents + 3 * std_accidents
lower_bound = mean_accidents - 3 * std_accidents

outliers = daily_data[(daily_data['ACCIDENT_COUNT'] > upper_bound) | 
                      (daily_data['ACCIDENT_COUNT'] < lower_bound)]
print(f"\nRemoving {len(outliers)} outlier days with extreme accident counts...")
daily_data = daily_data[(daily_data['ACCIDENT_COUNT'] <= upper_bound) & 
                        (daily_data['ACCIDENT_COUNT'] >= lower_bound)]

print(f"Dataset shape after outlier removal: {daily_data.shape}")

# =============================================================================
# SECTION 7: TRAIN/TEST SPLIT WITH TEMPORAL ORDER
# =============================================================================

print("\nCreating train/test split with temporal ordering...")

# Sort by date to maintain temporal order
daily_data = daily_data.sort_values('Date').reset_index(drop=True)

# Use 80% for training, 20% for testing
split_idx = int(len(daily_data) * 0.8)
df_train = daily_data[:split_idx]
df_test = daily_data[split_idx:]

print(f'Training data set length: {len(df_train)}')
print(f'Testing data set length: {len(df_test)}')

# Verify temporal split
print(f"Training date range: {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Testing date range: {df_test['Date'].min()} to {df_test['Date'].max()}")

# =============================================================================
# SECTION 8: ENHANCED POISSON REGRESSION MODELING
# =============================================================================

print("\nSetting up enhanced Poisson regression model...")

# Setup the regression expression with enhanced features
expr = """ACCIDENT_COUNT ~ Day + Q("DayOfWeek") + Q("Month_sin") + Q("Month_cos") + 
          Q("Temperature(F)") + Q("Temp_Squared") + Q("Humidity(%)") + 
          Precip_log + Wind_log + Wind_Precip + Temp_Humidity +
          C(Season) + C(Year)"""

# Set up the X and y matrices for the training and testing data sets
y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

# Ensure X_test has the same columns as X_train (in the same order)
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Add missing columns to X_test with zeros if needed
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0

# Reorder X_test columns to match X_train
X_test = X_test[X_train.columns]

print(f"\nTraining features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# Check column names
print("\nTraining feature columns:")
print(X_train.columns.tolist())
print("\nTesting feature columns:")
print(X_test.columns.tolist())

print("\nTraining enhanced Poisson regression model...")
poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

print("\n" + "="*70)
print("ENHANCED POISSON REGRESSION RESULTS")
print("="*70)
print(poisson_training_results.summary())

# =============================================================================
# SECTION 9: MODEL PREDICTIONS AND EVALUATION
# =============================================================================

print("\nMaking predictions on test data...")

# Get predictions
poisson_predictions = poisson_training_results.get_prediction(X_test)
predictions_summary_frame = poisson_predictions.summary_frame()

# Extract predicted and actual values
predicted_counts = predictions_summary_frame['mean']
actual_counts = y_test['ACCIDENT_COUNT']

# Calculate evaluation metrics
rmse = np.sqrt(np.mean((actual_counts - predicted_counts) ** 2))
mae = np.mean(np.abs(actual_counts - predicted_counts))

print(f"\nModel Performance on Test Set:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Calculate R-squared for predictions
ss_res = np.sum((actual_counts - predicted_counts) ** 2)
ss_tot = np.sum((actual_counts - np.mean(actual_counts)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"Prediction R-squared: {r_squared:.4f}")

# =============================================================================
# SECTION 10: VISUALIZATION
# =============================================================================

print("\nCreating visualizations...")

# Plot 1: Predicted versus actual counts over time
plt.figure(figsize=(15, 6))
plt.plot(df_test['Date'], predicted_counts, 'go-', label='Predicted counts', markersize=3, alpha=0.7)
plt.plot(df_test['Date'], actual_counts, 'ro-', label='Actual counts', markersize=3, alpha=0.7)
plt.title('Predicted versus Actual Daily Accident Counts (Enhanced Model)')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('enhanced_predicted_vs_actual_time.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Scatter plot of Actual versus Predicted counts
plt.figure(figsize=(8, 6))
plt.scatter(predicted_counts, actual_counts, alpha=0.6)
plt.plot([0, max(max(predicted_counts), max(actual_counts))], 
         [0, max(max(predicted_counts), max(actual_counts))], 'r--', lw=2)
plt.xlabel('Predicted Accident Counts')
plt.ylabel('Actual Accident Counts')
plt.title('Scatter Plot: Actual vs Predicted Accident Counts (Enhanced Model)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('enhanced_predicted_vs_actual_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Residuals plot
residuals = actual_counts - predicted_counts
plt.figure(figsize=(10, 6))
plt.scatter(predicted_counts, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Accident Counts')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Accident Counts (Enhanced Model)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('enhanced_residuals_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# SECTION 11: MODEL INTERPRETATION
# =============================================================================

print("\nInterpreting model coefficients...")

# Extract coefficients
coef_df = pd.DataFrame({
    'Variable': poisson_training_results.params.index,
    'Coefficient': poisson_training_results.params.values,
    'Std_Error': poisson_training_results.bse.values,
    'P_Value': poisson_training_results.pvalues.values
})

# Calculate incidence rate ratios (IRR)
coef_df['IRR'] = np.exp(coef_df['Coefficient'])
coef_df['IRR_Lower'] = np.exp(coef_df['Coefficient'] - 1.96 * coef_df['Std_Error'])
coef_df['IRR_Upper'] = np.exp(coef_df['Coefficient'] + 1.96 * coef_df['Std_Error'])

print("\nIncidence Rate Ratios (IRR) with 95% Confidence Intervals:")
print(coef_df[['Variable', 'IRR', 'IRR_Lower', 'IRR_Upper', 'P_Value']].to_string(index=False))

# Print key significant findings
print("\nKey Significant Findings (p < 0.05):")
significant_coefs = coef_df[coef_df['P_Value'] < 0.05]
for _, row in significant_coefs.iterrows():
    if row['Variable'] == 'Intercept':
        continue
    effect = "increase" if row['IRR'] > 1 else "decrease"
    percent_change = abs(row['IRR'] - 1) * 100
    print(f"  - {row['Variable']}: {effect} accidents by {percent_change:.2f}% per unit change (p = {row['P_Value']:.2e})")

# =============================================================================
# SECTION 12: MODEL GOODNESS OF FIT
# =============================================================================

print("\nModel Goodness of Fit:")
print(f"Deviance: {poisson_training_results.deviance:.2f}")
print(f"Pearson chi2: {poisson_training_results.pearson_chi2:.2f}")

# Calculate pseudo R-squared
null_deviance = poisson_training_results.null_deviance
deviance = poisson_training_results.deviance
pseudo_r2 = 1 - (deviance / null_deviance)
print(f"Pseudo R-squared: {pseudo_r2:.4f}")

# Check for overdispersion
pearson_chi2 = np.sum(poisson_training_results.resid_pearson**2)
dispersion = pearson_chi2 / poisson_training_results.df_resid
print(f"Dispersion statistic: {dispersion:.2f}")
if dispersion > 1.5:
    print("Warning: Evidence of overdispersion. Consider Negative Binomial model.")

print("\nEnhanced analysis complete!")
print("Generated files:")
print("  - enhanced_predicted_vs_actual_time.png")
print("  - enhanced_predicted_vs_actual_scatter.png")
print("  - enhanced_residuals_plot.png")