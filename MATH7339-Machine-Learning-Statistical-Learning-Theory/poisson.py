import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("US ACCIDENTS POISSON REGRESSION ANALYSIS")
print("=" * 80)


print("\n1. LOADING DATA (2016-2021)...")

# Load data
CSV_PATH = "US_Accidents_March23.csv"

columns_needed = [
    'Start_Time', 'State',
    'Temperature(F)', 'Precipitation(in)', 
    'Wind_Speed(mph)', 'Humidity(%)'
]

df = pd.read_csv(
    CSV_PATH,
    usecols=columns_needed,
    parse_dates=['Start_Time'],
    date_format='mixed'
)

# Filter to 2016-2021
df = df[(df['Start_Time'].dt.year >= 2016) & (df['Start_Time'].dt.year <= 2021)]
df['Date'] = df['Start_Time'].dt.date

print(f"Total accidents (2016-2021): {len(df):,}")

print("\n2. IMPUTING MISSING VALUES...")

weather_vars = ['Temperature(F)', 'Precipitation(in)', 'Wind_Speed(mph)', 'Humidity(%)']

# State-level imputation for better accuracy
for var in weather_vars:
    # First try state median
    state_median = df.groupby('State')[var].transform('median')
    df[var].fillna(state_median, inplace=True)
    # Then global median for any remaining
    df[var].fillna(df[var].median(), inplace=True)

print("Imputation complete using state-level medians with global fallback")

print("\n3. CREATING US DAILY SUMMARY...")

# Create daily aggregation
us_daily = df.groupby('Date').agg({
    'Temperature(F)': 'mean',
    'Precipitation(in)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Humidity(%)': 'mean'
}).reset_index()

accident_counts = df.groupby('Date').size().reset_index(name='accident_count')
us_daily = us_daily.merge(accident_counts, on='Date')
us_daily['Date'] = pd.to_datetime(us_daily['Date'])

# Add day of week for color coding
us_daily['DayOfWeek'] = us_daily['Date'].dt.dayofweek
us_daily['IsWeekend'] = (us_daily['DayOfWeek'] >= 5).astype(int)

# Create a composite weather severity score for visualization
# Normalize each weather component
scaler = StandardScaler()
us_daily['Temp_norm'] = scaler.fit_transform(us_daily[['Temperature(F)']])
us_daily['Precip_norm'] = scaler.fit_transform(us_daily[['Precipitation(in)']])
us_daily['Wind_norm'] = scaler.fit_transform(us_daily[['Wind_Speed(mph)']])
us_daily['Humidity_norm'] = scaler.fit_transform(us_daily[['Humidity(%)']])

# Weather Severity Score (higher = more severe conditions)
us_daily['Weather_Severity'] = (
    us_daily['Precip_norm'] * 0.35 +  # Precipitation most impactful
    us_daily['Wind_norm'] * 0.25 +     # Wind second
    np.abs(us_daily['Temp_norm']) * 0.2 +  # Temperature extremes
    np.abs(us_daily['Humidity_norm'] - 0.5) * 0.2  # Humidity extremes
)

mean_accidents = us_daily['accident_count'].mean()
print(f"\nUS Daily Summary Statistics:")
print(f"  Total days: {len(us_daily)}")
print(f"  Mean daily accidents: {mean_accidents:.1f}")
print(f"  Std dev: {us_daily['accident_count'].std():.1f}")
print(f"  Min: {us_daily['accident_count'].min()}")
print(f"  Max: {us_daily['accident_count'].max()}")

# Print descriptive statistics for the daily weather metrics
daily_weather_cols = ['Temperature(F)', 'Precipitation(in)', 'Wind_Speed(mph)', 'Humidity(%)']
daily_weather_stats = us_daily[daily_weather_cols].agg(['mean', 'std', 'min', 'max'])
print("\nUS Daily Aggregated Weather Metrics (mean across incidents per day):")
print(daily_weather_stats.round(2).to_string())

# Provide the full daily time series aggregated dataset
us_daily_time_series = us_daily[['Date'] + daily_weather_cols + ['accident_count']].sort_values('Date')
us_daily_display = us_daily_time_series.copy()
us_daily_display[daily_weather_cols] = us_daily_display[daily_weather_cols].round(2)
print("\nUS Daily Aggregated Time Series (2016-2021):")
print(us_daily_display.to_string(index=False))


print("\n4. GENERATING PLOT 1: US DAILY SUMMARY...")

fig = plt.figure(figsize=(14, 7))

# Create scatter plot with weather severity as color
scatter = plt.scatter(us_daily['Date'], us_daily['accident_count'], 
                     c=us_daily['Weather_Severity'], cmap='RdYlBu_r',
                     s=15, alpha=0.6, edgecolors='none')

# Add rolling average line
window_size = 30
rolling_mean = us_daily['accident_count'].rolling(window=window_size, center=True).mean()
plt.plot(us_daily['Date'], rolling_mean, 'k-', linewidth=2, 
         label=f'{window_size}-day rolling average', alpha=0.8)

# Mark weekends with subtle background shading (simplified approach)
weekend_dates = us_daily[us_daily['IsWeekend'] == 1]['Date'].values
for date in weekend_dates[::2]:  # Only mark every other weekend to reduce visual clutter
    plt.axvspan(date, date + pd.Timedelta(days=2),
                alpha=0.02, color='gray')

# Formatting
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Accident Count', fontsize=12)
plt.title('US Daily Accidents (2016-2021) with Weather Severity\n' + 
          'Color indicates weather severity (red=severe, blue=mild), gray bands=weekends',
          fontsize=14)
plt.colorbar(scatter, label='Weather Severity Index')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Add annotations for notable patterns
# Find highest accident day
max_idx = us_daily['accident_count'].idxmax()
max_date = us_daily.loc[max_idx, 'Date']
max_count = us_daily.loc[max_idx, 'accident_count']
plt.annotate(f'Peak: {max_count} accidents\n{max_date.strftime("%Y-%m-%d")}',
            xy=(max_date, max_count), xytext=(max_date + pd.Timedelta(days=100), max_count),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('us_daily_accidents_weather.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n5. FITTING POISSON REGRESSION FOR CALIFORNIA...")

# Select California as it has the most data
STATE = 'CA'
state_df = df[df['State'] == STATE].copy()

# Create daily summary for state
state_daily = state_df.groupby('Date').agg({
    'Temperature(F)': 'mean',
    'Precipitation(in)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Humidity(%)': 'mean'
}).reset_index()

state_counts = state_df.groupby('Date').size().reset_index(name='accident_count')
state_daily = state_daily.merge(state_counts, on='Date')
state_daily['Date'] = pd.to_datetime(state_daily['Date'])

print("\n6. FEATURE ENGINEERING...")

# Transform features to improve model performance
state_daily['Precipitation_log'] = np.log1p(state_daily['Precipitation(in)'])
state_daily['Wind_Speed_sqrt'] = np.sqrt(state_daily['Wind_Speed(mph)'])

# Temperature deviation from ideal (65°F)
state_daily['Temp_deviation'] = np.abs(state_daily['Temperature(F)'] - 65)

# Interaction: High humidity + precipitation (visibility proxy)
state_daily['Humid_Precip'] = state_daily['Humidity(%)'] * state_daily['Precipitation_log']

# Temporal features
state_daily['DayOfWeek'] = state_daily['Date'].dt.dayofweek
state_daily['Month'] = state_daily['Date'].dt.month
state_daily['Year'] = state_daily['Date'].dt.year
state_daily['IsWeekend'] = (state_daily['DayOfWeek'] >= 5).astype(int)
state_daily['time_trend'] = np.arange(len(state_daily))

# Normalize continuous features to prevent numerical issues
# Create new columns with '_norm' suffix to avoid overwriting
scaler_state = StandardScaler()
continuous_features = ['Temperature(F)', 'Wind_Speed_sqrt', 'Humidity(%)', 
                       'Temp_deviation', 'Humid_Precip']
for feature in continuous_features:
    state_daily[f'{feature}_norm'] = scaler_state.fit_transform(state_daily[[feature]])

# Check for and handle any remaining NaN values
state_daily = state_daily.fillna(0)

print(f"\n{STATE} Statistics:")
print(f"  Total days: {len(state_daily)}")
print(f"  Mean daily accidents: {state_daily['accident_count'].mean():.1f}")
print(f"  Std dev: {state_daily['accident_count'].std():.1f}")

print(f"\n7. FITTING POISSON MODEL...")

# Train-test split (80-20)
split_idx = int(len(state_daily) * 0.8)
train_data = state_daily[:split_idx].copy()
test_data = state_daily[split_idx:].copy()

# Prepare features for model
# First, let's see what normalized columns we actually have
available_cols = train_data.columns.tolist()

# Build feature list based on what's actually available
feature_cols = []
if 'Temperature(F)_norm' in available_cols:
    feature_cols.append('Temperature(F)_norm')
else:
    feature_cols.append('Temperature(F)')
    
feature_cols.append('Precipitation_log')

if 'Wind_Speed_sqrt_norm' in available_cols:
    feature_cols.append('Wind_Speed_sqrt_norm')
else:
    feature_cols.append('Wind_Speed_sqrt')
    
if 'Humidity(%)_norm' in available_cols:
    feature_cols.append('Humidity(%)_norm')
else:
    feature_cols.append('Humidity(%)')
    
if 'Temp_deviation_norm' in available_cols:
    feature_cols.append('Temp_deviation_norm')
else:
    feature_cols.append('Temp_deviation')
    
if 'Humid_Precip_norm' in available_cols:
    feature_cols.append('Humid_Precip_norm')
else:
    feature_cols.append('Humid_Precip')
    
feature_cols.extend(['IsWeekend', 'time_trend'])

print(f"Selected features: {feature_cols}")

# Add month dummies (excluding one for reference)
train_month_dummies = pd.get_dummies(train_data['Month'], prefix='Month', drop_first=True)
test_month_dummies = pd.get_dummies(test_data['Month'], prefix='Month', drop_first=True)

X_train = pd.concat([train_data[feature_cols], train_month_dummies], axis=1)
X_test = pd.concat([test_data[feature_cols], test_month_dummies], axis=1)

# Debug: Check data types before conversion
non_numeric_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'O':
        non_numeric_cols.append(col)
        
if non_numeric_cols:
    print(f"WARNING: Found non-numeric columns: {non_numeric_cols}")
    print(f"Attempting to convert to numeric...")

# Fill any NaN values with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Add constant
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Ensure test has same columns as train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Convert everything to float64 explicitly
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

y_train = train_data['accident_count'].values.astype('float64')
y_test = test_data['accident_count'].values.astype('float64')

print(f"\nModel data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_train dtypes: {X_train.dtypes.unique()}")
print(f"  y_train dtype: {y_train.dtype}")

# Fit Poisson model using GLM for expanded diagnostics
glm_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
poisson_training_results = glm_model.fit()

print("\nModel Summary:")
print(f"  Log-likelihood: {poisson_training_results.llf:.2f}")
print(f"  AIC: {poisson_training_results.aic:.2f}")
print(f"  BIC: {poisson_training_results.bic:.2f}")

print("\nFull GLM Poisson summary:")
print(poisson_training_results.summary())

# Get predictions with confidence intervals (GLM)
poisson_predictions = poisson_training_results.get_prediction(X_test)
predictions_summary_frame = poisson_predictions.summary_frame()
print("\nPrediction summary frame:")
print(predictions_summary_frame)

print("\n8. GENERATING PLOT 2: PREDICTED VS ACTUAL...")

# Extract predictions and actual values
pred_col = next((col for col in ['mean', 'predicted', 'predicted_mean']
                if col in predictions_summary_frame.columns), None)
if pred_col is None:
    raise KeyError("Prediction column not found in summary frame")

predicted_counts = predictions_summary_frame[pred_col].values.astype('float64')
actual_counts = y_test

# Create the plot following the Brooklyn Bridge example style
fig = plt.figure(figsize=(14, 7))
fig.suptitle(f'Predicted versus Actual Daily Accident Counts in {STATE}', fontsize=14, fontweight='bold')

# Plot predicted and actual counts
predicted_line, = plt.plot(test_data['Date'], predicted_counts, 'go-', 
                          label='Predicted counts', linewidth=1.5, markersize=4, alpha=0.7)
actual_line, = plt.plot(test_data['Date'], actual_counts, 'ro-', 
                       label='Actual counts', linewidth=1.5, markersize=4, alpha=0.7)

# Add confidence interval as shaded area
lower_ci_col = next((col for col in ['mean_ci_lower', 'ci_lower', 'predicted_mean_ci_lower']
                     if col in predictions_summary_frame.columns), None)
upper_ci_col = next((col for col in ['mean_ci_upper', 'ci_upper', 'predicted_mean_ci_upper']
                     if col in predictions_summary_frame.columns), None)

if lower_ci_col is None or upper_ci_col is None:
    raise KeyError("Confidence interval columns not found in summary frame")

lower_ci = predictions_summary_frame[lower_ci_col].values.astype('float64')
upper_ci = predictions_summary_frame[upper_ci_col].values.astype('float64')
plt.fill_between(test_data['Date'], lower_ci, upper_ci,
                alpha=0.2, color='green', label='95% Confidence Interval')

# Calculate and display metrics
rmse = np.sqrt(np.mean((actual_counts - predicted_counts)**2))
mae = np.mean(np.abs(actual_counts - predicted_counts))
r2 = 1 - np.sum((actual_counts - predicted_counts)**2) / np.sum((actual_counts - np.mean(actual_counts))**2)

# Add text box with metrics
textstr = f'RMSE: {rmse:.1f}\nMAE: {mae:.1f}\nR²: {r2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Accident Count', fontsize=12)
plt.legend(handles=[predicted_line, actual_line], loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{STATE}_poisson_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
