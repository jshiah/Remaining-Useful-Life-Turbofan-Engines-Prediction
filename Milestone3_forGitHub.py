#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

column_names = [
    "unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]  # 3 operational settings + 21 sensors


# In[3]:


df1 = pd.read_csv('/Users/JoyceShiah/Desktop/CMAPSSData/train_FD001.txt', sep="\s+", header=None, names=column_names, index_col=False)


# need `index_col = False` in order to correctly format `unit` and `time` columns (parse as integers) 

# In[4]:


# Display dataset info
# print(df1.info())
print(df1.head())


# ### Step 1: Drop Unnecessary Columns
# - drop if: columns contain only constant values
# - before dropping them, verify if either provide meaningful variance

# In[5]:


print(df1.nunique())


# ### Step 2: Check for Missing Values

# In[6]:


print(df1.isnull().sum())


# ### Step 3: Sort Data by `unit` and `time` [REVIEW THIS MORE]
# Since each unit (engine) runs for a certain time until engine failure, sorting ensures correct sequencing:

# In[7]:


df1 = df1.sort_values(by=["unit", "time"])
df1.head()


# ### Step 4: Calculate assigned Remaining Useful Life (RUL = number of time steps left before engine failure) value
# - calculated RUL value --> target variable for model
# - `RUL = Final Cycle − Current Cycle`
# - formula for target RUL calculation: `.transform(lambda x: x.max() - x)`
#     - group the data by each **engine** unit.
#     - each engine has **multiple time steps**, starting from time = 1 and increasing until failure
#     - for each engine, this finds the **maximum time value** (max = last recorded cycle before failure)
# - `x.max() - x`
#     - finds how many time steps are left before failure
#     - e.g. RUL = 150 − 125 = 25
#         - if an engine runs for 150 cycles and the current row has time = 125
# - `.transform()` ensures calculation is applied within **each engine unit**, maintaining the original DataFrame structure
# 
# **Why this step of assigning RUL values (serves as the target variable for training your XGBoost model) is essential for building the adaptive sliding window approach:**
# 
# - goal is to predict RUL at each time step, so without calculating RUL first, you wouldn't have a supervised learning target for XGBoost
# - adaptive sliding window model **learns from past sensor readings to predict future RUL**, so you need a **well-defined RUL label** for training
# - adaptive sliding window approach dynamically adjusts the window size based on engine degradation rates. This means:
#     - if an engine is **degrading slowly**, a longer history (larger window) is useful
#     - If an engine is **degrading rapidly**, a shorter history (smaller window) is needed to focus on recent changes
# - analyzing sensor behavior is critical for success in engine sustainability: without RUL, you wouldn't know how close an engine is to failure, making it impossible to adjust window size adaptively
# 
# 
# **Difference between Fixed vs. Adaptive Sliding Window Approach:**
# 
# - Since this is comparing fixed vs. adaptive sliding windows, both need a consistent target variable --> RUL
# - **Fixed Window:** uses a constant-length past sequence for prediction
#     - fixed number of past time steps (e.g., the last 30 or 50 readings)
# - **Adaptive Window:** uses RUL trends to adjust window size dynamically, so if RUL weren’t calculated first, it would be unclear when to adjust the window size, making adaptive modeling ineffective

# In[8]:


df1["RUL"] = df1.groupby("unit")["time"].transform(lambda x: x.max() - x)


# In[9]:


df1.head()


# ### Step 4: Visualization of the Calculated RUL distribution 
# - compute RUL for each engine unit in FD001 Train dataset

# In[10]:


# find max cycle per unit 
max_cycle_per_unit = df1.groupby("unit")["time"].max()

# compute RUL by merging with the main dataframe to compute RUL
df1["RUL"] = df1["unit"].map(max_cycle_per_unit) - df1["time"]

plt.figure(figsize=(8, 5))
plt.hist(df1["RUL"], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Remaining Useful Life (RUL)")
plt.ylabel("Frequency")
plt.title("Distribution of RUL in Dataset")
plt.grid(True)
plt.show()


# ### Discussion of graph:
# - right-skewed distribution
# - however, from 0 to around 120 RUL, there is a sudden spike every 20-30 RUL increments that go up to frequency of 800
# 
# #### From ONLINE: 
# - suggests that many engines have similar degradation patterns and tend to fail around certain lifecycle points. This could be due to:
#     - Consistent Maintenance Schedules – If engines are serviced at regular intervals, similar RUL values could appear frequently.
#     - Predefined Operating Conditions – If all engines follow the same operational settings, their degradation trends might align.
#     - Simulation Artifacts – Since this dataset is simulated (NASA C-MAPSS), certain degradation behaviors might be embedded in the data.
#     - Binning Effects – If the dataset groups engine cycles into bins, RUL distributions might show step-like patterns.

# ### Step 4.1: Going further into analysis of graph
# Check `Frequency` table for RUL, to see if the RUL increments are artificially grouped

# In[11]:


df1["RUL"].value_counts().sort_index().head(50)  # Check the first 50 RUL values


# Compare with `Engine` IDs

# In[85]:


df1.groupby("RUL")["unit"].nunique().head(50)  # See how many unique engines share the same RUL


# Overlay a Kernel Density Estimate (KDE)

# ### Discussion of analysis 
# - for every Remaining Useful Life (RUL) value from 0 to 49, there are exactly 100 rows in the dataset where that RUL value appears
# - since each row corresponds to an engine at a specific time step, this suggests that at every RUL step from 0 to 49, 100 different engines are recorded at that exact RUL value
# - NASA C-MAPSS dataset follows a structured simulation, ensuring that each engine degrades in a uniform way
# - all 100 engines reach RUL = 0 at some point, and they pass through the same RUL values at the same frequency during degradation
# 
# ### From GPT: 
# Key Implications
# - Structured Data Distribution: The dataset follows a strict degradation pattern, unlike real-world scenarios where failures can occur at random times.
# - Fixed Window Approach Simplicity: A fixed sliding window approach might be sufficient since degradation follows a predictable trend.
# - Adaptive Window Re-Evaluation: If all engines degrade uniformly, an adaptive window based on sensor trends may not provide much additional benefit unless sensor behavior varies significantly across engines.

# ## KDE Visualization
# - visualize the distribution of data points in a smooth manner
#     - KDE takes a set of data points and smooths them into a continuous curve
#     - uses a kernel (a small window) around each data point to contribute to the overall density estimation
#  
# Since the data used for modeling from the NASA dataset is primarily continuous, 
# - features:
#     - sensor measurements --> temperature, pressure, rotational speed, fuel flow
#     - operational settings --> are typically recorded as continuous numerical values
#     - RUL Values: continuous variable representing the estimated time until a failure occurs, which can also vary smoothly based on the operational conditions of the engines
#  
# **So, KDE is a preferred visualization over histograms:** 
# - KDE is especially useful for continuous data where you expect the distribution to change smoothly
# - can capture the density's shape without being constrained by discrete bin edges
# - from Seaborn documentation: KDE can produce a plot that is less cluttered and more interpretable, especially when drawing multiple distributions

# In[13]:


import seaborn as sns

plt.figure(figsize=(8, 5))
sns.kdeplot(df1["RUL"], fill=True, bw_adjust=1)
plt.xlabel("Remaining Useful Life (RUL)")
plt.ylabel("Density")
plt.title("Smoothed RUL Distribution (KDE Plot)")
plt.grid(True)
plt.show()


# ### Discussion of KDE plot
# - the flat KDE density between 20 to 127 RUL suggests that the distribution of RUL values in this range is evenly spread out
# - gradually tapering off @ RUL = 128
# - the counts slightly decrease (99 instead of 100), and after RUL 135–150, we see a more noticeable decline.
# - This suggests that:
#     - a fixed number of engines were observed up to RUL 127.
#     - Every unit had at least 128 cycles before failure.
#  
# ### From ONLINE:
# - Beyond RUL 127, fewer engines exist in the dataset at those RUL values.
# - Some engines might have started degradation later or had shorter lifespans, leading to fewer observations at higher RUL values.
# The KDE plot remained flat because of the uniform distribution between 20 and 120 RUL.
# - The slow drop after 128 RUL marks the start of the natural failure progression, where engines stop appearing at certain RUL values.

# ### Step 4.2: Analyze RUL distribution beyond 50 and check for similar patterns:

# In[14]:


df1["RUL"].value_counts().sort_index()[100:150]  # See counts for RUL values between 100-150


# In[83]:


# df1.groupby("RUL")["unit"].nunique()[100:150]


# ### Step 4.3: Calculate the correlation matrix for the features to identify relationships between sensor data, operational settings, and RUL. 
# - heatmap to visualize the correlations
# 
# 
# ### From ONLINE: 
# - Understanding Relationships: A heatmap visually represents the correlation coefficients between different sensors, allowing you to quickly see which sensors have strong or weak relationships with each other. This can help identify redundant sensors or those that provide unique information.
# 
# - Feature Selection: If you find that certain sensors are highly correlated, you might consider removing one of them from your analysis to reduce dimensionality. Keeping correlated features may not add significant value to your predictive model, while removing them can simplify the model and improve interpretability.
# 
# - Insight into Engine Behavior: High correlation between sensors might indicate that they are measuring related physical phenomena. For instance, if sensor 11 (e.g., temperature) and sensor 14 (e.g., pressure) are highly correlated, it may suggest that as temperature increases, pressure also tends to increase, reflecting a consistent relationship in engine operations.
# 
# - Identifying Patterns: Patterns in correlations can help uncover the underlying mechanics of the engine's performance. For instance, if two sensors correlate strongly during specific operational conditions, it might indicate how the engine responds to those conditions.

# In[16]:


plt.figure(figsize=(12, 8))
correlation_matrix = df1.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Start with a Common Threshold: Begin with a threshold of 0.5, as it's a commonly used benchmark for indicating a moderate to strong correlation.

# In[17]:


# Calculate the correlation matrix
correlation_matrix = df1.corr()

# Set a threshold for filtering
threshold = 0.5

# find large correlations
significant_correlations = correlation_matrix[(correlation_matrix >= threshold) | (correlation_matrix <= -threshold)]

print(significant_correlations.stack().reset_index().rename(columns={0: 'correlation'}))


# Display 10 highest and bottom 10 correlations

# In[18]:


# calculate the correlation matrix
correlation_matrix = df1.corr()

# set threshold for filtering
threshold = 0.5

# find large correlations
significant_correlations = correlation_matrix[(correlation_matrix >= threshold) | (correlation_matrix <= -threshold)]

# find significant correlations and reset index
significant_correlations = significant_correlations.stack().reset_index().rename(columns={0: 'correlation'})

# filter out self-correlations (correlation of a variable with itself)
significant_correlations = significant_correlations[significant_correlations['level_0'] != significant_correlations['level_1']]

# top 10 highest correlations
top_10_correlations = significant_correlations.nlargest(10, 'correlation')

# bottom 10 lowest correlations (most negative)
bottom_10_correlations = significant_correlations.nsmallest(10, 'correlation')

print("Top 10 Highest Correlations:")
print(top_10_correlations)

print("\nBottom 10 Lowest Correlations:")
print(bottom_10_correlations)


# ### Discussion of Top 10, Bottom 10 Correlation Analysis
# Top 10 Highest Correlations:
# - pairs like `sensor_9` and `sensor_14` have a very high correlation of 0.963
#     - shows that as one sensor's readings increase, the other sensor's readings tend to increase as well --> suggesting a strong linear relationship
# - **From ONLINE:** high correlations (like those above 0.8) might suggest redundancy in your features, as they provide similar information. You may consider selecting one of the correlated sensors for modeling to reduce multicollinearity.
# 
# Bottom 10 Lowest Correlations:
# - pairs such as `sensor_11` and `sensor_12`, have a strong negative correlation of -0.847
#     - as one sensor's readings increase, the other sensor's readings tend to decrease --> inverse relationship
# - **From ONLINE:** pairs with large negative correlations might indicate interesting relationships that could be worth investigating further, as they could reveal contrasting behavior in the system

# In[19]:


# # Testing different thresholds
# thresholds = [0.3, 0.5, 0.7, 0.9]
# for threshold in thresholds:
#     significant_correlations = correlation_matrix[(correlation_matrix >= threshold) | (correlation_matrix <= -threshold)]
#     print(f"\nSignificant correlations at threshold {threshold}:")
#     print(significant_correlations.stack().reset_index().rename(columns={0: 'correlation'}))


# ### Step 4.4: Analyze Sensor Trends 
# **FROM ONLINE:** Stable Readings: If a sensor's reading is represented by a horizontal line, it suggests that the sensor's output is relatively stable over the observed time period. This could mean that the sensor's operating condition isn't changing much, which might be expected for certain operational settings.
# 
# Lack of Variation: If multiple sensors show very little variation, it might indicate that the engine is operating under stable conditions or that there are issues with the sensors themselves, such as calibration problems.
# 
# Diverse Operating Conditions: If some sensors have varying readings (not horizontal), it can indicate that those sensors are reacting to changing operational conditions, which could be valuable for understanding engine performance and health.
# 
# Anomalies: If any sensor's reading dramatically deviates from the horizontal line, this could signal an anomaly or event worth investigating further, such as a sudden increase in temperature or pressure.

# In[20]:


# sensors to analyze (all sensors from 1 to 21)
key_sensors = [f'sensor_{i}' for i in range(1, 22)]

plt.figure(figsize=(15, 10))
for sensor in key_sensors:
    plt.plot(df1['time'], df1[sensor], label=sensor)

plt.title('Sensor Trend Analysis Over Time')
plt.xlabel('Time')
plt.ylabel('Sensor Values')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid()
plt.show()


# ### Sanity Check on # of columns

# In[21]:


len(df1.columns)


# NOTE: WILL SKIP OVER `Normalizing/Standardizing Sensor Readings` Step (for now, will evaluate XGBoost accuracy first w/ raw data)
# 

# ## To address TA's feedback from Milestone 2:

# TA comments: "No discussion on feature extraction: The report states that raw sensor data is sufficient, but it does not explore whether transformations (e.g., moving averages, trend indicators) could improve model performance."
# 
# ### Chosen Feature extraction method: Trend indicator
# - create features that indicate whether a sensor reading is increasing or decreasing over time
# - e.g. the difference between the current reading and the reading from a certain number of time steps ago could serve as a trend indicator
# 
# ## Create new dataframe `df1_eng` (feature-engineering, has 48 cols)
# ## `df1` = raw data only from NASA
# ## `df1_eng` = raw data + additional trend features

# In[22]:


df1_eng = df1.copy()  # create a separate copy for engineered features

# sensors analyze (sensor_1 to sensor_21)
sensors = [f'sensor_{i}' for i in range(1, 22)]

# calculate trend indicators, store in df_eng
for sensor in sensors:
    df1_eng[f'{sensor}_trend'] = df1_eng[sensor].diff()

df1_eng.fillna(0, inplace=True)

# sanity check: that df1 is unchanged
print("df1 shape:", df1.shape)
print("df_eng shape:", df1_eng.shape)

# Preview df_eng to confirm engineered features were added
print(df1_eng[[sensors[0], f'{sensors[0]}_trend']].head(10))  # Example for sensor_1


# ### Sanity Check on # of columns

# In[23]:


len(df1.columns)
# df1.columns


# **Note:** From now on, we are only dealing with `df1_eng` for Trend Indicators work

# In[24]:


# sensor to analyze 
sensor = 'sensor_1'

# calculate the trend indicator for sensor_1
df1_eng[f'{sensor}_trend'] = df1_eng[sensor].diff()

# fill NaN values resulting from the diff() method
df1_eng.fillna(0, inplace=True)

# preview the updated DataFrame between indices 100 to 150 for sensor_1 and its trend
# print(df1.loc[100:150, [sensor, f'{sensor}_trend']])


# In[25]:


len(df1.columns)


# ### Discussion of Analyzing Sensor_1_Trend only b/w indices 100 to 150:
# sensor_1 has constant values (518.67) for the indices you're checking --> trend indicator showing a difference of 0.0 for all those entries. 
# - The diff() method calculates the difference between consecutive rows, and since there are no changes in the values, the trend remains zero.
# 
# **Next action:** Check different sensors or a broader range of indices to see if there are varying values over time. 
# - sensors with changing values --> may give more meaningful trend indicators

# ### Now, trying between index of 100 to 150, since RUL decreased from 100 to 99 at RUL 128

# In[26]:


# # Select the relevant sensor columns for trend analysis
# sensors_to_analyze = df1_eng.columns[2:]  # Exclude 'unit' and 'time'

# # Calculate moving averages and differences for trend indicators
# for sensor in sensors_to_analyze:
#     df1_eng[f'{sensor}_moving_avg'] = df1_eng[sensor].rolling(window=5).mean()  # 5-point moving average
#     df1_eng[f'{sensor}_diff'] = df1_eng[sensor].diff()  # Difference from the previous value

# # Display the trend indicators for sensors between indices 100 and 150
# trend_indicators = df1_eng.loc[100:150, [*sensors_to_analyze, *[f'{sensor}_moving_avg' for sensor in sensors_to_analyze], *[f'{sensor}_diff' for sensor in sensors_to_analyze]]]

# print(trend_indicators)


# ### Sanity Check on # of columns

# In[27]:


len(df1.columns)


# ### Discussion of Analyzing Sensor_1_Trend only b/w indices 100 to 150:
# - Moving Averages: The rolling mean smooths out short-term fluctuations and highlights longer-term trends in the data. The window size of 5 means it averages the current and the previous four values.
# - Differences: The diff() method calculates the difference between consecutive values, giving you insight into how quickly sensor readings are changing, which can be particularly useful for identifying sudden changes or anomalies.

# ### Visualize trend indicators using Seaborn
# A good way to do this is by plotting the original sensor readings alongside their moving averages and differences. Here's an example of how to create these visualizations:
# 
# Plotting Original Sensor Readings and Moving Averages:
# Plotting Differences:

# In[82]:


# # Set the style for Seaborn
# sns.set(style="whitegrid")

# # Define the sensors to visualize (sensor_1 to sensor_21)
# sensors_to_plot = [f'sensor_{i}' for i in range(1, 22)]

# # Create a figure with subplots for each sensor
# num_sensors = len(sensors_to_plot)
# fig, axes = plt.subplots(num_sensors, 2, figsize=(15, num_sensors * 4), sharex=True)

# # Loop through each sensor to plot original readings and moving averages, and differences
# for i, sensor in enumerate(sensors_to_plot):
#     # Plot original readings and moving average
#     sns.lineplot(x=df1['time'], y=df1[sensor], ax=axes[i, 0], label='Original Reading', color='blue')
#     sns.lineplot(x=df1['time'], y=df1[f'{sensor}_moving_avg'], ax=axes[i, 0], label='5-Point Moving Average', color='orange')
#     axes[i, 0].set_title(f'Trend Analysis of {sensor}')
#     axes[i, 0].set_ylabel('Sensor Values')
#     axes[i, 0].legend()
#     axes[i, 0].grid()

#     # Plot the differences
#     sns.lineplot(x=df1['time'], y=df1[f'{sensor}_diff'], ax=axes[i, 1], label='Difference', color='red')
#     axes[i, 1].set_title(f'Difference of {sensor} Readings')
#     axes[i, 1].set_ylabel('Difference')
#     axes[i, 1].legend()
#     axes[i, 1].grid()

# # Set common labels
# plt.xlabel('Time')
# plt.tight_layout()
# plt.show()


# ### Discussion on Graphs between `SENSOR_11` and `SENSOR_12`
# Sensor Trends Over Time
# - Some sensors show an increasing trend (e.g., sensor_11), while others (e.g., sensor_12) appear to have a slight downward trend
# 
# Variability and Noise
# - The raw readings have fluctuations, which are smoothed by the moving average. If the moving average closely follows the raw data, it suggests a relatively stable signal. If it lags or smooths too much, the sensor may have a lot of noise.
# - Certain sensors show more variation toward the end of their life cycles, possibly indicating degradation (e.g., sensor_11).
# 
# #### Difference (Rate of Change):
# - The right column shows the first-order difference (change from the previous time step).
# If the difference fluctuates around zero with low variance, the sensor readings are stable. If the variance increases, it may indicate wear or failure approaching.
# - The difference plots for sensor_11 and sensor_12 show an increase in fluctuations towards the end, which may indicate degradation or an operational shift.
# 
# #### Possible Takeaways (From GPT):
# - Stable Sensors: Some sensors have nearly constant readings, meaning they might not be useful for predictive modeling.
# - Degrading Sensors: Those with a clear trend and increasing variation in the difference plot could be valuable for predicting Remaining Useful Life (RUL).
# - Sensor Noise: If some sensors show too much randomness in both plots, they may not contribute meaningful information.

# ### Discussion on Graphs between `SENSOR_11` and `SENSOR_12`
# 
# #### Sensor_14 Trend Analysis
# - Trend Analysis Plot (Left Plot)
#     - The sensor reading remains relatively stable for most of the cycle but shows an upward drift towards the end.
#     - A sudden drop at the very end might indicate an engine failure or a shutdown.
#     - The moving average closely follows the trend, indicating relatively low noise.
#  
# - Sensor_14 Difference Plot (Right Plot)
#     - The differences are mostly small but increase in magnitude toward the end.
#     - A sharp negative difference is visible at the end, aligning with the sudden drop in the trend graph.
#     - This suggests that Sensor_14 might be capturing degradation patterns, making it a useful feature for RUL prediction.
# #### Sensor_15 
# - Trend Analysis (Left Plot):
#     - There is a gradual increase in the sensor reading over time.
#     - The variance in sensor readings increases over time, suggesting degradation effects or operating condition changes.
#     - The moving average captures this trend well but smooths out the short-term fluctuations.
# - Difference Plot (Right Plot):
#     - The difference values remain small and stable at the beginning but increase in magnitude over time.
#     - The fluctuations become more significant in the later cycles, possibly indicating an accelerating degradation process.
#     - This increasing variance suggests that Sensor_15 might also be important for tracking wear and failure progression.
#  
# ### Key Takeaways
# - Sensor_14 shows a sudden failure signature (a sharp drop), which might make it a strong predictor for identifying the last cycles before failure.
# - Sensor_15 shows increasing variance in differences over time, which could be a useful feature for tracking gradual degradation.

# #### To identify the most relevant sensors:
# Feature Importance from XGBoost (Recommended)
# - Train an initial XGBoost model on all sensor data
# - Compare the feature importance scores
# - Keep only the top N most important sensors (choosing N = 10)

# ## Method 1
# #### Preparing X_train and y_train for RUL Prediction (USING RAW FEATURES (from NASA) ONLY)
# 

# In[29]:


len(df1.columns)


# In[30]:


len(df1_eng.columns)


# In[31]:


# "RAW features only" Feature Importance model 

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# define raw features
sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
op_settings = ["op_setting_1", "op_setting_2", "op_setting_3"]
feature_columns = sensor_columns + op_settings

df1['RUL'] = df1.groupby('unit')['time'].transform(lambda x: x.max() - x)  # RUL calculation for target variable

# split data into features and target
X = df1[feature_columns]
y = df1['RUL']

# split data into training and validation sets
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_raw = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_raw.fit(X_train_raw, y_train_raw)

y_pred_raw = xgb_raw.predict(X_val_raw)
rmse_raw = mean_squared_error(y_val_raw, y_pred_raw, squared=False)

print(f"Baseline Model (Raw Features) RMSE: {rmse_raw:.4f}")


# In[32]:


import matplotlib.pyplot as plt

# Plot feature importance
xgb.plot_importance(xgb_raw)
plt.title('Feature Importance')
plt.show()


# #### Discussion of Feature Importance XGBoost model on RAW Features dataframe only 
# Top Features Contributing to RUL Prediction
# - Most Significant Sensors:
#     - sensor_2: F score = 540.0
#     - sensor_4: F score = 457.0
#     - sensor_3: F score = 426.0
#     - sensor_9: F score = 410.0
#     - sensor_7: F score = 393.0
# - Additional Noteworthy Sensors:
#     - sensor_14: F score = 380.0
#     - sensor_15: F score = 371.0
#     - sensor_21: F score = 351.0
#     - sensor_11: F score = 330.0
#     - sensor_12: F score = 320.0
#     - sensor_20: F score = 295.0
# - Operational Settings:
#     - op_setting_1: F score = 291.0
#     - op_setting_2: F score = 220.0
#     - op_setting_2: F score = 147.0
# 
# ### Compared to Wang, T., Yu, J., Siegel, D., & Lee, J. (2008):
# Common sensors: 2, 3, 4, 7, 11, 12, 15, 20, 21
# - XGBoost Feature Importance analysis shows strong alignment with Wang et al.'s manual selection, which is a good start.
#     - This novel model independently identified similar trends to previously published models.
# - **Next action:** Identify if`sensor_` & `sensor_14` could capture short-term variations that improve RMSE (weren’t manually chosen by Wang et al.)
# 
# Differences
# - XGBoost found `sensor_9` & `sensor_14` important, while Wang et al. did not select either
# - Wang et al. stated in their paper they included only trend-consistent sensors, whereas XGBoost picks the most predictive ones based on tree splits (Wang et al. 2008).

# ### Train Models & Compare RMSE Values
# - Model 1: Wang et al.’s selected sensors: (2, 3, 4, 7, 11, 12, 15, 20, 21)
# - Model 2: XGBoost’s top-ranked sensors → (2, 3, 4, 7, 9, 11, 12, 14, 15, 21)
# - Model 3: All sensors (`sensor_1` to `sensor_21`)

# #### Compute Feature Correlation
# Objective: check whether `sensor_9` and `sensor_14` are highly correlated with Wang et al.’s selected sensors. Potentially use sensors in another model to compare with Wang's selected sensors.
# - Whichever has the lowest RMSE = best subset 

# In[33]:


# sensors from Wang et al. (2008 published model)
wang_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_20", "sensor_21"]

# sensors XGBoost feature importance analysis model identified that Wang et al. did NOT
extra_sensors = ["sensor_9", "sensor_14"]

# relevant sensors
selected_sensors = wang_sensors + extra_sensors

# compute correlation matrix, find corr coefficients
correlation_matrix = df1[selected_sensors].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Selected Sensors")
plt.show()


# #### Discussion: Yes, sensor 9 or sensor 14 have a very high correlation (0.963) with Wang et al.’s sensors, so they may be redundant. 

# #### Step 2: Train Three Models & Compare RMSE
# Train and evaluate three models using XGBoost

# In[34]:


from sklearn.metrics import mean_squared_error
import numpy as np

# define feature sets
X_wang = df1[wang_sensors]
X_xgb = df1[list(set(wang_sensors + extra_sensors))]  
X_all = df1[[f"sensor_{i}" for i in range(1, 22)]] 

y_train = df1["RUL"]

# initialize models
model_wang = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model_xgb = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model_all = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)

model_wang.fit(X_wang, y_train)
model_xgb.fit(X_xgb, y_train)
model_all.fit(X_all, y_train)

# predictions
y_pred_wang = model_wang.predict(X_wang)
y_pred_xgb = model_xgb.predict(X_xgb)
y_pred_all = model_all.predict(X_all)

# Compute RMSE
rmse_wang = np.sqrt(mean_squared_error(y_train, y_pred_wang))
rmse_xgb = np.sqrt(mean_squared_error(y_train, y_pred_xgb))
rmse_all = np.sqrt(mean_squared_error(y_train, y_pred_all))

print(f"RMSE using Wang et al. Sensors: {rmse_wang:.4f}")
print(f"RMSE using XGBoost Selected Sensors: {rmse_xgb:.4f}")
print(f"RMSE using All Sensors: {rmse_all:.4f}")


# # Important Discussion: Regarding Occam's Razor
# XGBoost’s selected 10 sensors (RMSE = 29.00) outperform Wang et al.’s selection of sensors(RMSE = 33.55)
# - Suggests that: `sensor_9` and `sensor_14` provide valuable predictive information that Wang et al. overlooked.
# - VERDICT: The XGBoost feature importance analysis was important and effective in selecting relevant sensors.
# 
# 
# PROS of going simpler:
# - Uses fewer but more informative sensors reduces model complexity
# - Faster training
# - Less computational resources used
# - Reduces overfitting risk
# 
# Only SLIGHT RMSE difference b/w 10 chosen (most influential) sensors from XGBoost Feature Importance model and applying ALL 21 sensors (29.0049 vs 28.7453). Nothing critical, so only using 10 best sensors is preferred. 
# - Implies that many of the other sensors don’t add much value
# - Keeping fewer, highly relevant sensors can simplify the model while maintaining strong performance
#     - aligns with **Occam's razor** --> simplest strategy is best for explaining data more accurately

# In[37]:


from xgboost import XGBRegressor

model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)


# In[38]:


# Define feature sets for all 3 xgb models
X_wang = df1[wang_sensors]
X_xgb = df1[list(set(wang_sensors + extra_sensors))] 
X_all = df1[[f"sensor_{i}" for i in range(1, 22)]]  # all 21 sensors

y_train = df1["RUL"]

# Initialize XGBoost models
model_wang = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model_xgb = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model_all = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)

# train models
model_wang.fit(X_wang, y_train)
model_xgb.fit(X_xgb, y_train)
model_all.fit(X_all, y_train)

y_pred_wang = model_wang.predict(X_wang)
y_pred_xgb = model_xgb.predict(X_xgb)
y_pred_all = model_all.predict(X_all)

# compute RMSE
rmse_wang = np.sqrt(mean_squared_error(y_train, y_pred_wang))
rmse_xgb = np.sqrt(mean_squared_error(y_train, y_pred_xgb))
rmse_all = np.sqrt(mean_squared_error(y_train, y_pred_all))

# compare RMSE values
print(f"RMSE using Wang et al. Sensors: {rmse_wang:.4f}")
print(f"RMSE using XGBoost Selected Sensors: {rmse_xgb:.4f}")
print(f"RMSE using All Sensors: {rmse_all:.4f}")


# In[80]:


# Sanity check on y_train
y_train.shape


# #### Feature Engineering
# Normalization/Standardization to see if can improve performance, especially since sensor measurements have different ranges and units.

# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale the features
X_wang_scaled = scaler.fit_transform(X_wang)
X_xgb_scaled = scaler.fit_transform(X_xgb)
X_all_scaled = scaler.fit_transform(X_all)


# In[41]:


import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective="regression", n_estimators=200, random_state=42)
model_lgb.fit(X_wang_scaled, y_train)
y_pred_lgb = model_lgb.predict(X_wang_scaled)

rmse_lgb = np.sqrt(mean_squared_error(y_train, y_pred_lgb))
print(f"RMSE using LightGBM: {rmse_lgb:.4f}")


# #### Discussion: RMSE using LightGBM is still high at 38.1647

# In[42]:


import xgboost as xgb
from xgboost import XGBRegressor
model_wang = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500, random_state=42)


# In[43]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
                           param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)

grid_search.fit(X_wang_scaled, y_train)

# Display the best parameters found
print(f"Best parameters for XGBoost: {grid_search.best_params_}")

# Train the model with the best parameters
model_best = grid_search.best_estimator_
model_best.fit(X_wang_scaled, y_train)
y_pred_best = model_best.predict(X_wang_scaled)

rmse_best = np.sqrt(mean_squared_error(y_train, y_pred_best))
print(f"RMSE using tuned XGBoost: {rmse_best:.4f}")


# #### Use Cross-Validation w/ XGBoost to Evaluate Model Performance
# Instead of single training-test split, trying cross-validation for potentially better estimate of model performance & reduce risk of overfitting.

# In[44]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb.XGBRegressor(objective="reg:squarederror", random_state=42), 
                         X_wang_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')

print(f"Cross-validated RMSE: {-scores.mean():.4f}")


# ## `X_wang_scaled` applies sensors → (2, 3, 4, 7, 11, 12, 15, 20, 21)
# #### Stacking (RMSE = 20.4869)

# In[45]:


from sklearn.ensemble import StackingRegressor


# In[46]:


# X_wang_scaled
estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
]
model_stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
model_stack.fit(X_wang_scaled, y_train)
y_pred_stack = model_stack.predict(X_wang_scaled)

rmse_stack = np.sqrt(mean_squared_error(y_train, y_pred_stack))
print(f"RMSE using Stacking with Wang et al's sensors: {rmse_stack:.4f}")


# #### Linear Regression (RMSE = 44.7125)

# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

y_train = df1["RUL"]

linear_model = LinearRegression()

linear_model.fit(X_xgb_scaled, y_train)

y_pred_lr = linear_model.predict(X_xgb_scaled)

rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_lr))
print(f"RMSE using Linear Regression: {rmse_lr:.4f}")


# #### Random Forest Regressor (RMSE = 17.3627)

# In[48]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_wang_scaled, y_train)

y_pred_rf = rf_model.predict(X_wang_scaled)

rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_rf))
print(f"RMSE using Random Forest: {rmse_rf:.4f}")


# #### Gradient Boosting Regressor (RMSE = 44.5983)

# In[49]:


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

gb_model.fit(X_wang_scaled, y_train)

y_pred_gb = gb_model.predict(X_wang_scaled)

rmse_gb = np.sqrt(mean_squared_error(y_train, y_pred_gb))
print(f"RMSE using Gradient Boosting: {rmse_gb:.4f}")


# #### K-Nearest Neighbors (KNN) Regressor (RMSE = 40.7499)

# In[50]:


from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_wang_scaled, y_train)

y_pred_knn = knn_model.predict(X_wang_scaled)

rmse_knn = np.sqrt(mean_squared_error(y_train, y_pred_knn))
print(f"RMSE using KNN: {rmse_knn:.4f}")


# #### ElasticNet (Regularized Linear Regression) (RMSE = 46.7193)

# In[51]:


from sklearn.linear_model import ElasticNet

elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

elasticnet_model.fit(X_wang_scaled, y_train)

y_pred_en = elasticnet_model.predict(X_wang_scaled)

rmse_en = np.sqrt(mean_squared_error(y_train, y_pred_en))
print(f"RMSE using ElasticNet: {rmse_en:.4f}")


# ## `X_xgb_scaled` applies XGBoost’s top-ranked sensors → (2, 3, 4, 7, 9, 11, 12, 14, 15, 21)
# #### Stacking (RMSE = 18.1376)

# In[52]:


# X_xgb_scaled
estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
]
model_stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
model_stack.fit(X_xgb_scaled, y_train)
y_pred_stack = model_stack.predict(X_xgb_scaled)

rmse_stack = np.sqrt(mean_squared_error(y_train, y_pred_stack))
print(f"RMSE using Stacking with XGBoost selected sensors: {rmse_stack:.4f}")


# #### Linear Regression (RMSE = 44.7125)

# In[53]:


y_train = df1["RUL"]

# Initialize Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_xgb_scaled, y_train)

# Make predictions
y_pred_lr = linear_model.predict(X_xgb_scaled)

# Calculate RMSE
rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_lr))
print(f"RMSE using Linear Regression: {rmse_lr:.4f}")


# #### Random Forest Regressor (RMSE = 15.7201 | BEST SO FAR!)
# Still better than all 21 sensors' RF model (w/ RMSE of 15.5778) b/c of Occam's Razor
# - Random forests are known to handle non-linear relationships

# In[54]:


# Initialize RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_xgb_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_xgb_scaled)

# Calculate RMSE
rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_rf))
print(f"RMSE using Random Forest: {rmse_rf:.4f}")


# #### Gradient Boosting Regressor (RMSE = 40.5690)

# In[55]:


# Initialize Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_xgb_scaled, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_xgb_scaled)

# Calculate RMSE
rmse_gb = np.sqrt(mean_squared_error(y_train, y_pred_gb))
print(f"RMSE using Gradient Boosting: {rmse_gb:.4f}")


# #### K-Nearest Neighbors (KNN) Regressor (RMSE = 36.9434)

# In[56]:


# Initialize KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn_model.fit(X_xgb_scaled, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_xgb_scaled)

# Calculate RMSE
rmse_knn = np.sqrt(mean_squared_error(y_train, y_pred_knn))
print(f"RMSE using KNN: {rmse_knn:.4f}")


# #### ElasticNet (Regularized Linear Regression) (RMSE = 44.7193)

# In[57]:


elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

elasticnet_model.fit(X_xgb_scaled, y_train)

y_pred_en = elasticnet_model.predict(X_xgb_scaled)

rmse_en = np.sqrt(mean_squared_error(y_train, y_pred_en))
print(f"RMSE using ElasticNet: {rmse_en:.4f}")


# ## `X_all_scaled` = all 21 sensors applied into model 
# #### Stacking (RMSE = 18.0911)

# In[58]:


# X_all_scaled

estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
]
model_stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
model_stack.fit(X_all_scaled, y_train)
y_pred_stack = model_stack.predict(X_all_scaled)

rmse_stack = np.sqrt(mean_squared_error(y_train, y_pred_stack))
print(f"RMSE using Stacking with ALL 21 sensors: {rmse_stack:.4f}")


# #### Random Forest (RMSE = 15.5778)

# In[59]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_all_scaled, y_train)

y_pred_rf = rf_model.predict(X_all_scaled)

rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_rf))
print(f"RMSE using Random Forest: {rmse_rf:.4f}")


# #### Gradient Boosting Regressor (RMSE = 40.326)

# In[60]:


gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

gb_model.fit(X_all_scaled, y_train)

y_pred_gb = gb_model.predict(X_all_scaled)

rmse_gb = np.sqrt(mean_squared_error(y_train, y_pred_gb))
print(f"RMSE using Gradient Boosting: {rmse_gb:.4f}")


# #### K-Nearest Neighbors (KNN) Regressor (RMSE = 37.2119)

# In[61]:


knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_all_scaled, y_train)

y_pred_knn = knn_model.predict(X_all_scaled)

rmse_knn = np.sqrt(mean_squared_error(y_train, y_pred_knn))
print(f"RMSE using KNN: {rmse_knn:.4f}")


# #### ElasticNet (Regularized Linear Regression) (RMSE = 44.6704)

# In[62]:


elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

elasticnet_model.fit(X_all_scaled, y_train)

y_pred_en = elasticnet_model.predict(X_all_scaled)

rmse_en = np.sqrt(mean_squared_error(y_train, y_pred_en))
print(f"RMSE using ElasticNet: {rmse_en:.4f}")


# ## Visualize: Plot & Compare RMSE between Simpler Models 

# In[63]:


# horizontal
import seaborn as sns
import matplotlib.pyplot as plt

rmse_xgb_values = {
    'Random Forest': 15.7201, 
    'Stacking (XGB+RF+LR)': 18.1376,  
    'Gradient Boosting': 40.5690, 
    'KNN': 36.9434,  
    'ElasticNet': 44.7193,  
    'Linear Regression': 44.7125,  
}

# convert the RMSE values to pd df for seaborn
rmse_xgb_df = pd.DataFrame(list(rmse_xgb_values.items()), columns=['Model', 'RMSE'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=rmse_xgb_df, color='royalblue')

plt.title("XGBoost Selected Features' Model Comparison: RMSE Values", fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('RMSE', fontsize=14)

plt.xticks(rotation=45, ha='right')

for index, row in rmse_xgb_df.iterrows():
    plt.text(index, row['RMSE'] + 0.5, f"{row['RMSE']:.2f}", ha='center', color='black')

plt.tight_layout()
plt.show()


# In[64]:


# vertical
rmse_xgb_values = {
    'Random Forest': 15.7201, 
    'Stacking (XGB+RF+LR)': 18.1376,  
    'Gradient Boosting': 40.5690, 
    'KNN': 36.9434,  
    'ElasticNet': 44.7193,  
    'Linear Regression': 44.7125,  
}

rmse_xgb_df = pd.DataFrame(list(rmse_xgb_values.items()), columns=['Model', 'RMSE'])

plt.figure(figsize=(10, 6))
sns.barplot(x='RMSE', y='Model', data=rmse_xgb_df, color='royalblue')

plt.title("XGBoost Selected Features' Model Comparison: RMSE Values", fontsize=16)
plt.xlabel('RMSE', fontsize=14)
plt.ylabel('Models', fontsize=14)

plt.xticks(rotation=45, ha='right')

for index, row in rmse_xgb_df.iterrows():
    plt.text(row['RMSE'] + 0.5, index, f"{row['RMSE']:.4f}", va='center', color='black')

plt.show()


# ### 2nd Feedback from TA: 
# Handling of temporal dependencies: Since engine degradation follows a time series pattern, no mention is made of temporal features (e.g., cumulative degradation metrics, rolling window statistics).
