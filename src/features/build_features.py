import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
import matplotlib as mpl
from sklearn.cluster import KMeans

#  %matplotlib inline

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

subset = df[df["set"] == 35]

subset = df[df["set"] == 35]["gyr_y"]
subset.plot()

for col in predictor_columns:
    df[col] = df[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 5]["acc_y"].plot()

duration = df[ df["set"] == 1 ].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    
    df.loc[(df["set"] == s), "duration"] = duration.seconds


duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5     # No. of repititions
duration_df.iloc[1] / 10

df[df["set"] == 15]["acc_y"].plot()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()
# stepsize of 200ms

fs = 1000 / 200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), fancybox=True, shadow=True)


#---------------------------------------------------------

df_lowpass = df.copy()
fs = 1000 / 200
cutoff = 1.3

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1,len(predictor_columns)+1), pc_values,)
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 22]

subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = (df_squared["acc_x"]**2) + (df_squared["acc_y"]**2) + (df_squared["acc_z"]**2) 
gyr_r = (df_squared["gyr_x"]**2) + (df_squared["gyr_y"]**2) + (df_squared["gyr_z"]**2) 

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 22]

subset[["acc_r","gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r","gyr_r"]

#window size
ws = int(1000 / 200)
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")


df_temporal_list = []
for s in df_temporal["set"].unique():
    
    subset = df_temporal[df_temporal["set"] == s].copy()
    
    for col in predictor_columns:
        # this loop was not necessary.
        # U could just use predictor_columns and the work would be done...
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)


subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].reset_index(drop=True).plot()

subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()



# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# Sampling rate - no. of samples per sec
fs = int(1000 / 200)
# Window Size
ws = int(2800 / 200)  # avg length of repition was 2.8s

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# Visualize Results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14"
    ]
].plot()
# pse - power spectral entropy



df_freq_list = []

for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformation to set - {s}")
    
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
df_freq.info()
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
# In order to deal with the overlapping windows, we will get rid of some part of the data.
# If u look at teh literature, an allowance of 50% is recommended.
# Which means, in our case, we get rid of 50% of the data by skipping every other row.
# It will result in huge data loss, but it has been shown to payoff in the long run by making your models less prone to overfitting.
# We have enough data points to afford the loss.

# Getting every 2nd row
df_freq = df_freq[::2]


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# K-means Clustering from Scikit-Learn

cluster_columns = ["acc_x","acc_y","acc_z"]
k_value = range(2,10)
inertias = []

df_cluster = df_freq.copy()

for k in k_value:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_value, inertias)
plt.xlabel("k")
plt.ylabel("Sum of Squared Distances")
plt.show()

# Elbow at k = 5

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
    
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
    
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster = df_cluster.drop(["duration"], axis = 1)

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")


