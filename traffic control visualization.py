import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
csv_file_path = 'C:/Users/Debabrata/Desktop/traffic.csv'
data = pd.read_csv(csv_file_path)
print(data.head())
data = data.dropna()


data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Hour'] = data['DateTime'].dt.hour
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['Month'] = data['DateTime'].dt.month

scaler = StandardScaler()
data[['Junction', 'Vehicles', 'Hour', 'DayOfWeek', 'Month']] = scaler.fit_transform(data[['Junction', 'Vehicles', 'Hour', 'DayOfWeek', 'Month']])

from scipy import stats
data = data[(np.abs(stats.zscore(data[['Junction', 'Vehicles', 'Hour', 'DayOfWeek', 'Month']])) < 3).all(axis=1)]

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['Junction', 'Vehicles', 'Hour', 'DayOfWeek', 'Month']])
data['pca_1'] = data_pca[:, 0]
data['pca_2'] = data_pca[:, 1]

kmeans = KMeans(n_clusters=5)
data['kmeans_cluster'] = kmeans.fit_predict(data[['pca_1', 'pca_2']])

dbscan = DBSCAN(eps=0.3, min_samples=10)
data['dbscan_cluster'] = dbscan.fit_predict(data[['pca_1', 'pca_2']])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Junction', y='Vehicles', hue='Hour', palette='viridis', data=data)
plt.title('Traffic Density Visualization by Hour')
plt.xlabel('Junction')
plt.ylabel('Vehicles')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca_1', y='pca_2', hue='kmeans_cluster', palette='tab10', data=data)
plt.title('Traffic Clustering with KMeans')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca_1', y='pca_2', hue='dbscan_cluster', palette='tab10', data=data)
plt.title('Traffic Clustering with DBSCAN')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Vehicles', hue='DayOfWeek', palette='tab10', data=data, errorbar=None)
plt.title('Traffic Density by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Vehicles')
plt.legend(title='Day of Week', loc='upper right', labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='Vehicles', data=data)
plt.title('Traffic Density by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Vehicles')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Vehicles', data=data, errorbar=None)
plt.title('Traffic Density by Month')
plt.xlabel('Month')
plt.ylabel('Vehicles')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

processed_csv_file = '/mnt/data/processed_traffic_data.csv'
data.to_csv(processed_csv_file, index=False)
print(f"Processed data saved to {processed_csv_file}")
