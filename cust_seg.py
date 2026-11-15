import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


df = pd.read_csv("customer_segmentation.csv")
print(df.head())
df.info()
print(df.isnull().sum())

df["Income"] = df["Income"].fillna(df["Income"].median())
df = df[df["Income"] < 600000]  # remove extreme outlier

df["Marital_Status"] = df["Marital_Status"].replace(
    {"Alone": "Single", "Absurd": "Single", "YOLO": "Single", "Together": "Married"}
)
df = df.drop(columns=["Z_CostContact", "Z_Revenue", "ID"])


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
df["Age"] = 2014 - df["Year_Birth"]
df["Customer_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days

mnt = ["MntWines", "MntFruits", "MntMeatProducts",
       "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["Total_Spent"] = df[mnt].sum(axis=1)
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Total_Purchases"] = df[["NumWebPurchases", "NumCatalogPurchases",
                            "NumStorePurchases"]].sum(axis=1)
df["Total_Campaigns"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                            "AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1)
df = df[df["Age"] < 100]


sns.histplot(df["Age"], bins = 30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df["Income"], bins = 30, kde=True)
plt.title("Income Distribution")
plt.show()

sns.histplot(df["Total_Spent"], bins = 30, kde=True)
plt.title("Total Spent Distribution")
plt.show()

sns.boxplot(x="Education", y = "Income", data=df)
plt.xticks(rotation = 45)
plt.title("Income by Education Level")
plt.show()

corr_m = df[["Income", "Age", "Recency", "Total_Spent", "NumWebPurchases", "NumStorePurchases"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_m, annot = True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

features = ["Age", "Income", "Total_Spent", "Total_Purchases", "Recency",
            "Children", "NumWebVisitsMonth", "NumDealsPurchases",
            "Total_Campaigns", "Customer_Days"]
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

wcss, sil = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(2, 11), wcss, "o-")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("k")
ax[1].plot(range(2, 11), sil, "o-")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("k")
plt.show()

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
df["Cluster"] = kmeans.labels_


pcs = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Cluster"], palette="viridis")
plt.title("Customer Segments (PCA)")
plt.show()

profile = df.groupby("Cluster")[features].mean().round(1)
print(profile)

sns.boxplot(x="Cluster", y="Total_Spent", data=df)
plt.title("Total Spent by Cluster")
plt.show()

sns.boxplot(x="Cluster", y="Income", data=df)
plt.title("Income by Cluster")
plt.show()


df.to_csv("customers_with_clusters.csv", index=False)
print("Saved customers_with_clusters.csv")

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved kmeans_model.pkl and scaler.pkl")

