import matplotlib.pyplot as plt
import pandas as pd
datas=pd.read_csv("car_data.csv")
x=datas.iloc[:,[8,9]].values
print(x)
from sklearn.cluster import KMeans
wcss_list=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
    plt.plot(range(1,10))
    plt.title("elbow graph")
    plt.xlabel("length")
    plt.ylabel("wcss_list")
    plt.show()
    kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
    y_predict=kmeans.fit_predict(x)
    print(y_predict)
    plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c="green",label="cluster1")
    plt.scatter(x[y_predict ==1,0], x[y_predict == 1,1], s=100, c="blue", label="cluster2")

    plt.scatter(x[y_predict == 2,0], x[y_predict == 2,1], s=100, c="yellow", label="cluster3")
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="black",label="cluster")
    plt.title("cluster data")
    plt.xlabel("wheelbase")
    plt.ylabel("length")
    plt.legend()
    plt.show()
