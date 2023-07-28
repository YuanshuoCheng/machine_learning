from datasets.dataset import DataSet
import numpy as np
from sklearn.cluster import KMeans
from utils import Visualization

if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()

    X_train = np.concatenate([data[:300],data[-300:]])
    Y_train = np.concatenate([target[:300],target[-300:]])
    #X_train = (X_train-np.min(X_train,axis=0))/(np.max(X_train,axis=0)-np.min(X_train,axis=0))
    color = ['red','blue','green','yellow']
    model = KMeans(n_clusters=2)
    model.fit(X_train)
    centers = model.cluster_centers_
    result = model.predict(X_train)
    vis = Visualization(colors=color)
    vis.fit(X_train)
    vis.savefig(Y_train,result,'kmeans_sklearn_.png')



