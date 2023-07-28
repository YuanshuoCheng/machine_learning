from datasets.dataset import DataSet
import numpy as np
from sklearn.cluster import DBSCAN
from utils import Visualization


if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()

    X_train = np.concatenate([data[:300],data[-300:]])
    Y_train = np.concatenate([target[:300],target[-300:]])
    X_train = (X_train-np.min(X_train,axis=0))/(np.max(X_train,axis=0)-np.min(X_train,axis=0))
    color = ['red','blue','green','yellow']
    model = DBSCAN(eps=0.3, min_samples=25)
    model.fit(X_train)
    result = model.labels_
    vis = Visualization(colors=color)
    vis.fit(X_train)
    vis.savefig(Y_train,result,'dbscan_.png')