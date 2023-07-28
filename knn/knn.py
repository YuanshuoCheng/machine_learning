import numpy as np
from datasets.dataset import DataSet
from sklearn.model_selection import train_test_split

def distance(data1,data2):
    data1 = data1.reshape((data1.shape[0],1,data1.shape[1]))
    dis = np.sum((data1-data2)**2,axis=2)
    return dis**0.5

class KNN:
    def __init__(self,k):
        self.k = k
        self.data = None
        self.targets = None
    def fit(self,data,targets):
        self.data = data
        self.targets = targets
    def predict(self,data):
        dis = distance(data,self.data)
        res = np.argsort(dis,axis=1)
        top_k_i = res[:,:self.k]
        cla = []
        for i in range(top_k_i.shape[0]):
            t = np.bincount(self.targets[top_k_i[i]])
            cla.append(np.argmax(t))
        return np.array(cla)


if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.3)
    knn = KNN(k=9)
    knn.fit(X_train,y_train)
    res = knn.predict(X_test)
    score = np.sum(res==y_test)/len(y_test)
    print(score)