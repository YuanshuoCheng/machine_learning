from datasets.dataset import DataSet
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

class Visualization:
    def __init__(self,colors):
        self.colors = colors
        self.x_norm = None
    def fit(self,data):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=2021)
        X_tsne = tsne.fit_transform(data)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        self.x_norm = (X_tsne - x_min) / (x_max - x_min)
    def show(self,gt,pred):
        plt.figure(figsize=(8, 8))
        for i in range(self.x_norm.shape[0]):
            plt.text(self.x_norm[i, 0], self.x_norm[i, 1], str(gt[i]), color=self.colors[pred[i]],
                     fontdict={'weight': 'bold', 'size': 8})
        plt.xticks([])
        plt.yticks([])
        plt.show()
    def savefig(self,gt,pred,file_name):
        plt.figure(figsize=(8, 8))
        for i in range(self.x_norm.shape[0]):
            plt.text(self.x_norm[i, 0], self.x_norm[i, 1], str(gt[i]), color=self.colors[pred[i]],
                     fontdict={'weight': 'bold', 'size': 8})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(file_name)


class Metrics:
    def __init__(self,n_classes=2):
        self.n_classes = n_classes
        self.confusion_matrix = [[0 for i in range(n_classes)] for j in range(n_classes)]
        self.labels = []
        self.preds = []
        self.ALL = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.TP = None

    def empty(self):
        self.confusion_matrix = [[0 for i in range(self.n_classes)] for j in range(self.n_classes)]
        self.labels = []
        self.preds = []

    def update(self,preds,labels):
        for i in range(len(labels)):
            self.confusion_matrix[labels[i]][preds[i]]+=1

    def count(self):
        confusion_matrix = np.array(self.confusion_matrix)
        self.ALL = np.sum(confusion_matrix)
        self.TN = confusion_matrix[0,0]
        self.FP = confusion_matrix[0,1]
        self.FN = confusion_matrix[1,0]
        self.TP = confusion_matrix[1,1]

    def accuracy(self):
        res = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        return res

    def recall(self):
        res = self.TP/(self.TP+self.FN)
        return res

    def precision(self):
        res = self.TP/(self.TP+self.FP)
        return res



if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()

    X_train = np.concatenate([data[:200],data[-200:]])
    Y_train = np.concatenate([target[:200],target[-200:]])
    X_train = (X_train-np.min(X_train,axis=0))/(np.max(X_train,axis=0)-np.min(X_train,axis=0))
    vis = Visualization(['red','blue'])
    vis.fit(X_train)
    vis.show(Y_train,Y_train)
