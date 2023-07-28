from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
class Visualization:
    def __init__(self,colors):
        self.colors = colors
        self.x_norm = None
    def fit(self,data):
        if data.shape[-1]>2:
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=2021)
            X_tsne = tsne.fit_transform(data)
        else:
            X_tsne = data
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        self.x_norm = (X_tsne - x_min) / (x_max - x_min)
    def savefig(self,gt,name):
        plt.figure(figsize=(8, 8))
        for i in range(self.x_norm.shape[0]):
            plt.scatter(self.x_norm[i, 0], self.x_norm[i, 1],color=self.colors[gt[i]])#, str(gt[i]), color=self.colors[pred[i]],
                     #fontdict={'weight': 'bold', 'size': 8})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(name)
if __name__ == '__main__':
    data = load_iris()
    x = data['data']
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    y = data['target']
    com = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[]]
    for c in com:
        if len(c)>0:
            x_ = x[:,c]
            name = str(c[0])+str(c[1])+'.png'
        else:
            x_ = x
            name = 'pca'+'.png'
        print(x_.shape)
        vis = Visualization(colors=['red','blue','yellow'])
        vis.fit(x_)
        vis.savefig(y,name)