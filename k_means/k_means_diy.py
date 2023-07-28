from datasets.dataset import DataSet
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from utils import Visualization
from sklearn.model_selection import train_test_split

def dis(m,v): # 计算欧式距离返回n*1大小矩阵，为n条数据到v的距离
    return np.sqrt(np.sum((m-v)**2,axis=1))

class KMEANS:
    def __init__(self,n_clusters,rad_seed=2021):
        self.n_clusters = n_clusters # 聚类中心个数
        self.centers = [None]*self.n_clusters # 存放聚类中心
        self.res = None
        self.its = 0 # 记录迭代次数
        self.rad_seed = rad_seed
    def fit(self,data):
        if self.rad_seed is not False:
            np.random.seed(self.rad_seed)
        for i in range(self.n_clusters):
            self.centers[i] = data[i] # 从数据中随意选取n个数作为n个聚类中心的初始值

        while(True): # 开始迭代求聚类中心
            self.its+=1
            distance = []
            for i in range(self.n_clusters): # 计算所有数据到每个聚类中心的距离
                distance.append(dis(data,self.centers[i]))
            distance = np.array(distance).T
            cla = np.argmin(distance,axis=1) # 按照距离最近原则将数据分配给各个聚类中心成为n_clusters簇
            centers = [None]*self.n_clusters
            for i in range(self.n_clusters): # 求每个簇的均值作为新的一组聚类中心
                cluster = data[cla==i]
                centers[i] = np.mean(cluster,axis=0)
            # 如果两次迭代中聚类中心没变，说明训练完成
            is_same = np.sum([np.abs(i-j)==0 for i,j in zip(centers,self.centers)])==data.shape[1]*self.n_clusters
            self.centers = centers
            if is_same:
                self.res = cla
                return self.res,self.its,self.centers
    def predict(self,data):
        distance = []
        for i in range(self.n_clusters):  # 计算所有数据到每个聚类中心的距离
            distance.append(dis(data, self.centers[i]))
        distance = np.array(distance).T
        cla = np.argmin(distance, axis=1)  # 按照距离最近原则将数据分配给各个聚类中心成为n_clusters簇
        return cla

if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()

    X_train = np.concatenate([data[:300], data[-300:]])
    y_train = np.concatenate([target[:300], target[-300:]])

    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))

    color = ['blue', 'red']
    kmeans = KMEANS(n_clusters=2)
    _,n_its,centers = kmeans.fit(X_train)

    test_data = X_train
    test_target = y_train
    result=kmeans.predict(test_data)
    vis = Visualization(colors=color)
    vis.fit(X_train)
    vis.savefig(test_target, result, 'kmeans_diy.png')
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=2021)
    # X_tsne = tsne.fit_transform(test_data)
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(test_target[i]), color=color[result[i]],
    #              fontdict={'weight': 'bold', 'size': 8})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()






