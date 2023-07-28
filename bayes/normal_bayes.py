import numpy as np
from datasets.dataset import DataSet
from sklearn.model_selection import train_test_split
from decision_tree.utils import Visualization

class NormalBayesClassifier:
    def __init__(self,n_class):
        self.pc = None # 数据中每个类的概率
        self.n_class = n_class
        self.sigma = [] # 两个类的数据的协方差矩阵
        self.mu = [] # 两个类的数据的均值向量

    def fit(self,data_all,targets):
        cla_count = np.bincount(targets)
        self.pc = cla_count/np.sum(cla_count)
        for i in range(cla_count.shape[0]):
            data = data_all[targets==i]
            mu = np.mean(data,axis=0)
            self.mu.append(mu)
            sigma = np.cov(data,rowvar=False)
            self.sigma.append(sigma)
    def _P(self,x):
        n = x.shape[1]
        p = []
        for i in range(self.n_class): # 按公式求当前样本属于每个类的概率
            a = 1/((2*np.pi)**(n/2)*np.linalg.det(self.sigma[i])**(1/2))
            t = np.mat(x - self.mu[i]) * np.linalg.pinv(self.sigma[i])
            b = np.exp(-(1/2)*np.sum(np.multiply(t,x-self.mu[i]),axis=1))
            p.append(a*b*self.pc[i])
        return np.array(p)

    def predict(self,data):
        p = self._P(data)
        cla = np.argmax(p,axis=0)
        return cla.reshape((data.shape[0],))


if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.3)
    nb = NormalBayesClassifier(n_class=2)
    nb.fit(X_train,y_train)
    res = nb.predict(X_test)
    score = np.sum(res==y_test)/len(y_test)
    print(score)
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './nor.png')