import numpy as np
from datasets.dataset import DataSet
from sklearn.model_selection import train_test_split
from decision_tree.utils import Visualization

class NaiveBayesClassifier:
    def __init__(self,n_class):
        self.pc = None
        self.dis_i = []
        self.con_i = []
        self.n_feats = 0
        self.dis_feat_count = None
        self.con_feat_count = None
        self.n_class = n_class

    def fit(self,data_all,targets,dis_i=[]): # dis_i为离散特征的索引
        cla_count = np.bincount(targets)
        self.pc = cla_count/np.sum(cla_count) # 训练集中每个类的概率
        self.dis_i = dis_i
        self.con_i = list(range(data_all.shape[1]))
        for i in dis_i:
            self.con_i.remove(i) # 删除离散特征的索引，留下连续特征的索引
        self.n_feats = data_all.shape[1]
        all_dis_feat_count = [] # 存放每个类数据中离散特征的统计
        all_con_feat_count = [] # 存放每个类数据中连续特征的统计
        max_nv = 0
        for i in dis_i:
            max_nv = max(max_nv,np.max(np.unique(data_all[:,i])))
        max_nv = int(max_nv+1)
        for cla in np.unique(targets):
            data = data_all[targets==cla] # 取出一个类的数据
            dis_feat_count = []
            con_feat_count = []
            for i in range(data.shape[1]):
                data_i = data[:,i]
                if i in dis_i: # 离散特征
                    data_i = data_i.astype(np.int32)
                    count = np.bincount(data_i)+1 # 拉普拉斯平滑
                    dis_feat_count.append(count) # 一个类的所有特征的值的统计
                else: # 如果是连续特征则计算均值和标准差
                    mu = np.mean(data_i)
                    sigma = np.std(data_i,ddof=1)
                    con_feat_count.append([mu,sigma])
            for i in range(len(dis_feat_count)):
                count_i = dis_feat_count[i]
                if count_i.shape[0]<max_nv: # 这里统一数组的长度，便于转化为ndarray计算
                    dif = max_nv-count_i.shape[0]
                    dis_feat_count[i] = np.concatenate([count_i,np.zeros(dif)])
            all_dis_feat_count.append(np.array(dis_feat_count))
            all_con_feat_count.append(np.array(con_feat_count))
        self.dis_feat_count = np.array(all_dis_feat_count)
        self.con_feat_count = np.array(all_con_feat_count)

    def _P(self,x,i): # 计算第i个特征属于两个类的概率
        if i in self.dis_i:
            index = self.dis_i.index(i)
            x = x.astype(np.int32)
            # 求离散特征的概率
            p = self.dis_feat_count[:,index,x].reshape((x.shape[0],self.n_class))/np.sum(self.dis_feat_count[:,index],axis=1)
        elif i in self.con_i:
            index = self.con_i.index(i)
            t = self.con_feat_count[:,index]
            mu = t[:,0]
            sigma = t[:,1]
            # 求连续特征的概率，以正态分布概率密度函数值近似概率值
            p = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x.reshape((x.shape[0],1))-mu)**2/(2*sigma**2))
        return p*self.pc

    def predict(self,data):
        p = np.ones((data.shape[0],self.n_class))
        for i in range(data.shape[1]):
            feat = data[:,i]
            p = p*self._P(feat,i) #朴素假设，总的概率等于每个特征的概率的累乘
        cla = np.argmax(p,axis=1)
        return cla


if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.3)
    nb = NaiveBayesClassifier(n_class=2)
    nb.fit(X_train,y_train,dis_i = [11])
    res = nb.predict(X_test)
    score = np.sum(res==y_test)/len(y_test)
    print('Acc:',score)
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './rf_diy_res_in_testset.png')

