import numpy as np
from datasets.dataset import DataSet
from sklearn.model_selection import train_test_split
from utils import Visualization,Metrics

def Gini(targets):
    N = len(targets)+1e-5
    count = np.bincount(targets)
    return (1-((np.sum(count**2))/(N**2)))

class Node:
    def __init__(self,max_sections=10):
        self.type = None
        self.threshold = None
        self.feat = None
        self.left = None
        self.right = None
        self.label = None
        self.max_sections = max_sections # 寻找最优阈值的时候最多尝试max_sections次
    def fit(self,data,targets,stop_n):
        count = np.bincount(targets)
        condition = len(data)<=stop_n or np.max(count)/np.sum(count)>0.99 # 是否终止分裂
        if condition: # 叶节点，选择流入当前节点的数据中类别最多的类作为本叶子节点的标签
            self.type = 'leaf'
            self.label = np.argmax(count)
        else:
            self.type = 'root'
            best_feat_i = 0
            best_hold = 0
            best_l_i = None
            best_r_i = None
            G = None
            for i in range(len(data[0])):
                feat_i = data[:,i]
                # 如果流入当前节点的数据小于max_sections，则阈值在这些数据中寻找，否则等距取# 如果流入当前节点的数据小于max_sections个数并寻找最优阈值
                if len(targets)<=self.max_sections:
                    holds = feat_i
                else:
                    holds = np.linspace(np.min(feat_i),np.max(feat_i),self.max_sections)
                for hold in holds:
                    l_i = (feat_i<=hold)
                    r_i = (feat_i>hold)
                    new_g = (len(l_i)*Gini(targets[l_i])+len(r_i)*Gini(targets[r_i]))/len(targets)
                    if G is None or new_g<G:
                        G = new_g
                        best_feat_i = i
                        best_hold = hold
                        best_l_i = l_i
                        best_r_i = r_i
            self.feat = best_feat_i
            self.threshold = best_hold
            self.left = Node(self.max_sections) # 递归建立左子树
            self.left.fit(data[best_l_i],targets[best_l_i],stop_n)
            self.right = Node(self.max_sections) # 递归建立右子树
            self.right.fit(data[best_r_i],targets[best_r_i],stop_n)

    def decide(self,data):
        if self.type == 'leaf': # 如果是叶子节点返回标签
            return self.label
        if data[self.feat]<=self.threshold: # 如果不是叶子节点，数据继续流动
            return self.left.decide(data)
        else:
            return self.right.decide(data)

class DecisionTree:
    def __init__(self,stop_n=32,max_sections=10):
        self.stop_n = stop_n
        self.root = Node(max_sections)
    def fit(self,data,targets):
        self.root.fit(data,targets,self.stop_n)

    def predict(self,data):
        res = []
        for data_i in data:
            res.append(self.root.decide(data_i))
        return np.array(res)
    def score(self,x_test,y_test):
        res = self.predict(x_test)
        return np.sum(res==y_test)/len(y_test)

class RandomForest:
    def __init__(self,n_estimators=32,n_samples=100,seed=2021,stop_n=32,max_sections=10):
        self.n_samples = n_samples
        self.n_estimators = n_estimators
        self.seed=seed
        self.stop_n = stop_n # 决策树叶子节点最多流入的样本数
        self.max_section = max_sections
        self.estimators = [DecisionTree(stop_n=self.stop_n,max_sections=self.max_section) for i in range(self.n_estimators)]
        # 多棵决策树
    def fit(self,data,targets):
        for i in range(self.n_estimators):
            sample_id = np.random.choice(len(data),self.n_samples) # 随机抽样并训练每科决策树
            self.estimators[i].fit(data[sample_id],targets[sample_id])
    def predict(self,data,xx=None):
        res = [] # 存放每科决策树的结果，最终结果少数服从多数
        for estimator in self.estimators:
            res.append(estimator.predict(data))
        res = np.array(res)
        res = np.array([np.argmax(np.bincount(res[:,i])) for i in range(res.shape[1])])
        return res
    def score(self,x_test,y_test):
        res = self.predict(x_test,y_test)
        return np.sum(res==y_test)/len(y_test)

def run_decision_tree(X_train,X_test,y_train,y_test):
    metrics = Metrics()
    dt = DecisionTree(stop_n=9,max_sections=10)
    dt.fit(X_train,y_train)
    res = dt.predict(X_test)
    metrics.update(res, y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './dt_diy_res_in_testset.png')

def run_random_forest(X_train,X_test,y_train,y_test):
    metrics = Metrics()
    rf = RandomForest(stop_n=9,max_sections=10,n_estimators=50,n_samples=200)
    rf.fit(X_train,y_train)
    res = rf.predict(X_test)
    metrics.update(res, y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './rf_diy_res_in_testset.png')

if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.3)
    print('======== Decision tree experiment START ========')
    run_decision_tree(X_train, X_test, y_train, y_test)
    print('======== Decision tree experiment END ========', '\n')
    print('======== Random forest experiment START ========')
    run_random_forest(X_train, X_test, y_train, y_test)
    print('======== Random forest experiment END ========', '\n')
