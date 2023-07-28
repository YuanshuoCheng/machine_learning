import numpy as np
from datasets.dataset import DataSet
from utils import Visualization,Metrics
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self,C=1,tol = 0.001,random_seed=2021,gamma=10,type_k='rbf'):
        self.b = None
        self.alpha = None
        self.C = C # 软间隔
        self.tol = tol # 松弛变量
        self._K = None # 经过核函数映射后内积
        self._targets = None # 标签[-1,1]
        self.gamma = gamma
        self.support_vec = None
        self.support_vec_alpha = None
        self.support_vec_targets = None
        self._type_k = type_k
        np.random.seed(random_seed)

    def fit(self,data,targets,max_iter_steps=100):
        self._targets = self._nor_targets(targets) # 标签映射到[-1,1]
        if self._type_k == 'rbf': # 选择核函数
            self._K = self._rbf_K(data)
        self.alpha = np.zeros(data.shape[0])
        self.b = 0
        parameter_changed = 1 # flag，记录本次迭代是否优化了参数

        for iter in range(max_iter_steps):
            if parameter_changed == 0: # 没有优化参数，此时迭代完成退出
                break
            parameter_changed = 0
            first_check = (self.alpha>0) * (self.alpha<self.C)
            check_i_range = np.concatenate([np.where(first_check)[0],np.where(~first_check)[0]],axis=0)
            for i in check_i_range:
                if not self._is_meet_KKT(i):
                    j = self.select_j(i)
                    if self._targets[i]==self._targets[j]: # 对同号和不同号两种情况下计算alpha的上界和下界
                        L = max(0,self.alpha[i]+self.alpha[j]-self.C)
                        H = min(self.C,self.alpha[j]+self.alpha[i])
                    else:
                        L = max(0,self.alpha[j]-self.alpha[i])
                        H = min(self.C,self.C+self.alpha[j]-self.alpha[i])
                    if L == H:
                        continue
                    eta = self._K[i,i]+self._K[j,j]-2*self._K[i,j]
                    alpha_j_new = self.alpha[j]+(self._targets[j]*(self._cal_E(i)-self._cal_E(j)))/eta # 更新alpha的值
                    if alpha_j_new >H:
                        alpha_j_new = H
                    elif alpha_j_new<L:
                        alpha_j_new = L
                    alpha_i_new = self.alpha[i]+self._targets[i]*self._targets[j]*(self.alpha[j]-alpha_j_new)

                    b1_new = -self._cal_E(i) - self._targets[i] * self._K[i,i] * (alpha_i_new - self.alpha[i]) \
                            - self._targets[j] * self._K[j,i] * (alpha_j_new - self.alpha[j]) + self.b
                    b2_new = -self._cal_E(j) - self._targets[i] * self._K[i,j] * (alpha_i_new - self.alpha[i]) \
                            - self._targets[j] * self._K[j,j] * (alpha_j_new - self.alpha[j]) + self.b

                    if (alpha_i_new > 0) and (alpha_i_new < self.C):
                        b_new = b1_new
                    elif (alpha_j_new > 0) and (alpha_j_new < self.C):
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2
                    if (abs(self.alpha[j] - alpha_j_new)>1e-5):
                        parameter_changed+=1
                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new
                    self.b = b_new
            print('iter:%d, param changed:%d'%(iter,parameter_changed))


        support_vec_i = self.alpha>0
        self.support_vec_alpha = self.alpha[support_vec_i]
        self.support_vec = data[support_vec_i]
        self.support_vec_targets = self._targets[support_vec_i]
        self._targets = None
        self._K = None

    def _rbf_K(self,data):
        dif = data.reshape((data.shape[0],1,data.shape[1]))-data.reshape((1,data.shape[0],data.shape[1]))
        l2 = np.sum(dif**2,axis=2)
        res = np.exp(-1 * l2 / (2 * self.gamma**2))
        return res

    def select_j(self,i):
        alpha_j = -1
        E_i = self._cal_E(i)

        max_E_abs = -1
        for j in range(self._targets.shape[0]):
            E_j = self._cal_E(j)
            E_abs = np.fabs(E_i-E_j)
            if E_abs>max_E_abs:
                alpha_j = j
                max_E_abs = E_abs
        if alpha_j == -1:
            alpha_j = np.random.randint(len(self._targets))
            while alpha_j == i:
                alpha_j = np.random.randint(len(self._targets))
        return alpha_j

    def _cal_E(self,i):
        res = np.dot(self._targets*self.alpha,self._K[:,i])+self.b-self._targets[i]
        return res

    def _is_meet_KKT(self,i):
        res = False
        y_i = self._targets[i]
        g_xi = np.dot(self.alpha*self._targets,self._K[i,:].T)+self.b
        if (np.fabs(self.alpha[i]) < self.tol) and (y_i * g_xi >= 1):
            res = True
        elif (np.fabs(self.alpha[i] - self.C) < self.tol) and (y_i * g_xi <= 1):
            res = True
        elif (self.alpha[i] > -self.tol) and (self.alpha[i] < (self.C + self.tol)) \
                and (np.fabs(y_i * g_xi - 1) < self.tol):
            res = True
        #res = False
        return res

    def _nor_targets(self,targets):
        a = -1
        b = 1
        neg = 0
        pos = 1
        k =(b - a) / (pos - neg)
        return a+k*(targets-neg)
    def _recover_targets(self,nored_targets):
        a = -1
        b = 1
        neg = 0
        pos = 1
        return neg+(pos-neg)/(b-a)*(nored_targets-a)

    def predict(self,data):
        k = self._cal_k(self.support_vec,data)
        #print(k)
        res = np.sign(np.dot(self.support_vec_alpha*self.support_vec_targets,k)+self.b)
        return self._recover_targets(res)

    def _cal_k(self, sup_v, data):
        result = None
        if self._type_k == 'rbf':
            dif = sup_v.reshape((sup_v.shape[0],1,sup_v.shape[1])) - data
            l2 = np.sum(dif**2,axis=2)
            result = np.exp(-1 * l2 / (2 * self.gamma ** 2))
        return result


if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.3)
    svm = SVM(C=50,tol = 1e-16,random_seed=2021,gamma=10,type_k='rbf')
    svm.fit(X_train[:500],y_train[:500],max_iter_steps=5)
    res = svm.predict(X_test).astype(np.int32)
    metrics = Metrics()
    metrics.update(res, y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red', 'blue','green'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './svm_diy_res_in_testset.png')