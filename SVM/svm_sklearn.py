import numpy as np
from sklearn.svm import SVC
from datasets.dataset import DataSet
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Visualization,Metrics



def run_svm_bin_linear(X_train,X_test,y_train,y_test):
    print('===== 二分类-线性核： =====')
    svm = SVC(C=200, kernel='linear')
    svm.fit(X_train, y_train)
    res = svm.predict(X_test)
    metrics = Metrics()
    metrics.update(res, y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './svm_sklearn_linear_res_in_testset.png')

def run_svm_bin_rbf(X_train,X_test,y_train,y_test):
    print('===== 二分类-高斯核： =====')
    svm = SVC(C=200, kernel='rbf')
    svm.fit(X_train, y_train)
    res = svm.predict(X_test)
    metrics = Metrics()
    metrics.update(res, y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './svm_sklearn_rbf_res_in_testset.png')

def run_svm_multi_1_rest(X_train,X_test,y_train,y_test):
    print('===== 多分类，1对剩余： =====')
    svm = SVC(C=200, kernel='rbf',decision_function_shape='ovr')
    svm.fit(X_train, y_train)
    res = svm.predict(X_test)
    print('acc:{:.4f}'.format(svm.score(X_test,y_test)))
    vis = Visualization(colors=['red', 'blue','green'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './svm_sklearn_1_rest_res_in_testset.png')

def run_svm_multi_1_1(X_train,X_test,y_train,y_test):
    print('===== 多分类，1对1： =====')
    svm = SVC(C=200, kernel='rbf', decision_function_shape='ovo')
    svm.fit(X_train, y_train)
    res = svm.predict(X_test)
    print('acc:{:.4f}'.format(svm.score(X_test, y_test)))
    vis = Visualization(colors=['red', 'blue', 'green'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './svm_sklearn_1_1_res_in_testset.png')

if __name__ == '__main__':
    np.random.seed(2021)
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data,target,target_head,data_head = dataset.get_data()
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    X_train,X_test,y_train,y_test = train_test_split(data,target,random_state=2021,test_size=0.3)
    run_svm_bin_linear(X_train, X_test, y_train, y_test)
    run_svm_bin_rbf(X_train, X_test, y_train, y_test)

    iris = datasets.load_iris()
    data = iris['data']
    target = iris['target']
    permutation = list(np.random.permutation(len(data)))
    data = data[permutation]
    target = target[permutation]
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.5)

    run_svm_multi_1_rest(X_train, X_test, y_train, y_test)
    run_svm_multi_1_1(X_train, X_test, y_train, y_test)

