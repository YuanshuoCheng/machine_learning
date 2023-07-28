from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from datasets.dataset import DataSet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import Visualization,Metrics


def run_decision_tree(X_train,X_test,y_train,y_test):
    metrics = Metrics()
    disiontree = DecisionTreeClassifier(random_state=0,criterion='gini',)
    disiontree.fit(X_train,y_train)
    res = disiontree.predict(X_test)
    metrics.update(res,y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red','blue'])
    vis.fit(X_test)
    vis.savefig(y_test,res,'./dt_sklearn_res_in_testset.png')
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(
        disiontree,
        feature_names=data_head,
        class_names=target_head,
        filled=True
    )
    fig.savefig("decistion_tree.png")

def run_random_forest(X_train,X_test,y_train,y_test):
    metrics = Metrics()
    rf = RandomForestClassifier(random_state=0,n_estimators=100)
    rf.fit(X_train, y_train)
    res = rf.predict(X_test)
    metrics.update(res,y_test)
    metrics.count()
    print('acc:{:.4f}'.format(metrics.accuracy()))
    print('recall:{:.4f}'.format(metrics.recall()))
    print('precision:{:.4f}'.format(metrics.precision()))
    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, res, './rf_sklearn_res_in_testset.png')


if __name__ == '__main__':
    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data,target,target_head,data_head = dataset.get_data()
    X_train,X_test,y_train,y_test = train_test_split(data,target,random_state=2021,test_size=0.3)
    print('======== Decision tree experiment START ========')
    run_decision_tree(X_train,X_test,y_train,y_test)
    print('======== Decision tree experiment END ========','\n')
    print('======== Random forest experiment START ========')
    run_random_forest(X_train,X_test,y_train,y_test)
    print('======== Random forest experiment END ========','\n')