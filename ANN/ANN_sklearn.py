from sklearn.neural_network import MLPClassifier
from datasets.dataset import DataSet
from sklearn.model_selection import train_test_split
import numpy as np
from utils import Visualization,Metrics


np.random.seed(2021)
dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
data,target,target_head,data_head = dataset.get_data()
X_train,X_test,y_train,y_test = train_test_split(data,target,random_state=2021,test_size=0.3)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train,axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test,axis=0)
mp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(16,16,16,16,16), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
mp.fit(X_train, y_train)
res = mp.predict(X_test)
metrics = Metrics()
metrics.update(res, y_test)
metrics.count()
print('acc:{:.4f}'.format(metrics.accuracy()))
print('recall:{:.4f}'.format(metrics.recall()))
print('precision:{:.4f}'.format(metrics.precision()))
vis = Visualization(colors=['red', 'blue'])
vis.fit(X_test)
vis.savefig(y_test, res, './ann_sklearn_res_in_testset.png')
