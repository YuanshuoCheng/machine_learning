from nn import Linear,Relu,SGD,CrossEntropyLoss
import numpy as np
from utils import Metrics,DataLoader,Visualization
from sklearn.model_selection import train_test_split
from datasets.dataset import DataSet


class MyModel:
    def __init__(self):
        self.linear1 = Linear(12,64)
        self.relu1 = Relu()
        self.linear2 = Linear(64,32)
        self.relu2 = Relu()
        self.linear3 = Linear(32, 2)
    def __call__(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    np.random.seed(2021)
    batch_size = 16
    lr = 0.001
    classes = 2
    epoch_n = 200

    dataset = DataSet('F:\PycharmProjects\machine_leatning\datasets\winequalityN.csv')
    data, target, target_head, data_head = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2021, test_size=0.3)
    train_loader = DataLoader(X_train,y_train,batch_size)
    test_loader = DataLoader(X_test, y_test, batch_size)

    loss_fn = CrossEntropyLoss()
    model = MyModel()
    opt = SGD(model, lr)
    for epoch in range(epoch_n):
        epoch_loss = 0
        cnt = 0
        train_loader.shuffle()
        for batch_x,batch_y in train_loader:
            batch_y_one_hot = np.eye(classes)[batch_y]
            res = model(batch_x)
            loss = loss_fn(res, batch_y_one_hot)
            opt.backward(loss_fn)
            opt.step()
            epoch_loss+=loss
            cnt+=1
        if epoch%10 == 0:
            print('epoch:%d || loss:%.6f'%(epoch,epoch_loss/cnt))

    metrics_test = Metrics()
    total_res = np.array([])
    for batch_x, batch_y in test_loader:
        batch_y_one_hot = np.eye(2)[batch_y]
        res_onehot = model(batch_x)
        res = np.argmax(res_onehot, axis=1)
        total_res = np.concatenate([total_res,res])
        metrics_test.update(res, batch_y)
    metrics_test.count()
    print('acc:{:.4f}'.format(metrics_test.accuracy()))
    print('recall:{:.4f}'.format(metrics_test.recall()))
    print('precision:{:.4f}'.format(metrics_test.precision()))

    vis = Visualization(colors=['red', 'blue'])
    vis.fit(X_test)
    vis.savefig(y_test, total_res.astype(np.int32), './ann_diy_res_in_testset.png')