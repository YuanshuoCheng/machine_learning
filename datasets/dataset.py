import pandas as pd
import numpy as np


class DataSet:
    def __init__(self,path,mode='cla',rad_seed = 2021):
        data = pd.read_csv(path).dropna(axis=0, how='any')
        data1 = data[:2000]
        data2 = data[-2000:]
        data = pd.concat([data1, data2]).reset_index().drop(['index'], axis=1)
        data = data.replace('white', 0).replace('red', 1)
        if mode == 'cla':
            self.target_head = 'type'
        elif mode == 'reg':
            self.target_head = 'residual sugar'
        self.data_head = data.columns.to_list()
        self.data_head.remove(self.target_head)
        self.target = data[self.target_head].to_numpy()
        self.data = data[self.data_head].to_numpy()
        if rad_seed is not False:
            np.random.seed(rad_seed)
        permutation = list(np.random.permutation(len(self.data)))
        self.data = self.data[permutation]
        self.target = self.target[permutation]

    def get_data(self):
        return self.data,self.target,self.target_head,self.data_head