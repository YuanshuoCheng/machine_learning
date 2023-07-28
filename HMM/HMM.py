import numpy as np
import re
import json


class HMM:
    def __init__(self):
        self.pi = None
        self.A = None
        self.B = None
        self.label2code = None
        self.word2code = None
        self.code2label = None
    def train(self,txt_path):
        x,y = self._path2tranData(txt_path)
        self.word2code = self._wordDic(x)
        self.label2code,self.code2label = self._labelDic(y)
        self.A = self._getA(y)
        self.pi = self._getPi(y)
        self.B = self._getB(x,y)

    def _getA(self,y):
        n_label = len(self.code2label)
        n_y = len(y)
        A = np.ones((n_label,n_label))
        for i in range(n_y-1):
            cur = y[i]
            next = y[i+1]
            A[self.label2code[cur]][self.label2code[next]] += 1
        #A = (A/A.sum(axis=1))
        #A = np.ones((n_label, n_label))
        s = np.sum(A,axis=1)
        s = s.reshape((s.shape[0],1))
        A = np.log(A/s)
        return A

    def _getPi(self,y):
        n_label = len(self.code2label)
        n_y = len(y)
        pi = np.zeros(n_label)
        for i in y:
            pi[self.label2code[i]]+=1
        pi = np.log(pi/n_y)
        return pi
    def _getB(self,x,y):
        n_word = len(self.word2code)
        n_labels = len(self.code2label)
        n_xy = len(x)
        B = np.ones((n_labels,n_word))
        for i in range(n_xy):
            cur_x = x[i]
            cur_y = y[i]
            B[self.label2code[cur_y]][self.word2code[cur_x]] += 1
        #B = np.ones((n_labels, n_word))
        s = B.sum(axis=1)
        s = s.reshape(s.shape[0],1)
        B = np.log(B / s)
        return B
    def tag(self,data):
        seq_len, num_labels = len(data), len(self.code2label)
        scores = self.pi.reshape((-1, 1))+self.B[:,self.word2code[data[0]]].reshape((-1, 1))
        paths = []
        for word in data[1:]:
            if word not in self.word2code.keys():
                scores_repeat = np.repeat(scores, num_labels, axis=1)
                # observe当前时刻t的每个标签的观测分数
                #observe = self.B[:, self.word2code[word]].reshape((1, -1))
                #observe_repeat = np.repeat(observe, num_labels, axis=0)
                # 从t-1时刻到t时刻最优分数的计算，这里需要考虑转移分数trans
                M = scores_repeat + self.A
                # 寻找到t时刻的最优路径
                scores = np.max(M, axis=0).reshape((-1, 1))
                idxs = np.argmax(M, axis=0)
                # 路径保存
                paths.append(idxs.tolist())
            else:
                # scores 表示起始0到t-1时刻的每个标签的最优分数
                scores_repeat = np.repeat(scores, num_labels, axis=1)
                # observe当前时刻t的每个标签的观测分数
                observe = self.B[:,self.word2code[word]].reshape((1, -1))
                observe_repeat = np.repeat(observe, num_labels, axis=0)
                # 从t-1时刻到t时刻最优分数的计算，这里需要考虑转移分数trans
                M = scores_repeat + self.A + observe_repeat
                # 寻找到t时刻的最优路径
                scores = np.max(M, axis=0).reshape((-1, 1))
                idxs = np.argmax(M, axis=0)
                # 路径保存
                paths.append(idxs.tolist())

        best_path = [0] * seq_len
        best_path[-1] = np.argmax(scores)
        # 最优路径回溯
        for i in range(seq_len - 2, -1, -1):
            idx = best_path[i + 1]
            best_path[i] = paths[i][idx]
        res = [self.code2label[i] for i in best_path]
        return res

    def _path2tranData(self,data_path):
        '''
            ./data.txt -> '他/n 是/v 好/ad 人/n !/w' -> [ ['他', '是', '好', '人', '!'],
                                                        ['n', 'v', 'ad', 'n', 'w'] ]
        '''
        with open(data_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        txt = re.split('\s',txt)
        words = []
        labels = []
        for pair in txt:
            if '/' not in pair:
                continue
            pair_lis = pair.split('/')
            words.append(pair_lis[0])
            labels.append(pair_lis[1])
        return [words,labels]

    def _labelDic(self,labels):
        '''
        :param labels: ['n', 'v', 'ad', 'n', 'w']
        :return: {'n':0, 'v':1, 'ad':2, 'w':3},['n', 'v', 'ad', 'w']
        '''
        uniqLabel = list(set(labels))
        code = list(range(len(uniqLabel)))
        dic = dict(zip(uniqLabel,code))
        return dic,uniqLabel
    def _wordDic(self,words):
        '''
        :param word: ['他', '是', '好', '人', '!']
        :return: ['他':0 , '是': 1, '好': 2, '人': 3, '!': 4]
        '''
        uniqWords = list(set(words))
        code = list(range(len(uniqWords)))
        dic = dict(zip(uniqWords,code))
        return dic
    def save(self,path):
        '''
        self.pi = None
        self.A = None
        self.B = None
        self.label2code = None
        self.word2code = None
        self.code2label = None
        :param path:
        :return:
        '''
        weights_dic = {'pi':self.pi.tolist(),'A':self.A.tolist(),'B':self.B.tolist(),'label2code':self.label2code,
                       'word2code':self.word2code,'code2label':self.code2label}
        with open(path,'w',encoding='utf-8') as fo:
            json.dump(weights_dic,fo)
    def load(self,path):
        with open(path,'r',encoding='utf-8') as fo:
            weights = json.load(fo)
        self.pi = np.array(weights['pi'])
        self.A = np.array(weights['A'])
        self.B = np.array(weights['B'])
        self.word2code = weights['word2code']
        self.code2label = weights['code2label']
        self.label2code = weights['label2code']
    def clear(self):
        self.pi = None
        self.A = None
        self.B = None
        self.label2code = None
        self.word2code = None
        self.code2label = None



