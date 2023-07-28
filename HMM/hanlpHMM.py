from pyhanlp import *
import re
from tests.book.ch07.pku import PKU199801_TRAIN

HMMPOSTagger = JClass('com.hankcs.hanlp.model.hmm.HMMPOSTagger')
AbstractLexicalAnalyzer = JClass('com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer')
PerceptronSegmenter = JClass('com.hankcs.hanlp.model.perceptron.PerceptronSegmenter')
FirstOrderHiddenMarkovModel = JClass('com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModel')
SecondOrderHiddenMarkovModel = JClass('com.hankcs.hanlp.model.hmm.SecondOrderHiddenMarkovModel')

def get_test_data(path):
    with open(path,'r',encoding='utf-8') as fo:
        txt = re.split('\s',fo.read())
        x = []
        y = []
        for i in txt:
            if '/' not in i:
                continue
            pair = i.split('/')
            x.append(pair[0])
            y.append(pair[1])
    return x,y

def cal_acc(pred,targets):
    n = len(pred)
    correct = 0
    for i in range(n):
        if pred[i] == targets[i]:
            correct+=1
    return correct/n

def train_hmm_pos(corpus, model):
    tagger = HMMPOSTagger(model)  # 创建词性标注器
    #print(corpus)
    #tagger.train(corpus)  # 训练
    tagger.train('./train.txt')
    x,y = get_test_data('./test_from_train.txt')
    # print(len(x))
    # 词性标注器不负责分词，所以只接受分词后的单词序列
    pred = tagger.tag(x)
    print(pred)
    print(cal_acc(pred,y))
    # 加上analyzer可以同时执行分词和词性标注
    # analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(), tagger)  # 构造词法分析器
    # print(analyzer.analyze("他的希望是希望上学").translateLabels())  # 分词+词性标注
    return tagger

tagger = train_hmm_pos(PKU199801_TRAIN, FirstOrderHiddenMarkovModel())