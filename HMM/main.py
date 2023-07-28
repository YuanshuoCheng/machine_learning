import numpy as np
import re
from HMM import HMM
import jieba
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import ttkthemes
from tkinter.ttk import Label, Button
from PIL import Image
from tkinter import StringVar,ttk
from tkinter.ttk import Label,Entry,Button
from tkinter.filedialog import askopenfilename
from HMM import HMM
import os.path
import warnings
warnings.filterwarnings("ignore")


def get_paired_test_data(path):
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

def get_test_data(path):
    with open(path, 'r', encoding='utf-8') as fo:
        txt = fo.read()
    return txt

def test(model,path):
    x, y = get_paired_test_data(path)
    pred = model.tag(x)
    acc = cal_acc(pred, y)
    res = ['%s/%s'%(x[i],pred[i]) for i in range(len(x))]
    res = ' '.join(res)
    return acc,res
def tag(model,path):
    x = get_test_data(path)
    words = jieba.lcut(x)
    pred = model.tag(words)
    res = ['%s/%s'%(words[i],pred[i]) for i in range(len(words))]
    return ' '.join(res)


class Application:
    def __init__(self):
        # ==== 窗口 ====
        self.TITLE = '基于HMM的科幻小说文本词性标注'
        self.window = None
        self.ws = 640
        self.hs = 480
        self.W = int(self.ws * 0.8)
        self.H = int(self.hs * 0.8)
        self.hmm = HMM()
        self.window = tk.Tk()
        style = ttkthemes.ThemedStyle(self.window)  # 设置需要设置主题的窗口
        style.set_theme("plastik")
        self.window.title(self.TITLE)
        content_style = ttk.Style()
        content_style.configure("A1.TLabel", foreground="red",font=('微软雅黑',12))
        # logo_style = ttk.Style()
        # logo_style.configure("B1.TLabel", foreground="#0D3771")
        x = (self.ws / 2) - (self.W / 2)
        y = (self.hs / 2) - (self.H / 2)
        self.window.geometry('%dx%d+%d+%d' % (self.W, self.H, x, y))
        self.sub_frames = []

        self.init_head_frame()
        # self.logo = ImageTk.PhotoImage(Image.open('./logo.png'))
        # self.label_logo = Label(self.window,style="B1.TLabel",image=self.logo,compound='left',
        #                         width=10,anchor='center')
        # self.label_logo.grid(column=1,row=5,pady=20)

        self.window.mainloop()

    def init_head_frame(self):
        self.top_frame = ttk.Frame(self.window)
        self.mid_frame = ttk.Frame(self.window)
        self.bot_frome = ttk.Frame(self.window)
        self.head = Label(self.top_frame, text="基于HMM的科幻小说文本词性标注", width=30,
                          compound=tk.CENTER,font=('微软雅黑',15))
        self.train_path = StringVar()
        self.test_path = StringVar()
        self.tag_path = StringVar()

        self.train_label = Label(self.mid_frame, text="模型训练：", width=10)
        self.input_train = Entry(self.mid_frame, textvariable=self.train_path, width=40)
        self.btn_choose_train = Button(self.mid_frame, text='选择训练数据', width=15,
                                    command=lambda: self.fun_choose_train())
        self.test_label = Label(self.mid_frame, text="模型测试：", width=10)
        self.input_test = Entry(self.mid_frame, textvariable=self.test_path, width=40)
        self.btn_choose_test = Button(self.mid_frame, text='选择测试数据', width=15,
                                    command=lambda: self.fun_choose_test())
        self.tag_label = Label(self.mid_frame, text="文本标注：", width=10)
        self.input_tag = Entry(self.mid_frame, textvariable=self.tag_path, width=40)
        self.btn_choose_tag = Button(self.mid_frame, text='选择标注数据', width=15,
                                    command=lambda: self.fun_choose_tag())
        self.btn_train = Button(self.mid_frame, text='训练模型', width=10,
                                    command=lambda: self.train())
        self.btn_test = Button(self.mid_frame, text='测试模型', width=10,
                                    command=lambda: self.test())
        self.btn_tag = Button(self.mid_frame, text='文本标注', width=10,
                                    command=lambda: self.tag())
        self.outlabel = Label(self.bot_frome,text='',style='A1.TLabel')
        self.head.grid(column=1,row=0,pady=40)

        self.train_label.grid(column=0,row=1,padx=5,pady=5)
        self.input_train.grid(column=1,row=1,padx=5,pady=5)
        self.btn_choose_train.grid(column=2,row=1,padx =5,pady=5)

        self.test_label.grid(column=0,row=2,padx =5,pady=5)
        self.input_test.grid(column=1,row=2,padx =5,pady=5)
        self.btn_choose_test.grid(column=2,row=2,padx =5,pady=5)

        self.tag_label.grid(column=0,row=3,padx =5,pady=5)
        self.input_tag.grid(column=1,row=3,padx =5,pady=5)
        self.btn_choose_tag.grid(column=2,row=3,padx =5,pady=5)

        self.btn_train.grid(column=0,row=4,pady=30)
        self.btn_test.grid(column=1, row=4,pady=30)
        self.btn_tag.grid(column=2, row=4,pady=30)

        self.outlabel.pack()

        self.top_frame.pack()
        self.mid_frame.pack()
        self.bot_frome.pack(pady=20)

    def fun_choose_train(self):
        self.outlabel['text'] = ''
        path_ = askopenfilename()
        self.train_path.set(path_)

    def fun_choose_test(self):
        self.outlabel['text'] = ''
        path_ = askopenfilename()
        self.test_path.set(path_)
    def fun_choose_tag(self):
        self.outlabel['text'] = ''
        path_ = askopenfilename()
        self.tag_path.set(path_)

    def train(self):
        self.outlabel['text'] = ''
        train_path = self.train_path.get()
        if os.path.isfile(train_path):
            try:
                self.hmm.clear()
                self.hmm.train(train_path)
                out_path = './hmm.param'
                self.hmm.save(out_path)
                self.outlabel['text'] = '训练完成，模型参数输出至：%s！'%out_path
            except:
                self.outlabel['text'] = '训练文件格式错误！'
        else:
            self.outlabel['text'] = '请选择训练数据文件！'
    def test(self):
        self.outlabel['text'] = ''
        if os.path.isfile('./hmm.param'):
            try:
                self.hmm.load('./hmm.param')
                test_path = self.test_path.get()
                acc,res = test(self.hmm,test_path)
                out_path = './test_out.txt'
                with open(out_path,'w',encoding='utf-8') as fo:
                    fo.write(res)
                self.outlabel['text'] = '测试完成，模型准确率为：%.4f，标注结果输出至：%s'%(acc,out_path)
            except:
                self.outlabel['text'] = '测试数据格式错误或无测试数据！'
        else:
            self.outlabel['text'] = '模型参数加载失败，请确认参数文件"./hmm.param"存在！'
    def tag(self):
        self.outlabel['text'] = ''
        tag_path = self.tag_path.get()
        if os.path.isfile('./hmm.param'):
            try:
                self.hmm.load('./hmm.param')
                res = tag(self.hmm,tag_path)
                out_path = './tag_out.txt'
                with open(out_path,'w',encoding='utf-8') as fo:
                    fo.write(res)
                self.outlabel['text'] = '标注完成，标注结果输出至：%s' % out_path
            except:
                self.outlabel['text'] = '待标注数据格式错误或无待标注数据！'
        else:
            self.outlabel['text'] = '模型参数加载失败，请确认参数文件"./hmm.param"存在！'

if __name__ == '__main__':
    app = Application()





