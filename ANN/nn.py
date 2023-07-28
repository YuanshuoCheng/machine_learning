import numpy as np


class Module:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass


class Linear(Module):
    def __init__(self,in_planes,out_planes):
        super(Linear, self).__init__()
        self.w = np.random.randn(in_planes+1,out_planes)
        self.grad = None
    def __call__(self,x):
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        self.grad = x
        res = np.dot(x,self.w)
        return res


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.grad = None

    def __call__(self, pred, target,eps=1e-8):
        exps = np.exp(pred-np.max(pred,axis=1).reshape(pred.shape[0],1))
        logits = exps / (np.sum(exps,axis=1).reshape(exps.shape[0],1)+eps)
        loss = -np.sum(target*np.log(logits+eps),axis=1)
        self.grad = logits-target
        return loss.mean()

    def backward(self):
        return self.grad


class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.grad = None
    def __call__(self,x):
        x[x<0] = 0.0
        self.grad = np.ones_like(x)
        self.grad[x<=0] = 0.0
        return x


class SGD:
    def __init__(self,model,lr):
        self.lr = lr
        self.models=[]
        for att in model.__dict__:
            t = model.__dict__[att]
            if t.__class__.__base__ == Module:
                self.models.append(t)
    def backward(self,loss_fn):
        bs = self.models[0].grad.shape[0]
        i = np.random.randint(bs)
        last_grad = loss_fn.grad[i] # 大小为该层神经元的个数，即该层的输出维度
        for model in self.models[::-1]:
            if 'w' in model.__dict__:
                model.grad = model.grad[i].reshape((model.grad[i].shape[0],1))*last_grad
                last_grad = np.sum((last_grad*model.w[:-1]),axis=1)
            else:
                last_grad = last_grad*model.grad[i]

    def step(self):
        for model in self.models[::-1]:
            if 'w' in model.__dict__:
                model.w = model.w-model.grad*self.lr

class MSELoss:
    def __init__(self):
        self.grad = None
    def __call__(self, x, y):
        dx = 2 * (x - y) / x.shape[1]
        self.grad = dx
        return np.sum(np.square(x - y)) / x.shape[0]


# class SoftMax(Module):
#     def __init__(self):
#         super(SoftMax, self).__init__()
#         self.grad = None
#     def __call__(self,x):
#         exps = np.exp(x)
#         res = exps / np.sum(exps)
#         self.grad = res/(1-res)
#         return res