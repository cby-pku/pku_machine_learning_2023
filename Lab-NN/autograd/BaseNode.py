from .Init import * 

class Node(object):
    def __init__(self, name, *params):
        # 节点可学习参数的梯度，self.grad[i]为self.params[i]的梯度  
        self.grad = []
        # 节点保存的临时数据，如输入数据、中间结果、输出数据等
        self.cache = []
        # 节点的名字
        self.name = name
        # 节点的可训练参数
        self.params = list(params)

    def flush(self):
        '''
        清空梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X):
        '''
        正向传播。（该类为基类，无需实现具体函数，请在子类中实现。）
        '''
        pass


    def backward(self, grad):
        '''
        反向传播，grad为loss对当前节点的输出值的梯度。（该类为基类，无需实现具体函数，请在子类中实现。）
        '''
        pass
