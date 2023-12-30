from .BaseNode import Node
from typing import List
import numpy as np

class Graph(object):
    '''
    计算图类
    '''
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes

    def flush(self):        
        for node in self.nodes:
            node.flush()

    def forward(self, X):
        """
        正向传播
        @param X: n*d 输入样本
        @return: 最后一层的输出
        """
        for node in self.nodes:
            X = node.forward(X)
        return X

    def backward(self, grad=None):
        """
        反向传播
        @param grad: 前向传播所输出的output对应的梯度
        @return: 无返回值
        """
        # TODO: Please implement the backward function for the computational graph, which can back propagate the gradients
        # from the loss node to the head node.
        if grad is None:
            # FIXME 2
            grad = np.ones_like(self.nodes[-1].cache[0])
            raise ValueError("Gradient should be provided for backward propagation")
        for node in reversed(self.nodes):
            grad = node.backward(grad)

        
    
    def optimstep(self, lr):
        """
        利用计算好的梯度对参数进行更新
        @param lr: 超参数，学习率
        @return: 无返回值
        """  
        # TODO: Please implement the optimstep function for a computational graph, 
        # which can update the parameters of each node based on their gradients.
        for node in self.nodes:
            #node.optimstep(lr)
            if len(node.params) > 0:
                for i in range(len(node.params)):
                    node.params[i] -= lr * node.grad[i]
            #    for param, grad in zip(node.params,node.grad):
            #        node.param -= lr*grad
        
        
