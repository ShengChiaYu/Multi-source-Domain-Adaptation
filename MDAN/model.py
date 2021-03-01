import os
import csv
from PIL import Image
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

    
class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

# Model
class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        #取掉model的最後1層
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.resnet_layer(x)
        x = torch.squeeze(torch.squeeze(x,dim=2),dim=2)       
        return x
    
class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self,feature_ex,num_classes=345,num_domains=3):
        super(MDANet, self).__init__()
        self.feature_dim = 2048
        self.num_domains = num_domains
        
        self.num_classes=num_classes
        # Parameters of hidden, resnet, feature learning component.
        self.hiddens = feature_ex
        
        
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.feature_dim, self.num_classes)
        
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([nn.Linear(self.feature_dim, 2) for _ in range(self.num_domains)])
        
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
        
        
    def forward(self, sinputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains):
            sh_relu[i] = self.hiddens(sh_relu[i])            
        th_relu = F.relu(self.hiddens(th_relu))
        
        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.softmax(sh_relu[i]), dim=1))
            
        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
        return logprobs, sdomains, tdomains

    def inference(self, inputs):
        h_relu = inputs
        
        h_relu = F.relu(self.hiddens(h_relu))
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs
