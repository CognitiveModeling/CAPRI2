"""
Implementation of a single-layered Mixture Density Network
producing a Gaussian distribution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.autograd import Variable
import math
import torch.optim as optim


def weights_init(m):
    """
    Initialize weights normal distributed with sd = 0.01
    :param m: weight matrix
    :return: normal distributed weights
    """
    m.weight.data.normal_(0.0, 0.01)


class Multivariate_Gaussian_Network(nn.Module):
    min_p = 1e-11                 
    
    def __init__(self, input_dim, output_dim):
        """
        Initialization
        :param input_dim: dimensionality of input
        :param output_dim: dimensionality of output
        """
        super(Multivariate_Gaussian_Network, self).__init__()
        self.fcMu = nn.Linear(input_dim, output_dim)
        weights_init(self.fcMu)
        self.fcSigma = nn.Linear(input_dim, output_dim)
        weights_init(self.fcSigma)
        self.gauss_loss_func = torch.nn.GaussianNLLLoss()                                                        

    def forward(self, x):
        """
        Forward pass of input
        :param x: input
        :return: mu, Sigma of resulting output distribution
        """
        mu = self.fcMu(x)
        # Sigma determined with ELUs + 1 + p to ensure values > 0
        # small p > 0 avoids that Sigma == 0
        sigma = F.elu(self.fcSigma(x)) + 1 + self.min_p
        return mu, sigma

    def get_optimizer(self, learning_rate, momentum_term=0.0, type='SGD'):
        """
        :param learning_rate: learning rate of optimizer
        :param momentum_term: momentum term used of optimizer
        :param type: which optimizer to use, 'SGD' or 'Adam'                                         
        :return: optimizer of the network
        """
        if type == 'Adam':
            return optim.Adam(self.parameters(), lr=learning_rate, eps=1e-04)
        return optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum_term)
    
    def loss_criterion(self, output, label, tanh = False):
        """
        Loss function, i.e., negative log likelihood
        :param output: output (mu, Sigma) of the network
        :param label: nominal output
        :param tanh: process likelihood by tanh                                               
        :return: negative log likelihood of nominal label under output distribution
        """
        if tanh:
            return self._loss_criterion_tanh(output, label)
        return self.gauss_loss_func(input=output[0].unsqueeze(0), target=label.unsqueeze(0), var=output[1].unsqueeze(0))

    def batch_loss_criterion(self, output, label, tanh=False):
        """
        Loss function, i.e., negative log likelihood  for batched outputs
        :param output: output (mu, Sigma) of the network , each is a batch
        :param label: batch of target output
        :param tanh: process likelihood by tanh
        :return: negative log likelihood of nominal label under output distribution
        """
        if tanh:
            return self._batch_loss_criterion_tanh(output, label)
        return self.gauss_loss_func(input=output[0], target=label, var=output[1])

    def _batch_loss_criterion_tanh(self, output, label):
        """
        Loss function applied for batched outputs squeezed by tanh
        :param output: output (mu, Sigma) of the network, each is a batch
        :param label: batch of target outputs
        :return: negative log likelihood of nominal label under output distribution
        """
        mu = output[0]
        sigma = torch.diag_embed(output[1], offset=0, dim1=-2, dim2=-1)
        distr = torch.distributions.MultivariateNormal(mu, sigma)
        return torch.mean(torch.tanh(-1 * distr.log_prob(label) * (1.0 / 100)) * 100)

    def _loss_criterion_tanh(self, output, label):
        """
        Loss function, i.e., negative log likelihood squeezed by tanh
        :param output: output (mu, Sigma) of the network
        :param label: nominal output
        :return: negative log likelihood of nominal label under output distribution
        """
        mu = output[0]                      
        sigma = torch.diag(output[1])
        distr = torch.distributions.MultivariateNormal(mu, sigma)
        # negative log likelihood is squashed by tanh * 100 to avoid loss > 100
        # multiplied by constant factor c = 100
        return torch.tanh(-1 * distr.log_prob(label)*(1.0/100)) * 100
