import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import numpy as np


BW_BLK_SIZE=8   # -1 for full diag, >1 for block diag


#================================== Batch Norm =========================================
def batch_norm(X, gamma, beta, running_mean, running_var, eps, momentum):
    # Expect X to be either 2D ([N, D]) or 4D ([B, C, H, W])
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
        shape = (1, X.shape[1])      # For fully connected: reshape to (1, D)
    else:
        shape = (1, X.shape[1], 1, 1)  # For convolution: reshape to (1, C, 1, 1)
    
    # When in evaluation mode (gradients disabled), use stored running statistics
    if not torch.is_grad_enabled():
        X_hat = (X - running_mean.view(shape)) / torch.sqrt(running_var.view(shape) + eps)
    else:
        # Compute the mean and variance for the current mini-batch
        if len(X.shape) == 2:
            # For 1D, compute across the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # For 2D, compute mean/var over batch, height and width so that mean/var have shape [C]
            mean = X.mean(dim=(0, 2, 3))
            var = ((X - mean.view(1, -1, 1, 1)) ** 2).mean(dim=(0, 2, 3))
        
        # Normalize using the computed batch statistics (reshaped for broadcast)
        X_hat = (X - mean.view(shape)) / torch.sqrt(var.view(shape) + eps)
        # Update running statistics in place to preserve their original shape ([D] or [C])
        running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean)
        running_var.copy_((1.0 - momentum) * running_var + momentum * var)
    
    # Scale and shift
    Y = gamma.view(shape) * X_hat + beta.view(shape)
    return Y, running_mean, running_var

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # gamma and beta are learnable parameters, stored as shape [num_features]
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # Running mean and variance buffers initialized with the same shape
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, X):
        # Ensure running_mean and running_var are on the same device as X
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_var = self.running_var.to(X.device)
        
        # Update batchnorm statistics and normalize the input
        Y, self.running_mean, self.running_var = batch_norm(
            X, self.gamma, self.beta, self.running_mean,
            self.running_var, self.eps, self.momentum)
        return Y

#=========================================================================



def comp_avg_corr(x):
    with torch.no_grad():
        # flatten x from [B,C,H,W] to [C,B*H*W]
        x_f= x.permute(1,0,2,3).reshape(x.shape[1],-1).contiguous().detach()
        # compute corr matrix
        corr_matrix = torch.corrcoef(x_f)
        # Extract upper triangular part (excluding diagonal)
        upper_tri = torch.triu(corr_matrix, diagonal=1)
        # Compute average of cross-correlation coefficients
        avg_corr = upper_tri.abs().sum() / ((upper_tri.numel() - upper_tri.diag().numel())/2)
    return avg_corr

def comp_cov_cond(x):
    # flatten x from [B,C,H,W] to [C,B*H*W]
    with torch.no_grad():
        x_f= x.permute(1,0,2,3).reshape(x.shape[1],-1).contiguous().detach()
        cov_cond=torch.linalg.cond(torch.cov(x_f))
    return cov_cond




def get_rank(x):
    x_f= x.permute(1,0,2,3).reshape(x.shape[1],-1).contiguous().detach()
    return torch.linalg.matrix_rank(x_f)/x.shape[1]


def cov_to_corr(cov_matrix):
    # Compute the standard deviations
    std = torch.sqrt(torch.diag(cov_matrix))
    
    # Compute the correlation matrix
    corr_matrix = cov_matrix / torch.outer(std, std)
    return corr_matrix


def corr_to_cov(corr_matrix,std):
    # Compute the standard deviations
    D = torch.diag(std)
    # D=torch.outer(std,std)
    # cov=corr_matrix * torch.outer(std,std)
    return D@corr_matrix@D

def fix_corr(corr):
    a=0.9+0.1*torch.exp(-(abs(corr)/0.9)**10)
    a=a.clone()  # so not to lose the gradients in backprop
    torch.diagonal(a).fill_(1.0)
    return a*corr


# def fix_cov(covmat):
#     a=torch.ones_like(covmat)*0.9
#     a.fill_diagonal_(1.0)
#     # a=a.clone()  # so not to lose the gradients in backprop
#     # torch.diagonal(a).fill_(1.0)
#     return a*covmat

def fix_cov(covmat,fix_factor=0.9):
    # Create a tensor of fix_factor with the same shape as covmat
    a = torch.ones_like(covmat) * fix_factor
    
    # Handle both 2D and 3D cases for setting diagonal to 1.0
    if covmat.dim() == 2:
        # For 2D case [D,D]
        a.fill_diagonal_(1.0)
    else:
        # For 3D case [B,D,D]
        # Set diagonal to 1.0 for each matrix in the batch
        eye = torch.eye(covmat.size(-1), device=covmat.device)
        a = a * (1 - eye) + eye
    
    return a * covmat

# num_channels is the number of channels in each group (bw block size)
def get_grp_ch(num_features,num_groups=1,num_channels=-1):
    if num_channels == -1:
        num_channels = (num_features - 1) // num_groups + 1
    num_groups = num_features // num_channels
    while num_features % num_channels != 0:
        # num_channels //= 2
        num_channels -= 1
        num_groups = num_features // num_channels
    assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
        num_groups)
    return num_groups,num_channels


#=================================Cholesky=================================

#---------full diagonal-------------
def cholesky_batch_full_diag(X, running_mean=None, running_cov=None, eps=1e-5, momentum=0.1,cov_warmup=False):
    # Use is_grad_enabled to determine whether we are in training mode
    assert len(X.shape) in (2, 4)
    n_features = X.shape[1]
    cov_I = torch.eye(n_features).to(running_cov.device)     
    if torch.is_grad_enabled():
        # Training mode: compute batch statistics and update running values.
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            cov = torch.cov(X.T, correction=0)
        else:
            mean = X.mean(dim=(0, 2, 3))
            Xtmp = X.view(X.shape[0], X.shape[1], -1)
            Xtmp = Xtmp.permute(1, 0, 2).reshape(X.shape[1], -1)
            cov = torch.cov(Xtmp, correction=0) + eps * cov_I
        with torch.no_grad():
            running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean)
            if cov_warmup:
                x_var = torch.diag_embed(torch.diag(cov))
                running_cov.copy_((1.0 - momentum) * x_var + momentum * cov)
            else:
                running_cov.copy_((1.0 - momentum) * running_cov + momentum * cov)
        norm_mean = mean  # Or you can use running_mean, depending on your design.
    else:
        # Evaluation mode: use stored running statistics directly.
        norm_mean = running_mean

    # Compute the Cholesky factor from the running covariance.
    L = torch.linalg.cholesky(running_cov)
    
    if len(X.shape) == 2:
        X_hat = (X - norm_mean.view(1, n_features)).T
        Y = torch.linalg.solve_triangular(L, X_hat, upper=False).T
    else:
        X_hat = X - norm_mean.view(1, n_features, 1, 1)
        X_hat = X_hat.permute(1, 0, 2, 3).reshape(n_features, -1)
        Y = X
        Y = torch.linalg.solve_triangular(L, X_hat, upper=False)
        Y = Y.reshape(n_features, X.shape[0], X.shape[2], X.shape[3]).permute(1, 0, 2, 3)
    return Y, running_mean.detach(), running_cov.detach()

class BWCholeskyBlock_full_diag(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,momentum=0.1,eps=1e-5,pre_bias_block=None,num_bias_features=None):
        super().__init__()
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively 
        self.n_features=num_features
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.momentum = momentum
        self.eps = eps
        self.cov_warmup=False
        # self.gamma = nn.Parameter(torch.ones(num_features))
        # The variables that are not model parameters are initialized to 0 and 1
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_cov', torch.eye(num_features))
        # self.register_buffer('running_cov', torch.ones(num_features))   # debug - pseudo cholesky
        self.pre_bias_block=pre_bias_block

        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
        # Save the updated running_mean and moving_var
        Y, self.running_mean, self.running_cov = cholesky_batch_full_diag(
            X, self.running_mean, self.running_cov, eps=self.eps, momentum=self.momentum,cov_warmup=self.cov_warmup)
        if self.pre_bias_block is not None:
            Y=self.pre_bias_block(Y)
        # add the bias
        shape = (1,self.n_bias_features) if len(X.shape)==2 else (1,self.n_bias_features,1,1)
        Y += self.beta.view(shape)
        return Y

#---------block diagonal-------------
def cholesky_batch_block_diag(X, running_mean=None, running_cov=None, n_channels=-1, eps=1e-5, momentum=0.1,cov_warmup=False,fix_factor=0.9):
    # Use is_grad_enabled to determine whether we are in training mode
    assert len(X.shape) in (2, 4)
    n_features = X.shape[1]
    if n_channels==-1:
        n_channels=n_features
    n_groups=n_features//n_channels
    x = X.transpose(0, 1).contiguous().view(n_groups, n_channels, -1)    
    cov_I = torch.eye(n_channels).expand(n_groups, n_channels, n_channels).clone().to(running_cov.device)     
    _, d, m = x.size()
    if torch.is_grad_enabled():
        mean = x.mean(-1, keepdim=True)
        xc = x - mean
        cov = torch.baddbmm(cov_I, xc, xc.transpose(1, 2), beta=eps, alpha=1. / m)

        # In training mode, the current mean and variance are used
        # Update the mean and variance using moving average
        with torch.no_grad():
            running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean)
            # running_mean = mean         # no running mean (alpha = momentum = 1)
            
            # during warmup, we're not updating running_cov but using a statistics of current batch 
            if cov_warmup:
                # x_var = torch.diag_embed(torch.diag(cov))
                x_var = torch.eye(d).unsqueeze(0).to(cov)*cov
                running_cov.copy_((1.0 - momentum) * x_var + momentum * cov)
            # when warm up is done, running_cov is updated only during training
            # elif torch.is_grad_enabled():    # debug : temporarily disabling this check. 
            else:       # debug: temporary allow running_cov to be updated during both train and validation
                running_cov.copy_((1.0 - momentum) * running_cov + momentum * cov)
            # fix the cov matrix
            running_cov=fix_cov(running_cov,fix_factor)
    else:
        # Evaluation mode: use stored running statistics directly.
        xc = x - running_mean
    
    # note that we're using running_cov also during training
    L = torch.linalg.cholesky(running_cov)
    if len(X.shape) == 2:
        Y = torch.linalg.solve_triangular(L,xc,upper=False).reshape(X.shape[1],X.shape[0]).permute(1,0)
    else:
        Y = torch.linalg.solve_triangular(L,xc,upper=False).reshape(X.shape[1],X.shape[0],X.shape[2],X.shape[3]).permute(1,0,2,3)
    Y = Y.contiguous()
    return Y, running_mean.detach(), running_cov.detach()


class BWCholeskyBlock(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,num_groups=1, num_channels=BW_BLK_SIZE,momentum=0.1,eps=1e-5,pre_bias_block=None,num_bias_features=None,fix_factor=0.9):
        super().__init__()
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively 
        self.n_features=num_features
        self.num_groups,self.num_channels=get_grp_ch(num_features,num_groups,num_channels)
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.momentum = momentum
        self.eps = eps
        self.cov_warmup=False
        # self.gamma = nn.Parameter(torch.ones(num_features))
        # The variables that are not model parameters are initialized to 0 and 1
        self.register_buffer('running_mean', torch.zeros(self.num_groups, self.num_channels, 1))
        self.register_buffer('running_cov', torch.eye(self.num_channels).expand(self.num_groups, self.num_channels, self.num_channels).clone())
        self.pre_bias_block=pre_bias_block
        self.fix_factor=fix_factor
        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
        # Save the updated running_mean and moving_var
        Y, self.running_mean, self.running_cov = cholesky_batch_block_diag(
            X, self.running_mean, self.running_cov, self.num_channels,eps=self.eps, momentum=self.momentum,cov_warmup=self.cov_warmup,fix_factor=self.fix_factor)
        if self.pre_bias_block is not None:
            Y=self.pre_bias_block(Y)
        # add the bias
        shape = (1,self.n_bias_features) if len(X.shape)==2 else (1,self.n_bias_features,1,1)
        Y += self.beta.view(shape)
        return Y

#=============================Iter Norm=====================================

def iter_norm_batch(X, running_mean=None, running_wm=None, T=10, eps=1e-5, momentum=0.1,n_channels=-1,apply_fix_cov=False):
    # Use is_grad_enabled to determine whether we are in training mode
    if n_channels==-1:
        n_channels=X.size(1)
    g = X.size(1) // n_channels
    x = X.transpose(0, 1).contiguous().view(g, n_channels, -1)
    _, d, m = x.size()
    if torch.is_grad_enabled():
        # calculate centered activation by subtracted mini-batch mean
        mean = x.mean(-1, keepdim=True)
        xc = x - mean
        # calculate covariance matrix
        P = [None] * (T + 1)
        P[0] = torch.eye(d).to(X).expand(g, d, d).clone()
        # Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
        Sigma = torch.baddbmm(P[0], xc, xc.transpose(1, 2), beta=eps, alpha=1. / m)  # =torch.cov(xc,correction=0)
        if apply_fix_cov:
            # corr=cov_to_corr(Sigma[0])
            # xc_std=xc[0].std(axis=-1)
            # corr=fix_corr(corr)
            # Sigma[0]=corr_to_cov(corr,xc_std)
            Sigma[0]=fix_cov(Sigma[0])

        # reciprocal of trace of Sigma: shape [g, 1, 1]
        rTr = (Sigma * P[0]).sum(1, keepdim=True).sum(2, keepdim=True).reciprocal_()
        Sigma_N = Sigma * rTr
        for k in range(T):
            # P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, P[k].bmm(P[k]).bmm(P[k]), Sigma_N)
            P[k + 1] = torch.baddbmm(P[k], P[k].bmm(P[k]).bmm(P[k]), Sigma_N, beta=1.5, alpha=-0.5)
        wm = P[T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
        # self.running_mean += momentum * ( mean.detach() - self.running_mean)
        # self.running_wm += momentum * ( wm.detach() - self.running_wm)
        running_mean = (1-momentum)*running_mean + momentum * mean.detach()
        running_wm = (1-momentum)*running_wm + momentum * wm.detach() 
    else:
        xc = x - running_mean
        wm = running_wm
    xn = wm.matmul(xc)
    X_hat = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
    return X_hat, running_mean.data, running_wm.data

class IterNormMod(nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=-1, T=10, dim=4, eps=1e-5, momentum=0.1, affine=True, *args, **kwargs):
        super(IterNormMod, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.fix_cov=kwargs.get('fix_cov',False)

        if num_channels == -1:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*shape))
            self.bias = nn.Parameter(torch.Tensor(*shape))

        if not self.affine:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels).clone())
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, X: torch.Tensor):
        eps = 1e-5
        momentum = self.momentum
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_wm = self.running_wm.to(X.device)

        X_hat,self.running_mean,self.running_wm = iter_norm_batch(X,
                                                                  self.running_mean,
                                                                  self.running_wm,
                                                                  self.T,
                                                                  momentum=self.momentum,
                                                                  n_channels=self.num_channels,
                                                                  apply_fix_cov=self.fix_cov)

        # affine
        if self.affine:
            X_hat = X_hat * self.weight
            X_hat = X_hat + self.bias
        return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, dim={dim}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)

class BWItnBlock(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,T=10,momentum=0.1,eps=1e-5,pre_bias_block=None,num_bias_features=None):
        super().__init__()
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively 
        self.T=T
        self.n_features=num_features
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.momentum = momentum
        self.eps = eps

        # note: currently assuming no grouping of features so we have only one group
        num_groups=1
        self.num_channels = num_features
        # The variables that are not model parameters are initialized to 0 and 1
        self.register_buffer('running_mean', torch.zeros(num_groups, self.num_channels, 1))
        self.register_buffer('running_cov', torch.eye(self.num_channels).expand(num_groups, self.num_channels, self.num_channels).clone())
        self.pre_bias_block=pre_bias_block

        # self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
        # Save the updated running_mean and moving_var
        X_hat,self.running_mean,self.running_cov = iter_norm_batch(X,
                                                                  self.running_mean,
                                                                  self.running_cov,
                                                                  self.T,
                                                                  momentum=self.momentum,
                                                                  n_channels=self.num_channels,
                                                                  apply_fix_cov=False)
        if self.pre_bias_block is not None:
            X_hat=self.pre_bias_block(X_hat)
        # add the bias
        shape = (1,self.n_bias_features) if len(X.shape)==2 else (1,self.n_bias_features,1,1)
        X_hat += self.beta.view(shape)
        return X_hat



#==============================Select Whitening===============================
def should_use_batch_whitening(N, H, W, C, momentum=0.99, threshold=5000):
    """Determine if BatchWhitening should be used based on number of samples
    
    Args:
        N (int): Number of samples
        H (int): Height of the image
        W (int): Width of the image
        C (int): Number of channels
        threshold (int): Threshold for using BatchWhitening

    The threshold is chosen based on the fact that we need enough samples for stable computation of the covariance matrix.
    the error in computing the covariance matrix of vector with C elements from V=N*H*W samples is ~ sqrt(C*C/(2*V))
    so for error of ~0.01 we need V > C*C/2/0.01^2 = 5000*C*C
    """
    print('-----------BW triage------------ \n')
    n_samples = N * H * W / (1-momentum)
    n_ch=C
    print(f'N,H,W,C,mu={N,H,W,n_ch,momentum} --> n_samples = {n_samples}, th={threshold*n_ch*n_ch}')
    return n_samples >= threshold*n_ch*n_ch




def get_batch_whitening_config(N, H, W, C, momentum=0.99, threshold=0.01):
    """Set Batch Whitening configuration based on number of samples
    
    Args:
        N (int): Number of samples
        H (int): Height of the image
        W (int): Width of the image
        C (int): Number of channels
        threshold (float): Threshold for using BatchWhitening

    The mechanism:
    the effective number of samples is N*H*W/(1-momentum)
    this number should be >= blk_size*blk_size/2*threshold^2

    we can control blk_size. so we want to find what is the blk_size that satisfies the condition above.
    so we solve for blk_size in the equation:
    N*H*W/(1-momentum) >= blk_size*blk_size/2*threshold^2
    blk_size >= sqrt(2*N*H*W/(1-momentum)*threshold^2)
    now, if blk_size < 2 , we set blk_size to 1, which means using batchnorm.
    if blk_size > C we clip it to C which means using the whole channel as a group.
    in between, we use blk_size as nearest power of 2 to the blk_size.
    """
    print('-----------BW triage------------ \n')
    n_samples = N * H * W / (1-momentum)

    blk_size = int(np.sqrt(2*n_samples*threshold*threshold))
    print(f'raw block size: {blk_size}')
    new_mom=momentum
    if blk_size < 2:
        blk_size = 1
    elif blk_size > C:
        blk_size = C
        # new_mom=max(0,1-(2*threshold*threshold*N*H*W)/(C*C))
    else:
        blk_size = 2**int(np.log2(blk_size))
    print(f'N,H,W,C,mu={N,H,W,C,momentum} --> blk_size = {blk_size}, momentum={new_mom}')
    return blk_size,new_mom



# BatchWhiteningBlock=BWItnBlock
# BatchWhiteningBlock=BWCholeskyBlock
# BatchWhiteningBlock=BWCholeskyBlock_full_diag
if BW_BLK_SIZE==-1:
    BatchWhiteningBlock=BWCholeskyBlock_full_diag
else:
    BatchWhiteningBlock=BWCholeskyBlock
BatchNorm=nn.BatchNorm2d
# BatchNorm=MyBatchNorm2d

