import torch
from torch import nn
from torch.nn import functional as F


def comp_avg_corr(x):
    with torch.no_grad():
        # flatten x from [B,C,H,W] to [C,B*H*W]
        x_f= x.permute(1,0,2,3).reshape(x.shape[1],-1).detach()
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
        x_f= x.permute(1,0,2,3).reshape(x.shape[1],-1).detach()
        cov_cond=torch.linalg.cond(torch.cov(x_f))
    return cov_cond




def get_rank(x):
    x_f= x.permute(1,0,2,3).reshape(x.shape[1],-1).detach()
    return torch.linalg.matrix_rank(x_f)/x.shape[1]

def fix_cov(c):
    a=0.9+0.1*torch.exp(-(c/0.9)**10)
    # torch.diagonal(a).fill_(1.0)
    return a*c

def batch_orthonorm(X, running_mean=None, running_cov=None, eps=1e-5, momentum=0.1,cov_warmup=False):
    # Use is_grad_enabled to determine whether we are in training mode
    assert len(X.shape) in (2, 4)
    n_features = X.shape[1]

    if len(X.shape) == 2:
        # When using a fully connected layer, calculate the mean and
        # variance on the feature dimension
        # shape = (1, n_features)
        mean = X.mean(dim=0)
        cov = torch.cov(X.T,correction=0)        
        # var = ((X - mean) ** 2).mean(dim=0)
    else:
        # When using a two-dimensional convolutional layer, calculate the
        # mean and covariance on the channel dimension (axis=1). Here we
        # need to maintain the shape of X, so that the broadcasting
        # operation can be carried out later
        # shape = (1, n_features, 1, 1)
        mean = X.mean(dim=(0, 2, 3))
        Xtmp = X.view(X.shape[0],X.shape[1],-1)
        Xtmp = Xtmp.permute(1,0,2).reshape(X.shape[1],-1)
        cov = torch.cov(Xtmp,correction=0) 
        # var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    # In training mode, the current mean and variance are used
    # Update the mean and variance using moving average
    # running_mean = (1.0 - momentum) * running_mean + momentum * mean
    running_mean = mean         # no running mean (alpha = momentum = 1)
    cov_I = torch.eye(n_features).to(running_cov.device)     
    # during warmup, we're not updating running_cov but using a statistics of current batch 
    if cov_warmup:
        x_var = torch.diag_embed(torch.diag(cov))
        running_cov = (1.0 - momentum) * x_var + momentum * cov
    # when warm up is done, running_cov is updated only during training
    elif torch.is_grad_enabled():    
        running_cov = (1.0 - momentum) * running_cov + momentum * cov
    L = torch.linalg.cholesky(running_cov + eps*cov_I)
    if len(X.shape) == 2:
        X_hat = (X-running_mean.view(1,n_features)).T
        Y = torch.linalg.solve_triangular(L,X_hat,upper=False).T
    else:
        X_hat = X-running_mean.view(1,n_features,1,1)
        X_hat = X_hat.permute(1,0,2,3).reshape(X.shape[1],-1)
        Y = torch.linalg.solve_triangular(L,X_hat,upper=False).reshape(X.shape[1],X.shape[0],X.shape[2],X.shape[3]).permute(1,0,2,3)
    return Y, running_mean.data, running_cov.data



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
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
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
        P[0] = torch.eye(d).to(X).expand(g, d, d)
        # Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
        Sigma = torch.baddbmm(P[0], xc, xc.transpose(1, 2), beta=eps, alpha=1. / m)  # =torch.cov(xc,correction=0)
        if apply_fix_cov:
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
        xc = x - running_mean.view(1,n_channels,1)
        wm = running_wm
    xn = wm.matmul(xc)
    X_hat = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
    return X_hat, running_mean.data, running_wm.data
