import torch
import numpy as np


""" Amortized Stein Variational Gradient Descent Ops """
def batch_rbf_old(x, y, h_min=1e-3):
    """
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D)
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D)
        h_min(`float`): Minimum bandwidth.
    """
    Kx, D = x.shape[-2:]
    Ky, D2 = y.shape[-2:]
    assert D == D2
    leading_shape = x.shape[:-2]
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    dist_sq = torch.sum(diff**2, -1)
    input_shape = (*leading_shape, *[Kx * Ky])
    values, _ = torch.topk(dist_sq.view(*input_shape), k=(Kx * Ky // 2 + 1))  # ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)
    h = medians_sq / np.log(Kx)  # ... (shape)
    h = torch.max(h, torch.tensor([h_min]).cuda()).detach()
    h = h.unsqueeze(-1).unsqueeze(-1)
    kappa = torch.exp(-dist_sq / h)  # ... x Kx x Ky
    h = h.unsqueeze(-1)
    kappa = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1
    kappa_grad = -2 * diff / h * kappa
    return kappa, kappa_grad


def batch_rbf(x, y, h_min=1e-3):
    """
        x (tensor): A tensor of shape (Nx, B, D) containing Nx particles
        y (tensor): A tensor of shape (Ny, B, D) containing Ny particles
        h_min(`float`): Minimum bandwidth.
    """
    Nx, Bx, Dx = x.shape 
    Ny, By, Dy = y.shape
    assert Bx == By
    assert Dx == Dy

    diff = x.unsqueeze(1) - y.unsqueeze(0) # Nx x Ny x B x D
    dist_sq = torch.sum(diff**2, -1).mean(dim=-1) # Nx x Ny
    values, _ = torch.topk(dist_sq.view(-1), k=dist_sq.nelement()//2+1)
    median_sq = values[-1]
    h = median_sq / np.log(Nx)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    # Nx x Ny
    kappa = torch.exp(-dist_sq / h)
    # Nx x Ny x B x D
    kappa_grad = torch.einsum('ij,ijkl->ijkl', kappa, -2 * diff / h)
    return kappa, kappa_grad


def score_func(x, h_min=1e-3):
    N, D = x.shape
    z_x = torch.rand_like(x) * 1e-10
    z_x += x
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    dist_sq = torch.sum(diff**2, -1) # N x N
    values, _ = torch.topk(dist_sq.view(-1), k=dist_sq.nelement()//2+1)
    median_sq = values[-1]
    h = median_sq / np.log(N)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    kappa = torch.exp(-dist_sq/h)
    I = torch.eye(N).cuda()
    kappa_inv = torch.inverse(kappa + 1e-3 * I)
    kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2*diff/h)

    return kappa_inv @ kappa_grad


def approx_jacobian_trace(fx, x):
    """Hutchinson's trace Jacobian estimator O(1) call to autograd,
        used by "\"minmax\" method"""
    eps = torch.randn_like(fx)
    jvp = torch.autograd.grad(
            fx,
            x,
            grad_outputs=eps,
            retain_graph=True,
            create_graph=True)[0]
    if eps.shape[-1] == jvp.shape[-1]:
        tr_jvp = torch.einsum('bi,bi->b', jvp, eps)
    else:
        tr_jvp = torch.einsum('bi,bj->b', jvp, eps)
    return tr_jvp
