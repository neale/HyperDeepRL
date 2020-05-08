import torch
import numpy as np


""" Amortized Stein Variational Gradient Descent Ops """
def batch_rbf_xy(x, y, h_min=1e-3):
    """
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.
    """
    Kx, D = x.shape[-2:]
    Ky, D2 = y.shape[-2:]
    assert D == D2
    leading_shape = x.shape[:-2]
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    # ... x Kx x Ky x D
    dist_sq = torch.sum(diff**2, -1)
    input_shape = (*leading_shape, *[Kx * Ky])
    #values, _ = torch.topk(dist_sq.view(*input_shape), k=(Kx * Ky // 2 + 1))  # ... x floor(Ks*Kd/2)
    #medians_sq = values[..., -1]  # ... (shape) (last element is the median)
    medians_sq, _ = torch.median(dist_sq.view(*input_shape), dim=1)  # ... x floor(Ks*Kd/2)
    h = medians_sq / np.log(Kx)  # ... (shape)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    h = h.detach()  # Just in case.
    h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1)
    # ... x 1 x 1
    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky
    # Construct the gradient
    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    # ... x 1 x 1 x 1
    kappa_expanded = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D
    return kappa, kappa_grad

