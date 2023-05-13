import numpy as np
import torch
from opt_einsum import contract
from time import time

from ..util_old import helpers

def iprd(X, Y):
    """Inner product between two tensors."""
    return torch.inner(X.flatten(), Y.flatten())


def coupled_nmf_model_torch(
        X, Y, rank, alpha,
        min_iters=3, max_iters=200, tol=1e-5, inner_iters=3,
        verbose=False, outfileprefix=None
    ):
    """
    X : matrix data
        (neurons x neuron_time_bins)

    Y : tensor data
        (facepoints x frequency bins x face_time_bins)

    rank : int
        Number of components.

    alpha : float
        Number between zero and one. If zero, fits only the
        face data. If one, fits only the neural data. Intermediate
        values interpolate.
    """

    # Determine float32 vs float64.
    assert X.dtype == Y.dtype
    dtype = X.dtype

    # Check the neural and face data are on the same device.
    assert X.device == Y.device
    device = X.device

    # Check alpha
    assert alpha >= 0
    assert alpha <= 1

    # Data dimensions.
    nn, ntx = X.shape
    nfp, nfq, nty = Y.shape
    assert ntx == nty
    nt = ntx
    _idx = np.arange(rank)

    # This is the only case we care about for now.
    assert ntx >= nty

    # Neuron factors
    W = torch.rand(rank, nn, dtype=dtype).to(device)

    # Temporal factors
    H = torch.rand(rank, nt, dtype=dtype).to(device)

    # Face factors
    U = torch.rand(rank, nfp, dtype=dtype).to(device)

    # Frequency factors
    V = torch.rand(rank, nfq, dtype=dtype).to(device)

    # Re-scale neural and temporal factors to match neural data norm
    Xnrm = torch.norm(X)
    dn = torch.sqrt(iprd(W @ W.T, H @ H.T))
    W *= torch.sqrt(Xnrm / dn)
    H *= torch.sqrt(Xnrm / dn)

    # Cached matrix products.
    HHt = H @ H.T
    WWt = W @ W.T
    XHt = X @ H.T
    WX = W @ X

    # Compute loss on matrix data
    neural_losses = [
        ((
            (Xnrm ** 2)
            + iprd(WWt, HHt)
            - 2 * iprd(W.T, XHt)
        ) / (Xnrm ** 2)).item()
    ]

    # Re-scale frequency and face factors to match face tensor norm
    Ynrm = torch.norm(Y)
    dn = torch.sqrt(torch.sum(HHt * (U @ U.T) * (V @ V.T)))
    U *= torch.sqrt(Ynrm / dn)
    V *= torch.sqrt(Ynrm / dn)

    # Cached matrix products.
    UUt = U @ U.T
    VVt = V @ V.T
    #Y_VHt = contract("ijk,rj,rk->ir", Y, V, H)
    Y_VHt = contract("ijk,rjk->ir", Y, V[:, :, None] * H[:, None, :])
    G = VVt * HHt

    # Compute loss on face tensor.
    face_losses = [
        ((
            (Ynrm ** 2)
            + torch.sum(UUt * VVt * HHt)
            - 2 * iprd(U.T, Y_VHt)
        ) / (Ynrm ** 2)).item()
    ]
    total_losses = [
        alpha * neural_losses[-1] +
        (1 - alpha) * face_losses[-1]
    ]
    if verbose:
        print(
            f"Total Loss (0): {total_losses[-1]} \t"
            f"Neural Loss (0): {neural_losses[-1]} \t"
            f"Face Loss (0): {face_losses[-1]}"
        )
        last_time = time()

    # Weighting factors
    s1 = (alpha / (Xnrm ** 2))
    s2 = ((1 - alpha) / (Ynrm ** 2))

    # Main loop.
    for itr in range(max_iters):

        # Use a different ordering for parameter updates each iteration.
        PRM = np.random.permutation(rank)

        # === UPDATE U === #
        for j in range(inner_iters):        
            for p in PRM:
                idx = (_idx != p)
                U[p] = (Y_VHt[:, p] - torch.matmul(G[idx][:, p], U[idx])) / G[p, p]
                U[p].relu_()

        # Update cached matrix products
        UUt = U @ U.T
        G = UUt * HHt
        Y_UHt = contract("ijk,ri,rk->jr", Y, U, H)
        # Y_UHt = contract("ijk,rik->jr", Y, U[:, :, None] * H[:, None, :])
        # Y_UV is updated later.

        # === UPDATE V === #
        for j in range(inner_iters):        
            for p in PRM:
                idx = (_idx != p)
                V[p] = (Y_UHt[:, p] - torch.matmul(G[idx][:, p], V[idx])) / G[p, p]
                V[p].relu_()

        # Update cached matrix products
        VVt = V @ V.T
        G = UUt * VVt
        # Y_VHt is updated later.
        Y_UV = contract("ijk,ri,rj->kr", Y, U, V)
        # Y_UV = contract("ijk,rij->kr", Y, U[:, :, None] * V[:, None, :])

        # === UPDATE W === #
        for j in range(inner_iters):
            for p in PRM:
                idx = (_idx != p)
                W[p] = (XHt[:, p] - torch.matmul(HHt[idx][:, p], W[idx])) / HHt[p, p]
                W[p].relu_()

        # Update cached matrix products.
        WWt[:] = W @ W.T
        WX[:] = W @ X

        # === UPDATE H === #
        for j in range(inner_iters):
            for p in PRM:

                idx = (_idx != p)

                # Contribution from neural data.
                ztx = WX[p] - torch.matmul(WWt[idx][:, p], H[idx])

                # Contribution from face data.
                zty = Y_UV[:, p] - torch.matmul(G[idx][:, p], H[idx])

                # Update params.
                torch.divide(
                    s1 * ztx + s2 * zty,
                    s1 * WWt[p, p] + s2 * G[p, p],
                    out=H[p]
                )
                H[p].relu_()

        # Update cached matrix products.
        HHt = H @ H.T
        XHt = X @ H.T
        Y_VHt = contract("ijk,rj,rk->ir", Y, V, H)
        # Y_VHt = contract("ijk,rjk->ir", Y,  V[:, :, None] * H[:, None, :])
        # Y_UHt is updated later.
        G = VVt * HHt

        # === COMPUTE LOSSES === #
        neural_losses.append(
            ((
                (Xnrm ** 2)
                + iprd(WWt, HHt)
                - 2 * iprd(W.T, XHt)
            ) / (Xnrm ** 2)).item()
        )
        face_losses.append(
            ((
                (Ynrm ** 2)
                + torch.sum(UUt * G)
                - 2 * iprd(U.T, Y_VHt)
            ) / (Ynrm ** 2)).item()
        )
        total_losses.append(
            alpha * neural_losses[-1] +
            (1 - alpha) * face_losses[-1]
        )

        if verbose:
            print(
                f"Total Loss ({itr + 1}): {total_losses[-1]} \t"
                f"Neural Loss ({itr + 1}): {neural_losses[-1]} \t"
                f"Face Loss ({itr + 1}): {face_losses[-1]} \t"
                f"Iteration Time ({itr + 1}): {time() - last_time}"
            )
            last_time = time()

        # Save progress.
        if outfileprefix is not None:
            np.savez(
                outfileprefix + "_losses.npz",
                total_losses=total_losses,
                neural_losses=neural_losses,
                face_losses=face_losses,
                alpha=alpha,
                rank=rank
            )
            np.savez(
                outfileprefix + "_factors.npz",
                neural=W,
                time=H,
                face=U,
                frequency=V
            )

        # Check convergence.
        if ((itr + 1) >= min_iters) and ((total_losses[-2] - total_losses[-1]) < tol):
            if verbose:
                print("Converged!")
            break

    # Package and return results.
    losses = {
        "total_losses": np.array(total_losses),
        "neural_losses": np.array(neural_losses),
        "face_losses": np.array(face_losses),
    }
    factors = {
        "neural": W,
        "time": H,
        "face": U,
        "frequency": V
    }
    return losses, factors


def coupled_nmf_wrapper(config_filepath):
    config = helpers.load_config(config_filepath)

    rank = config['Neural']['rank']
    alpha = config['Neural']['alpha']
    max_iters = config['Neural']['max_iters']
    tol = config['Neural']['tol']
    inner_iters = config['Neural']['inner_iters']

    for session in config['General']['sessions']:
        neural_tensor = helpers.load_nwb_ts(session['nwb'],'Neural', 'neural_tensor')
        face_tensor = helpers.load_nwb_ts(session['nwb'], 'Neural', 'face_tensor')
        losses, factors = coupled_nmf_model_torch(neural_tensor, face_tensor,
                                                  rank=rank, alpha=alpha,
                                                  min_iters=3, max_iters=max_iters, tol=tol,
                                                  inner_iters=inner_iters,
                                                  verbose=False, outfileprefix=None)
        for key, value in factors:
            helpers.create_nwb_ts(session['nwb'], 'Neural', f'factors_{key}', value, 1.0)

