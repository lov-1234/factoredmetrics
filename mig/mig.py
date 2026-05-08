import math
import torch
from helpers.densities import log_gaussian
from einops import rearrange


def compute_log_q_z_given_x(z, mean, logvar):
    """
    Computes log q(z_{i,j} | x_n) for all i, n, j.
    
    Args:
        z:      (N, D)
        mean:   (N, D)
        logvar: (N, D)

    Returns:
        log_q_z_given_x: (N, N, D)
            log_q_z_given_x[i, n, j] = log q(z[i, j] | x_n)
    """
    z_eval = rearrange(z, "i d -> i 1 d")              # (N, 1, D)
    mean_comp = rearrange(mean, "n d -> 1 n d")        # (1, N, D)
    logvar_comp = rearrange(logvar, "n d -> 1 n d")    # (1, N, D)
    log_q_z_given_x = log_gaussian(
        z_eval,
        mean_comp,
        logvar_comp,
    )  # (N, N, D)
    return log_q_z_given_x


def calculate_entropy_zj_from_log_q(log_q_z_given_x):
    """
    Estimate H(z_j) = -E_{q(z_j)}[log q(z_j)].

    Args:
        log_q_z_given_x: (N, N, D)

    Returns:
        entropy_zj: (D,)
    """
    N = log_q_z_given_x.shape[0]
    # log q(z_{i,j}) = log(1/N sum_n q(z_{i,j} | x_n))
    log_q_zj = torch.logsumexp(log_q_z_given_x, dim=1) - math.log(N)  # (N, D)
    # H(z_j) = - mean_i log q(z_{i,j})
    entropy_zj = -log_q_zj.mean(dim=0)  # (D,)
    return entropy_zj


def calculate_first_mi_terms(log_q_z_given_x, factors):
    """
    Compute the first term of MI:
        E_{q(z_j, v_k)}[log q(z_j | v_k)]
    for all latent dimensions j and all factors k.

    Args:
        log_q_z_given_x: (N, N, D)
            log_q_z_given_x[i, n, j] = log q(z[i, j] | x_n)

        factors: (N, K)
            factors[i, k] = value of factor k for sample i

    Returns:
        first_terms: (D, K)
            first_terms[j, k] = E[log q(z_j | v_k)]
    """

    device = log_q_z_given_x.device
    factors = factors.to(device)

    N, _, D = log_q_z_given_x.shape
    K = factors.shape[1]

    first_terms = torch.zeros(D, K, device=device)

    for k in range(K):
        factor_k = factors[:, k]  # (N,)
        unique_values = torch.unique(factor_k)

        # We accumulate over samples i.
        first_term_sum_k = torch.zeros(D, device=device)

        for value in unique_values:
            # Rows i whose actual factor value is `value`
            row_mask = factor_k == value       # (N,)

            # Mixture components n with factor value `value`
            component_mask = factor_k == value  # (N,)

            count = component_mask.sum().item()

            # selected_log_q shape:
            # (num_samples_with_value, count, D)
            selected_log_q = log_q_z_given_x[row_mask][:, component_mask, :]

            # log q(z_{i,j} | v_k=value)
            # = log(1/count sum_{n: v_k(n)=value} q(z_{i,j} | x_n))
            #
            # Shape: (num_samples_with_value, D)
            log_q_z_given_vk = (
                torch.logsumexp(selected_log_q, dim=1)
                - math.log(count)
            )

            # Sum over evaluation samples i with this factor value.
            first_term_sum_k += log_q_z_given_vk.sum(dim=0)

        # Average over all N samples.
        first_terms[:, k] = first_term_sum_k / N

    return first_terms


def estimate_mi_matrix_density(z, mean, logvar, factors):
    """
    Estimate MI matrix M[j, k] = I(z_j ; v_k).

    Args:
        z:       (N, D)
        mean:    (N, D)
        logvar:  (N, D)
        factors: (N, K)

    Returns:
        mi_matrix: (D, K)
        entropy_zj: (D,)
        first_terms: (D, K)
    """
    log_q_z_given_x = compute_log_q_z_given_x(z, mean, logvar)
    entropy_zj = calculate_entropy_zj_from_log_q(log_q_z_given_x)  # (D,)
    first_terms = calculate_first_mi_terms(
        log_q_z_given_x=log_q_z_given_x,
        factors=factors,
    )  # (D, K)
    mi_matrix = first_terms + entropy_zj[:, None]  # (D, K)

    return mi_matrix, entropy_zj, first_terms


def entropy_discrete(labels, eps=1e-12):
    """
    Empirical entropy H(v_k) for one discrete factor.

    Args:
        labels: (N,)

    Returns:
        entropy: scalar tensor
    """
    _, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * torch.log(probs + eps)).sum()


def compute_mig_from_mi_matrix(mi_matrix, factors, eps=1e-12):
    """
    Compute MIG from MI matrix.

    Args:
        mi_matrix: (D, K)
            mi_matrix[j, k] = I(z_j ; v_k)

        factors: (N, K)

    Returns:
        mig: scalar tensor
        mig_per_factor: (K,)
        factor_entropies: (K,)
    """
    device = mi_matrix.device
    factors = factors.to(device)
    D, K = mi_matrix.shape
    factor_entropies = torch.stack([
        entropy_discrete(factors[:, k], eps=eps)
        for k in range(K)
    ])  # (K,)
    sorted_mi, _ = torch.sort(mi_matrix, dim=0, descending=True)
    top1 = sorted_mi[0]  # (K,)
    if D > 1:
        top2 = sorted_mi[1]
    else:
        top2 = torch.zeros_like(top1)
    mig_per_factor = (top1 - top2) / (factor_entropies + eps)
    mig = mig_per_factor.mean()
    return mig, mig_per_factor, factor_entropies


"""
Usage:
mi_matrix, entropy_zj, first_terms = estimate_mi_matrix_density(
    z=z,
    mean=mean,
    logvar=logvar,
    factors=factors,
)

mig, mig_per_factor, factor_entropies = compute_mig_from_mi_matrix(
    mi_matrix=mi_matrix,
    factors=factors,
)

print("MI matrix:", mi_matrix.shape)          # (D, K)
print("Entropy z_j:", entropy_zj.shape)       # (D,)
print("First terms:", first_terms.shape)      # (D, K)
print("MIG:", mig.item())
print("MIG per factor:", mig_per_factor)
"""