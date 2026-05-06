# Factored Metrics

Factored Metrics is a repository for implementing evaluation metrics for disentangled latent variable models. The goal is to provide clear, reusable, and well-tested implementations of metrics used to evaluate models such as TCVAE, RF-VAE, Beta-VAE, Factor-VAE, and related approaches.

## Purpose

Disentangled representation learning aims to learn latent variables that capture independent factors of variation in data. This repository focuses on metric implementations that help quantify how well a learned representation separates those generative factors.

The project is currently intended to support metrics such as:

- Mutual Information Gap (MIG)
- DCI Disentanglement, Completeness, and Informativeness
- One factor fixed evaluation
- One factor varied evaluation

Additional metrics may be added as the repository evolves.

## Planned Scope

This repository will collect metric implementations, utilities, and examples for evaluating latent variable models. The intended workflow is:

1. Train or load a disentangled latent variable model.
2. Extract latent representations for a dataset with known or controllable factors of variation.
3. Compute one or more disentanglement metrics.
4. Compare results across models, training runs, or metric families.

## Target Models

The metrics in this project are designed to be useful for evaluating models including, but not limited to:

- TCVAE
- RF-VAE
- Beta-VAE
- Factor-VAE
- Other variational autoencoder based disentanglement methods

## Repository Status

This project is under active development. Implementations, examples, documentation, and tests will be added over time.

## Installation

Installation instructions will be added once the package structure and dependencies are finalized.

For development, clone the repository:

```bash
git clone <repository-url>
cd factoredmetrics
```

## Usage

Usage examples will be added as metric implementations become available.

The intended API will likely follow a simple pattern:

```python
from factoredmetrics.mig import mig

score = mig(latents, factors)
print(score)
```

The exact API may change while the project is being developed.

## Metrics

Notation used below:

- `z_j` is the `j`-th learned latent dimension.
- `v_k` is the `k`-th ground-truth factor of variation.
- `I(z_j; v_k)` is the mutual information between a latent dimension and a factor.
- `H(v_k)` is the entropy of a ground-truth factor.

### Mutual Information Gap (MIG)

MIG measures whether each ground-truth factor of variation is captured primarily by a single latent dimension. For each factor `v_k`, the latent dimensions are ranked by mutual information. If `j_k^(1)` and `j_k^(2)` are the indices of the top two latent dimensions for factor `v_k`, then:

```text
MIG = (1 / K) * sum_k [ (I(z_{j_k^(1)}; v_k) - I(z_{j_k^(2)}; v_k)) / H(v_k) ]
```

Higher MIG values generally indicate stronger disentanglement because the most informative latent dimension for each factor is separated from the second-most informative one.

MIG was introduced by Chen et al. in [Isolating Sources of Disentanglement in Variational Autoencoders](https://papers.nips.cc/paper/7527-isolating-sources-of-disentanglement-in-variational-autoencoders), NeurIPS 2018.

### DCI

DCI evaluates disentangled representations using three criteria:

- Disentanglement: whether each latent dimension captures at most one factor.
- Completeness: whether each factor is captured by a small number of latent dimensions.
- Informativeness: whether the representation is useful for predicting the factors.

DCI is usually computed from an importance matrix `R`, where `R_{j,k}` measures how important latent dimension `z_j` is for predicting factor `v_k`, often using regressors such as LASSO or random forests.

For disentanglement, normalize each row of `R`:

```text
p_{j,k} = R_{j,k} / sum_k R_{j,k}
D_j = 1 - H_K(p_j)
D = sum_j rho_j * D_j
rho_j = (sum_k R_{j,k}) / (sum_{j,k} R_{j,k})
```

For completeness, normalize each column of `R`:

```text
q_{j,k} = R_{j,k} / sum_j R_{j,k}
C_k = 1 - H_D(q_k)
C = sum_k eta_k * C_k
eta_k = (sum_j R_{j,k}) / (sum_{j,k} R_{j,k})
```

Here `H_K` and `H_D` are normalized entropy terms. Informativeness is reported through the prediction quality of the model used to predict the factors from the latents, for example prediction error, accuracy, or `R^2`, depending on the factor type and implementation.

DCI was introduced by Eastwood and Williams in [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ), ICLR 2018.

### One Factor Fixed

This evaluation studies how latent representations behave when one generative factor is held fixed while others vary.

For a selected factor `v_k`, collect `L` samples where `v_k` is fixed and the other factors vary:

```text
v^(i) = (v_k, v_{-k}^(i)), i = 1, ..., L
```

Encode the samples and compute the empirical variance of each latent dimension:

```text
s_j^2 = Var_i[z_j(x^(i))]
m = argmin_j s_j^2
```

The index `m` is treated as the latent dimension most associated with the fixed factor. A majority-vote classifier can then map `m` to the factor index `k`; the final score is the classification accuracy.

This family of metrics comes from the beta-VAE evaluation protocol in Higgins et al., [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl&noteId=Sy2fzU9gl), ICLR 2017. The variance-based majority-vote version is from Kim and Mnih, [Disentangling by Factorising](https://proceedings.mlr.press/v80/kim18b), ICML 2018.

### One Factor Varied

This evaluation studies how latent representations respond when a single generative factor is varied while others are fixed.

For a selected factor `v_k`, collect `L` samples where only `v_k` varies and all other factors are fixed:

```text
v^(i) = (v_k^(i), v_{-k}), i = 1, ..., L
```

Then compute the empirical variance of each latent dimension:

```text
s_j^2 = Var_i[z_j(x^(i))]
m = argmax_j s_j^2
```

The index `m` is treated as the latent dimension most responsive to the varied factor. As in the one-factor-fixed metric, a majority-vote classifier can map `m` to the factor index `k`, and the final score is the classification accuracy.

The one-factor-varied variant is described in Kim et al., [Relevance Factor VAE: Learning and Identifying Disentangled Factors](https://www.catalyzex.com/paper/relevance-factor-vae-learning-and-identifying), 2019, and is also summarized in the Bayes-Factor-VAE evaluation discussion by Kim et al., [Bayes-Factor-VAE: Hierarchical Bayesian Deep Auto-Encoder Models for Factor Disentanglement](https://openaccess.thecvf.com/content_ICCV_2019/html/Kim_Bayes-Factor-VAE_Hierarchical_Bayesian_Deep_Auto-Encoder_Models_for_Factor_Disentanglement_ICCV_2019_paper.html), ICCV 2019.

## References

- Chen, R. T. Q., Li, X., Grosse, R., and Duvenaud, D. [Isolating Sources of Disentanglement in Variational Autoencoders](https://papers.nips.cc/paper/7527-isolating-sources-of-disentanglement-in-variational-autoencoders). NeurIPS 2018.
- Eastwood, C., and Williams, C. K. I. [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ). ICLR 2018.
- Higgins, I., Matthey, L., Pal, A., Burgess, C. P., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl&noteId=Sy2fzU9gl). ICLR 2017.
- Kim, H., and Mnih, A. [Disentangling by Factorising](https://proceedings.mlr.press/v80/kim18b). ICML 2018.
- Kim, M., Wang, Y., Sahu, P., and Pavlovic, V. [Relevance Factor VAE: Learning and Identifying Disentangled Factors](https://www.catalyzex.com/paper/relevance-factor-vae-learning-and-identifying). 2019.
- Kim, M., Wang, Y., Sahu, P., and Pavlovic, V. [Bayes-Factor-VAE: Hierarchical Bayesian Deep Auto-Encoder Models for Factor Disentanglement](https://openaccess.thecvf.com/content_ICCV_2019/html/Kim_Bayes-Factor-VAE_Hierarchical_Bayesian_Deep_Auto-Encoder_Models_for_Factor_Disentanglement_ICCV_2019_paper.html). ICCV 2019.

## Contributing

Contributions are welcome as the project develops. Useful contributions may include:

- New metric implementations
- Bug fixes
- Tests and validation scripts
- Documentation improvements
- Reproducible examples

## License

License information will be added later.
