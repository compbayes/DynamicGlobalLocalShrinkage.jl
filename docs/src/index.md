```@meta
CurrentModule = DynamicGlobalLocalShrinkage
```

# DynamicGlobalLocalShrinkage

Documentation for dynamic global-local shrinkage process priors for state-space models  [DynamicGlobalLocalShrinkage](https://github.com/compbayes/DynamicGlobalLocalShrinkage.jl).

## Installation
Install from the Julia package manager (via Github) by typing `]` in the Julia REPL:
```
] add git@github.com:compbayes/DynamicGlobalLocalShrinkage.jl.git
```

## Usage

The Gibbs sampler in the package samples from the joint posterior $p(h_{1:T}, \xi_{0:T}, \mu, \phi, \sigma_\eta \vert y_{1:T})$ in the following model:
```math
\begin{align*}
  y_t &= \nu_t, \quad \nu_t \sim N(0,\exp(h_t)) \\
  h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
\end{align*}
```
The shape parameter $\alpha$ is a known input, but future versions of the package will add an update step for it.

The individual updating steps of the package is intended to be used as components in sampler for other models. The documentation contains examples of a **time series regression** with dynamic shrinkage process prior
```math
\begin{align*}
  y_t &= \boldsymbol{x}_t^\top \boldsymbol{\beta} + \nu_t, \quad \nu_t \sim N(0,\exp(h_t)) \\
  h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
\end{align*}
```
and the **local level model** with dynamic shrinkage process prior
```math
\begin{align*}
  y_t &= x_t + \varepsilon_t, \quad \varepsilon_t \sim N(0,\sigma^2_\varepsilon) \\
  x_t &= x_{t-1} + \nu_t, \quad \nu_t \sim N(0,\exp(h_t)) \\
  h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\alpha, 0, \sigma_\eta). \\
\end{align*}
```

## Sampling details

The package implements a Gibbs sampler for dynamic global-local shrinkage process priors which can be used in state-space models. The algorithm is essentially the one presented in Kowal et al. (2019)[^1] but some minor changes due to a somewhat different prior for the hyperparameters. The sampler in Kowal et al. (2019) builds on the work of Kim et al. (1998)[^2] and Kastner and Fruhwirth-Schnatter (2014)[^3] for stochastic volatility models, and Rue (2001)[^4] for fast sampling of high-dimensional Gaussian distributions with sparse precision matrix.

The sampling steps for each parameter block is exported as separate functions as the package is mostly intended to be used as building blocks in other (state-space) packages.

The Gibbs sampler in the package samples from the joint posterior in the following model:
```math
\begin{align*}
  y_t &= \mu_y + \nu_t, \quad \nu_t \sim N(0,\exp(h_t)) \\
  h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\alpha, 0, \sigma_\eta) \\
\end{align*}
```
where $Z(\alpha,\beta, \mu, \sigma_\eta)$ is the Z-distribution, and the package is restricted to the symmetric case $Z(\alpha,\alpha, 0, \sigma_\eta)$.

The package uses two established data augmentation tricks to make Gibbs sampling possible.

**Trick 1 - Make the log-volatility additive**. Since $y_t = \exp(h_t/2)\tilde{\nu}_t, \, \tilde{\nu}_t \sim N(0,1)$, we can square and take logs to obtain a model where the log-volatility enter additively:

```math
y^\star_t = h_t + \zeta_t \quad \zeta_t \sim \log\chi^2_1,
```

where $y^\star_t = \log (y_t -\mu_y)^2$ and $\log\chi^2_1$ is the distribution of the log of a chi-squared variable with 1 degree of freedom. This distribution can be approximated by a mixture of normals distribution with known parameters: $\zeta_t \overset{\mathrm{approx}}{\sim}\mathrm{MoN}(\boldsymbol{m},\boldsymbol{v}, \boldsymbol{\omega})$, where $\boldsymbol{m}$ is the vector with $K$ means, $\boldsymbol{v}$ is the vector with variances and $\boldsymbol{\omega}$ is the vector with weights for the mixture components. The 10-component mixture of normals approximation from Omori et al. (2007)[^5] is the default in the package. The mixture of normal can be written using the latent mixture allocation variables $I_t$: 
```math
\begin{align*}
\zeta_t \mid (I_t = k) & \overset{\mathrm{ind}}{\sim} N(m_k,v_k) \\
I_t & \overset{\mathrm{iid}}{\sim} \mathrm{Categorical}(\omega_1,\ldots,\omega_K)
\end{align*}
```

**Trick 2 - Express the Z-distribution as a normal scale mixture**. The Z-distribution can be expressed as mean-variance mixture of Normals. In the special case with a symmetric Z-distribution $\alpha=\beta$ with zero mean, we can express the distribution as a scale mixture of normals with a variance following a Polya-Gamma distribution

```math
\begin{equation*}
  Z(\alpha, \alpha, 0, \sigma^2_\eta) = \int N(0,\sigma^2_\eta \xi^{-1})\mathrm{PG}(\xi \vert 2\alpha,0)\, d \xi
\end{equation*}
```

The original parameters $\mu_y, h_{1:T}, \mu, \phi$ have been expanded to $\mu_y, h_{1:T}, \mu, \phi, \xi_{0:T}, I_{1:T}$, where $I_{1:T}$ is a vector with the mixture component indicators in the mixture of normals approximation to the log chi-square distribution. 

Hence, the Gibbs sampling algorithm in the package samples from the augmented model
```math
\begin{align*}
  y^\star_t &= h_t + \zeta_t, \quad \zeta_t \mid (I_t = k) \overset{\mathrm{ind}}{\sim} N(m_k, v_k) \\
  I_t & \overset{\mathrm{iid}}{\sim} \mathrm{Categorical}(\omega_1,\ldots,\omega_K) \\
  h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \vert \xi_t \overset{\mathrm{ind}}{\sim} N(0,\sigma^2_\eta \xi_t^{-1}) \\
  \xi_t & \overset{\mathrm{iid}}{\sim} \mathrm{PG}(2\alpha,0)
\end{align*}
```

The complete **Gibbs sampling algorithm** consists of the following steps:
- Sample the mixture allocation $I_{1:T}$ independently from categorical distributions
- Sample the log-volatility path $h_{0:T}$ jointly using a sparse multivariate normal distribution
- Sample $\mu$ from a normal distribution
- Sample $\phi$ from a truncated normal distribution
- Sample the Polya-Gamma variables $\xi_{1:T}$ from a Polya-Gamma distribution
- Sample the mean $\mu_y$ from a normal distribution


## References

[^1]: Kowal, D. R., Matteson, D. S., and Ruppert, D. (2019). Dynamic shrinkage processes. *Journal of the Royal Statistical Society: Series B* (Statistical Methodology)*, 81(4):781–804.
[^2]: Kim, S., Shephard, N., and Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. *The Review of Economic Studies*, 65(3):361–393.
[^3]: Kastner, G. and Fruhwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *Computational Statistics & Data Analysis*, 76:408–423.
[^4]: Rue, H. (2001). Fast sampling of Gaussian Markov random fields. *Journal of the Royal Statistical Society: Series B*, 63(2):325–338.
[^5]: Omori, Y., Chib, S., Shephard, N., and Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *Journal of Econometrics*, 140(2):425–449.