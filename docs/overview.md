# Overview

**agricola** is a framework for conducting genome-wide association studies
in admixed individuals. Like [regenie](https://rgcgithub.github.io/regenie),
**agricola** uses whole-genome regression to capture polygenic effects and correct
for cryptic relatedness. Like [Tractor](https://atkinson-lab.github.io/Tractor-tutorial/),
**agricola** then calculates ancestry-specific effect estimates, conditioned
on local ancestry and whole-genome predictions. This two-step procedure is
described in greater detail below.

```mermaid
flowchart TB 

%% ---- STEP 1 INPUTS ----
A["Pruned common variants<br/>Plink2 .pgen"]
B1["Local ancestry (.lanc)"]
C1["Phenotypes<br/>Covariates"]

%% ---- STEP 1 ----
D[\"Step 1 (level 0 ridge)"/]
E[\"Step 1 (level 1 ridge"/]
F["LOCO predictions"]

%% ---- STEP 2 INPUTS ----
G["Full variant set<br/>Plink2 .pgen"]
B2["Local ancestry (.lanc)"]
C2["Phenotypes<br/>Covariates"]

%% ---- STEP 2 ----
H[\"Step 2: local ancestry-conditioned<br/>single-variant tests"/]

%% ---- FLOWS ----
A --> D
B1 --> D
C1 --> D
D --> E
E --> F

G --> H
B2 --> H
F --> H
C2 --> H

%% ---------- STYLING ----------
classDef geno fill:#FFF8F0;
classDef pheno fill:#9DD9D2;
classDef loco fill:#FF8811;
classDef agricola fill:#F4D06F;

class A,B1,G,B2 geno;
class C1,C2 pheno;
class F loco;
class D,E,H agricola;
```

## agricola

### Step 1: Whole-Genome Regression

Step 1 of **agricola** closely follows the approach taken in **regenie**.
The goal here is to calculate leave-one-chromosome-out polygenic
scores using a reasonable subset of genetic markers. This is accomplished using
two layers of ridge regression. In the level 0 ridge regression, separate
regressions are performed using blocks of consecutive markers. This accounts
for linkage disequilibrium within blocks and yields a reduced set of genetic
predictors. In the level 1 ridge regression, the level 0 predictors are used
to calculate cross-validated whole-genome LOCO predictions for each phenotype.


#### Level 0 Ridge Regression

```mermaid
flowchart TB

    GENOME["Set of M common variants"]
    PREDS[" "]
    PRED["(N, 5M/B) matrix of predictors"]
    PHENO["Phenotype"]
    GENOME ==> |Divide genome into blocks of size B| BLOCKS ==> |Per-block regressions for 5 penalties| PREDS ==> PRED
    PHENO ==> PREDS

    subgraph BLOCKS[" "]
        direction LR
        A[Block 1]
        B[Block 2]
        C[...]
        D[Block M/B]

        A --- B
        B --- C
        C --- D
    end

    subgraph PREDS[" "]
        direction LR
        E["Block 1 predictions"]
        F["Block 2 predictions"]
        G[...]
        H["Block M/B predictions"]

        E --- F
        F --- G
        G --- H
    end

    linkStyle 4 stroke-width:0px;
    linkStyle 5 stroke-width:0px;
    linkStyle 6 stroke-width:0px;
    linkStyle 7 stroke-width:0px;
    linkStyle 8 stroke-width:0px;
    linkStyle 9 stroke-width:0px;

```

In level 0 of step 1, the genome is divided into blocks of consecutive SNPs
and ridge regression is used to generate per-block predictions.

First, a subset of $M$ (e.g. 500,000) variants is chosen to capture common
variation. Supposing block-size $B$ and sample size $N$, **agricola** reads
variants into $J=\lceil M/B \rceil$ blocks. For a given $N\times J$ block of
genotypes $G_B$ and $N\times 1$ phenotype $Y$, a series of ridge regressions is
performed for pre-specified penalties $\lambda_1, \lambda_2, \cdots, \lambda_A$:


$$\hat{Y}^{(a)}_B = G_B(G_B^TG_B + \lambda_a I_B)^{-1}G_B^TY$$

These block-specific predictors are aggregated to obtain
a $N\times JA$ set of genome-wide predictors, $W$:

$$W = \begin{pmatrix}\hat{Y}^{(1)}_1 & \cdots & \hat{Y}^{(A)}_1 & \cdots &  \hat{Y}^{(1)}_J & \cdots & \hat{Y}^{(A)}_J \end{pmatrix}$$

This procedure is followed for both quantitative and binary traits.
An $N\times C$ covariate matrix $X$ may be optionally supplied. In this case,
$X$ is regressed out of $Y$ and $G$, so that $Y$ and $G$ are
mean-centered and orthogonal to $X$. $Y$ and $G$ are additionally
standardized to have variance equal to 1.

#### Level 1 Ridge Regression

```mermaid
flowchart LR

A["Level 0 predictions"]
B["Phenotype"]
C["Cross-validated<br/>whole-genome predictions"]


    subgraph LOCO[" "]
        direction TB
        D["Chr1 LOCO prediction"]
        E["Chr2 LOCO prediction"]
        F[...]
        G["Chr22 LOCO prediction"]

        D --- E
        E --- F
        F --- G
    end

A ==> C
B ==> C
C ==> LOCO
```

In level 1 of step 1, a second level of ridge regression is performed using the
predictions from level 0. Predictions are obtained for penalties
$\gamma_1, \cdots, \gamma_A$ and a final model is selected via cross-validation.

For quantitative traits, it is assumed that the phenotype $Y$ has been
residualized by $X$ and standardized as in level 0. For the model
$Y = W\beta + \epsilon$, we fit the following
ridge regression for each penalty $\gamma_a$.

$$\hat{\beta}^{(a)} = (W^TW + \lambda_a I_J)^{-1}W^TY$$

The optimal $\gamma_a$ is selected by minimizing mean-squared error with k-fold
cross-validation. For each chromosome $c$, a subset of predictors in $W$
corresponding to this chromosome, $W_c$ may be defined. Thus the
chromosome-specific predictor is given by the subset of coefficients $\hat{\beta}^{(a)}_c$.
A final set of leave-one-chromosome-out (LOCO) predictors is given by
$\begin{pmatrix}\hat{Y}_c\end{pmatrix}$, where:

$$\hat{Y}_c = W\hat{\beta}^{(a)}-W_c\hat{\beta}^{(a)}_c$$


For binary traits, logistic ridge regression is used to estimate
$\hat{\beta}^{(a)}$ in the model $\text{logit}(p) = \mu + W\beta$.
Here, $\hat{\mu}$ represents the effect on $Y$ from the covariates $X$
and is estimated separately using the logistic regression model
$\text{logit}(p) = \mu$. $\hat{\mu}$ is then provided as an un-penalized offset
in the logistic ridge regression model above.

The LOCO predictors (on the linear scale) are given as:

$$\hat{\eta}_c = \hat{\mu} + W\hat{\beta}^{(a)}-W_c\hat{\beta}^{(a)}_c$$

Here, the optional parameter $\gamma_a$ is selected by maximizing log-likelihood
with k-fold cross-validation. Optionally, leave-one-out cross-validation may
be used for rare phenotypes if specified by the user.

### Step 2: Single-Variant Tests

In step 2, variants are tested for association with phenotypes under several
possible models, depending on the user specification.

#### Notation

Before defining the tests that can be performed using **agricola**, let
the following notation serve as a reference:



| Notation      | Shape |  Description |
| --- | --- | --- |
| \(N\) | (1,) | Sample size (max sample size across phenotypes) |
| \(X\) | (N,C) | The covariates |
| \(Y\) | (N,) | The phenotype. For quantitative traits, \(Y\) has been orthogonalized with respect to \(X\) and standardized to have mean zero and unit variance. |
| \(\hat{Y}_c\) | (N,) | The LOCO prediction (linear scale) from step 1 |
| \(G\) | (N,K) | The ancestry-deconvoluted genotypes at a given locus. This is the number of haplotypes from ancestry \(k\) with the alternate allele. |
| \(\tilde{G}\) | (N,K) | The ancestry-deconvoluted genotypes, orthogonalized by \(X\). |
| \(H\) | (N,) | The genotypes at a given locus. This is the total number of haplotypes with the alternate allele. |
| \(\tilde{H}\) | (N,) | The genotypes, orthogonalized by \(X\) |
| \(L\) | (N,K-1) | The local ancestry at a given locus. \(L_k\) is the number of haplotypes from ancestry \(k\). One ancestry is omitted since it can be determined from the other $K-1$. |
| \(\tilde{L}\) | (N,K-1) | The local ancestry, orthogonalized by \(X\) |


#### Tests (Quantitative Traits)

For quantitative traits,
the associations are tested with respect to the residualized phenotype
$\tilde{Y} = Y - \hat{Y}_c$. The full model is given by:


\[\tilde{Y} = \tilde{L}\alpha + \tilde{G}\beta + \epsilon,\]

where \(\epsilon \sim \text{Normal}(0_N, \sigma^2I_N)\).

The following models can be tested using **agricola**


| Model Name | Mean Model | Estimand | \(H_0\) | Local Ancestry Conditioned? |
| --- | --- | --- | --- | --- |
| \(\text{agricola}_{\text{lanc}, \text{het}}\) | \(E[\tilde{Y}] = \tilde{L}\alpha + \tilde{G}\beta\) | \(\beta\) | \(\beta = 0\)  | Yes |
| \(\text{agricola}_{\text{nolanc}, \text{het}}\) | \(E[\tilde{Y}] = \tilde{G}\beta\) | \(\beta\) | \(\beta = 0\)  | No |
| \(\text{agricola}_{\text{lanc}, \text{hom}}\) | \(E[\tilde{Y}] = \tilde{L}\alpha + \tilde{H}\gamma\) | \(\gamma\) | \(\gamma = 0\)  | Yes |
| \(\text{agricola}_{\text{nolanc}, \text{hom}} \) | \(E[\tilde{Y}] = \tilde{H}\gamma\) | \(\gamma\) | \(\gamma = 0\)  | No |


Each null hypothesis can be tested using either a Rao's score test
(`--test-type score`) or Wald test (`--test-type wald`). The choice of local
ancestry adjustment can be specified with `--adjust-lanc` or `--no-adjust-lanc`.
For a given run, both the heterogeneous and homogeneous models
(e.g. \(\text{agricola}_{\text{lanc}, \text{het}}\) and \(\text{agricola}_{\text{lanc}, \text{hom}}\)) are fit.

#### Tests (Binary Traits)

For binary traits, the LOCO prediction
$\hat{Y}_c$ is provided as an offset in logistic regression models of the form
$\text{logit}(p) = \hat{Y}_c + Z\beta$. As with quantitative traits,
the following tests can be performed (either with Wald or Rao's score test).


| Model Name | Mean Model | Estimand | \(H_0\) | Local Ancestry Conditioned? |
| --- | --- | --- | --- | --- |
| \(\text{agricola}_{\text{lanc}, \text{het}}\) | \(\text{logit}(E[Y]) = \hat{Y}_c + \tilde{L}\alpha + \tilde{G}\beta\) | \(\beta\) | \(\beta = 0\)  | Yes |
| \(\text{agricola}_{\text{nolanc}, \text{het}}\) | \(\text{logit}(E[Y]) = \hat{Y}_c + \tilde{G}\beta\) | \(\beta\) | \(\beta = 0\)  | No |
| \(\text{agricola}_{\text{lanc}, \text{hom}}\) | \(\text{logit}(E[Y]) = \hat{Y}_c + \tilde{L}\alpha + \tilde{H}\gamma\) | \(\gamma\) | \(\gamma = 0\)  | Yes |
| \(\text{agricola}_{\text{nolanc}, \text{hom}} \) | \(\text{logit}(E[Y]) = \hat{Y}_c + \tilde{H}\gamma\) | \(\gamma\) | \(\gamma = 0\)  | No |


## Miscellaneous

**agricola** currently supports only autosomes. Analysis with sex chromosomes
is an area of future development.
