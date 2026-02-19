# Overview

**lagga** is a framework for conducting genome-wide association studies
in admixed individuals. Like [regenie](https://rgcgithub.github.io/regenie),
**lagga** uses whole-genome regression to capture polygenic effects and correct
for cryptic relatedness. Like [Tractor](https://atkinson-lab.github.io/Tractor-tutorial/),
**lagga** then calculates ancestry-specific effect estimates, conditioned
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
classDef lagga fill:#F4D06F;

class A,B1,G,B2 geno;
class C1,C2 pheno;
class F loco;
class D,E,H lagga;
```

## lagga

### Step 1: whole-genome regression

Step 1 of **lagga** closely follows the approach taken in **regenie**.
The goal here is to calculate leave-one-chromosome-out polygenic
scores using a reasonable subset of genetic markers. This is accomplished using
two layers of ridge regression. In the level 0 ridge regression, separate
regressions are performed using blocks of consecutive markers. This accounts
for linkage disequilibrium within blocks and yields a reduced set of genetic
predictors. In the level 1 ridge regression, the level 0 predictors are used
to calculate cross-validated whole-genome LOCO predictions for each phenotype.


#### Level 0 ridge regression

```mermaid
flowchart TB

    GENOME["Set of M common variants"]
    PREDS[" "]
    PRED["(N, 5M/B) matrix of predictors"]
    GENOME ==> |Divide genome into blocks of size B| BLOCKS ==> |Per-block regressions for 5 penalties| PREDS ==> PRED

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

    linkStyle 3 stroke-width:0px;
    linkStyle 4 stroke-width:0px;
    linkStyle 5 stroke-width:0px;
    linkStyle 6 stroke-width:0px;
    linkStyle 7 stroke-width:0px;
    linkStyle 8 stroke-width:0px;

```

#### Level 1 ridge regression

### Step 2: single-variant tests
