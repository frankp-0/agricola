# Overview

**lagga** is a framework for conducting genome-wide association studies
in admixed individuals. Like [regenie](https://rgcgithub.github.io/regenie),
**lagga** uses whole-genome regression to capture polygenic effects and correct
for cryptic relatedness. Like [Tractor](https://atkinson-lab.github.io/Tractor-tutorial/),
**lagga** then calculates ancestry-specific effect estimates, conditioned
on local ancestry. We describe this two-step process in greater detail below.

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
