# MIT License
# Copyright (c) 2026 Franklin Ockerman
# See LICENSE.txt file for full license text

import pytest
import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
from agricola._internal.cli import app
import jax

runner = CliRunner()


@pytest.fixture
def toy_data(tmp_path: Path):
    N = 20
    P = 4
    C = 2

    ## Phenotypes
    pheno_file = tmp_path / "pheno.tsv"
    key0 = jax.random.key(8899134)
    df_traits = pd.DataFrame(jax.random.normal(key0, shape=(N, P)))
    df_traits.columns = ["trait" + str(i) for i in range(0, 4)]
    df_traits.insert(0, "#IID", [f"Sample_{i + 1}" for i in range(len(df_traits))])  # pyright: ignore
    df_traits.to_csv(pheno_file, sep="\t", index=False, header=True)

    ## Covariates
    covar_file = tmp_path / "covar.tsv"
    key1 = jax.random.key(13100)
    df_covars = pd.DataFrame(jax.random.normal(key1, shape=(N, C)))
    df_covars.columns = ["covar" + str(i) for i in range(0, 2)]
    df_covars.insert(0, "#IID", [f"Sample_{i + 1}" for i in range(len(df_covars))])  # pyright: ignore
    df_covars.to_csv(covar_file, sep="\t", index=False, header=True)

    ## Plinks file
    plinks = ["tests/data/chr" + str(chr) for chr in range(20, 23)]
    plinks_file = tmp_path / "plinks.txt"
    with open(plinks_file, "w") as f:
        for plink in plinks:
            f.write(f"{plink}\n")

    ## lancs file
    lancs = ["tests/data/chr" + str(chr) + ".lanc" for chr in range(20, 23)]
    lancs_file = tmp_path / "lancs.txt"
    with open(lancs_file, "w") as f:
        for lanc in lancs:
            f.write(f"{lanc}\n")

    ## outs file
    outs = [str(tmp_path / "chr") + str(chr) for chr in range(20, 23)]
    outs_file = tmp_path / "outs.txt"
    with open(outs_file, "w") as f:
        for out in outs:
            f.write(f"{out}\n")

    return {
        "pheno_file": str(pheno_file),
        "covar_file": str(covar_file),
        "plink_list": plinks_file,
        "lanc_list": lancs_file,
        "step1_prefix": str(tmp_path / "out"),
        "outs_file": outs_file,
    }


def test_step1_toy(toy_data):
    result = runner.invoke(
        app,
        [
            "step1",
            "--plink-list",
            toy_data["plink_list"],
            "--lanc-list",
            toy_data["lanc_list"],
            "--out-prefix",
            toy_data["step1_prefix"],
            "--pheno-file",
            toy_data["pheno_file"],
            "--covar-file",
            toy_data["covar_file"],
        ],
    )
    assert result.exit_code == 0


def test_step2_toy(toy_data):
    result = runner.invoke(
        app,
        [
            "step2",
            "--plink-list",
            toy_data["plink_list"],
            "--lanc-list",
            toy_data["lanc_list"],
            "--step1-prefix",
            "tests/data/step1_pred",
            "--out-list",
            toy_data["outs_file"],
            "--pheno-file",
            toy_data["pheno_file"],
            "--covar-file",
            toy_data["covar_file"],
        ],
    )
    assert result.exit_code == 0

    with open(toy_data["outs_file"], "r") as f:
        outs_files = f.readlines()

    file_20 = outs_files[0].strip() + "_trait0.parquet"
    assert Path(file_20).exists()


def test_allsteps_toy(toy_data):
    result = runner.invoke(
        app,
        [
            "all-steps",
            "--plink-list",
            toy_data["plink_list"],
            "--lanc-list",
            toy_data["lanc_list"],
            "--out-list",
            toy_data["outs_file"],
            "--pheno-file",
            toy_data["pheno_file"],
            "--covar-file",
            toy_data["covar_file"],
        ],
    )
    assert result.exit_code == 0

    with open(toy_data["outs_file"], "r") as f:
        outs_files = f.readlines()

    file_20 = outs_files[0].strip() + "_trait0.parquet"
    assert Path(file_20).exists()
