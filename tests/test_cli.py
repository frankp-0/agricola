import pytest
import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
from lagga.cli import app
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

    return {
        "pheno_file": str(pheno_file),
        "covar_file": str(covar_file),
        "plink_prefix": ",".join(
            ["tests/data/chr" + str(chr) for chr in range(20, 23)]
        ),
        "lanc_file": ",".join(
            ["tests/data/chr" + str(chr) + ".lanc" for chr in range(20, 23)]
        ),
        "step1_prefix": str(tmp_path / "out"),
        "step2_prefix": ",".join(
            [str(tmp_path / "chr") + str(chr) for chr in range(20, 23)]
        ),
    }


def test_step1_toy(toy_data):
    result = runner.invoke(
        app,
        [
            "step1",
            "--plink-prefix",
            toy_data["plink_prefix"],
            "--lanc-file",
            toy_data["lanc_file"],
            "--out-prefix",
            toy_data["step1_prefix"],
            "--pheno-file",
            toy_data["pheno_file"],
            "--covar-file",
            toy_data["covar_file"],
        ],
    )
    assert result.exit_code == 0

    out_file = toy_data["step1_prefix"] + ".pkl"
    assert Path(out_file).exists()


def test_step2_toy(toy_data):
    result = runner.invoke(
        app,
        [
            "step2",
            "--plink-prefix",
            toy_data["plink_prefix"],
            "--lanc-file",
            toy_data["lanc_file"],
            "--step1-prefix",
            "tests/data/step1_pred",
            "--out-prefix",
            toy_data["step2_prefix"],
            "--pheno-file",
            toy_data["pheno_file"],
            "--covar-file",
            toy_data["covar_file"],
        ],
    )
    assert result.exit_code == 0

    file_20 = toy_data["step2_prefix"][0]
    assert Path(file_20).exists()
