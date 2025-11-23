from logging import raiseExceptions
from ._cpp import read_rfmix, read_flare
import pandas as pd
from pgenlib import PvarReader, PgenReader
from .data import _get_info
import numpy as np


def convert_lanc(file: str, file_fmt: str, plink_prefix: str, output: str):
    ## Read input local ancestry file to pandas DataFrame
    if file_fmt == "FLARE":
        df = pd.DataFrame(read_flare(file))
    elif file_fmt == "RFMix":
        df = pd.DataFrame(read_rfmix(file))
    else:
        raise ValueError("Please specify either `FLARE` or `RFMix` input")

    ## Read plink files
    pvar = PvarReader(bytes(plink_prefix + ".pvar", "utf8"))
    pgen = PgenReader(bytes(plink_prefix + ".pgen", "utf8"))
    n_variants = pvar.get_variant_ct()

    n_skip = 0
    with open(plink_prefix + ".psam") as psam:
        for line in psam:
            if line.startswith("#IID") | line.startswith("#FID"):
                break
            n_skip += 1

    ## Variant plink info
    df_pvar = _get_info(pvar, range(n_variants))  # variant info

    ## Sample plink info
    df_psam = pd.read_csv(
        plink_prefix + ".psam", sep="\\s+", skiprows=n_skip, dtype=str
    )
    samples = df_psam["#IID"]

    if not samples.isin(df["sample"]).all():
        raise ValueError("Not all pgen samples exist in local ancestry input")

    ## Filter input to ordered plink samples
    df = df[df["sample"].isin(samples)].copy()

    ## Sort df by sample, chrom, spos
    df["sample"] = pd.Categorical(df["sample"], categories=samples, ordered=True)
    df = df.sort_values(by=["sample", "chrom", "spos"]).reset_index(drop=True)

    ## Exclude tracts starting after or ending before pgen range
    min_pvar = np.min(df_pvar["pos"])
    max_pvar = np.max(df_pvar["pos"])
    tracts_mask = (df["spos"] < max_pvar) & (df["epos"] > min_pvar)
    df = df[tracts_mask]

    ## Clip tracts positions to pgen start, end
    df["epos"] = df["epos"].clip(upper=max_pvar)
    df["spos"] = df["spos"].clip(lower=min_pvar)

    ## Get index of first pvar pos >= tract epos
    df["idx"] = np.searchsorted(df_pvar["pos"].values, df["epos"].values, side="left")

    ## If multiple tracts have same idx, pick last one
    df = (
        df.sort_values(["sample", "chrom", "idx"])
        .groupby(["sample", "chrom", "idx"], as_index=False, observed=True)
        .tail(1)  # last row per group
    )

    ## Get .lanc file lines
    df["switch"] = (
        df["idx"]
        .astype(str)
        .str.cat(df["anc0"].astype(str), sep=":")
        .str.cat(df["anc1"].astype(str))
    )
    lines = (
        df.groupby(["sample", "chrom"], observed=True)["switch"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index(drop=True)
    )

    ## Write output
    header = f"{len(df_pvar)} {len(df_psam)}"
    with open(output, "w") as f:
        f.write(header + "\n" + "\n".join(lines.astype(str)))
