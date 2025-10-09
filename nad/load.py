import os
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd


def load_census(census_path: os.PathLike) -> pd.DataFrame:
    return pd.read_csv(
        census_path,
        usecols=[
            "ENTIDAD",
            "MUN",
            "LOC",
            "AGEB",
            "MZA",
            "NOM_LOC",
            "POBTOT",
            "P_0A2",
            "P_3A5",
            "P_60YMAS",
            "P18YM_PB",
            "P_18YMAS",
            "GRAPROES",
            "TVIVPARHAB",
        ],
    )


def get_census_level(
    census: pd.DataFrame,
    level: Literal["ageb", "block"],
) -> pd.DataFrame:
    if level == "ageb":
        return (
            census.query("NOM_LOC == 'Total AGEB urbana'")
            .assign(
                CVEGEO=lambda df: (
                    df["ENTIDAD"].astype(str).str.zfill(2)
                    + df["MUN"].astype(str).str.zfill(3)
                    + df["LOC"].astype(str).str.zfill(4)
                    + df["AGEB"].astype(str).str.zfill(4)
                ),
            )
            .drop(columns=["ENTIDAD", "MUN", "LOC", "AGEB", "NOM_LOC"])
            .set_index("CVEGEO")
            .replace("*", np.nan)
            .astype(float)
            .assign(
                P_0A5=lambda df: df["P_0A2"] + df["P_3A5"],
            )
        )

    if level == "block":
        return (
            census.assign(
                CVEGEO=lambda df: (
                    df["ENTIDAD"].astype(str).str.zfill(2)
                    + df["MUN"].astype(str).str.zfill(3)
                    + df["LOC"].astype(str).str.zfill(4)
                    + df["AGEB"].astype(str).str.zfill(4)
                    + df["MZA"].astype(str).str.zfill(3)
                ),
            )
            .drop(columns=["ENTIDAD", "MUN", "LOC", "AGEB", "MZA", "NOM_LOC"])
            .set_index("CVEGEO")
            .replace(["*", "N/D"], np.nan)
            .astype(float)
            .assign(
                P_0A5=lambda df: df["P_0A2"] + df["P_3A5"],
            )
        )

    err = f"Level '{level}' not recognized. Use 'ageb' or 'block'."
    raise ValueError(err)


def load_equip_df(path: os.PathLike, *, name: str) -> gpd.GeoDataFrame:
    return (
        gpd.read_file(path)
        .reset_index(drop=True)
        .filter(["geometry"])
        .to_crs("EPSG:6372")
        .assign(equipamiento=name)
    )
