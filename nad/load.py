import os
from collections.abc import Iterable
from pathlib import Path
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


def load_ageb_geometry(path: os.PathLike) -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            path,
        )
        .drop(columns=["POBTOT"])
        .set_index("CVEGEO")
    )


def load_block_geometry(
    path: os.PathLike,
    ageb_cvegeos: Iterable[str],  # noqa: ARG001
) -> gpd.GeoDataFrame:
    return (
        gpd.read_file(path)
        .assign(CVEGEO_AGEB=lambda x: x.CVEGEO.str[:13])
        .query("CVEGEO_AGEB in @ageb_cvegeos")
        .set_index("CVEGEO")
        .to_crs("EPSG:6372")
    )


def load_equip_df(
    path: os.PathLike,
    *,
    name: str | None = None,
    extra_cols: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    if extra_cols is None:
        extra_cols = []

    out = (
        gpd.read_file(path)
        .reset_index(drop=True)
        .filter(["geometry", *extra_cols])
        .to_crs("EPSG:6372")
    )

    if name is not None:
        out = out.assign(equipamiento=name)

    return out


def load_medical_equip_df(path: os.PathLike) -> pd.DataFrame:
    salud_equipamientos = {
        "CLÍNICA": "centro_salud",
        "HOSPITAL": "hospital",
        "CENTRO DE SALUD": "centro_salud",
    }

    return (
        gpd.read_file(path)
        .assign(equipamiento="unidad_medica")
        .reset_index(drop=True)
        .to_crs("EPSG:6372")
        .query("CATEGORIA == 'PÚBLICO'")
        .query("TIPO.isin(@salud_equipamientos.keys())")
        .assign(equipamiento=lambda df: df["TIPO"].map(salud_equipamientos))
        .filter(["geometry", "equipamiento"])
    )


def load_all_equipments(data_path: os.PathLike) -> pd.DataFrame:
    data_path = Path(data_path)

    return pd.concat(
        [
            load_equip_df(data_path / "datos" / "Guarderias", name="guarderia"),
            load_equip_df(data_path / "datos" / "PreescolarWgs84", name="preescolar"),
            load_equip_df(data_path / "datos" / "PrimariasWgs84", name="primaria"),
            load_equip_df(
                data_path / "datos" / "Secundarias_Wgs84",
                name="secundaria",
            ),
            load_equip_df(
                data_path / "datos" / "Preparatorias_Wgs84",
                name="preparatoria",
            ),
            load_equip_df(data_path / "datos" / "Superior_Wgs84", name="universidad"),
            load_equip_df(data_path / "datos" / "Auditorios_Wgs84", name="auditorio"),
            load_equip_df(data_path / "datos" / "Bibliotecas", name="biblioteca"),
            load_equip_df(data_path / "datos" / "CinesWGS84", name="cine"),
            load_equip_df(
                data_path / "datos" / "pparquesWgs84",
                name="parque",
                extra_cols=["CLASIFICAC"],
            )
            .query("CLASIFICAC == 'PARQUE'")
            .drop(columns=["CLASIFICAC"]),
            load_equip_df(
                data_path / "datos" / "Uni_DeportivasWgs84",
                name="unidad_deportiva",
            ),
            load_medical_equip_df(data_path / "datos" / "Unidad_Medica_Wgs84"),
        ],
        ignore_index=True,
    )
