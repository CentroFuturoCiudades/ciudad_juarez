import geopandas as gpd
import pandas as pd


def intersect_gdf_with_equipments(
    df: gpd.GeoDataFrame,
    df_equip: gpd.GeoDataFrame,
) -> pd.DataFrame:
    index_name = str(df.index.name) if df.index.name else "index"
    return (
        df[["geometry"]]
        .sjoin(df_equip, how="inner", predicate="contains")
        .reset_index(names="index_col")
        .groupby(["index_col", "equipamiento"])
        .size()
        .reset_index()
        .pivot_table(
            index="index_col",
            columns="equipamiento",
            values=0,
            fill_value=0,
        )
        .rename_axis(index=index_name)
        .fillna(0)
    )
