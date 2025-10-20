import geopandas as gpd
import pandas as pd


def calculate_tdpa_exposure(
    df: gpd.GeoDataFrame,
    df_streets: gpd.GeoDataFrame,
    threshold_km: int = 500,
) -> pd.Series:
    df_temp = df.copy().reset_index()
    dist_cols = [f"distance_{row['ID']}" for index, row in df_streets.iterrows()]
    # get distance from blocks to all major streets
    df_temp[dist_cols] = df_temp.geometry.centroid.apply(
        lambda g: df_streets.geometry.distance(g),
    )

    df_temp = pd.melt(
        df_temp[["id", "geometry", *dist_cols]],
        id_vars=["id", "geometry"],
        var_name="station_id",
        value_name="distance_km",
    )
    df_temp["station_id"] = (
        df_temp["station_id"].replace({"distance_": ""}, regex=True).astype(int)
    )
    df_temp = df_temp.merge(
        df_streets[["ID", "TDPA_mean"]],
        left_on="station_id",
        right_on="ID",
        how="left",
    )

    results = {}
    for cve, group in df_temp.groupby("id"):
        close = group[group["distance_km"] <= threshold_km]

        if not close.empty:
            value = close["TDPA_mean"].sum() / threshold_km
        else:
            min_dist = group["distance_km"].min()
            closest = group[group["distance_km"] == min_dist]
            value = closest["TDPA_mean"].sum() / min_dist

        results[cve] = value

    return pd.Series(results)
