import os

import geopandas as gpd
import pandas as pd
import rasterio as rio
import rasterio.mask as rio_mask


def get_flooded_area(raster_path: os.PathLike, df: gpd.GeoDataFrame) -> pd.DataFrame:
    flooded_rows = []

    with rio.open(
        raster_path,
        "r",
    ) as ds:
        pixel_area = ds.res[0] * ds.res[1]

        df_reprojected = df[["geometry"]].to_crs(ds.crs)
        for idx, geom in df_reprojected["geometry"].items():
            masked, _ = rio_mask.mask(ds, [geom], crop=True)
            flooded_area = (masked == 5).sum() * pixel_area
            flooded_area_frac = flooded_area / geom.area
            flooded_rows.append(
                {
                    "index": idx,
                    "flooded_area": flooded_area,
                    "flooded_area_frac": flooded_area_frac,
                },
            )

    return pd.DataFrame(flooded_rows).set_index("index")
