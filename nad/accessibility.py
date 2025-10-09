import geopandas as gpd
import numpy as np
import pandana as pdna
import pandas as pd


def create_network(gdf_vialidades, results_path, max_distance):
    gdf_edges = gdf_vialidades.explode(index_parts=False)

    gdf_edges["u_geom"] = gdf_edges.geometry.apply(lambda geom: geom.coords[0])
    gdf_edges["v_geom"] = gdf_edges.geometry.apply(lambda geom: geom.coords[-1])

    gdf_nodes = pd.DataFrame(
        pd.concat([gdf_edges["u_geom"], gdf_edges["v_geom"]])
        .drop_duplicates()
        .tolist(),
        columns=["x", "y"],
    ).reset_index(drop=True)
    gdf_nodes["node_id"] = gdf_nodes.index

    coord_to_id = {
        (row.x, row.y): row.node_id for row in gdf_nodes.itertuples(index=False)
    }
    gdf_edges["u"] = gdf_edges["u_geom"].map(coord_to_id)
    gdf_edges["v"] = gdf_edges["v_geom"].map(coord_to_id)

    gdf_edges["distance"] = gdf_edges.geometry.length
    gdf_edges = gdf_edges[["u", "v", "distance"]].reset_index(drop=True)
    gdf_nodes = gdf_nodes[["node_id", "x", "y"]]

    net = pdna.Network(
        gdf_nodes["x"],
        gdf_nodes["y"],
        gdf_edges["u"],
        gdf_edges["v"],
        gdf_edges[["distance"]],
    )
    net.precompute(max_distance)
    net.save_hdf5(results_path / "network.h5")
    return net


def get_proximities(
    net: pdna.Network,
    coords: pd.DataFrame,
    poi_type: str,
    max_distance: float,
    num_pois: int = 1,
) -> pd.Series:
    net.set_pois(poi_type, max_distance, num_pois, coords["x"], coords["y"])
    res = (
        net.nearest_pois(
            distance=max_distance,
            category=poi_type,
            num_pois=num_pois,
            include_poi_ids=True,
        )
        .rename(columns={x: f"distance{x}" for x in range(1, num_pois + 1)})
        .reset_index(names="origin_id")
    )
    return pd.wide_to_long(
        res,
        stubnames=["distance", "poi"],
        i="origin_id",
        j="num_poi",
        sep="",
    ).rename(columns={"poi": "destination_id"})


def get_accessibility_metrics(
    net: pdna.Network,
    destinations: gpd.GeoDataFrame,
    weights: pd.Series,
    amenity: str,
    max_distance: float,
    num_pois: int,
    walk_speed: float,
    radius: float,
    adjustment_factor: float,
):
    proximities = get_proximities(
        net,
        destinations.get_coordinates(),
        amenity,
        max_distance=max_distance,
        num_pois=num_pois,
    )
    proximities["time"] = (proximities["distance"] / 1000) * 60 / walk_speed
    proximities = proximities.join(weights, on="origin_id")
    proximities = proximities.join(destinations[["capacity"]], on="destination_id")
    proximities["gravity"] = 1 / np.exp(
        1 / (radius / adjustment_factor) * proximities["distance"],
    )
    proximities["reach"] = proximities["gravity"] * proximities[weights.name]
    destinations = (
        proximities.groupby("destination_id")
        .agg({"reach": "sum", "capacity": "first"})
        .assign(
            opportunities_ratio=lambda df: df.apply(
                lambda x: x["capacity"] / x["reach"] if x["reach"] > 0 else 0,
                axis=1,
            ),
        )
        .query("opportunities_ratio > 0")
    )
    proximities["accessibility_score"] = proximities.apply(
        lambda x: destinations["opportunities_ratio"].loc[x["destination_id"]]
        * x["gravity"]
        if x["destination_id"] in destinations["opportunities_ratio"].index
        else 0,
        axis=1,
    )
    accessibility_scores = (
        proximities.reset_index()
        .groupby("origin_id")
        .agg({"accessibility_score": "sum", "time": "min"})
    )
    return accessibility_scores
