"""Convert raw trip parquet files into centroid-coordinate parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import geopandas as gpd
import pandas as pd
from pyproj import Transformer


TRIP_REQUIRED_COLUMNS = (
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "total_amount",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw yellow taxi parquet files into location-centroid coordinates."
    )
    parser.add_argument(
        "--trip-parquet",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more raw trip parquet files.",
    )
    parser.add_argument(
        "--zones",
        required=True,
        metavar="PATH",
        help="Taxi zones file (GeoJSON/GeoParquet/Shapefile/zip path).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output parquet path (default: project's data/viagens_lat_long.parquet, "
            "using TVAE/config.py REAL_DATA_PATH."
        ),
    )
    parser.add_argument(
        "--timezone",
        default=None,
        help="Optional timezone to localize/convert pickup datetime.",
    )
    return parser.parse_args()


def _resolve_output_path(raw_output: str | None) -> Path:
    if raw_output:
        return Path(raw_output)
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / "data" / "viagens_lat_long.parquet").resolve()


def main() -> None:
    args = parse_args()
    start_all = time.time()

    # 1) Load raw trips and keep required columns.
    step_start = time.time()
    trip_paths = [Path(p) for p in args.trip_parquet]
    df = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in trip_paths],
        ignore_index=True,
    )
    missing_cols = [col for col in TRIP_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns missing in source parquet files: {missing_cols}")
    df = df[list(TRIP_REQUIRED_COLUMNS)].copy()
    print(f"[1] Trip data read time: {time.time() - step_start:.2f}s")

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    if args.timezone:
        df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].dt.tz_localize(
            args.timezone
        )

    monthly_counts = (
        df["tpep_pickup_datetime"]
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )
    print("\n# Rows per month")
    for month, n_rows in monthly_counts.items():
        print(f"{month}: {n_rows}")

    # 2) Read taxi-zone geometries.
    step_start = time.time()
    gdf = gpd.read_file(Path(args.zones))
    gdf = gdf[["LocationID", "geometry"]].dropna(subset=["LocationID", "geometry"])
    print(f"[2] Zones source CRS: {gdf.crs}")
    gdf = gdf.to_crs(epsg=4326)
    print(f"[2] Zones normalized CRS: {gdf.crs}")
    print(f"[2] Taxi-zone read time: {time.time() - step_start:.2f} seconds")

    # 3) Merge pickup and dropoff zones.
    step_start = time.time()
    df = df.merge(
        gdf,
        how="left",
        left_on="PULocationID",
        right_on="LocationID",
    ).rename(columns={"geometry": "PU_Geometry"})
    df = df.drop(columns=["LocationID"])

    df = df.merge(
        gdf,
        how="left",
        left_on="DOLocationID",
        right_on="LocationID",
    ).rename(columns={"geometry": "DO_Geometry"})
    df = df.drop(columns=["LocationID"])

    print(f"[3] Geometry merge time: {time.time() - step_start:.2f}s")

    # Keep only original fields + generated coordinates.
    df = df.drop(columns=[col for col in ("PU_Geometry", "DO_Geometry") if col in df.columns])

    # 4) Build centroid lookup for each zone.
    step_start = time.time()
    zones_3857 = gdf.to_crs("EPSG:3857")
    zones_3857["centroid"] = zones_3857.geometry.centroid
    mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    zone_lon_dict = {}
    zone_lat_dict = {}
    for zone_id, centroid in zip(zones_3857["LocationID"], zones_3857["centroid"]):
        zone_id_int = int(zone_id)
        if centroid is None or pd.isna(centroid.x) or pd.isna(centroid.y):
            zone_lon_dict[zone_id_int] = float("nan")
            zone_lat_dict[zone_id_int] = float("nan")
            continue

        lon, lat = mercator_to_wgs84.transform(float(centroid.x), float(centroid.y))
        zone_lon_dict[zone_id_int] = float(lon)
        zone_lat_dict[zone_id_int] = float(lat)

    print(f"[4] Zone centroid preprocessing time: {time.time() - step_start:.2f} seconds")

    # 5) Generate coordinate columns.
    step_start = time.time()
    trip_df = df.copy()
    trip_df["PU_longitude"] = trip_df["PULocationID"].map(zone_lon_dict)
    trip_df["PU_latitude"] = trip_df["PULocationID"].map(zone_lat_dict)
    trip_df["DO_longitude"] = trip_df["DOLocationID"].map(zone_lon_dict)
    trip_df["DO_latitude"] = trip_df["DOLocationID"].map(zone_lat_dict)

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trip_df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"[5] Coordinate generation + write time: {time.time() - step_start:.2f}s")
    print(f"Output written to: {output_path}")
    print(f"Total runtime: {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
