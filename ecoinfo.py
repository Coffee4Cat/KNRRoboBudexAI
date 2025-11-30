import rasterio
import pandas as pd
import random
import statistics
# ---------------------------------------------------------
# Efficient point extraction
# ---------------------------------------------------------


def extract_value(src, lat, lon):
    """Extract a single pixel value from a raster"""
    try:
        row, col = src.index(lon, lat)
    except Exception:
        return None

    if row < 0 or col < 0 or row >= src.height or col >= src.width:
        return None

    value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]

    # check against raster's NoData
    if src.nodata is not None and value == src.nodata:
        return None

    return value
# ---------------------------------------------------------
# Extract values for multiple rasters and coordinates
# ---------------------------------------------------------


def extract_multiple_to_df(
    raster_map,
    coordinates,
    named_mode=False,
    calculate_avgs=False,
    calculate_stdevs=False
):
    """
    raster_map: dict of value_name -> raster_path
    coordinates: dict(label -> (lat, lon)) OR list of (lat, lon)
    """
    rows = []

    if False:
        coord_iter = ((label, *coords)
                      for label, coords in coordinates.items())
    else:
        coord_iter = ((None, *coords) for coords in coordinates)

    # Extract values
    for label, lat, lon in coord_iter:
        row = {"lat": lat, "lon": lon}
        if label is not None:
            row["city"] = label

        for value_name, path in raster_map.items():
            with rasterio.open(path) as src:
                row[value_name] = extract_value(src, lat, lon)

        rows.append(row)

    # Build DF
    df = pd.DataFrame(rows)

    # Drop rows with any None (invalid raster hits)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    # Compute stats only on surviving rows
    if calculate_avgs or calculate_stdevs:
        # Pull cleaned values
        cleaned = {name: df[name].tolist() for name in raster_map}

        if calculate_avgs:
            for name, vals in cleaned.items():
                df[name + "_avg"] = sum(vals) / len(vals) if vals else None

        if calculate_stdevs:
            for name, vals in cleaned.items():
                df[name + "_stdev"] = (
                    statistics.stdev(vals) if len(vals) > 1 else None
                )

    return df
# ---------------------------------------------------------
# Get Parameters
# ---------------------------------------------------------


def get_default_params_map():
    """Get raster parameters like bounds, resolution, CRS"""
    return {
        "Wind efficiency": "data/USA_power-density_10m.tif",
        "Solar power": "data/PVOUT.tif",
        "Fiber optics": "data/usa.tif",
        "Temperature": "data/TEMP.tif",
        "Popilation density": "data/usa_pd_2020_1km.tif",
    }


def generate_random_points(n, north, east, south, west):
    """
    Returns a list of (lat, lon) random points inside the bounding box.
    north: max latitude
    south: min latitude
    east:  max longitude
    west:  min longitude
    """
    points = []
    for _ in range(n):
        lat = random.uniform(south, north)
        lon = random.uniform(west, east)
        points.append((lat, lon))
    return points


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    raster_map = get_default_params_map()

    city_coords = {
        "Dallas, TX": (32.7767, -96.7970),
        "New York, NY": (40.7128, -74.0060),
        "Los Angeles, CA": (34.0522, -118.2437),
        "Chicago, IL": (41.8781, -87.6298),
        "Miami, FL": (25.7617, -80.1918),
        "Moab , UT": (38.5733, -109.5498),
        "Many Farms, AZ": (35.2854, -109.5455),
        "Truth or Consequences, NM": (33.1286, -107.2522),
    }

    df = extract_multiple_to_df(
        raster_map, city_coords, named_mode=True)
    print(df)

    coordinates_nameless = [
        (32.7767, -96.7970),
        (40.7128, -74.0060),
        (34.0522, -118.2437),
        (41.8781, -87.6298),
        (25.7617, -80.1918),
        (38.5733, -109.5498),
        (35.2854, -109.5455),
        (33.1286, -107.2522),
    ]
    df2 = extract_multiple_to_df(raster_map, coordinates_nameless)
    print(df2)
    # df.to_csv("city_raster_values.csv", index=False)

    random_points = generate_random_points(
        n=5000,             # or any number you want
        north=49.38,
        south=24.52,
        west=-124.78,
        east=-66.95
    )

    df_random = extract_multiple_to_df(
        raster_map,
        random_points,
        named_mode=False,
        calculate_avgs=True,
        calculate_stdevs=True
    )

    for name in raster_map:
        vals = df_random[name].tolist()
        # Remove None values
        vals = [v for v in vals if v is not None]
        if vals:
            avg = sum(vals) / len(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"{name}: avg = {avg:.3f}, stdev = {stdev:.3f}")
        else:
            print(f"{name}: no valid points")
