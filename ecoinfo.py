import rasterio
import pandas as pd

# ---------------------------------------------------------
# Efficient point extraction
# ---------------------------------------------------------


def extract_value(src, lat, lon):
    """Extract a single pixel value from a raster"""
    row, col = src.index(lon, lat)
    return src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]

# ---------------------------------------------------------
# Extract values for multiple rasters and coordinates
# ---------------------------------------------------------


def extract_multiple_to_df(raster_map, coordinates, named_mode=False, calculate_avgs=False, calculate_stdevs=False):
    """
    raster_map: dict of value_name -> raster_path
    coordinates: dict of label -> (lat, lon)

    Returns: pandas DataFrame with cities as rows, rasters as columns
    """
    data = []
    if (named_mode):
        for city_name, (lat, lon) in coordinates.items():
            row = {"city": city_name, "lat": lat, "lon": lon}
            for value_name, path in raster_map.items():
                with rasterio.open(path) as src:
                    row[value_name] = extract_value(src, lat, lon)
            data.append(row)
    else:
        for (lat, lon) in coordinates:
            row = {"lat": lat, "lon": lon}
            for value_name, path in raster_map.items():
                with rasterio.open(path) as src:
                    row[value_name] = extract_value(src, lat, lon)
            data.append(row)

    if (calculate_avgs):
        avg_row = {"city": "Average" if named_mode else "Average",
                   "lat": None, "lon": None}
        for value_name in raster_map.keys():
            avg_row[value_name] = sum(d[value_name] for d in data) / len(data)
        data.append(avg_row)
    if (calculate_stdevs):
        import statistics
        stdev_row = {"city": "Stdev" if named_mode else "Stdev",
                     "lat": None, "lon": None}
        for value_name in raster_map.keys():
            stdev_row[value_name] = statistics.stdev(
                d[value_name] for d in data)
        data.append(stdev_row)
    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------
# Get Parameters
# ---------------------------------------------------------
def get_default_params_map():
    """Get raster parameters like bounds, resolution, CRS"""
    return {
        "Wind efficiency": "data/USA_capacity-factor_IEC2.tif",
        "Solar power": "data/PVOUT.tif",
        "Fiber optics": "data/usa.tif",
        "Temperature": "data/TEMP.tif",
        "Popilation density": "data/usa_pd_2020_1km.tif",
    }


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
        raster_map, city_coords, named_mode=True, calculate_avgs=True, calculate_stdevs=True)

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
