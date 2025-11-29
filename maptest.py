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


def extract_multiple_to_df(raster_map, coordinates):
    """
    raster_map: dict of value_name -> raster_path
    coordinates: dict of label -> (lat, lon)

    Returns: pandas DataFrame with cities as rows, rasters as columns
    """
    data = []
    for city_name, (lat, lon) in coordinates.items():
        row = {"city": city_name, "lat": lat, "lon": lon}
        for value_name, path in raster_map.items():
            with rasterio.open(path) as src:
                row[value_name] = extract_value(src, lat, lon)
        data.append(row)

    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    raster_map = {
        "Wind efficiency": "USA_capacity-factor_IEC2.tif",
        "Solar power": "PVOUT.tif",
        "Fiber optics": "usa.tif",
    }

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

    df = extract_multiple_to_df(raster_map, city_coords)

    print(df)

    df.to_csv("city_raster_values.csv", index=False)
