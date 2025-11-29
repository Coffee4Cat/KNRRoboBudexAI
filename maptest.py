import rasterio

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
def extract_multiple(raster_map, coordinates):
    """
    raster_map: dict of value_name -> raster_path
    coordinates: dict of label -> (lat, lon)

    Returns: dict[value_name][label] = value
    """
    results = {}
    for value_name, path in raster_map.items():
        with rasterio.open(path) as src:
            results[value_name] = {}
            for label, (lat, lon) in coordinates.items():
                val = extract_value(src, lat, lon)
                results[value_name][label] = val
    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    raster_map = {
        "IEC2": "USA_capacity-factor_IEC2.tif",
        "power_density": "USA_power-density_50m.tif",
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

    results = extract_multiple(raster_map, city_coords)

    for value_name, city_vals in results.items():
        print(f"\nValues for {value_name}:")
        for city, val in city_vals.items():
            print(f"{city}: {val:.3f}")
