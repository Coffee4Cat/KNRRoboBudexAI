import rasterio

# ---------------------------------------------------------
# Efficient point extraction
# ---------------------------------------------------------


def extract_value(src, lat, lon):
    """Extract a single pixel value from a raster without loading the whole file."""
    row, col = src.index(lon, lat)
    return src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    raster_path = "USA_capacity-factor_IEC2.tif"

    # Example U.S. cities (lat, lon)
    city_coords = {
        "Dallas, TX": (32.7767, -96.7970),
        "New York, NY": (40.7128, -74.0060),
        "Los Angeles, CA": (34.0522, -118.2437),
        "Chicago, IL": (41.8781, -87.6298),
        "Miami, FL": (25.7617, -80.1918)
    }

    # Extract values efficiently
    cities_with_values = []
    with rasterio.open(raster_path) as src:
        for city, (lat, lon) in city_coords.items():
            val = extract_value(src, lat, lon)
            cities_with_values.append((lon, lat, val))
            print(f"{city}: {val:.3f}")
