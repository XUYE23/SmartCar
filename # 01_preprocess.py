# 01_preprocess.py
# Reads the map, calculates terrain attributes, and saves them.

import numpy as np
import rasterio
from tqdm import tqdm
import config

def main():
    print("--- Starting Step 1: Terrain Preprocessing ---")
    
    try:
        with rasterio.open(config.MAP_ELEVATION_PATH) as src:
            elevation_map = src.read(1).astype(np.float32)
    except rasterio.errors.RasterioIOError:
        print(f"Error: Could not find or open map file at '{config.MAP_ELEVATION_PATH}'")
        return

    rows, cols = elevation_map.shape
    
    # Initialize attribute maps with float32 to save memory
    slope_map = np.zeros_like(elevation_map, dtype=np.float32)
    aspect_map = np.zeros_like(elevation_map, dtype=np.float32)
    nx_map = np.zeros_like(elevation_map, dtype=np.float32)
    ny_map = np.zeros_like(elevation_map, dtype=np.float32)
    nz_map = np.ones_like(elevation_map, dtype=np.float32) # Default is (0,0,1)

    eight_cellsize = 8 * config.CELLSIZE
    
    # Iterate over inner pixels
    for r in tqdm(range(1, rows - 1), desc="Preprocessing Map"):
        for c in range(1, cols - 1):
            # Extract 3x3 window
            z = elevation_map[r-1:r+2, c-1:c+2]
            a, b, _c = z[0, 0], z[0, 1], z[0, 2]
            d, _, f = z[1, 0], z[1, 1], z[1, 2]
            g, h, i = z[2, 0], z[2, 1], z[2, 2]

            # Calculate elevation change rates (dz/dx, dz/dy) with k-factor
            dz_dx = config.K_FACTOR * ((_c + 2*f + i) - (a + 2*d + g)) / eight_cellsize
            dz_dy = config.K_FACTOR * ((a + 2*b + _c) - (g + 2*h + i)) / eight_cellsize

            # Calculate slope
            grad = np.sqrt(dz_dx**2 + dz_dy**2)
            slope_map[r, c] = np.rad2deg(np.arctan(grad))

            # Calculate aspect
            if dz_dx == 0 and dz_dy == 0:
                 aspect_map[r, c] = -1 # Flat, special value
            else:
                aspect_rad = np.arctan2(dz_dx, dz_dy)
                aspect_deg = np.rad2deg(aspect_rad)
                # Convert from math angle to geographic angle (0 is North, clockwise)
                aspect_map[r, c] = (450 - aspect_deg) % 360

            # Calculate normal vector components
            # Normal vector is proportional to (-dz/dx, -dz/dy, 1)
            vec_mag = np.sqrt(dz_dx**2 + dz_dy**2 + 1)
            nx_map[r, c] = -dz_dx / vec_mag
            ny_map[r, c] = -dz_dy / vec_mag
            nz_map[r, c] = 1 / vec_mag

    # Save all processed maps into a single compressed file
    np.savez_compressed(
        config.PREPROCESSED_DATA_PATH,
        elevation=elevation_map,
        slope=slope_map,
        aspect=aspect_map,
        nx=nx_map,
        ny=ny_map,
        nz=nz_map
    )
    
    print(f"Preprocessing complete. Data saved to '{config.PREPROCESSED_DATA_PATH}'")

if __name__ == "__main__":
    main()