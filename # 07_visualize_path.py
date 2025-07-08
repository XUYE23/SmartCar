# 07_visualize_path.py
# Visualizes the optimized path on the terrain map.

import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import config

def main():
    print("\n--- Starting Step 7: Path Visualization ---")

    OPTIMIZED_PATH_FILE = "problem3_results_astar.xlsx" 
    
    X_COL_NAME = '栅格x坐标'
    Y_COL_NAME = '栅格y坐标'

    try:
        with rasterio.open(config.MAP_ELEVATION_PATH) as src:
            elevation_map = src.read(1)
            rows, cols = elevation_map.shape

        path_df = pd.read_excel(OPTIMIZED_PATH_FILE)
        path_df.columns = path_df.columns.str.strip()

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return
    
    start_point_p5 = {'x': 4698, 'y': 6162} 
    
    x_coords = [start_point_p5['x']] + path_df[X_COL_NAME].tolist()
    y_coords = [start_point_p5['y']] + path_df[Y_COL_NAME].tolist()

    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(15, 15))

    # 1. Plot the terrain map as a background
    im = ax.imshow(
        elevation_map, 
        cmap='terrain', 
        extent=[0, cols, 0, rows], 
        origin='lower'
    )

    # 2. Plot the optimized path over the map
    ax.plot(
        x_coords, 
        y_coords, 
        color='red', 
        linewidth=2.5, # Increased linewidth for better visibility
        marker='.',    # Add small markers for each waypoint
        markersize=5,
        label='Optimized Path (P5 to P6)'
    )

    # 3. Highlight Start and End points
    ax.scatter(
        x_coords[0], y_coords[0], 
        color='cyan', s=200, zorder=5, edgecolors='black', label='Start (P5)' # Changed color for better contrast
    )
    ax.scatter(
        x_coords[-1], y_coords[-1], 
        color='magenta', s=200, zorder=5, edgecolors='black', label='End (P6)' # Changed color for better contrast
    )

    # 4. Add a colorbar for elevation reference
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Elevation (m)', fontsize=12)

    # 5. Add titles and labels
    ax.set_title('Zoomed-in View of Optimized Path on Terrain Map', fontsize=20)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- FIX: Set plot limits to zoom in on the path area ---
    buffer = 100  # Set a buffer of 100 units (meters) around the path
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    # --- END OF FIX ---
    
    # 6. Save the figure
    output_filename = 'path_on_map_visualization_zoomed.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Visualization complete. Zoomed-in plot saved to '{output_filename}'")

if __name__ == "__main__":
    main()