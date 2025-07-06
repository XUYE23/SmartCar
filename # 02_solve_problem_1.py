# 02_solve_problem_1.py
# Solves problem 1: evaluates a given path.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
import utils

def main():
    print("\n--- Starting Step 2: Path Evaluation (Problem 1) ---")

    X_COL_NAME = '栅格x坐标'
    Y_COL_NAME = '栅格y坐标'
    HEADING_COL_NAME = '车头朝向(单位：度)'
    
    try:
        data = np.load(config.PREPROCESSED_DATA_PATH)
        slope_map = data['slope']
        elevation_map = data['elevation']
        nx_map, ny_map, nz_map = data['nx'], data['ny'], data['nz']

        path_df = pd.read_excel(config.PATH_P1_P2_PATH)
        bad_areas_df = pd.read_excel(config.BAD_AREAS_PATH)

        path_df.columns = path_df.columns.str.strip()
        bad_areas_df.columns = bad_areas_df.columns.str.strip()

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return
    except KeyError as e:
        print(f"Column name error: {e}. Please check if the Excel column names match the code.")
        return

    bad_areas_set = set(zip(bad_areas_df[X_COL_NAME], bad_areas_df[Y_COL_NAME]))
    
    total_distance = 0.0
    total_time = 0.0
    total_epsilon = 0.0
    safety_time = 0.0
    
    # --- FIX: Initialize all plot lists correctly ---
    time_series = [0.0]
    distance_series = [0.0]
    elevation_series = []
    velocity_series = []
    slope_series = []
    
    # Process first point for all series to ensure matching lengths
    x0, y0 = path_df.iloc[0][X_COL_NAME], path_df.iloc[0][Y_COL_NAME]
    r0, c0 = utils.coords_to_rc(x0, y0, config.MAP_DIMENSION)
    s0 = slope_map[r0, c0]
    
    elevation_series.append(elevation_map[r0, c0])
    slope_series.append(s0)
    velocity_series.append(utils.get_velocity_ms(s0) * 3.6)

    path_len = len(path_df)
    for i in tqdm(range(1, path_len), desc="Evaluating Path P1-P2"):
        prev = path_df.iloc[i-1]
        curr = path_df.iloc[i]
        
        r_curr, c_curr = utils.coords_to_rc(curr[X_COL_NAME], curr[Y_COL_NAME], config.MAP_DIMENSION)
        r_prev, c_prev = utils.coords_to_rc(prev[X_COL_NAME], prev[Y_COL_NAME], config.MAP_DIMENSION)

        s_curr = slope_map[r_curr, c_curr]
        s_prev = slope_map[r_prev, c_prev]
        
        n_curr = np.array([nx_map[r_curr, c_curr], ny_map[r_curr, c_curr], nz_map[r_curr, c_curr]])
        n_prev = np.array([nx_map[r_prev, c_prev], ny_map[r_prev, c_prev], nz_map[r_prev, c_prev]])

        delta_L = abs(curr[X_COL_NAME] - prev[X_COL_NAME]) + abs(curr[Y_COL_NAME] - prev[Y_COL_NAME])
        
        h_curr = curr[HEADING_COL_NAME]
        h_prev = prev[HEADING_COL_NAME]
        angle_diff = abs(h_curr - h_prev)
        delta_theta = min(angle_diff, 360 - angle_diff)
        
        omega = utils.get_omega(delta_L, delta_theta)
        
        D_i = omega * config.CELLSIZE
        v_i_ms = utils.get_velocity_ms(s_curr)
        T_i = D_i / v_i_ms if v_i_ms > 0 else float('inf')
        
        s_bar_i = (s_prev + s_curr) / 2.0
        dot_product = np.clip(np.dot(n_curr, n_prev), -1.0, 1.0)
        phi_i = np.arccos(dot_product)
        epsilon_i = s_bar_i * phi_i
        
        total_distance += D_i
        total_time += T_i
        total_epsilon += epsilon_i
        if (curr[X_COL_NAME], curr[Y_COL_NAME]) in bad_areas_set:
            safety_time += T_i
        
        # Append to plot lists
        distance_series.append(total_distance)
        time_series.append(total_time)
        elevation_series.append(elevation_map[r_curr, c_curr])
        slope_series.append(s_curr)
        velocity_series.append(v_i_ms * 3.6)

    print("\n--- Problem 1 Results ---")
    print(f"Total Mileage: {total_distance / 1000:.4f} km")
    print(f"Total Time ( 时效性 ): {total_time / 3600:.4f} hours")
    print(f"Total Stability ( 平稳性 ): {total_epsilon:.4f}")
    print(f"Total Time in Bad Areas ( 安全性 ): {safety_time:.4f} seconds")
    
    # --- FIX: Plot 4 subplots ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)
    fig.suptitle('Path P1-P2 Performance Analysis', fontsize=16)

    # Time vs. Mileage
    axs[0].plot(distance_series, time_series, 'b-')
    axs[0].set_ylabel('Cumulative Time (s)')
    axs[0].set_title('Time vs. Mileage')
    axs[0].grid(True)
    
    # Elevation vs. Mileage
    # Note: distance_series has n+1 points, others have n. We plot against the correct slice.
    axs[1].plot(distance_series, elevation_series, 'g-')
    axs[1].set_ylabel('Elevation (m)')
    axs[1].set_title('Elevation vs. Mileage')
    axs[1].grid(True)
    
    # --- NEW PLOT: Slope vs. Mileage ---
    axs[2].plot(distance_series, slope_series, 'm-')
    axs[2].set_ylabel('Instantaneous Slope (degrees)')
    axs[2].set_title('Slope vs. Mileage')
    axs[2].grid(True)

    # Velocity vs. Mileage
    axs[3].plot(distance_series, velocity_series, 'r-')
    axs[3].set_xlabel('Cumulative Mileage (m)')
    axs[3].set_ylabel('Instantaneous Velocity (km/h)')
    axs[3].set_title('Velocity vs. Mileage')
    axs[3].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(config.PROBLEM1_PLOT_PATH)
    print(f"Plots saved to '{config.PROBLEM1_PLOT_PATH}'")

if __name__ == "__main__":
    main()