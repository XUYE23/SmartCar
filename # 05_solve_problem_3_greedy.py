# 05_solve_problem_3_greedy.py
# Solves problem 3 using a Greedy Algorithm.

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils

def main():
    print("\n--- Starting Step 5: Path Optimization (Problem 3 - Greedy Algorithm) ---")

    X_COL_NAME = '栅格x坐标'
    Y_COL_NAME = '栅格y坐标'
    
    try:
        path_df = pd.read_excel(config.PATH_P5_P6_PATH)
        path_df.columns = path_df.columns.str.strip()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return
    except KeyError as e:
        print(f"Column name error: {e}. Please check if the Excel column names match the code.")
        return

    if len(path_df) < 2:
        print("Error: Path file must contain at least a start point and one waypoint.")
        return
        
    start_point_coords = path_df.iloc[0][[X_COL_NAME, Y_COL_NAME]].values
    waypoints_df = path_df.iloc[1:].copy()
    waypoints_coords = waypoints_df[[X_COL_NAME, Y_COL_NAME]].values
    
    n = len(waypoints_coords)
    
    # --- Greedy Algorithm Implementation ---
    
    optimal_headings = np.zeros(n, dtype=int)
    total_mileage = 0.0

    move_to_heading = {
        (0, 1): 0, (1, 1): 45, (1, 0): 90, (1, -1): 135,
        (0, -1): 180, (-1, -1): 225, (-1, 0): 270, (-1, 1): 315
    }

    # Step 1: Handle the first move from P5 to L1
    prev_p = start_point_coords
    curr_p = waypoints_coords[0] # L1
    
    dx, dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]
    
    if (dx, dy) not in move_to_heading:
        print(f"Error: Initial move from {prev_p} to {curr_p} is invalid.")
        return
        
    # For the first move, we assume no turn penalty (delta_theta = 0)
    delta_L = abs(dx) + abs(dy)
    omega = utils.get_omega(delta_L, 0)
    total_mileage += omega * config.CELLSIZE
    
    # The heading at L1 is determined by the move to get there
    prev_heading = move_to_heading[(dx, dy)]
    optimal_headings[0] = prev_heading

    # Step 2: Iterate through the rest of the waypoints
    for i in tqdm(range(1, n), desc="Greedy Path Calculation"):
        prev_p = waypoints_coords[i-1] # e.g., L1
        curr_p = waypoints_coords[i]   # e.g., L2

        dx, dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]
        
        if (dx, dy) not in move_to_heading:
            print(f"Error: Move from L{i} {prev_p} to L{i+1} {curr_p} is invalid.")
            return

        # Greedy choice: The current heading is fixed by the required move.
        # The cost is determined by the turn from the previous heading.
        h_curr = move_to_heading[(dx, dy)]
        
        delta_L = abs(dx) + abs(dy)
        angle_diff = abs(h_curr - prev_heading)
        delta_theta = min(angle_diff, 360 - angle_diff)
        
        # This is greedy because we commit to this turn without looking ahead.
        omega = utils.get_omega(delta_L, delta_theta)
        total_mileage += omega * config.CELLSIZE
        
        # Update state for the next iteration
        prev_heading = h_curr
        optimal_headings[i] = h_curr

    # --- End of Algorithm ---

    # Create and save results
    result_df = waypoints_df.copy()
    result_df['无人车车头方向(度)'] = optimal_headings
    result_df.to_excel("problem3_results_greedy.xlsx", index=False)
    
    print("\n--- Problem 3 Results (Greedy Algorithm) ---")
    print(f"Path found with total mileage: {total_mileage:.4f} meters")
    print(f"Greedy heading sequence saved to 'problem3_results_greedy.xlsx'")

if __name__ == "__main__":
    main()