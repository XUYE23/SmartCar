# 04_solve_problem_3.py
# Solves problem 3: finds the optimal heading sequence for minimal mileage.

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils

def main():
    print("\n--- Starting Step 4: Path Optimization (Problem 3) ---")

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
        print(f"Actual path columns: {repr(path_df.columns.tolist())}")
        return

    if len(path_df) < 2:
        print("Error: Path file must contain at least a start point and one waypoint.")
        return
        
    start_point_coords = path_df.iloc[0][[X_COL_NAME, Y_COL_NAME]].values
    waypoints_df = path_df.iloc[1:].copy()
    waypoints_coords = waypoints_df[[X_COL_NAME, Y_COL_NAME]].values
    
    n = len(waypoints_coords)
    H = len(config.HEADINGS)

    dp = np.full((n, H), float('inf'))
    backpointer = np.full((n, H), -1, dtype=int)

    # --- Initialization for the P5 -> L1 move ---
    p_start = start_point_coords
    p_L1 = waypoints_coords[0]

    delta_L_init = abs(p_L1[0] - p_start[0]) + abs(p_L1[1] - p_start[1])
    omega_init = utils.get_omega(delta_L_init, 0) # Assume delta_theta=0 for the first move
    cost_init = omega_init * config.CELLSIZE
    
    if cost_init == float('inf'):
        print(f"Error: The initial move from P5 {p_start} to L1 {p_L1} is invalid (delta_L={delta_L_init}).")
        return

    # --- FIX: Define a precise mapping from move vector to required heading ---
    move_to_heading = {
        (0, 1): 0, (1, 1): 45, (1, 0): 90, (1, -1): 135,
        (0, -1): 180, (-1, -1): 225, (-1, 0): 270, (-1, 1): 315
    }

    # Populate the first row of dp table (costs to arrive at L1)
    # The heading must match the direction of the first move
    dx_init, dy_init = p_L1[0] - p_start[0], p_L1[1] - p_start[1]
    if (dx_init, dy_init) in move_to_heading:
        h_init = move_to_heading[(dx_init, dy_init)]
        h_init_idx = config.HEADINGS.index(h_init)
        dp[0, h_init_idx] = cost_init
    # --- END OF FIX ---

    # DP iterations for the rest of the path (L1 -> L2, L2 -> L3, ...)
    for i in tqdm(range(1, n), desc="Optimizing Path P5-P6"):
        prev_p = waypoints_coords[i-1] # e.g., L1
        curr_p = waypoints_coords[i]   # e.g., L2

        dx, dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]
        delta_L = abs(dx) + abs(dy)
        
        # A given move (dx, dy) determines a UNIQUE required heading
        if (dx, dy) not in move_to_heading:
            # This move is impossible (e.g., dx=2), so this row in dp will remain inf
            # This will be caught by the final check
            continue
            
        h_curr = move_to_heading[(dx, dy)]
        h_curr_idx = config.HEADINGS.index(h_curr)
        
        # We only need to calculate costs for this one possible current heading.
        # We iterate through all possible previous headings to find the cheapest way to get here.
        for h_prev_idx, h_prev in enumerate(config.HEADINGS):
            if dp[i-1, h_prev_idx] == float('inf'):
                continue

            min_angle_diff = min(abs(h_curr - h_prev), 360 - abs(h_curr - h_prev))
            if min_angle_diff > 90:
                continue

            omega = utils.get_omega(delta_L, min_angle_diff)
            if omega == float('inf'):
                # This case should not be reached if delta_L is always 1 or 2
                continue
            
            cost = omega * config.CELLSIZE
            new_total_cost = dp[i-1, h_prev_idx] + cost

            if new_total_cost < dp[i, h_curr_idx]:
                dp[i, h_curr_idx] = new_total_cost
                backpointer[i, h_curr_idx] = h_prev_idx
    
    # Backtracking
    optimal_headings = np.zeros(n, dtype=int)
    min_final_cost = np.min(dp[n-1, :])
    
    if min_final_cost == float('inf'):
        print("Error: No valid path could be found through the waypoints.")
        # Add some debug info to find where it failed
        for i in range(n):
            if np.all(dp[i,:] == float('inf')):
                print(f"Path became impossible at step {i+1} (from L{i} to L{i+1}).")
                p_prev = waypoints_coords[i-1] if i > 0 else start_point_coords
                p_curr = waypoints_coords[i]
                print(f"Move from {p_prev} to {p_curr} could not be completed from any previous valid state.")
                break
        return
        
    last_h_idx = np.argmin(dp[n-1, :])
    optimal_headings[n-1] = config.HEADINGS[last_h_idx]

    for i in range(n - 2, -1, -1):
        last_h_idx = backpointer[i + 1, last_h_idx]
        optimal_headings[i] = config.HEADINGS[last_h_idx]

    # Create and save results
    result_df = waypoints_df.copy() # Use copy to avoid SettingWithCopyWarning
    result_df['无人车车头方向(度)'] = optimal_headings
    result_df.to_excel(config.PROBLEM3_RESULTS_PATH, index=False)
    
    print("\n--- Problem 3 Results ---")
    print(f"Optimal path found with minimum mileage: {min_final_cost:.4f} meters")
    print(f"Optimal heading sequence saved to '{config.PROBLEM3_RESULTS_PATH}'")


if __name__ == "__main__":
    main()