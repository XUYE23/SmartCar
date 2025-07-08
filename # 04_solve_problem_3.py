# 04_solve_problem_3.py (Corrected version with full DP logic and fixed indentation)
# Solves problem 3 by correctly modeling the state transitions.

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils

def build_turn_logic():
    """
    Builds a lookup table for vehicle movement rules based on Attachment 3.
    Returns a dictionary: logic[h_prev][(dx, dy)] = list_of_possible_h_curr
    """
    logic = {h: {} for h in config.HEADINGS}
    
    for h_prev in config.HEADINGS:
        # 1. Straight move
        s_rad = np.deg2rad(h_prev)
        s_dx, s_dy = int(round(np.sin(s_rad))), int(round(np.cos(s_rad)))
        s_h_curr_options = [(h_prev - 45 + 360) % 360, h_prev, (h_prev + 45 + 360) % 360]
        logic[h_prev][(s_dx, s_dy)] = s_h_curr_options

        # 2. Left-turning move
        h_turn_l = (h_prev - 45 + 360) % 360
        l_rad = np.deg2rad(h_turn_l)
        l_dx, l_dy = int(round(np.sin(l_rad))), int(round(np.cos(l_rad)))
        l_h_curr_options = [(h_turn_l - 45 + 360) % 360, h_turn_l]
        logic[h_prev][(l_dx, l_dy)] = l_h_curr_options
            
        # 3. Right-turning move
        h_turn_r = (h_prev + 45 + 360) % 360
        r_rad = np.deg2rad(h_turn_r)
        r_dx, r_dy = int(round(np.sin(r_rad))), int(round(np.cos(r_rad)))
        r_h_curr_options = [h_turn_r, (h_turn_r + 45 + 360) % 360]
        logic[h_prev][(r_dx, r_dy)] = r_h_curr_options
            
    return logic


def main():
    print("\n--- Starting Step 4: Path Optimization (Problem 3 - Correct DP) ---")

    X_COL_NAME = '栅格x坐标'
    Y_COL_NAME = '栅格y坐标'
    
    try:
        path_df = pd.read_excel(config.PATH_P5_P6_PATH)
        path_df.columns = path_df.columns.str.strip()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return

    start_point_coords = path_df.iloc[0][[X_COL_NAME, Y_COL_NAME]].values
    waypoints_df = path_df.iloc[1:].copy()
    waypoints_coords = waypoints_df[[X_COL_NAME, Y_COL_NAME]].values
    
    n = len(waypoints_coords)
    H = len(config.HEADINGS)

    dp = np.full((n, H), float('inf'))
    backpointer = np.full((n, H), -1, dtype=int)

    turn_logic = build_turn_logic()
    
    # --- Initialization for the P5 -> L1 move ---
    p_start = start_point_coords
    p_L1 = waypoints_coords[0]
    dx_init, dy_init = p_L1[0] - p_start[0], p_L1[1] - p_start[1]
    
    # Assumption: The first move's turn penalty is zero (delta_theta = 0).
    delta_L_init = abs(dx_init) + abs(dy_init)
    cost_init = utils.get_omega(delta_L_init, 0) * config.CELLSIZE
    
    if cost_init == float('inf'):
        print(f"Error: The initial move from P5 {p_start} to L1 {p_L1} is invalid.")
        return

    # To get to L1, we must have made the move (dx_init, dy_init).
    # We must find all h_prev at P5 that could have resulted in this move.
    # For each such h_prev, we find its possible h_curr options at L1.
    for h_prev in config.HEADINGS:
        # Check if the required initial move is a valid one from this h_prev
        if (dx_init, dy_init) in turn_logic[h_prev]:
            # If so, get the list of possible next headings at L1
            possible_h_curr_list = turn_logic[h_prev][(dx_init, dy_init)]
            
            for h_curr in possible_h_curr_list:
                h_curr_idx = config.HEADINGS.index(h_curr)
                # The cost is the same for all initial possibilities
                dp[0, h_curr_idx] = cost_init
        
    # --- Corrected DP iterations ---
    for i in tqdm(range(1, n), desc="Optimizing Path (Correct DP)"):
        prev_p = waypoints_coords[i-1]
        curr_p = waypoints_coords[i]
        required_dx, required_dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]
        
        for h_prev_idx, h_prev in enumerate(config.HEADINGS):
            if dp[i-1, h_prev_idx] == float('inf'):
                continue

            if (required_dx, required_dy) not in turn_logic[h_prev]:
                continue
            
            possible_h_curr_list = turn_logic[h_prev][(required_dx, required_dy)]
            
            for h_curr in possible_h_curr_list:
                h_curr_idx = config.HEADINGS.index(h_curr)
                
                delta_L = abs(required_dx) + abs(required_dy)
                angle_diff = abs(h_curr - h_prev)
                delta_theta = min(angle_diff, 360 - angle_diff)
                
                cost = utils.get_omega(delta_L, delta_theta) * config.CELLSIZE
                new_total_cost = dp[i-1, h_prev_idx] + cost

                if new_total_cost < dp[i, h_curr_idx]:
                    dp[i, h_curr_idx] = new_total_cost
                    backpointer[i, h_curr_idx] = h_prev_idx

    # Backtracking (this part remains the same)
    optimal_headings = np.zeros(n, dtype=int)
    min_final_cost = np.min(dp[n-1, :])
    
    if min_final_cost == float('inf'):
        print("Error: No valid path could be found through the waypoints.")
        return
        
    last_h_idx = np.argmin(dp[n-1, :])
    optimal_headings[n-1] = config.HEADINGS[last_h_idx]

    for i in range(n - 2, -1, -1):
        last_h_idx = backpointer[i + 1, last_h_idx]
        optimal_headings[i] = config.HEADINGS[last_h_idx]

    result_df = waypoints_df.copy()
    result_df['无人车车头方向(度)'] = optimal_headings
    result_df.to_excel("problem3_results_correct_dp.xlsx", index=False)
    
    print("\n--- Problem 3 Results (Correct DP) ---")
    print(f"Optimal path found with minimum mileage: {min_final_cost:.4f} meters")
    print(f"Optimal heading sequence saved to 'problem3_results_correct_dp.xlsx'")

if __name__ == "__main__":
    main()