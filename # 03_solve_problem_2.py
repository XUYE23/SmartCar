# 03_solve_problem_2.py (Corrected version with correct turn logic)
# Solves problem 2: validates a path for compliance.

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils

def build_correct_turn_rules():
    """
    Builds a correct and complete lookup table for valid state transitions.
    Returns a dictionary where rules[h_prev] is a set of valid (dx, dy, h_curr) tuples.
    """
    rules = {h: set() for h in config.HEADINGS}
    
    # Iterate through every possible starting heading
    for h_prev in config.HEADINGS:
        
        # 1. Consider a STRAIGHT move
        s_rad = np.deg2rad(h_prev)
        s_dx, s_dy = int(round(np.sin(s_rad))), int(round(np.cos(s_rad)))
        # After a straight move, 3 next headings are possible
        possible_h_curr_s = [(h_prev - 45 + 360) % 360, h_prev, (h_prev + 45 + 360) % 360]
        for h_curr in possible_h_curr_s:
            rules[h_prev].add((s_dx, s_dy, h_curr))
            
        # 2. Consider a 45-DEGREE LEFT turning move
        # The move itself is towards the direction (h_prev - 45)
        h_turn_l = (h_prev - 45 + 360) % 360
        l_rad = np.deg2rad(h_turn_l)
        l_dx, l_dy = int(round(np.sin(l_rad))), int(round(np.cos(l_rad)))
        # After a turning move, 2 next headings are possible
        possible_h_curr_l = [(h_turn_l - 45 + 360) % 360, h_turn_l]
        for h_curr in possible_h_curr_l:
            rules[h_prev].add((l_dx, l_dy, h_curr))
            
        # 3. Consider a 45-DEGREE RIGHT turning move
        h_turn_r = (h_prev + 45 + 360) % 360
        r_rad = np.deg2rad(h_turn_r)
        r_dx, r_dy = int(round(np.sin(r_rad))), int(round(np.cos(r_rad)))
        possible_h_curr_r = [h_turn_r, (h_turn_r + 45 + 360) % 360]
        for h_curr in possible_h_curr_r:
            rules[h_prev].add((r_dx, r_dy, h_curr))
            
    return rules

def main():
    print("\n--- Starting Step 3: Path Validation (Problem 2) ---")
    
    X_COL_NAME = '栅格x坐标'
    Y_COL_NAME = '栅格y坐标'
    HEADING_COL_NAME = '车头朝向(单位：度)'
    ID_COL_NAME = '编号'

    try:
        data = np.load(config.PREPROCESSED_DATA_PATH)
        slope_map = data['slope']
        path_df = pd.read_excel(config.PATH_P3_P4_PATH)
        path_df.columns = path_df.columns.str.strip()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return
    except KeyError as e:
        print(f"Column name error: {e}. Please check if the Excel column names match the code.")
        return

    # Use the new, correct function to build the rules
    turn_rules = build_correct_turn_rules()
    violations = []
    
    # 1. Check for slope violations
    for idx, row in tqdm(path_df.iterrows(), total=len(path_df), desc="Checking Slopes"):
        r, c = utils.coords_to_rc(row[X_COL_NAME], row[Y_COL_NAME], config.MAP_DIMENSION)
        if slope_map[r, c] > config.MAX_SLOPE_DEG:
            violations.append({
                '栅格编号1': row[ID_COL_NAME],
                '栅格编号2': '-',
                '错误类型': '超过最大通行坡度'
            })
            
    # 2. Check for turn/move violations
    for i in tqdm(range(1, len(path_df)), desc="Checking Turns"):
        prev = path_df.iloc[i-1]
        curr = path_df.iloc[i]
        
        dx = curr[X_COL_NAME] - prev[X_COL_NAME]
        dy = curr[Y_COL_NAME] - prev[Y_COL_NAME]
        
        h_prev = prev[HEADING_COL_NAME]
        h_curr = curr[HEADING_COL_NAME]

        # The state transition to check
        move = (dx, dy, h_curr)
        
        # Check if this transition is in the set of allowed moves from h_prev
        if move not in turn_rules[h_prev]:
            violations.append({
                '栅格编号1': prev[ID_COL_NAME],
                '栅格编号2': curr[ID_COL_NAME],
                '错误类型': '车头方向错误'
            })
            
    # Save results to Excel
    if violations:
        violations_df = pd.DataFrame(violations)
        violations_df.to_excel(config.PROBLEM2_RESULTS_PATH, index=False)
        print(f"{len(violations)} violations found. Results saved to '{config.PROBLEM2_RESULTS_PATH}'")
    else:
        print("No violations found in the path.")

if __name__ == "__main__":
    main()