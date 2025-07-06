# 03_solve_problem_2.py
# Solves problem 2: validates a path for compliance.

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils

def build_turn_rules():
    # This function remains unchanged
    rules = {h: set() for h in config.HEADINGS}
    rules[0].add((0, 1, 0)); rules[90].add((1, 0, 90)); rules[180].add((0, -1, 180)); rules[270].add((-1, 0, 270))
    rules[45].add((1, 1, 45)); rules[135].add((1, -1, 135)); rules[225].add((-1, -1, 225)); rules[315].add((-1, 1, 315))
    for h in config.HEADINGS:
        h_left = (h - 45 + 360) % 360
        h_right = (h + 45) % 360
        if h in [0, 90, 180, 270]:
            dx, dy = {0:(0,1), 90:(1,0), 180:(0,-1), 270:(-1,0)}[h]
            rules[h].add((dx, dy, h_left)); rules[h].add((dx, dy, h_right))
        else:
            dx, dy = {45:(1,1), 135:(1,-1), 225:(-1,-1), 315:(-1,1)}[h]
            rules[h].add((dx, dy, h_left)); rules[h].add((dx, dy, h_right))
    return rules

def main():
    print("\n--- Starting Step 3: Path Validation (Problem 2) ---")
    
    # Define clean column names
    X_COL_NAME = '栅格x坐标'
    Y_COL_NAME = '栅格y坐标'
    HEADING_COL_NAME = '车头朝向(单位：度)'
    ID_COL_NAME = '编号'

    try:
        data = np.load(config.PREPROCESSED_DATA_PATH)
        slope_map = data['slope']
        path_df = pd.read_excel(config.PATH_P3_P4_PATH)

        # --- FIX: Clean column names to remove leading/trailing whitespace ---
        path_df.columns = path_df.columns.str.strip()

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return
    except KeyError as e:
        print(f"Column name error: {e}. Please check if the Excel column names match the code.")
        print(f"Expected path columns like: ['{ID_COL_NAME}', '{X_COL_NAME}', '{Y_COL_NAME}', '{HEADING_COL_NAME}']")
        print(f"Actual path columns: {path_df.columns.tolist()}")
        return

    turn_rules = build_turn_rules()
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
        
        move = (dx, dy, curr[HEADING_COL_NAME])
        
        if move not in turn_rules[prev[HEADING_COL_NAME]]:
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