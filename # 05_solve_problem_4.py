# 05_solve_problem_4.py
# Solves problem 4 by implementing a generic A* search algorithm.
# Version 5: Uses a more aggressive weight for the heuristic to ensure completion.

import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import config
import utils

# --- A* Core Algorithm and Helper functions (NO CHANGES FROM PREVIOUS VERSION) ---

def reconstruct_path(parents, current_state, start_coords):
    path = [];
    while current_state in parents: path.append(current_state); current_state = parents[current_state]
    path.append(current_state); path.reverse(); return path

def a_star_search(start_coords, end_coords, cost_function, heuristic_function, terrain_data, turn_logic, weight=1.0):
    open_set = []
    for h in config.HEADINGS:
        start_state = (start_coords[0], start_coords[1], h)
        g_cost = 0; h_cost = heuristic_function(start_state, end_coords)
        f_cost = g_cost + weight * h_cost
        heapq.heappush(open_set, (f_cost, g_cost, start_state))
    g_costs = {(start_coords[0], start_coords[1], h): 0 for h in config.HEADINGS}
    parents, closed_set = {}, set()
    pbar = tqdm(desc=f"A* Search {start_coords}=>{end_coords} (W={weight})")
    while open_set:
        pbar.update(1)
        _, g_cost, current_state = heapq.heappop(open_set)
        if current_state in closed_set: continue
        closed_set.add(current_state)
        if (current_state[0], current_state[1]) == end_coords:
            pbar.close(); print(f"Goal {end_coords} reached! Reconstructing path...")
            return reconstruct_path(parents, current_state, start_coords)
        h_prev = current_state[2]
        for move_def in turn_logic[h_prev].values():
            (dx, dy), possible_h_next_list = move_def
            x_next, y_next = current_state[0] + dx, current_state[1] + dy
            if not (0 <= x_next < config.MAP_DIMENSION and 0 <= y_next < config.MAP_DIMENSION): continue
            r_next, c_next = utils.coords_to_rc(x_next, y_next, config.MAP_DIMENSION)
            if terrain_data['slope'][r_next, c_next] > config.MAX_SLOPE_DEG: continue
            for h_next in possible_h_next_list:
                next_state = (x_next, y_next, h_next)
                if next_state in closed_set: continue
                step_cost = cost_function(current_state, next_state, terrain_data)
                new_g_cost = g_cost + step_cost
                if next_state not in g_costs or new_g_cost < g_costs[next_state]:
                    g_costs[next_state] = new_g_cost
                    h_cost = heuristic_function(next_state, end_coords)
                    f_cost = new_g_cost + weight * h_cost
                    heapq.heappush(open_set, (f_cost, new_g_cost, next_state))
                    parents[next_state] = current_state
    pbar.close(); print("No path found."); return None

def heuristic_null(state, end_coords): return 0
def heuristic_time(state, end_coords):
    dx = abs(state[0] - end_coords[0]); dy = abs(state[1] - end_coords[1])
    euclidean_dist_m = np.sqrt(dx**2 + dy**2) * config.CELLSIZE
    return euclidean_dist_m / (30.0 / 3.6)
def heuristic_mileage(state, end_coords):
    dx = abs(state[0] - end_coords[0]); dy = abs(state[1] - end_coords[1])
    return (dx + dy + (np.sqrt(2) - 2) * min(dx, dy)) * config.CELLSIZE
def cost_stability(state_prev, state_curr, terrain_data):
    r_prev, c_prev = utils.coords_to_rc(state_prev[0], state_prev[1], config.MAP_DIMENSION)
    r_curr, c_curr = utils.coords_to_rc(state_curr[0], state_curr[1], config.MAP_DIMENSION)
    s_prev = terrain_data['slope'][r_prev, c_prev]; s_curr = terrain_data['slope'][r_curr, c_curr]
    n_prev = np.array([terrain_data['nx'][r_prev, c_prev], terrain_data['ny'][r_prev, c_prev], terrain_data['nz'][r_prev, c_prev]])
    n_curr = np.array([terrain_data['nx'][r_curr, c_curr], terrain_data['ny'][r_curr, c_curr], terrain_data['nz'][r_curr, c_curr]])
    s_bar = (s_prev + s_curr) / 2.0; dot_product = np.clip(np.dot(n_prev, n_curr), -1.0, 1.0)
    return s_bar * np.arccos(dot_product)
def cost_time(state_prev, state_curr, terrain_data):
    mileage_cost = cost_mileage(state_prev, state_curr, terrain_data)
    r_curr, c_curr = utils.coords_to_rc(state_curr[0], state_curr[1], config.MAP_DIMENSION)
    velocity = utils.get_velocity_ms(terrain_data['slope'][r_curr, c_curr])
    return mileage_cost / velocity if velocity > 0 else float('inf')
def cost_mileage(state_prev, state_curr, terrain_data):
    dx = abs(state_curr[0] - state_prev[0]); dy = abs(state_curr[1] - state_prev[1])
    delta_L = dx + dy; h_prev = state_prev[2]; h_curr = state_curr[2]
    angle_diff = abs(h_curr - h_prev); delta_theta = min(angle_diff, 360 - angle_diff)
    return utils.get_omega(delta_L, delta_theta) * config.CELLSIZE

def build_turn_logic_for_astar():
    logic = {h: {} for h in config.HEADINGS}
    for h_prev in config.HEADINGS:
        s_rad = np.deg2rad(h_prev); s_dx, s_dy = int(round(np.sin(s_rad))), int(round(np.cos(s_rad)))
        s_h_curr_options = [(h_prev - 45 + 360) % 360, h_prev, (h_prev + 45 + 360) % 360]
        logic[h_prev]['straight'] = ((s_dx, s_dy), s_h_curr_options)
        h_turn_l = (h_prev - 45 + 360) % 360; l_rad = np.deg2rad(h_turn_l)
        l_dx, l_dy = int(round(np.sin(l_rad))), int(round(np.cos(l_rad)))
        l_h_curr_options = [(h_turn_l - 45 + 360) % 360, h_turn_l]
        logic[h_prev]['left_turn'] = ((l_dx, l_dy), l_h_curr_options)
        h_turn_r = (h_prev + 45 + 360) % 360; r_rad = np.deg2rad(h_turn_r)
        r_dx, r_dy = int(round(np.sin(r_rad))), int(round(np.cos(r_rad)))
        r_h_curr_options = [h_turn_r, (h_turn_r + 45 + 360) % 360]
        logic[h_prev]['right_turn'] = ((r_dx, r_dy), r_h_curr_options)
    return logic
def evaluate_final_path(path, terrain_data, bad_areas_set):
    total_mileage, total_time, total_stability, safety_time = 0, 0, 0, 0
    for i in range(1, len(path)):
        state_prev, state_curr = path[i-1], path[i]
        step_mileage = cost_mileage(state_prev, state_curr, terrain_data)
        step_time = cost_time(state_prev, state_curr, terrain_data)
        step_stability = cost_stability(state_prev, state_curr, terrain_data)
        total_mileage += step_mileage; total_time += step_time; total_stability += step_stability
        if (state_curr[0], state_curr[1]) in bad_areas_set: safety_time += step_time
    return {"平稳性(度·弧度)": total_stability, "里程(米)": total_mileage, "行驶时长(秒)": total_time, "安全性(秒)": safety_time}

# --- MAIN FUNCTION ---
def main():
    print("\n--- 开始执行步骤 5: 最优路径规划 (问题四) ---")
    start_time_total = time.time()
    
    try:
        print("正在将预处理的地形数据加载到内存中... (可能需要一些时间)")
        with np.load(config.PREPROCESSED_DATA_PATH) as data:
            terrain_data = {'slope': data['slope'][:], 'nx': data['nx'][:], 'ny': data['ny'][:], 'nz': data['nz'][:]}
        print("数据加载成功。")
        bad_areas_df = pd.read_excel(config.BAD_AREAS_PATH)
        bad_areas_df.columns = bad_areas_df.columns.str.strip()
        bad_areas_set = set(zip(bad_areas_df['栅格x坐标'], bad_areas_df['栅格y坐标']))
    except FileNotFoundError as e:
        print(f"错误: 未找到所需的数据文件: {e.filename}"); return

    turn_logic = build_turn_logic_for_astar()

    # --- FIX: Increased the heuristic weight for faster, near-optimal results ---
    AGGRESSIVE_WEIGHT = 3.0 # A higher value makes the search more "greedy"
    tasks = [
        {"name": "C3-Z4", "objective": "时效性最好", "start_coords": (5075, 6987), "end_coords": (8570, 4707),
         "cost_func": cost_time, "heuristic_func": heuristic_time, "output_file": "problem4_C3_Z4_results.xlsx", "weight": AGGRESSIVE_WEIGHT},
        {"name": "C5-Z7", "objective": "路程最短", "start_coords": (3279, 5169), "end_coords": (6907, 2189),
         "cost_func": cost_mileage, "heuristic_func": heuristic_mileage, "output_file": "problem4_C5_Z7_results.xlsx", "weight": AGGRESSIVE_WEIGHT},
        {"name": "C6-Z5", "objective": "平稳性最好", "start_coords": (5922, 4615), "end_coords": (8154, 3875),
         "cost_func": cost_stability, "heuristic_func": heuristic_null, "output_file": "problem4_C6_Z5_results.xlsx", "weight": 1.0}
    ]

    results_summary = []
    for task in tasks:
        end_coords = task.get('end_coords') # Simplified key access
        print(f"\n--- 正在执行任务: {task['name']} ({task['objective']}) ---")
        start_time_task = time.time()
        path = a_star_search(
            task['start_coords'], end_coords, task['cost_func'], task['heuristic_func'],
            terrain_data, turn_logic, weight=task['weight']
        )
        end_time_task = time.time()
        print(f"任务 {task['name']} 搜索完成，耗时 {end_time_task - start_time_task:.2f} 秒。")

        if path:
            path_df = pd.DataFrame([{'栅格x坐标': s[0], '栅格y坐标': s[1], '车头朝向(单位：度)': s[2]} for s in path])
            path_df.to_excel(task['output_file'], index=False)
            print(f"路径已保存至 '{task['output_file']}'")
            metrics = evaluate_final_path(path, terrain_data, bad_areas_set)
            metrics["路径起点-终点"] = task['name']
            results_summary.append(metrics)
        else:
            print(f"未能为任务 {task['name']} 找到路径。")

    if results_summary:
        print("\n\n--- 最终路径评估汇总 (表 3) ---")
        summary_df = pd.DataFrame(results_summary)[["路径起点-终点", "平稳性(度·弧度)", "里程(米)", "行驶时长(秒)", "安全性(秒)"]]
        print(summary_df.to_string())

    print(f"\n脚本总执行时间: {time.time() - start_time_total:.2f} 秒。")

if __name__ == "__main__":
    main()