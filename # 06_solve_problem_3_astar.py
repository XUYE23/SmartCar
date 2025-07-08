# 06_solve_problem_3_astar.py
# Solves problem 3 using the A* Algorithm.

import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import utils
import heapq # For the priority queue (Open List)
from collections import namedtuple

# Define a Node structure for the A* search
Node = namedtuple('Node', ['f_cost', 'g_cost', 'waypoint_index', 'heading', 'parent'])

def heuristic_cost_estimate(current_index, waypoints_coords, move_to_heading):
    """
    Heuristic function (h_cost): estimates the cost from current node to goal.
    It's an admissible heuristic as it calculates the cost with zero turn penalty.
    """
    h = 0.0
    for i in range(current_index, len(waypoints_coords) - 1):
        p1 = waypoints_coords[i]
        p2 = waypoints_coords[i+1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        delta_L = abs(dx) + abs(dy)
        omega = utils.get_omega(delta_L, 0) # Minimum cost for this move
        h += omega * config.CELLSIZE
    return h

def main():
    print("\n--- Starting Step 6: Path Optimization (Problem 3 - A* Algorithm) ---")

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

    move_to_heading = {
        (0, 1): 0, (1, 1): 45, (1, 0): 90, (1, -1): 135,
        (0, -1): 180, (-1, -1): 225, (-1, 0): 270, (-1, 1): 315
    }

    # --- A* Algorithm Implementation ---
    
    open_list = [] # Priority queue
    closed_list = set() # Set of (waypoint_index, heading) tuples

    # Initialization: P5 -> L1
    p_start = start_point_coords
    p_L1 = waypoints_coords[0]
    dx_init, dy_init = p_L1[0] - p_start[0], p_L1[1] - p_start[1]

    if (dx_init, dy_init) not in move_to_heading:
        print(f"Error: Initial move from {p_start} to {p_L1} is invalid.")
        return

    h_L1 = move_to_heading[(dx_init, dy_init)]
    delta_L_init = abs(dx_init) + abs(dy_init)
    omega_init = utils.get_omega(delta_L_init, 0)
    g_cost_init = omega_init * config.CELLSIZE
    h_cost_init = heuristic_cost_estimate(0, waypoints_coords, move_to_heading)
    
    start_node = Node(g_cost_init + h_cost_init, g_cost_init, 0, h_L1, None)
    heapq.heappush(open_list, start_node)

    final_node = None

    with tqdm(total=n, desc="A* Path Search") as pbar:
        while open_list:
            current_node = heapq.heappop(open_list)
            
            pbar.n = current_node.waypoint_index + 1
            pbar.refresh()
            
            if (current_node.waypoint_index, current_node.heading) in closed_list:
                continue
            
            closed_list.add((current_node.waypoint_index, current_node.heading))

            # Goal check
            if current_node.waypoint_index == n - 1:
                final_node = current_node
                pbar.n = n # Ensure progress bar completes
                pbar.refresh()
                break

            # Expand neighbors
            i = current_node.waypoint_index
            prev_p = waypoints_coords[i]
            curr_p = waypoints_coords[i+1]
            dx, dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]

            if (dx, dy) not in move_to_heading:
                continue

            h_next = move_to_heading[(dx, dy)]
            
            # Since the next heading is fixed, we only have one neighbor to consider
            h_prev = current_node.heading
            delta_L = abs(dx) + abs(dy)
            angle_diff = abs(h_next - h_prev)
            delta_theta = min(angle_diff, 360 - angle_diff)

            if delta_theta > 90:
                continue
            
            move_cost = utils.get_omega(delta_L, delta_theta) * config.CELLSIZE
            
            g_cost_new = current_node.g_cost + move_cost
            h_cost_new = heuristic_cost_estimate(i + 1, waypoints_coords, move_to_heading)
            f_cost_new = g_cost_new + h_cost_new
            
            neighbor_node = Node(f_cost_new, g_cost_new, i + 1, h_next, current_node)
            heapq.heappush(open_list, neighbor_node)

    # --- End of Algorithm ---

    if final_node is None:
        print("Error: A* algorithm could not find a valid path.")
        return

    # Backtrack to find the path
    optimal_headings = np.zeros(n, dtype=int)
    current = final_node
    while current is not None:
        optimal_headings[current.waypoint_index] = current.heading
        current = current.parent

    result_df = waypoints_df.copy()
    result_df['无人车车头方向(度)'] = optimal_headings
    result_df.to_excel("problem3_results_astar.xlsx", index=False)
    
    print("\n--- Problem 3 Results (A* Algorithm) ---")
    print(f"Path found with total mileage: {final_node.g_cost:.4f} meters")
    print(f"A* heading sequence saved to 'problem3_results_astar.xlsx'")

if __name__ == "__main__":
    main()