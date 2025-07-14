# 06_solve_problem_5.py
# Solves problem 5: Location-Routing Problem with battery constraints.
# Version 4.1: Uncompressed and fully formatted final version.

import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
import os
from itertools import combinations
import math

# Assuming config.py and utils.py are in the same directory
import config
import utils

# --- Data Definitions (from problem description) ---
LOCATIONS = {
    'Z1': (10996, 6648), 'Z2': (9979, 5931), 'Z3': (9564, 4823),
    'Z4': (8570, 4707), 'Z5': (8154, 3875), 'Z6': (7739, 3390),
    'Z7': (6907, 2189), 'Z8': (6560, 1173), 'Z9': (7054, 410),
    'C1': (5333, 8223), 'C2': (8039, 7941), 'C3': (5075, 6987),
    'C4': (4698, 6162), 'C5': (3279, 5169), 'C6': (5922, 4615),
    'C7': (4305, 3413), 'C8': (3265, 2674), 'C9': (4166, 780),
}
WAREHOUSE_CAPACITIES = {
    'C1': 600, 'C2': 800, 'C3': 800, 'C4': 800, 'C5': 800,
    'C6': 800, 'C7': 600, 'C8': 800, 'C9': 500,
}
DAILY_DEMANDS = {
    'Z1': 510, 'Z2': 450, 'Z3': 285, 'Z4': 450, 'Z5': 675,
    'Z6': 450, 'Z7': 285, 'Z8': 675, 'Z9': 450,
}
PROBLEM_DEMANDS = {z: d * 1.5 for z, d in DAILY_DEMANDS.items()}
VEHICLE_CAPACITY = 500
BATTERY_CAPACITY = 100.0
COST_MATRIX_PATH = "p5_cost_matrix.json"
A_STAR_WEIGHT = 3.0
A_STAR_MAX_ITERATIONS = 20_000_000

# --- A* and Helper Functions ---
def get_consumption_rate(slope_deg):
    if 0 <= slope_deg < 10:
        return 1.0
    elif 10 <= slope_deg < 20:
        return 1.5
    elif 20 <= slope_deg <= 30:
        return 2.0
    else:
        return float('inf')

def cost_time_and_battery(state_prev, state_curr, terrain_data):
    dx = abs(state_curr[0] - state_prev[0])
    dy = abs(state_curr[1] - state_prev[1])
    delta_L = dx + dy
    
    h_prev = state_prev[2]
    h_curr = state_curr[2]
    angle_diff = abs(h_curr - h_prev)
    delta_theta = min(angle_diff, 360 - angle_diff)
    
    mileage_cost = utils.get_omega(delta_L, delta_theta) * config.CELLSIZE
    
    r_curr, c_curr = utils.coords_to_rc(state_curr[0], state_curr[1], config.MAP_DIMENSION)
    slope_curr = terrain_data['slope'][r_curr, c_curr]
    velocity = utils.get_velocity_ms(slope_curr)
    
    time_cost = mileage_cost / velocity if velocity > 0 else float('inf')
    consumption = (mileage_cost / 1000) * get_consumption_rate(slope_curr)
    
    return time_cost, consumption

def a_star_search_for_lrp(start_coords, end_coords, terrain_data, turn_logic, max_iterations):
    open_set = []
    for h in config.HEADINGS:
        start_state = (start_coords[0], start_coords[1], h)
        h_cost = np.sqrt((start_state[0] - end_coords[0])**2 + (start_state[1] - end_coords[1])**2) * config.CELLSIZE / (30.0/3.6)
        f_cost = 0 + A_STAR_WEIGHT * h_cost
        heapq.heappush(open_set, (f_cost, 0, 0, start_state)) # (f_cost, g_time, g_battery, state)
    
    g_costs = {(start_coords[0], start_coords[1], h): (0, 0) for h in config.HEADINGS} # (g_time, g_battery)
    parents, closed_set = {}, set()
    
    iteration_count = 0
    while open_set:
        if iteration_count > max_iterations:
            # print(f"\nWarning: A* search from {start_coords} to {end_coords} timed out.")
            return None, float('inf'), float('inf')
        iteration_count += 1
        
        _, g_time, g_battery, current_state = heapq.heappop(open_set)
        
        if current_state in closed_set:
            continue
        closed_set.add(current_state)
        
        if (current_state[0], current_state[1]) == end_coords:
            path = []
            temp_node = current_state
            while temp_node in parents:
                path.append(temp_node)
                temp_node = parents[temp_node]
            path.append(temp_node)
            path.reverse()
            return path, g_time, g_battery
            
        h_prev = current_state[2]
        for move_def in turn_logic[h_prev].values():
            (dx, dy), possible_h_next_list = move_def
            x_next, y_next = current_state[0] + dx, current_state[1] + dy
            
            if not (0 <= x_next < config.MAP_DIMENSION and 0 <= y_next < config.MAP_DIMENSION):
                continue
                
            r_next, c_next_coord = utils.coords_to_rc(x_next, y_next, config.MAP_DIMENSION)
            if terrain_data['slope'][r_next, c_next_coord] > config.MAX_SLOPE_DEG:
                continue
                
            for h_next in possible_h_next_list:
                next_state = (x_next, y_next, h_next)
                if next_state in closed_set:
                    continue
                    
                time_step, battery_step = cost_time_and_battery(current_state, next_state, terrain_data)
                if time_step == float('inf'):
                    continue
                    
                new_g_time = g_time + time_step
                new_g_battery = g_battery + battery_step
                
                if next_state not in g_costs or new_g_time < g_costs[next_state][0]:
                    g_costs[next_state] = (new_g_time, new_g_battery)
                    h_cost = np.sqrt((next_state[0] - end_coords[0])**2 + (next_state[1] - end_coords[1])**2) * config.CELLSIZE / (30.0/3.6)
                    f_cost = new_g_time + A_STAR_WEIGHT * h_cost
                    heapq.heappush(open_set, (f_cost, new_g_time, new_g_battery, next_state))
                    parents[next_state] = current_state
                    
    return None, float('inf'), float('inf')

def build_turn_logic_for_astar():
    logic = {h: {} for h in config.HEADINGS}
    for h_prev in config.HEADINGS:
        s_rad = np.deg2rad(h_prev)
        s_dx, s_dy = int(round(np.sin(s_rad))), int(round(np.cos(s_rad)))
        s_h_curr_options = [(h_prev - 45 + 360) % 360, h_prev, (h_prev + 45 + 360) % 360]
        logic[h_prev]['straight'] = ((s_dx, s_dy), s_h_curr_options)
        
        h_turn_l = (h_prev - 45 + 360) % 360
        l_rad = np.deg2rad(h_turn_l)
        l_dx, l_dy = int(round(np.sin(l_rad))), int(round(np.cos(l_rad)))
        l_h_curr_options = [(h_turn_l - 45 + 360) % 360, h_turn_l]
        logic[h_prev]['left_turn'] = ((l_dx, l_dy), l_h_curr_options)
        
        h_turn_r = (h_prev + 45 + 360) % 360
        r_rad = np.deg2rad(h_turn_r)
        r_dx, r_dy = int(round(np.sin(r_rad))), int(round(np.cos(r_rad)))
        r_h_curr_options = [h_turn_r, (h_turn_r + 45 + 360) % 360]
        logic[h_prev]['right_turn'] = ((r_dx, r_dy), r_h_curr_options)
    return logic

# --- Stage 1: Pre-computation ---
def precompute_costs(terrain_data, turn_logic):
    if os.path.exists(COST_MATRIX_PATH):
        print(f"Loading pre-computed cost matrix from '{COST_MATRIX_PATH}'...")
        with open(COST_MATRIX_PATH, 'r') as f:
            return json.load(f)
            
    print("--- Stage 1: Pre-computing Cost Matrix ---")
    points = list(LOCATIONS.keys())
    cost_matrix = {p1: {p2: {"time": float('inf'), "battery": float('inf')} for p2 in points if p1 != p2} for p1 in points}
    
    with tqdm(total=len(points)*(len(points)-1)//2, desc="Computing paths") as pbar:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p_start, p_end = points[i], points[j]
                coords_start, coords_end = LOCATIONS[p_start], LOCATIONS[p_end]
                
                _, time_cost, battery_cost = a_star_search_for_lrp(coords_start, coords_end, terrain_data, turn_logic, A_STAR_MAX_ITERATIONS)
                cost_matrix[p_start][p_end] = {"time": time_cost, "battery": battery_cost}
                
                _, time_cost_rev, battery_cost_rev = a_star_search_for_lrp(coords_end, coords_start, terrain_data, turn_logic, A_STAR_MAX_ITERATIONS)
                cost_matrix[p_end][p_start] = {"time": time_cost_rev, "battery": battery_cost_rev}
                pbar.update(1)
                
    with open(COST_MATRIX_PATH, 'w') as f:
        json.dump(cost_matrix, f, indent=4)
    print(f"Cost matrix saved to '{COST_MATRIX_PATH}'.")
    return cost_matrix

# --- Stage 2: Revised Warehouse Assignment ---
def assign_warehouses_revised(cost_matrix):
    print("\n--- Stage 2 (Revised): Assigning Fronts to Warehouses ---")
    fronts = list(PROBLEM_DEMANDS.keys())
    warehouses = list(WAREHOUSE_CAPACITIES.keys())
    
    service_options = []
    for z in fronts:
        for c in warehouses:
            cost_cz = cost_matrix.get(c, {}).get(z, {"time": float('inf'), "battery": float('inf')})
            cost_zc = cost_matrix.get(z, {}).get(c, {"time": float('inf'), "battery": float('inf')})
            
            is_reachable = cost_cz['time'] != float('inf') and cost_zc['time'] != float('inf')
            is_battery_sufficient = (cost_cz['battery'] + cost_zc['battery']) <= BATTERY_CAPACITY
            
            if is_reachable and is_battery_sufficient:
                service_options.append({'front': z, 'warehouse': c, 'cost': cost_cz['time']})

    service_options.sort(key=lambda x: x['cost'])
    
    assignments = {c: [] for c in warehouses}
    remaining_capacity = WAREHOUSE_CAPACITIES.copy()
    assigned_fronts = set()

    for option in service_options:
        z = option['front']
        c = option['warehouse']
        
        if z in assigned_fronts:
            continue
            
        if remaining_capacity[c] >= PROBLEM_DEMANDS[z]:
            assignments[c].append(z)
            remaining_capacity[c] -= PROBLEM_DEMANDS[z]
            assigned_fronts.add(z)
    
    unassigned = set(fronts) - assigned_fronts
    if unassigned:
        print(f"WARNING: Could not assign all fronts! Unassigned: {unassigned}")

    final_assignments = {c: z_list for c, z_list in assignments.items() if z_list}
    print("Warehouse assignment complete.")
    return final_assignments

# --- Stage 3: Robust VRP Solver ---
def solve_vrp_for_warehouse_revised(warehouse, customers, cost_matrix):
    if not customers:
        return []
        
    split_customers, customer_demands = [], {}
    for cust in customers:
        demand = PROBLEM_DEMANDS[cust]
        num_full_loads = int(demand // VEHICLE_CAPACITY)
        for i in range(num_full_loads):
            cust_id = f"{cust}_{i}"
            split_customers.append(cust_id)
            customer_demands[cust_id] = VEHICLE_CAPACITY
        if demand % VEHICLE_CAPACITY > 0.01:
            cust_id = f"{cust}_{num_full_loads}"
            split_customers.append(cust_id)
            customer_demands[cust_id] = demand % VEHICLE_CAPACITY
    
    def get_route_details(route):
        demand = sum(customer_demands[c] for c in route)
        path_nodes = [warehouse] + [c.split('_')[0] for c in route] + [warehouse]
        time, battery = 0, 0
        for i in range(len(path_nodes) - 1):
            leg = cost_matrix.get(path_nodes[i], {}).get(path_nodes[i+1])
            if not leg or leg['time'] == float('inf'):
                return float('inf'), float('inf'), float('inf')
            time += leg['time']
            battery += leg['battery']
        return demand, time, battery
    
    savings = {}
    for c1, c2 in combinations(split_customers, 2):
        oc1, oc2 = c1.split('_')[0], c2.split('_')[0]
        t_c1 = cost_matrix.get(warehouse,{}).get(oc1,{}).get('time', float('inf'))
        t_c2 = cost_matrix.get(warehouse,{}).get(oc2,{}).get('time', float('inf'))
        t_12 = cost_matrix.get(oc1,{}).get(oc2,{}).get('time', float('inf'))
        
        if any(t == float('inf') for t in [t_c1, t_c2, t_12]):
            saving_val = -float('inf')
        else:
            saving_val = t_c1 + t_c2 - t_12
        savings[(c1, c2)] = saving_val

    routes = [[c] for c in split_customers]
    sorted_savings = sorted(savings.items(), key=lambda item: item[1], reverse=True)

    for (c1, c2), saving_val in sorted_savings:
        if saving_val == -float('inf'):
            break
            
        r1_idx, r2_idx = -1, -1
        for i, r in enumerate(routes):
            if c1 in r: r1_idx = i
            if c2 in r: r2_idx = i
            
        if r1_idx != -1 and r2_idx != -1 and r1_idx != r2_idx:
            route1, route2 = routes[r1_idx], routes[r2_idx]
            
            # Try merging in both directions
            for R1, R2, C1_node, C2_node in [(route1, route2, c1, c2), (route2, route1, c2, c1)]:
                # Merge R1's end with R2's start
                if R1[-1] == C1_node and R2[0] == C2_node:
                    merged_route = R1 + R2
                    demand, _, battery = get_route_details(merged_route)
                    if demand <= VEHICLE_CAPACITY and battery <= BATTERY_CAPACITY:
                        # Update the correct original list
                        if R1 == route1:
                            routes[r1_idx] = merged_route
                            routes.pop(r2_idx)
                        else: # R1 == route2
                            routes[r2_idx] = merged_route
                            routes.pop(r1_idx)
                        break 
            else: # This else corresponds to the inner for-loop
                continue # If merge happened, outer loop continues
            break # If merge happened, break from the inner for-loop
            
    final_routes = []
    for route in routes:
        demand, time, battery = get_route_details(route)
        if time != float('inf'):
            final_routes.append({
                "route": [warehouse] + route + [warehouse], 
                "demand": demand, 
                "time": time, 
                "battery": battery
            })
    return final_routes

# --- Main Orchestration ---
def main():
    print("\n--- Starting Step 6: Logistics Planning (Problem 5) ---")
    try:
        print("Loading preprocessed terrain data into memory...")
        with np.load(config.PREPROCESSED_DATA_PATH) as data:
            terrain_data = {'slope': data['slope'][:], 'nx': data['nx'][:], 'ny': data['ny'][:], 'nz': data['nz'][:]}
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find '{config.PREPROCESSED_DATA_PATH}'. Please run 01_preprocess.py first.")
        return
    
    turn_logic = build_turn_logic_for_astar()
    cost_matrix = precompute_costs(terrain_data, turn_logic)
    assignments = assign_warehouses_revised(cost_matrix)

    print("\n--- Stage 3: Solving Vehicle Routing Problem for each Warehouse ---")
    full_plan = {}
    with tqdm(total=len(assignments), desc="Planning VRP") as pbar:
        for warehouse, customers in assignments.items():
            full_plan[warehouse] = solve_vrp_for_warehouse_revised(warehouse, customers, cost_matrix)
            pbar.update(1)

    print("\n\n--- FINAL LOGISTICS PLAN ---")
    total_vehicles, max_time = 0, 0
    for warehouse, routes in full_plan.items():
        if not routes:
            continue
        print(f"\nWarehouse: {warehouse}")
        print(f"  - Assigned Fronts: {assignments.get(warehouse, 'N/A')}")
        print(f"  - Total Demand: {sum(PROBLEM_DEMANDS[z] for z in assignments.get(warehouse, [])):.1f} kg / {WAREHOUSE_CAPACITIES.get(warehouse, 0)} kg")
        print(f"  - Deployed Vehicles: {len(routes)}")
        total_vehicles += len(routes)
        
        for i, route_info in enumerate(routes):
            route_str = " -> ".join([r.split('_')[0] for r in route_info['route']])
            print(f"    - Vehicle {i+1}: Route: {route_str}")
            print(f"      - Load: {route_info['demand']:.1f} kg, Time: {route_info['time']/3600:.2f} hrs, Battery: {route_info['battery']:.1f}%")
            if route_info['time'] > max_time:
                max_time = route_info['time']
    
    print("\n--- OVERALL PLAN SUMMARY ---")
    print(f"Activated Warehouses: {list(full_plan.keys())} ({len(full_plan)} in total)")
    print(f"Total Vehicles Deployed: {total_vehicles}")
    print(f"Makespan (Max Time): {max_time/3600:.2f} hours")

if __name__ == "__main__":
    main()