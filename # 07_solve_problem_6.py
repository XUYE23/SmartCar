# 07_solve_problem_6.py
# Solves problem 6: 15-day dynamic inventory-routing simulation.
# Version 3: Fixes KeyError by standardizing event data and state updates.

import heapq, json, os, pandas as pd
from tqdm import tqdm

# Assuming config.py and utils.py are in the same directory
import config
import utils

# --- Constants and Initial Setup ---
NUM_DAYS = 15
COST_MATRIX_PATH = "p5_cost_matrix.json"
OUTPUT_LOG_PATH = "problem6_vehicle_log.xlsx"
MAX_WAREHOUSE_CHARGERS, VEHICLE_CAPACITY, BATTERY_CAPACITY = 2, 500.0, 100.0
INVENTORY_MAX_DAYS, INVENTORY_MIN_DAYS = 3.5, 0.5
CHARGE_THRESHOLD, CHARGE_TIME_UNIT = 80.0, 2 * 3600
DAILY_DEMANDS = {'Z1':510,'Z2':450,'Z3':285,'Z4':450,'Z5':675,'Z6':450,'Z7':285,'Z8':675,'Z9':450}
INITIAL_VEHICLE_DEPLOYMENT = {f"V{i+1}":loc for i,loc in enumerate(["C1","C1","C2","C2","C2","C3","C3","C4","C5","C5","C6","C6","C7","C8","C8","C9","C9"])}

def format_time(seconds):
    if seconds is None or seconds == float('inf'): return None
    day = int(seconds // (24 * 3600)) + 1
    rem = seconds % (24 * 3600)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"Day {day}, {int(h):02d}:{int(m):02d}:{int(s):02d}"

class SystemState:
    def __init__(self, cost_matrix):
        self.time, self.cost_matrix = 0.0, cost_matrix
        self.inventory = {z: 2.0 * d for z, d in DAILY_DEMANDS.items()}
        self.vehicles = {v_id: {"loc": loc, "battery": 100.0, "status": "IDLE", "current_load": 0} for v_id, loc in INITIAL_VEHICLE_DEPLOYMENT.items()}
        self.warehouse_chargers = {c: [] for c in set(INITIAL_VEHICLE_DEPLOYMENT.values())}
        self.used_vehicles, self.activated_warehouses, self.trip_count = set(), set(), 0

class Event:
    def __init__(self, time, event_type, data):
        self.time, self.type, self.data = time, event_type, data
    def __lt__(self, other): return self.time < other.time

class Logger:
    def __init__(self): self.log_entries = []
    def record(self, entry):
        for k, v in entry.items():
            if "time" in k and isinstance(v, (int, float)): entry[k] = format_time(v)
        self.log_entries.append(entry)
    def save(self):
        if not self.log_entries: print("No logs were recorded."); return
        df = pd.DataFrame(self.log_entries)
        cols = {"vehicle_id":"无人车编号","start_warehouse":"出发仓库","load_kg":"载货数量(kg)","route":"行驶路线","arrival_front":"达到前沿阵地","delivered_kg":"送达物资数量(kg)","charge_start_time":"充电开始时间","charge_location":"所在仓库","charge_end_time":"充电结束时间"}
        df = df.rename(columns=cols)
        for col in cols.values():
            if col not in df.columns: df[col] = pd.NA
        df[list(cols.values())].to_excel(OUTPUT_LOG_PATH, index=False)
        print(f"\nSimulation log saved to '{OUTPUT_LOG_PATH}'")

def run_simulation():
    try:
        with open(COST_MATRIX_PATH, 'r') as f: cost_matrix = json.load(f)
    except FileNotFoundError: print(f"Error: Cost matrix '{COST_MATRIX_PATH}' not found."); return

    state, logger, event_queue = SystemState(cost_matrix), Logger(), []

    for day in range(1, NUM_DAYS + 1):
        day_start_time = (day - 1) * 24 * 3600
        state.time = day_start_time
        tqdm.write(f"\n--- Day {day} | Time: {format_time(state.time)} ---")

        # Daily Planning
        demands_today = []
        for z, inv in state.inventory.items():
            if inv - DAILY_DEMANDS[z] < INVENTORY_MIN_DAYS * DAILY_DEMANDS[z]:
                req = (INVENTORY_MAX_DAYS * DAILY_DEMANDS[z]) - inv
                if req > 0: demands_today.append({"front": z, "amount": req})
        
        if demands_today: tqdm.write(f"  - Demands identified for: {[d['front'] for d in demands_today]}")
        
        idle_vehicles = [v_id for v_id, v_data in state.vehicles.items() if v_data["status"] == "IDLE"]
        
        for demand in sorted(demands_today, key=lambda x: x['amount'], reverse=True):
            z_demand, amount_needed = demand["front"], demand["amount"]
            loads_to_deliver = []
            while amount_needed > 0:
                load = min(amount_needed, VEHICLE_CAPACITY); loads_to_deliver.append(load); amount_needed -= load

            for load in loads_to_deliver:
                best_vehicle, min_cost = None, float('inf')
                for v_id in idle_vehicles:
                    v_data = state.vehicles[v_id]; c_loc = v_data["loc"]
                    cost_out = cost_matrix.get(c_loc, {}).get(z_demand, {})
                    cost_ret = cost_matrix.get(z_demand, {}).get(c_loc, {})
                    if cost_out and cost_ret:
                        round_trip_battery = cost_out.get('battery', float('inf')) + cost_ret.get('battery', float('inf'))
                        if round_trip_battery <= v_data["battery"] and cost_out.get('time', float('inf')) < min_cost:
                            min_cost, best_vehicle = cost_out.get('time'), v_id
                
                if best_vehicle:
                    state.used_vehicles.add(best_vehicle); state.activated_warehouses.add(state.vehicles[best_vehicle]["loc"]); state.trip_count += 1
                    state.vehicles[best_vehicle].update({"status": "DISPATCHED", "current_load": load})
                    idle_vehicles.remove(best_vehicle)
                    # --- FIX: Simplified DEPART event data ---
                    heapq.heappush(event_queue, Event(state.time, "DEPART", {"vehicle_id": best_vehicle, "destination": z_demand}))

        # Daily Execution
        day_end_time = day * 24 * 3600
        with tqdm(total=len(event_queue), desc=f"  - Simulating Day {day} events") as pbar:
            while event_queue and event_queue[0].time < day_end_time:
                pbar.total = len(event_queue)
                pbar.update(1)
                event = heapq.heappop(event_queue)
                state.time = event.time
                v_id = event.data["vehicle_id"]
                v_data = state.vehicles[v_id]

                if event.type == "DEPART":
                    c_loc, z_loc = v_data["loc"], event.data["destination"]
                    logger.record({"vehicle_id":v_id, "start_warehouse":c_loc, "load_kg":v_data["current_load"], "route":f"{c_loc} -> {z_loc}"})
                    path_cost = cost_matrix[c_loc][z_loc]
                    v_data["battery"] -= path_cost["battery"]
                    arrival_time = state.time + path_cost["time"]
                    # --- FIX: Pass necessary info for next event ---
                    heapq.heappush(event_queue, Event(arrival_time, "ARRIVE_FRONT", {"vehicle_id":v_id, "arrival_front":z_loc, "origin_warehouse":c_loc}))

                elif event.type == "ARRIVE_FRONT":
                    z_loc, c_loc = event.data["arrival_front"], event.data["origin_warehouse"]
                    delivered_kg = v_data["current_load"]
                    logger.record({"vehicle_id":v_id, "arrival_front":z_loc, "delivered_kg":delivered_kg})
                    state.inventory[z_loc] += delivered_kg
                    v_data.update({"loc": z_loc, "current_load": 0})
                    path_cost = cost_matrix[z_loc][c_loc]
                    v_data["battery"] -= path_cost["battery"]
                    arrival_time = state.time + path_cost["time"]
                    heapq.heappush(event_queue, Event(arrival_time, "ARRIVE_WAREHOUSE", {"vehicle_id":v_id, "arrival_warehouse":c_loc}))

                elif event.type == "ARRIVE_WAREHOUSE":
                    c_loc = event.data["arrival_warehouse"]
                    v_data["loc"] = c_loc
                    if v_data["battery"] < CHARGE_THRESHOLD and len(state.warehouse_chargers.get(c_loc,[])) < MAX_WAREHOUSE_CHARGERS:
                        v_data["status"] = "CHARGING"; state.warehouse_chargers[c_loc].append(v_id)
                        heapq.heappush(event_queue, Event(state.time, "START_CHARGE", {"vehicle_id":v_id}))
                    else: v_data["status"] = "IDLE"

                elif event.type == "START_CHARGE":
                    c_loc = v_data["loc"]
                    logger.record({"charge_start_time":state.time, "charge_location":c_loc, "vehicle_id":v_id})
                    heapq.heappush(event_queue, Event(state.time + CHARGE_TIME_UNIT, "END_CHARGE", {"vehicle_id":v_id}))

                elif event.type == "END_CHARGE":
                    c_loc = v_data["loc"]
                    b = v_data["battery"]
                    new_b = 20 if b<20 else 50 if b<50 else 80 if b<80 else 100
                    v_data["battery"] = new_b
                    logger.record({"charge_end_time":state.time, "charge_location":c_loc, "vehicle_id":v_id})
                    if new_b < 100: heapq.heappush(event_queue, Event(state.time, "START_CHARGE", {"vehicle_id":v_id}))
                    else:
                        if v_id in state.warehouse_chargers.get(c_loc,[]): state.warehouse_chargers[c_loc].remove(v_id)
                        v_data["status"] = "IDLE"

        # End of Day
        for z in state.inventory: state.inventory[z] -= DAILY_DEMANDS[z]

    logger.save()
    print("\n--- 15-DAY SIMULATION FINAL SUMMARY ---")
    print(f"  - 启用中转仓库数量: {len(state.activated_warehouses)}")
    print(f"  - 无人车数量: {len(state.used_vehicles)}")
    print(f"  - 无人车运输趟次: {state.trip_count}")
    print(f"  - (Activated Warehouses: {sorted(list(state.activated_warehouses))})")
    print(f"  - (Used Vehicles: {sorted(list(state.used_vehicles))})")

def main():
    print("\n--- Starting Step 7: 15-Day Simulation (Problem 6) ---")
    run_simulation()
    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    main()