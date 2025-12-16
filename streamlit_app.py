from typing import Dict, List
import streamlit as st
import simpy
import random
import numpy as np
import math
import pandas as pd
from io import BytesIO
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as Doctorsates
from matplotlib.patches import Rectangle, Patch

# =============================
# Utilities
# =============================
def exp_time(mean):
    return 0.0 if mean <= 0 else random.expovariate(1.0/mean)

def normal_time(mean, cv=0.4):
    if mean <= 0:
        return 0.0
    sd = max(1e-6, mean * cv)
    return max(0.0, random.gauss(mean, sd))

def speed_multiplier_from_cv(cv):
    if cv <= 0:
        return 1.0
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = -0.5 * sigma**2
    return np.random.lognormal(mean=mu, sigma=sigma)

def draw_service_time(role_for_dist, mean, dist_map, cv_task):
    dist = dist_map.get(role_for_dist, "exponential")
    base = normal_time(mean, cv=0.4) if dist == "normal" else exp_time(mean)
    return base * speed_multiplier_from_cv(cv_task)

# =============================
# Work schedule helpers
# =============================
MIN_PER_HOUR = 60
DAY_MIN = 24 * MIN_PER_HOUR

def is_open(t_min, open_minutes):
    return (t_min % DAY_MIN) < open_minutes

def minutes_until_close(t_min, open_minutes):
    return max(0.0, open_minutes - (t_min % DAY_MIN))

def minutes_until_open(t_min, open_minutes):
    t_mod = t_min % DAY_MIN
    return 0.0 if t_mod < open_minutes else DAY_MIN - t_mod

def effective_open_minutes(sim_minutes, open_minutes):
    full_days = int(sim_minutes // DAY_MIN)
    remainder = sim_minutes % DAY_MIN
    return full_days * open_minutes + min(open_minutes, remainder)

# =============================
# Roles / constants
# =============================
ROLES = ["Administrative staff", "Nurse", "Doctors", "Other staff"]
DONE = "Done"

# =============================
# Metrics
# =============================
class Metrics:
    def __init__(self):
        self.time_stamps = []
        self.queues = {r: [] for r in ROLES}
        self.waits = {r: [] for r in ROLES}
        self.taps = {r: 0 for r in ROLES}
        self.completed = 0
        self.arrivals_total = 0
        self.arrivals_by_role = {r: 0 for r in ROLES}
        self.service_time_sum = {r: 0.0 for r in ROLES}
        self.loop_fd_insufficient = 0
        self.loop_fd_corrections = 0
        self.loop_nurse_insufficient = 0
        self.loop_nurse_corrections = 0
        self.loop_provider_insufficient = 0
        self.loop_provider_corrections = 0
        self.loop_backoffice_insufficient = 0
        self.loop_backoffice_corrections = 0
        self.events = []
        self.task_arrival_time: Dict[str, float] = {}
        self.task_completion_time: Dict[str, float] = {}

    def log(self, t, name, step, note="", arrival_t=None):
        self.events.append((t, name, step, note, arrival_t if arrival_t is not None else self.task_arrival_time.get(name)))

# =============================
# Step labels
# =============================
STEP_LABELS = {
    "ARRIVE": "Task arrived", "FD_QUEUE": "Administrative staff: queued", "FD_DONE": "Administrative staff: completed",
    "FD_INSUFF": "Administrative staff: missing info", "FD_RETRY_QUEUE": "Administrative staff: re-queued (info)",
    "FD_RETRY_DONE": "Administrative staff: re-done (info)", "NU_QUEUE": "Nurse: queued", "NU_DONE": "Nurse: completed",
    "NU_INSUFF": "Nurse: missing info", "NU_RECHECK_QUEUE": "Nurse: re-check queued",
    "NU_RECHECK_DONE": "Nurse: re-check completed", "PR_QUEUE": "Doctors: queued", "PR_DONE": "Doctors: completed",
    "PR_INSUFF": "Doctors: corrections needed", "PR_RECHECK_QUEUE": "Doctors: recheck queued",
    "PR_RECHECK_DONE": "Doctors: recheck done", "BO_QUEUE": "Other staff: queued", "BO_DONE": "Other staff: completed",
    "BO_INSUFF": "Other staff: corrections needed", "BO_RECHECK_QUEUE": "Other staff: recheck queued",
    "BO_RECHECK_DONE": "Other staff: recheck done", "DONE": "Task resolved"
}

def pretty_step(code):
    return STEP_LABELS.get(code, code)

# =============================
# Availability schedule generator
# =============================
def generate_availability_schedule(sim_minutes: int, role: str, minutes_per_day: int, seed_offset: int = 0) -> set:
    open_minutes_per_day = sim_minutes // (sim_minutes // DAY_MIN) if sim_minutes >= DAY_MIN else sim_minutes
    
    if minutes_per_day >= open_minutes_per_day:
        return set(range(int(sim_minutes)))
    if minutes_per_day <= 0:
        return set()
    
    local_random = random.Random(hash(role) + seed_offset)
    available_minutes = set()
    num_days = int(np.ceil(sim_minutes / DAY_MIN))
    
    BLOCK_SIZE = 10
    
    for day in range(num_days):
        day_start = day * DAY_MIN
        day_end = min((day + 1) * DAY_MIN, sim_minutes)
        day_length = min(day_end - day_start, open_minutes_per_day)
        
        blocks_needed = int(np.ceil(minutes_per_day / BLOCK_SIZE))
        total_blocks = day_length // BLOCK_SIZE
        
        if total_blocks <= 0:
            continue
            
        blocks_to_select = min(blocks_needed, total_blocks)
        selected_block_indices = local_random.sample(range(total_blocks), blocks_to_select)
        
        for block_idx in selected_block_indices:
            block_start = day_start + (block_idx * BLOCK_SIZE)
            block_end = min(block_start + BLOCK_SIZE, day_start + day_length)
            available_minutes.update(range(block_start, block_end))
        
        current_available = len([m for m in available_minutes if day_start <= m < day_start + day_length])
        if current_available < minutes_per_day and current_available < day_length:
            remaining_needed = min(minutes_per_day - current_available, day_length - current_available)
            unavailable = [m for m in range(day_start, day_start + day_length) if m not in available_minutes]
            if unavailable and remaining_needed > 0:
                extra_minutes = local_random.sample(unavailable, min(remaining_needed, len(unavailable)))
                available_minutes.update(extra_minutes)
    
    return available_minutes

# =============================
# System
# =============================
class CHCSystem:
    def __init__(self, env, params, metrics, seed_offset=0):
        self.env = env
        self.p = params
        self.m = metrics

        self.fd_cap = params["frontdesk_cap"]
        self.nu_cap = params["nurse_cap"]
        self.pr_cap = params["provider_cap"]
        self.bo_cap = params["backoffice_cap"]

        self.frontdesk = simpy.Resource(env, capacity=self.fd_cap) if self.fd_cap > 0 else None
        self.nurse = simpy.Resource(env, capacity=self.nu_cap) if self.nu_cap > 0 else None
        self.provider = simpy.Resource(env, capacity=self.pr_cap) if self.pr_cap > 0 else None
        self.backoffice = simpy.Resource(env, capacity=self.bo_cap) if self.bo_cap > 0 else None

        self.role_to_res = {
            "Administrative staff": self.frontdesk, "Nurse": self.nurse,
            "Doctors": self.provider, "Other staff": self.backoffice
        }
        
        avail_params = params.get("availability_per_day", {"Administrative staff": 480, "Nurse": 480, "Doctors": 480, "Other staff": 480})        
        self.availability = {
            role: generate_availability_schedule(params["sim_minutes"], role, avail_params.get(role, 60), seed_offset)
            for role in ROLES
        }

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
        if resource is None or mean_time <= 1e-12:
            return
        if role_for_dist is None:
            role_for_dist = role_account

        remaining = draw_service_time(role_for_dist, mean_time, self.p["dist_role"], self.p["cv_speed"])
        remaining += max(0.0, self.p["emr_overhead"].get(role_account, 0.0))

        open_minutes = self.p["open_minutes"]
        available_set = self.availability.get(role_account, set())

        while remaining > 1e-9:
            current_min = int(self.env.now)
            
            # If clinic is closed, wait until it opens
            if not is_open(self.env.now, open_minutes):
                yield self.env.timeout(minutes_until_open(self.env.now, open_minutes))
                continue
            
            # If staff member is not available at this minute, wait
            if len(available_set) > 0 and current_min not in available_set:
                yield self.env.timeout(1)
                continue
            
            # Calculate how much time until closing
            window = minutes_until_close(self.env.now, open_minutes)
            
            # If we're at or past closing, stop and wait for next day
            if window <= 0:
                yield self.env.timeout(minutes_until_open(self.env.now, open_minutes))
                continue
            
            # Calculate available work window considering staff availability
            if len(available_set) > 0:
                avail_window = 1
                check_min = current_min + 1
                while check_min in available_set and avail_window < window and avail_window < remaining:
                    avail_window += 1
                    check_min += 1
                work_chunk = min(remaining, window, avail_window)
            else:
                work_chunk = min(remaining, window)
            
            # Acquire resource and do work - but ONLY during open hours
            with resource.request() as req:
                t_req = self.env.now
                yield req
                self.m.waits[role_account].append(self.env.now - t_req)
                self.m.taps[role_account] += 1
                
                # Double-check we don't work past closing
                time_left_today = minutes_until_close(self.env.now, open_minutes)
                actual_work = min(work_chunk, time_left_today)
                
                if actual_work > 0:
                    yield self.env.timeout(actual_work)
                    self.m.service_time_sum[role_account] += actual_work
                    remaining -= actual_work

# =============================
# Routing helpers
# =============================
def sample_next_role(route_row: Dict[str, float]) -> str:
    keys = tuple(route_row.keys())
    vals = np.fromiter((max(0.0, float(route_row[k])) for k in keys), dtype=float)
    s = vals.sum()
    if s <= 0:
        return DONE
    probs = vals / s
    return random.choices(keys, weights=probs, k=1)[0]

# =============================
# Workflows per role
# =============================
def handle_role(env, task_id, s: CHCSystem, role: str):
    if role not in ROLES:
        return DONE

    res = s.role_to_res[role]

    if role == "Administrative staff":
        if res is not None:
            s.m.log(env.now, task_id, "FD_QUEUE", "")
            yield from s.scheduled_service(res, "Administrative staff", s.p["svc_frontdesk"])
            s.m.log(env.now, task_id, "FD_DONE", "")
            fd_loops = 0
            total_loops = 0
            max_loops = s.p["max_fd_loops"]
            
            while total_loops < max_loops:
                # Check insufficient info first
                if random.random() < s.p["p_fd_insuff"]:
                    total_loops += 1
                    s.m.loop_fd_insufficient += 1
                    s.m.log(env.now, task_id, "FD_INSUFF", f"Missing info loop #{total_loops}")
                    yield env.timeout(s.p["fd_insuff_delay"])
                    s.m.log(env.now, task_id, "FD_INSUFF_QUEUE", f"Loop #{total_loops}")
                    yield from s.scheduled_service(res, "Administrative staff", s.p["svc_frontdesk"] * 0.5)
                    s.m.log(env.now, task_id, "FD_INSUFF_DONE", f"Loop #{total_loops}")
                # If no insufficient info, check corrections
                elif random.random() < s.p["p_fd_corrections"]:
                    total_loops += 1
                    s.m.loop_fd_corrections += 1
                    s.m.log(env.now, task_id, "FD_corrections", f"corrections loop #{total_loops}")
                    yield env.timeout(s.p["fd_corrections_delay"])
                    s.m.log(env.now, task_id, "FD_corrections_QUEUE", f"Loop #{total_loops}")
                    yield from s.scheduled_service(res, "Administrative staff", s.p["svc_frontdesk"] * 0.33)
                    s.m.log(env.now, task_id, "FD_corrections_DONE", f"Loop #{total_loops}")
                else:
                    break  # No issues, exit loop
                    
    elif role == "Nurse":
        if res is not None:
            s.m.log(env.now, task_id, "NU_QUEUE", "")
            if random.random() < s.p["p_protocol"]:
                yield from s.scheduled_service(res, "Nurse", s.p["svc_nurse_protocol"], role_for_dist="NurseProtocol")
            else:
                yield from s.scheduled_service(res, "Nurse", s.p["svc_nurse"])
                s.m.log(env.now, task_id, "NU_DONE", "")
            nurse_loops = 0
            while (nurse_loops < s.p["max_nurse_loops"]) and (random.random() < s.p["p_nurse_insuff"]):
                nurse_loops += 1
                s.m.loop_nurse_insufficient += 1
                s.m.log(env.now, task_id, "NU_INSUFF", f"Back to FD loop #{nurse_loops}")
                if s.role_to_res["Administrative staff"] is not None:
                    s.m.log(env.now, task_id, "FD_QUEUE", f"After nurse loop #{nurse_loops}")
                    yield from s.scheduled_service(s.role_to_res["Administrative staff"], "Administrative staff", s.p["svc_frontdesk"])
                    s.m.log(env.now, task_id, "FD_DONE", f"After nurse loop #{nurse_loops}")
                s.m.log(env.now, task_id, "NU_RECHECK_QUEUE", f"Loop #{nurse_loops}")
                yield from s.scheduled_service(res, "Nurse", max(0.0, 0.5 * s.p["svc_nurse"]))
                s.m.log(env.now, task_id, "NU_RECHECK_DONE", f"Loop #{nurse_loops}")

    elif role == "Doctors":
        if res is not None:
            s.m.log(env.now, task_id, "PR_QUEUE", "")
            yield from s.scheduled_service(res, "Doctors", s.p["svc_provider"])
            s.m.log(env.now, task_id, "PR_DONE", "")
            
            # Combined loop counter for both types
            total_loops = 0
            max_loops = s.p["max_provider_loops"]
            
            while total_loops < max_loops:
                # Check insufficient info first
                if random.random() < s.p["p_provider_insuff"]:
                    total_loops += 1
                    s.m.loop_provider_insufficient += 1
                    s.m.log(env.now, task_id, "PR_INSUFF", f"Missing info loop #{total_loops}")
                    yield env.timeout(s.p["provider_insuff_delay"])
                    s.m.log(env.now, task_id, "PR_INSUFF_QUEUE", f"Loop #{total_loops}")
                    yield from s.scheduled_service(res, "Doctors", s.p["svc_provider"] * 0.5)
                    s.m.log(env.now, task_id, "PR_INSUFF_DONE", f"Loop #{total_loops}")
                # If no insufficient info, check corrections
                elif random.random() < s.p["p_provider_corrections"]:
                    total_loops += 1
                    s.m.loop_provider_corrections += 1
                    s.m.log(env.now, task_id, "PR_corrections", f"corrections loop #{total_loops}")
                    yield env.timeout(s.p["provider_corrections_delay"])
                    s.m.log(env.now, task_id, "PR_corrections_QUEUE", f"Loop #{total_loops}")
                    yield from s.scheduled_service(res, "Doctors", s.p["svc_provider"] * 0.33)
                    s.m.log(env.now, task_id, "PR_corrections_DONE", f"Loop #{total_loops}")
                else:
                    break  # No issues, exit loop

    elif role == "Other staff":
        if res is not None:
            s.m.log(env.now, task_id, "BO_QUEUE", "")
            yield from s.scheduled_service(res, "Other staff", s.p["svc_backoffice"])
            s.m.log(env.now, task_id, "BO_DONE", "")
            
            # Combined loop counter for both types
            total_loops = 0
            max_loops = s.p["max_backoffice_loops"]
            
            while total_loops < max_loops:
                # Check insufficient info first
                if random.random() < s.p["p_backoffice_insuff"]:
                    total_loops += 1
                    s.m.loop_backoffice_insufficient += 1
                    s.m.log(env.now, task_id, "BO_INSUFF", f"Missing info loop #{total_loops}")
                    yield env.timeout(s.p["backoffice_insuff_delay"])
                    s.m.log(env.now, task_id, "BO_INSUFF_QUEUE", f"Loop #{total_loops}")
                    yield from s.scheduled_service(res, "Other staff", s.p["svc_backoffice"] * 0.5)
                    s.m.log(env.now, task_id, "BO_INSUFF_DONE", f"Loop #{total_loops}")
                # If no insufficient info, check corrections
                elif random.random() < s.p["p_backoffice_corrections"]:
                    total_loops += 1
                    s.m.loop_backoffice_corrections += 1
                    s.m.log(env.now, task_id, "BO_corrections", f"corrections loop #{total_loops}")
                    yield env.timeout(s.p["backoffice_corrections_delay"])
                    s.m.log(env.now, task_id, "BO_corrections_QUEUE", f"Loop #{total_loops}")
                    yield from s.scheduled_service(res, "Other staff", s.p["svc_backoffice"] * 0.33)
                    s.m.log(env.now, task_id, "BO_corrections_DONE", f"Loop #{total_loops}")
                else:
                    break  # No issues, exit loop

    row = s.p["route_matrix"].get(role, {DONE: 1.0})
    nxt = sample_next_role(row)
    return nxt

def task_lifecycle(env, task_id: str, s: CHCSystem, initial_role: str):
    s.m.task_arrival_time[task_id] = env.now
    s.m.arrivals_total += 1
    s.m.arrivals_by_role[initial_role] += 1
    s.m.log(env.now, task_id, "ARRIVE", f"Arrived at {initial_role}", arrival_t=env.now)

    role = initial_role
    for _ in range(60):
        nxt = yield from handle_role(env, task_id, s, role)
        if nxt == DONE:
            s.m.completed += 1
            s.m.task_completion_time[task_id] = env.now
            s.m.log(env.now, task_id, "DONE", "Task completed")
            return
        role = nxt

    s.m.completed += 1
    s.m.task_completion_time[task_id] = env.now
    s.m.log(env.now, task_id, "DONE", "Max handoffs reached – forced completion")

def arrival_process_for_role(env, s: CHCSystem, role_name: str, rate_per_hour: int):
    i = 0
    lam = max(0, int(rate_per_hour)) / 60.0
    open_minutes = s.p["open_minutes"]
    
    while True:
        # If clinic is closed, wait until it opens
        current_time = env.now
        if not is_open(current_time, open_minutes):
            wait_time = minutes_until_open(current_time, open_minutes)
            yield env.timeout(wait_time)
        
        # Calculate time until closing
        time_until_close = minutes_until_close(env.now, open_minutes)
        
        # Generate inter-arrival time
        inter = random.expovariate(lam) if lam > 0 else 999999999
        
        # If arrival would happen after closing, wait until next day
        if inter >= time_until_close:
            # Wait until closing
            yield env.timeout(time_until_close)
            # Wait until next opening
            wait_open = minutes_until_open(env.now, open_minutes)
            yield env.timeout(wait_open)
            continue  # Start new iteration to generate new arrival
        
        # Normal arrival during open hours
        yield env.timeout(inter)
        i += 1
        task_id = f"{role_name[:2].upper()}-{i:05d}"
        env.process(task_lifecycle(env, task_id, s, initial_role=role_name))

def monitor(env, s: CHCSystem):
    while True:
        s.m.time_stamps.append(env.now)
        for r in ROLES:
            res = s.role_to_res[r]
            self_q = len(res.queue) + res.count if res is not None else 0
            s.m.queues[r].append(self_q)
        yield env.timeout(1)

# =============================
# Run single replication
# =============================
def run_single_replication(p: Dict, seed: int) -> Metrics:
    random.seed(seed)
    np.random.seed(seed)
    
    metrics = Metrics()
    env = simpy.Environment()
    system = CHCSystem(env, p, metrics, seed_offset=seed)

    for role in ROLES:
        rate = int(p["arrivals_per_hour_by_role"].get(role, 0))
        env.process(arrival_process_for_role(env, system, role, rate))

    env.process(monitor(env, system))  # ← Line 386
    env.run(until=p["sim_minutes"])
    
    return metrics

# =============================
# Burnout calculation (MBI + JD-R)
# =============================
def calculate_burnout(all_metrics: List[Metrics], p: Dict, active_roles: List[str]) -> Dict:
    """
    Burnout model with user-defined weights for each underlying factor:
      - Utilization (with threshold effects)
      - Availability Stress
      - corrections Percentage
      - Task Switching (queue volatility)
      - Incompletion Rate
      - Throughput Deficit
    """
    weights = p.get("burnout_weights", {
        "utilization": 8, "availability_stress": 1,
        "corrections": 8, "task_switching": 1,
        "incompletion": 1, "throughput_deficit": 1
    })
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        # If all weights are 0, return zero burnout
        return {
            "by_role": {role: {"overall": 0.0, "components": {}} for role in active_roles},
            "overall_clinic": 0.0
        }
    
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    burnout_scores = {}
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    num_days = max(1, p["sim_minutes"] / DAY_MIN)
    open_minutes_per_day = p["open_minutes"]

    for role in active_roles:
        capacity = {
            "Administrative staff": p["frontdesk_cap"],
            "Nurse": p["nurse_cap"],
            "Doctors": p["provider_cap"],
            "Other staff": p["backoffice_cap"]
        }[role]
        if capacity == 0:
            continue

        # Collect metrics across replications
        util_list = []
        corrections_pct_list = []
        queue_volatility_list = []
        completion_rate_list = []
        throughput_rate_list = []

        for metrics in all_metrics:
            # Utilization (0–1)
            total_service = metrics.service_time_sum[role]
            avail_minutes_per_day = p.get("availability_per_day", {}).get(role, open_minutes_per_day)
            total_available_capacity = capacity * num_days * avail_minutes_per_day
            util = total_service / max(1, total_available_capacity)
            util_list.append(min(1.0, util))

            # correctionsPct (0–1)
            loop_counts = {
                "Administrative staff": metrics.loop_fd_insufficient,
                "Nurse": metrics.loop_nurse_insufficient,
                "Doctors": metrics.loop_provider_insufficient,
                "Other staff": metrics.loop_backoffice_insufficient
            }
            loops = loop_counts.get(role, 0)
            svc_time = {
                "Administrative staff": p["svc_frontdesk"],
                "Nurse": p["svc_nurse"],
                "Doctors": p["svc_provider"],
                "Other staff": p["svc_backoffice"]
            }[role]
            estimated_corrections = loops * max(0.0, svc_time) * 0.5
            corrections_pct = (estimated_corrections / max(1, total_service)) if total_service > 0 else 0.0
            corrections_pct_list.append(min(1.0, corrections_pct))

            # Queue Volatility (0–1)
            queue_lengths = metrics.queues[role]
            if len(queue_lengths) > 1:
                q_mean = np.mean(queue_lengths)
                q_std = np.std(queue_lengths)
                q_cv = (q_std / max(1e-6, q_mean)) if q_mean > 0 else 0.0
                queue_volatility_list.append(min(1.0, q_cv))
            else:
                queue_volatility_list.append(0.0)

            # Same-day completion rate (0–1)
            done_ids = set(metrics.task_completion_time.keys())
            if len(done_ids) > 0:
                same_day = sum(
                    1 for k in done_ids
                    if int(metrics.task_arrival_time.get(k, 0) // DAY_MIN) ==
                       int(metrics.task_completion_time[k] // DAY_MIN)
                )
                completion_rate_list.append(same_day / len(done_ids))
            else:
                completion_rate_list.append(0.0)

            # Throughput rate (tasks/day)
            tasks_completed = len(done_ids)
            throughput_rate_list.append(tasks_completed / num_days)

        # Average metrics across replications
        avg_util = float(np.mean(util_list)) if util_list else 0.0
        avg_corrections = float(np.mean(corrections_pct_list)) if corrections_pct_list else 0.0
        avg_queue_volatility = float(np.mean(queue_volatility_list)) if queue_volatility_list else 0.0
        avg_completion_rate = float(np.mean(completion_rate_list)) if completion_rate_list else 0.0
        avg_throughput = float(np.mean(throughput_rate_list)) if throughput_rate_list else 0.0

        # Availability stress (0–1)
        avail_minutes_per_day = p.get("availability_per_day", {}).get(role, open_minutes_per_day)
        avail_stress = (open_minutes_per_day - float(avail_minutes_per_day)) / open_minutes_per_day
        avail_stress = min(max(avail_stress, 0.0), 1.0)

        # Non-linear transformations
        def transform_utilization(u):
            if u <= 0.75:
                return u / 0.75 * 0.5
            else:
                excess = (u - 0.75) / 0.25
                return 0.5 + 0.5 * (np.exp(2 * excess) - 1) / (np.exp(2) - 1)
        
        util_transformed = transform_utilization(avg_util)
        corrections_transformed = min(1.0, avg_corrections * 2.5)  # Linear scaling, amplified 2.5x
        volatility_transformed = np.sqrt(avg_queue_volatility)
        incompletion = 1.0 - avg_completion_rate
        incompletion_transformed = incompletion ** 0.7
        
        expected_throughput = p["arrivals_per_hour_by_role"].get(role, 1) * open_time_available / 60.0 / num_days
        throughput_ratio = avg_throughput / max(1e-6, expected_throughput)
        throughput_deficit = max(0.0, 1.0 - throughput_ratio)
        throughput_deficit = min(1.0, throughput_deficit)

        # Calculate component scores (0-100 each)
        components = {
            "utilization": 100.0 * util_transformed,
            "availability_stress": 100.0 * avail_stress,
            "corrections": 100.0 * corrections_transformed,
            "task_switching": 100.0 * volatility_transformed,
            "incompletion": 100.0 * incompletion_transformed,
            "throughput_deficit": 100.0 * throughput_deficit
        }

        # Calculate weighted burnout score
        burnout_score = sum(norm_weights[k] * components[k] for k in components.keys())

        burnout_scores[role] = {
            "overall": float(burnout_score),
            "components": {k: float(v) for k, v in components.items()}
        }

    clinic_burnout = np.mean([v["overall"] for v in burnout_scores.values()]) if burnout_scores else 0.0
    return {"by_role": burnout_scores, "overall_clinic": float(clinic_burnout)}

# =============================
# Visualization functions
# =============================
def plot_daily_utilization(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily utilization (% of available time spent working) by role.
    This directly corresponds to the utilization burnout component.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes_per_day = p["open_minutes"]
    
    for role in active_roles:
        daily_util_per_rep = [[] for _ in range(num_days)]
        
        capacity = {
            "Administrative staff": p["frontdesk_cap"],
            "Nurse": p["nurse_cap"],
            "Doctors": p["provider_cap"],
            "Other staff": p["backoffice_cap"]
        }[role]
        
        if capacity == 0:
            continue
        
        avail_minutes_per_day = p.get("availability_per_day", {}).get(role, open_minutes_per_day)
        
        role_queue_step = {
            "Administrative staff": "FD_QUEUE",
            "Nurse": "NU_QUEUE",
            "Doctors": "PR_QUEUE",
            "Other staff": "BO_QUEUE"
        }
        queue_step = role_queue_step.get(role)
        
        avg_service_time = {
            "Administrative staff": p.get("svc_frontdesk", 10),
            "Nurse": p.get("svc_nurse", 15),
            "Doctors": p.get("svc_provider", 20),
            "Other staff": p.get("svc_backoffice", 12)
        }.get(role, 10)
        
        for metrics in all_metrics:
            for d in range(num_days):
                day_start = d * DAY_MIN
                day_end = day_start + open_minutes_per_day
                
                # Count tasks served by this role today
                tasks_served_today = sum(1 for t, name, step, note, arr in metrics.events
                                        if step == queue_step and day_start <= t < day_end)
                
                # Estimate work time
                available_capacity_minutes = capacity * avail_minutes_per_day
                estimated_work_time = tasks_served_today * avg_service_time
                daily_util = min(1.0, estimated_work_time / max(1, available_capacity_minutes)) * 100
                
                daily_util_per_rep[d].append(daily_util)
        
        # Calculate mean and std
        means = [np.mean(daily_util_per_rep[d]) if daily_util_per_rep[d] else 0 
                for d in range(num_days)]
        stds = [np.std(daily_util_per_rep[d]) if len(daily_util_per_rep[d]) > 1 else 0 
               for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line
        ax.plot(x, means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)

    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
    ax.set_title('Daily Staff Utilization', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    return fig
    
def plot_queue_over_time(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes = p["open_minutes"]
    
    for role in active_roles:
        daily_queues = []
        for metrics in all_metrics:
            role_daily = []
            for day in range(num_days):
                end_of_open_time = day * DAY_MIN + open_minutes
                
                if len(metrics.time_stamps) > 0:
                    closest_idx = min(range(len(metrics.time_stamps)), 
                                    key=lambda i: abs(metrics.time_stamps[i] - end_of_open_time))
                    role_daily.append(metrics.queues[role][closest_idx])
                else:
                    role_daily.append(0)
            daily_queues.append(role_daily)
        
        if daily_queues:
            daily_array = np.array(daily_queues)
            mean_daily = np.mean(daily_array, axis=0)
            std_daily = np.std(daily_array, axis=0)
            
            x = np.arange(1, num_days + 1)
            
            # Plot line with markers
            ax.plot(x, mean_daily, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
            
            # Add confidence band
            ax.fill_between(x, mean_daily - std_daily, mean_daily + std_daily,
                          color=colors.get(role, '#333333'), alpha=0.15)
    
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Queue Length (end of day)', fontsize=11, fontweight='bold')
    ax.set_title('Backlog', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig
    
def plot_daily_throughput(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily throughput (tasks completed) by role over time with SD shading.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes = p["open_minutes"]
    
    for role in active_roles:
        daily_completed_lists = [[] for _ in range(num_days)]
        
        # Role prefix mapping
        role_prefix_map = {
            "Administrative staff": "AD",
            "Nurse": "NU",
            "Doctors": "DO",
            "Other staff": "OT"
        }
        prefix = role_prefix_map.get(role, "")
        
        for metrics in all_metrics:
            for d in range(num_days):
                start_t = d * DAY_MIN
                end_t = start_t + open_minutes
                
                # Count tasks for THIS ROLE completed during this day
                completed = sum(1 for task_id, ct in metrics.task_completion_time.items() 
                               if start_t <= ct < end_t and task_id.startswith(prefix))
                daily_completed_lists[d].append(completed)
        
        # Calculate mean and std for each day
        daily_means = [np.mean(daily_completed_lists[d]) for d in range(num_days)]
        daily_stds = [np.std(daily_completed_lists[d]) for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line with markers for this role
        ax.plot(x, daily_means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band (±1 SD)
        upper_bound = [daily_means[i] + daily_stds[i] for i in range(num_days)]
        lower_bound = [max(0, daily_means[i] - daily_stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower_bound, upper_bound, color=colors.get(role, '#333333'), alpha=0.1)
    
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasks Completed', fontsize=11, fontweight='bold')
    ax.set_title('Number Processed', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig
    
def plot_response_time_distribution(all_metrics: List[Metrics], p: Dict):
    """
    Histogram showing distribution of task completion times in 3-hour bins, auto-scaling up to simulation length.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
    # Collect all turnaround times across ALL replications
    all_turnaround_times = []
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        if len(done_ids) > 0:
            turnaround_times = [comp_times[k] - arr_times.get(k, comp_times[k]) for k in done_ids]
            all_turnaround_times.extend(turnaround_times)
    
    if not all_turnaround_times:
        st.warning("No completed tasks to plot")
        return fig
    
    # Determine simulation length in days and convert to hours
    sim_days = int(p["sim_minutes"] // DAY_MIN)
    max_hours = min(sim_days * 24, 336)  # Cap at 14 days (336 hours) only if sim is longer
    
    # Convert to hours and cap at max_hours
    all_turnaround_hours = [t / 60.0 for t in all_turnaround_times]
    
    # Filter out tasks that exceed max_hours
    all_turnaround_hours_filtered = [h for h in all_turnaround_hours if h < max_hours]
    all_turnaround_hours = all_turnaround_hours_filtered   

    # Define bins: 3-hour bins up to max_hours
    bin_edges_hours = np.arange(0, max_hours + 3, 3)
    
    # Create histogram
    counts, _ = np.histogram(all_turnaround_hours, bins=bin_edges_hours)
    
    # Create x-axis positions (left edge of each bin)
    bin_left_edges = bin_edges_hours[:-1]
    bin_width = 3  # Each bin is 3 hours wide
    
    # Create histogram bars
    ax.bar(bin_left_edges, counts, width=bin_width, 
           align='edge', color='#1f77b4', alpha=0.7, 
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Hours', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    
    # Adaptive title based on whether capped or not
    if sim_days > 14:
        ax.set_title('Distribution of Task Completion Times (capped at 14 days)', fontsize=12, fontweight='bold')
    else:
        ax.set_title('Distribution of Task Completion Times', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, max_hours)
    ax.set_ylim(bottom=0)
    
    # Format y-axis to use comma separator for thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Set x-axis ticks every 24 hours (1 day) to keep it readable
    ax.set_xticks(np.arange(0, max_hours + 24, 24))
    
    # Add vertical lines at day boundaries for clarity
    num_days = int(max_hours // 24)
    for day in range(1, num_days + 1):
        ax.axvline(x=day*24, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    return fig
    
def plot_completion_by_day(all_metrics: List[Metrics], p: Dict):
    """
    Bar chart showing number of tasks completed same day, +1 day, +2 days, etc. (matches simulation length).
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    # Determine simulation length in days
    sim_days = int(p["sim_minutes"] // DAY_MIN)
    
    # Create categories based on simulation length
    categories = ['Same Day']
    for i in range(1, sim_days):
        categories.append(f'+{i} Day{"s" if i > 1 else ""}')
    
    # Collect counts from each replication
    counts_per_rep = {cat: [] for cat in categories}
    
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        category_counts = {cat: 0 for cat in categories}
        
        for task_id in done_ids:
            arrival_time = arr_times.get(task_id, comp_times[task_id])
            completion_time = comp_times[task_id]
            
            arrival_day = int(arrival_time // DAY_MIN)
            completion_day = int(completion_time // DAY_MIN)
            days_diff = completion_day - arrival_day
            
            if days_diff == 0:
                category_counts['Same Day'] += 1
            elif days_diff < len(categories):
                category_counts[categories[days_diff]] += 1
            else:
                # If beyond simulation days, count in last category
                category_counts[categories[-1]] += 1
        
        for cat in categories:
            counts_per_rep[cat].append(category_counts[cat])
    
    # Calculate mean and std for each category
    means = [np.mean(counts_per_rep[cat]) for cat in categories]
    stds = [np.std(counts_per_rep[cat]) for cat in categories]
    
    # Create color gradient from green (good) to red (bad)
    num_categories = len(categories)
    colors = []
    for i in range(num_categories):
        if i == 0:
            colors.append('#2ecc71')  # Green for same day
        else:
            # Gradient from yellow to red based on position
            ratio = (i - 1) / max(1, num_categories - 2)
            if ratio < 0.25:
                colors.append('#f39c12')  # Yellow
            elif ratio < 0.50:
                colors.append('#e67e22')  # Orange
            elif ratio < 0.75:
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#c0392b')  # Dark red
    
    x = np.arange(len(categories))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    
    # Add error bars (visual only, no numbers)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6)
    
    ax.set_xlabel('Completion', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax.set_title('Delays in Days', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    
    # Rotate labels if too many categories
    if num_categories > 15:
        ax.set_xticklabels(categories, fontsize=8, rotation=45, ha='right')
    else:
        ax.set_xticklabels(categories, fontsize=9, rotation=45, ha='right')
    
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_daily_completion_rate(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily same-day completion rate BY ROLE.
    This directly corresponds to the incompletion burnout component.
    Incompletion burnout = 100 - completion rate shown here.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes_per_day = p["open_minutes"]
    
    for role in active_roles:
        daily_completion_rate_per_rep = [[] for _ in range(num_days)]
        
        # Determine which tasks belong to this role (by prefix)
        role_prefix_map = {
            "Administrative staff": "AD",
            "Nurse": "NU",
            "Doctors": "DO",
            "Other staff": "OT"
        }
        prefix = role_prefix_map.get(role, "")
        
        for metrics in all_metrics:
            for d in range(num_days):
                day_start = d * DAY_MIN
                day_end = day_start + open_minutes_per_day
                
                # Count tasks that belong to THIS ROLE and arrived today
                tasks_arrived_today = [k for k, at in metrics.task_arrival_time.items() 
                                      if day_start <= at < day_end and k.startswith(prefix)]
                
                if tasks_arrived_today:
                    tasks_completed_same_day = sum(1 for k in tasks_arrived_today 
                                                   if k in metrics.task_completion_time 
                                                   and metrics.task_completion_time[k] < day_end)
                    completion_rate = (tasks_completed_same_day / len(tasks_arrived_today)) * 100
                    daily_completion_rate_per_rep[d].append(completion_rate)
                else:
                    daily_completion_rate_per_rep[d].append(0.0)
        
        # Calculate mean and std across replications for each day
        means = [np.mean(daily_completion_rate_per_rep[d]) if daily_completion_rate_per_rep[d] else 0 
                for d in range(num_days)]
        stds = [np.std(daily_completion_rate_per_rep[d]) if len(daily_completion_rate_per_rep[d]) > 1 else 0 
               for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line for this role
        ax.plot(x, means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Same-Day Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Same-Day Task Completion', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(0, 105)
    
    # Add reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_daily_workload(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily task arrivals by role with utilization-based burnout thresholds.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    
    for role in active_roles:
        daily_arrivals_per_rep = [[] for _ in range(num_days)]
        
        for metrics in all_metrics:
            arr_times = metrics.task_arrival_time
            arrivals_by_role = metrics.arrivals_by_role
            
            for d in range(num_days):
                start_t = d * DAY_MIN
                end_t = (d + 1) * DAY_MIN
                
                # Count arrivals for this role on this day
                arrivals = sum(1 for task_id, at in arr_times.items() 
                             if start_t <= at < end_t and task_id.startswith(role[:2].upper()))
                daily_arrivals_per_rep[d].append(arrivals)
        
        # Calculate mean and std
        means = [np.mean(daily_arrivals_per_rep[d]) for d in range(num_days)]
        stds = [np.std(daily_arrivals_per_rep[d]) for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line
        ax.plot(x, means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    # Add utilization-based thresholds
    # Calculate capacity-based thresholds (tasks that would lead to 75% and 90% utilization)
    open_minutes = p["open_minutes"]
    
    # Add threshold lines (these are approximate based on average service times)
    y_max = ax.get_ylim()[1]
    
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('New Number Tasks', fontsize=11, fontweight='bold')
    ax.set_title('Daily Workload', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def plot_burnout_over_days(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing burnout score by role over days using DAILY metrics.
    Each day's burnout is calculated based ONLY on that day's activity.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes_per_day = p["open_minutes"]
    
    weights = p.get("burnout_weights", {
        "utilization": 8, "availability_stress": 1,
        "corrections": 8, "task_switching": 1,
        "incompletion": 1, "throughput_deficit": 1
    })
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        st.warning("All burnout weights are 0 - cannot plot burnout over time")
        return fig
    
    norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    for role in active_roles:
        daily_burnout_per_rep = [[] for _ in range(num_days)]
        
        capacity = {
            "Administrative staff": p["frontdesk_cap"],
            "Nurse": p["nurse_cap"],
            "Doctors": p["provider_cap"],
            "Other staff": p["backoffice_cap"]
        }[role]
        
        if capacity == 0:
            continue
        
        avail_minutes_per_day = p.get("availability_per_day", {}).get(role, open_minutes_per_day)
        
        # Map role to INSUFF step code
        loop_step_map = {
            "Administrative staff": "FD_INSUFF",
            "Nurse": "NU_INSUFF",
            "Doctors": "PR_INSUFF",
            "Other staff": "BO_INSUFF"
        }
        insuff_step = loop_step_map[role]
        
        for metrics in all_metrics:
            # Calculate burnout for EACH day independently
            for d in range(num_days):
                day_start = d * DAY_MIN
                day_end = day_start + open_minutes_per_day
                
                # 1. UTILIZATION - based on actual work time vs available time
                # Count tasks that were SERVED by this role today (queued today)
                role_queue_step = {
                    "Administrative staff": "FD_QUEUE",
                    "Nurse": "NU_QUEUE",
                    "Doctors": "PR_QUEUE",
                    "Other staff": "BO_QUEUE"
                }
                queue_step = role_queue_step.get(role)
                
                tasks_served_today = sum(1 for t, name, step, note, arr in metrics.events
                                        if step == queue_step and day_start <= t < day_end)
                
                available_capacity_minutes = capacity * avail_minutes_per_day
                
                # Get average service time for this role
                avg_service_time = {
                    "Administrative staff": p.get("svc_frontdesk", 10),
                    "Nurse": p.get("svc_nurse", 15),
                    "Doctors": p.get("svc_provider", 20),
                    "Other staff": p.get("svc_backoffice", 12)
                }.get(role, 10)
                
                estimated_work_time = tasks_served_today * avg_service_time
                daily_util = min(1.0, estimated_work_time / max(1, available_capacity_minutes))
                
                # 2. corrections - count INSUFF events TODAY ONLY
                daily_loops = sum(1 for t, name, step, note, arr in metrics.events 
                                 if step == insuff_step and day_start <= t < day_end)
                
                # Normalize: 5+ loops = maximum (1.0)
                daily_corrections = min(1.0, daily_loops / 5.0)
                
                # 3. TASK SWITCHING - queue volatility TODAY ONLY
                day_queue_samples = [metrics.queues[role][i] for i, t in enumerate(metrics.time_stamps) 
                                    if day_start <= t < day_end]
                
                if len(day_queue_samples) > 1:
                    q_std = np.std(day_queue_samples)
                    # Normalize by capacity instead of mean to avoid division issues
                    # High std relative to capacity = high volatility
                    daily_volatility = min(1.0, q_std / max(1, capacity * 3))
                else:
                    daily_volatility = 0.0
                
                # 4. INCOMPLETION - tasks that arrived at THIS ROLE today but didn't finish today
                role_step_map = {
                    "Administrative staff": "FD_QUEUE",
                    "Nurse": "NU_QUEUE", 
                    "Doctors": "PR_QUEUE",
                    "Other staff": "BO_QUEUE"
                }
                queue_step = role_step_map[role]

                # Find tasks that queued at this role today
                tasks_at_role_today = set()
                for t, name, step, note, arr in metrics.events:
                    if step == queue_step and day_start <= t < day_end:
                        tasks_at_role_today.add(name)

                if tasks_at_role_today:
                    # Of those tasks, how many completed (anywhere in the workflow) by end of day?
                    tasks_completed_same_day = sum(1 for task_id in tasks_at_role_today
                                                   if task_id in metrics.task_completion_time
                                                   and metrics.task_completion_time[task_id] < day_end)
                    daily_incompletion = 1.0 - (tasks_completed_same_day / len(tasks_at_role_today))
                else:
                    daily_incompletion = 0.0
                
                # 5. THROUGHPUT DEFICIT - completed today vs expected today
                # Map role to task prefix
                role_prefix_map = {
                    "Administrative staff": "AD",
                    "Nurse": "NU",
                    "Doctors": "DO",
                    "Other staff": "OT"
                }
                prefix = role_prefix_map.get(role, "")
                
                # Count completions for THIS ROLE today
                tasks_completed_today = sum(1 for task_id, ct in metrics.task_completion_time.items()
                                           if day_start <= ct < day_end and task_id.startswith(prefix))
                expected_daily = p["arrivals_per_hour_by_role"].get(role, 1) * open_minutes_per_day / 60.0
                
                if expected_daily > 0:
                    daily_throughput_deficit = max(0.0, min(1.0, 1.0 - (tasks_completed_today / expected_daily)))
                else:
                    daily_throughput_deficit = 0.0
                
                # 6. AVAILABILITY STRESS (constant each day)
                avail_stress = (open_minutes_per_day - float(avail_minutes_per_day)) / open_minutes_per_day
                avail_stress = min(max(avail_stress, 0.0), 1.0)
                
                # Convert to 0-100 scale (direct mapping, minimal transformation)
                components = {
                    "utilization": 100.0 * daily_util,
                    "availability_stress": 100.0 * avail_stress,
                    "corrections": 100.0 * daily_corrections,
                    "task_switching": 100.0 * daily_volatility,
                    "incompletion": 100.0 * daily_incompletion,
                    "throughput_deficit": 100.0 * daily_throughput_deficit
                }
                
                # Calculate weighted burnout score for this day
                burnout_score = sum(norm_weights[k] * components[k] for k in components.keys())
                daily_burnout_per_rep[d].append(burnout_score)
        
        # Calculate mean and std across replications for each day
        means = [np.mean(daily_burnout_per_rep[d]) if daily_burnout_per_rep[d] else 0 
                for d in range(num_days)]
        stds = [np.std(daily_burnout_per_rep[d]) if len(daily_burnout_per_rep[d]) > 1 else 0 
               for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line
        ax.plot(x, means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'),  linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    # Add burnout threshold lines
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Burnout Score (0-100)', fontsize=11, fontweight='bold')
    ax.set_title('Burnout', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig
    
def plot_rerouting_by_day(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily reroutes (inappropriate receipt) by role.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    
    for role in active_roles:
        daily_reroutes_per_rep = [[] for _ in range(num_days)]
        
        for metrics in all_metrics:
            # Count reroutes per day by looking at routing events
            daily_reroute_count = [0] * num_days
            
            for t, name, step, note, arr in metrics.events:
                day = int(t // DAY_MIN)
                if day < num_days:
                    # A reroute happens when a task moves between roles (not loops within same role)
                    # Look for QUEUE events that indicate task arrived at a role
                    if step in ["FD_QUEUE", "NU_QUEUE", "PR_QUEUE", "BO_QUEUE"]:
                        # Determine which role this queue belongs to
                        step_to_role = {
                            "FD_QUEUE": "Administrative staff",
                            "NU_QUEUE": "Nurse",
                            "PR_QUEUE": "Doctors",
                            "BO_QUEUE": "Other staff"
                        }
                        current_role = step_to_role.get(step)
                        
                        # Check if task initially arrived at a different role
                        initial_role = None
                        if name.startswith("AD"):
                            initial_role = "Administrative staff"
                        elif name.startswith("NU"):
                            initial_role = "Nurse"
                        elif name.startswith("DO"):
                            initial_role = "Doctors"
                        elif name.startswith("OT"):
                            initial_role = "Other staff"
                        
                        # If current role matches the role we're plotting and it's not the initial role
                        if current_role == role and initial_role != role:
                            daily_reroute_count[day] += 1
            
            for d in range(num_days):
                daily_reroutes_per_rep[d].append(daily_reroute_count[d])
        
        # Calculate mean and std
        means = [np.mean(daily_reroutes_per_rep[d]) for d in range(num_days)]
        stds = [np.std(daily_reroutes_per_rep[d]) for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line
        ax.plot(x, means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Reroutes', fontsize=11, fontweight='bold')
    ax.set_title('Inappropriate Receipt (Rerouting)', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def plot_missing_info_by_day(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily missing info callbacks by role.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurses': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    
    for role in active_roles:
        daily_missing_per_rep = [[] for _ in range(num_days)]
        
        for metrics in all_metrics:
            # Count missing info events per day
            daily_missing_count = [0] * num_days
            
            for t, name, step, note, arr in metrics.events:
                day = int(t // DAY_MIN)
                if day < num_days:
                    # Missing info events are INSUFF steps
                    step_to_role = {
                        "FD_INSUFF": "Administrative staff",
                        "NU_INSUFF": "Nurse",
                        "PR_INSUFF": "Doctors",
                        "BO_INSUFF": "Other staff"
                    }
                    
                    if step in step_to_role and step_to_role[step] == role:
                        daily_missing_count[day] += 1
            
            for d in range(num_days):
                daily_missing_per_rep[d].append(daily_missing_count[d])
        
        # Calculate mean and std
        means = [np.mean(daily_missing_per_rep[d]) for d in range(num_days)]
        stds = [np.std(daily_missing_per_rep[d]) for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line
        ax.plot(x, means, color=colors.get('Nurses' if role == 'Nurse' else role, '#333333'), linewidth=2.5, marker='o', markersize=6, label='Nurses' if role == 'Nurse' else role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    ax.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Missing Info Events', fontsize=11, fontweight='bold')
    ax.set_title('Missing Information or Corrections', fontsize=12, fontweight='bold')
    
    if num_days > 0:
        x_ticks = np.arange(1, num_days + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig
    
def plot_overtime_needed(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Bar chart showing additional hours per day needed to complete all tasks.
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    
    num_days = max(1, p["sim_minutes"] / DAY_MIN)
    
    overtime_lists = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        for role in active_roles:
            capacity = {
                "Administrative staff": p["frontdesk_cap"],
                "Nurse": p["nurse_cap"],
                "Doctors": p["provider_cap"],
                "Other staff": p["backoffice_cap"]
            }[role]
            
            if capacity > 0:
                total_work_needed = metrics.service_time_sum[role]
                avail_minutes_per_day = p.get("availability_per_day", {}).get(role, p["open_minutes"])
                capacity_per_day = capacity * avail_minutes_per_day
                total_capacity_available = capacity_per_day * num_days
                overtime_minutes = max(0, total_work_needed - total_capacity_available)
                
                if overtime_minutes > 0:
                    overtime_hours_total = overtime_minutes / 60.0
                    overtime_hours_per_day = overtime_hours_total / num_days
                    overtime_hours_per_person = overtime_hours_per_day / capacity
                else:
                    overtime_hours_per_person = 0.0
                
                overtime_lists[role].append(overtime_hours_per_person)
    
    means = [np.mean(overtime_lists[r]) for r in active_roles]
    stds = [np.std(overtime_lists[r]) for r in active_roles]
    
    colors = []
    for mean_ot in means:
        if mean_ot < 0.5:
            colors.append('#2ecc71')
        elif mean_ot < 1.0:
            colors.append('#f39c12')
        elif mean_ot < 2.0:
            colors.append('#e67e22')
        else:
            colors.append('#e74c3c')
    
    x = np.arange(len(active_roles))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6)
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Additional Hours per Day per Person', fontsize=10)
    ax.set_title('Additional FTE Needed to Avoid Backlog', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Nurses' if role == 'Nurse' else role for role in active_roles], fontsize=9, rotation=15, ha='right')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., height + max(0.05, std),
                    f'{mean:.1f}h\n±{std:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                    '0.0h',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='gray')
    
    plt.tight_layout()
    return fig

def help_icon(help_text: str, title: str = None):
    if title is None:
        title = "How is this calculated?"
    with st.expander(title):
        st.caption(help_text)

def aggregate_replications(p: Dict, all_metrics: List[Metrics], active_roles: List[str]):
    num_reps = len(all_metrics)
    
    def fmt_mean_std(values):
        m = np.mean(values)
        s = np.std(values, ddof=1) if len(values) > 1 else 0.0
        return f"{m:.1f} ± {s:.1f}"
    
    def fmt_mean_std_pct(values):
        m = np.mean(values)
        s = np.std(values, ddof=1) if len(values) > 1 else 0.0
        return f"{m:.1f}% ± {s:.1f}%"
    
    flow_avg_list = []
    flow_med_list = []
    same_day_list = []
    time_at_role_lists = {r: [] for r in ROLES}
    
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        if len(done_ids) > 0:
            tt = np.array([comp_times[k] - arr_times.get(k, comp_times[k]) for k in done_ids])
            flow_avg_list.append(float(np.mean(tt)))
            flow_med_list.append(float(np.median(tt)))
            
            for r in ROLES:
                time_at_role_lists[r].append(metrics.service_time_sum[r] / len(done_ids))
            
            same_day = sum(1 for k in done_ids 
                          if int(arr_times.get(k, 0) // DAY_MIN) == int(comp_times[k] // DAY_MIN))
            same_day_list.append(100.0 * same_day / len(done_ids))
        else:
            flow_avg_list.append(0.0)
            flow_med_list.append(0.0)
            same_day_list.append(0.0)
            for r in ROLES:
                time_at_role_lists[r].append(0.0)
    
    flow_df = pd.DataFrame([
        {"Metric": "Average turnaround time (minutes)", "Value": fmt_mean_std(flow_avg_list)},
        {"Metric": "Median turnaround time (minutes)", "Value": fmt_mean_std(flow_med_list)},
        {"Metric": "Same-day completion", "Value": fmt_mean_std_pct(same_day_list)}
    ])
    
    time_at_role_df = pd.DataFrame([
        {"Role": r, "Avg time at role (min) per completed task": fmt_mean_std(time_at_role_lists[r])}
        for r in active_roles
    ])
    
    q_avg_lists = {r: [] for r in ROLES}
    q_max_lists = {r: [] for r in ROLES}
    
    for metrics in all_metrics:
        for r in ROLES:
            q_avg_lists[r].append(np.mean(metrics.queues[r]) if len(metrics.queues[r]) > 0 else 0.0)
            q_max_lists[r].append(np.max(metrics.queues[r]) if len(metrics.queues[r]) > 0 else 0)
    
    queue_df = pd.DataFrame([
        {"Role": r, "Avg queue length": fmt_mean_std(q_avg_lists[r]), "Max queue length": fmt_mean_std(q_max_lists[r])}
        for r in active_roles
    ])
    
    corrections_pct_list = []
    loop_counts_lists = {"Administrative staff": [], "Nurse": [], "Doctors": [], "Other staff": []}
    
    for metrics in all_metrics:
        corrections_tasks = set()
        for t, name, step, note, _arr in metrics.events:
            if step.endswith("INSUFF") or "RECHECK" in step:
                corrections_tasks.add(name)
        
        done_ids = set(metrics.task_completion_time.keys())
        corrections_pct_list.append(100.0 * len(corrections_tasks & done_ids) / max(1, len(done_ids)))
        
        loop_counts_lists["Administrative staff"].append(metrics.loop_fd_insufficient)
        loop_counts_lists["Nurse"].append(metrics.loop_nurse_insufficient)
        loop_counts_lists["Doctors"].append(metrics.loop_provider_insufficient)
        loop_counts_lists["Other staff"].append(metrics.loop_backoffice_insufficient)
    
    corrections_overview_df = pd.DataFrame([
        {"Metric": "% tasks with any corrections", "Value": fmt_mean_std_pct(corrections_pct_list)}
    ])
    
    total_loops_list = [sum(loop_counts_lists[r][i] for r in ROLES) for i in range(num_reps)]
    loop_origin_df = pd.DataFrame([
        {"Role": r, "Loop Count": fmt_mean_std(loop_counts_lists[r]),
         "Share": fmt_mean_std_pct([100.0 * loop_counts_lists[r][i] / total_loops_list[i] 
                                   if total_loops_list[i] > 0 else 0.0 for i in range(num_reps)])}
        for r in active_roles
    ])
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    daily_arrivals_lists = [[] for _ in range(num_days)]
    daily_completed_lists = [[] for _ in range(num_days)]
    daily_from_prev_lists = [[] for _ in range(num_days)]
    daily_for_next_lists = [[] for _ in range(num_days)]
    
    for metrics in all_metrics:
        arr_times = metrics.task_arrival_time
        comp_times = metrics.task_completion_time
        
        for d in range(num_days):
            start_t = d * DAY_MIN
            end_t = (d + 1) * DAY_MIN
            
            arrivals_today = sum(1 for k, at in arr_times.items() if start_t <= at < end_t)
            completed_today = sum(1 for k, ct in comp_times.items() if start_t <= ct < end_t)
            from_prev = sum(1 for k, at in arr_times.items() 
                          if at < start_t and (k not in comp_times or comp_times[k] >= start_t))
            for_next = sum(1 for k, at in arr_times.items() 
                         if at < end_t and (k not in comp_times or comp_times[k] >= end_t))
            
            daily_arrivals_lists[d].append(arrivals_today)
            daily_completed_lists[d].append(completed_today)
            daily_from_prev_lists[d].append(from_prev)
            daily_for_next_lists[d].append(for_next)
    
    throughput_rows = []
    for d in range(num_days):
        throughput_rows.append({
            "Day": d + 1,
            "Total tasks that day": fmt_mean_std(daily_arrivals_lists[d]),
            "Completed tasks": fmt_mean_std(daily_completed_lists[d]),
            "Tasks from previous day": fmt_mean_std(daily_from_prev_lists[d]),
            "Tasks for next day": fmt_mean_std(daily_for_next_lists[d])
        })
    throughput_full_df = pd.DataFrame(throughput_rows)
    
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    denom = {
        "Administrative staff": max(1, p["frontdesk_cap"]) * open_time_available,
        "Nurse": max(1, p["nurse_cap"]) * open_time_available,
        "Doctors": max(1, p["provider_cap"]) * open_time_available,
        "Other staff": max(1, p["backoffice_cap"]) * open_time_available,
    }
    
    util_lists = {r: [] for r in ROLES}
    for metrics in all_metrics:
        for r in ROLES:
            util_lists[r].append(100.0 * metrics.service_time_sum[r] / max(1, denom[r]))
    
    util_overall_list = [np.mean([util_lists[r][i] for r in ROLES]) for i in range(num_reps)]
    
    util_rows = [{"Role": r, "Utilization": fmt_mean_std_pct(util_lists[r])} for r in active_roles]
    util_rows.append({"Role": "Overall", "Utilization": fmt_mean_std_pct(util_overall_list)})
    util_df = pd.DataFrame(util_rows)
    
    summary_row = {
        "Name": "",
        "Avg turnaround (min)": np.mean(flow_avg_list),
        "Median turnaround (min)": np.mean(flow_med_list),
        "Same-day completion (%)": np.mean(same_day_list),
        "corrections (% of completed)": np.mean(corrections_pct_list),
        "Utilization overall (%)": np.mean(util_overall_list),
    }
    summary_df = pd.DataFrame([summary_row])
    
    return {
        "flow_df": flow_df, "time_at_role_df": time_at_role_df, "queue_df": queue_df,
        "corrections_overview_df": corrections_overview_df, "loop_origin_df": loop_origin_df,
        "throughput_full_df": throughput_full_df, "util_df": util_df, "summary_df": summary_df
    }

def create_summary_table(all_metrics: List[Metrics], p: Dict, burnout_data: Dict, active_roles: List[str]):
    """
    Create a summary table organized by role (columns) and metric (rows).
    """
    num_reps = len(all_metrics)
    
    # Missing info by role (INSUFF events - correction loops)
    missing_info_by_role = {r: [] for r in active_roles}
    for metrics in all_metrics:
        role_missing = {r: 0 for r in active_roles}
        
        for t, name, step, note, _arr in metrics.events:
            step_to_role = {
                "FD_INSUFF": "Administrative staff",
                "NU_INSUFF": "Nurse",
                "PR_INSUFF": "Doctors",
                "BO_INSUFF": "Other staff"
            }
            if step in step_to_role:
                role_missing[step_to_role[step]] += 1
        
        for role in active_roles:
            missing_info_by_role[role].append(role_missing[role])
    
    # Re-routes by role (tasks that arrived at wrong role and got forwarded)
    reroutes_by_role = {r: [] for r in active_roles}
    for metrics in all_metrics:
        role_reroutes = {r: 0 for r in active_roles}
        
        for t, name, step, note, arr in metrics.events:
            # A reroute is when a task queues at a role that didn't originally receive it
            if step in ["FD_QUEUE", "NU_QUEUE", "PR_QUEUE", "BO_QUEUE"]:
                step_to_role = {
                    "FD_QUEUE": "Administrative staff",
                    "NU_QUEUE": "Nurse",
                    "PR_QUEUE": "Doctors",
                    "BO_QUEUE": "Other staff"
                }
                current_role = step_to_role[step]
                
                # Determine initial role from task ID prefix
                initial_role = None
                if name.startswith("AD"):
                    initial_role = "Administrative staff"
                elif name.startswith("NU"):
                    initial_role = "Nurse"
                elif name.startswith("DO"):
                    initial_role = "Doctors"
                elif name.startswith("OT"):
                    initial_role = "Other staff"
                
                # Count as reroute if current role != initial role
                # AND it's not a recheck queue (those are corrections, not reroutes)
                if current_role != initial_role and "RECHECK" not in step:
                    role_reroutes[current_role] += 1
        
        for role in active_roles:
            reroutes_by_role[role].append(role_reroutes[role])
    
    # Utilization by role
    num_days = p["sim_minutes"] / DAY_MIN
    util_by_role = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        for role in active_roles:
            capacity = {
                "Administrative staff": p["frontdesk_cap"],
                "Nurse": p["nurse_cap"],
                "Doctors": p["provider_cap"],
                "Other staff": p["backoffice_cap"]
            }[role]
            
            if capacity > 0:
                total_service = metrics.service_time_sum[role]
                avail_minutes_per_day = p.get("availability_per_day", {}).get(role, p["open_minutes"])
                total_available_capacity = capacity * num_days * avail_minutes_per_day
                util = 100.0 * min(1.0, total_service / max(1, total_available_capacity))
                util_by_role[role].append(util)
    
    # Burnout by role
    # Burnout by role - calculate as average of daily burnout scores
    burnout_by_role = {}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes_per_day = p["open_minutes"]
    
    weights = p.get("burnout_weights", {
        "utilization": 8, "availability_stress": 1,
        "corrections": 8, "task_switching": 1,
        "incompletion": 1, "throughput_deficit": 1
    })
    
    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
        
        for role in active_roles:
            capacity = {
                "Administrative staff": p["frontdesk_cap"],
                "Nurse": p["nurse_cap"],
                "Doctors": p["provider_cap"],
                "Other staff": p["backoffice_cap"]
            }[role]
            
            if capacity == 0:
                burnout_by_role[role] = 0.0
                continue
            
            avail_minutes_per_day = p.get("availability_per_day", {}).get(role, open_minutes_per_day)
            
            # Map role to INSUFF step code
            loop_step_map = {
                "Administrative staff": "FD_INSUFF",
                "Nurse": "NU_INSUFF",
                "Doctors": "PR_INSUFF",
                "Other staff": "BO_INSUFF"
            }
            insuff_step = loop_step_map[role]
            
            role_queue_step = {
                "Administrative staff": "FD_QUEUE",
                "Nurse": "NU_QUEUE",
                "Doctors": "PR_QUEUE",
                "Other staff": "BO_QUEUE"
            }
            queue_step = role_queue_step.get(role)
            
            avg_service_time = {
                "Administrative staff": p.get("svc_frontdesk", 10),
                "Nurse": p.get("svc_nurse", 15),
                "Doctors": p.get("svc_provider", 20),
                "Other staff": p.get("svc_backoffice", 12)
            }.get(role, 10)
            
            role_prefix_map = {
                "Administrative staff": "AD",
                "Nurse": "NU",
                "Doctors": "DO",
                "Other staff": "OT"
            }
            prefix = role_prefix_map.get(role, "")
            
            # Calculate daily burnout for each replication, then average
            all_daily_burnouts = []
            
            for metrics in all_metrics:
                for d in range(num_days):
                    day_start = d * DAY_MIN
                    day_end = day_start + open_minutes_per_day
                    
                    # 1. UTILIZATION
                    tasks_served_today = sum(1 for t, name, step, note, arr in metrics.events
                                            if step == queue_step and day_start <= t < day_end)
                    available_capacity_minutes = capacity * avail_minutes_per_day
                    estimated_work_time = tasks_served_today * avg_service_time
                    daily_util = min(1.0, estimated_work_time / max(1, available_capacity_minutes))
                    
                    # 2. corrections
                    daily_loops = sum(1 for t, name, step, note, arr in metrics.events 
                                     if step == insuff_step and day_start <= t < day_end)
                    daily_corrections = min(1.0, daily_loops / 5.0)
                    
                    # 3. TASK SWITCHING
                    day_queue_samples = [metrics.queues[role][i] for i, t in enumerate(metrics.time_stamps) 
                                        if day_start <= t < day_end]
                    if len(day_queue_samples) > 1:
                        q_std = np.std(day_queue_samples)
                        daily_volatility = min(1.0, q_std / max(1, capacity * 3))
                    else:
                        daily_volatility = 0.0
                    
                    # 4. INCOMPLETION
                    tasks_at_role_today = set()
                    for t, name, step, note, arr in metrics.events:
                        if step == queue_step and day_start <= t < day_end:
                            tasks_at_role_today.add(name)
                    
                    if tasks_at_role_today:
                        tasks_completed_same_day = sum(1 for task_id in tasks_at_role_today
                                                       if task_id in metrics.task_completion_time
                                                       and metrics.task_completion_time[task_id] < day_end)
                        daily_incompletion = 1.0 - (tasks_completed_same_day / len(tasks_at_role_today))
                    else:
                        daily_incompletion = 0.0
                    
                    # 5. THROUGHPUT DEFICIT
                    tasks_completed_today = sum(1 for task_id, ct in metrics.task_completion_time.items()
                                               if day_start <= ct < day_end and task_id.startswith(prefix))
                    expected_daily = p["arrivals_per_hour_by_role"].get(role, 1) * open_minutes_per_day / 60.0
                    
                    if expected_daily > 0:
                        daily_throughput_deficit = max(0.0, min(1.0, 1.0 - (tasks_completed_today / expected_daily)))
                    else:
                        daily_throughput_deficit = 0.0
                    
                    # 6. AVAILABILITY STRESS
                    avail_stress = (open_minutes_per_day - float(avail_minutes_per_day)) / open_minutes_per_day
                    avail_stress = min(max(avail_stress, 0.0), 1.0)
                    
                    # Calculate burnout for this day
                    components = {
                        "utilization": 100.0 * daily_util,
                        "availability_stress": 100.0 * avail_stress,
                        "corrections": 100.0 * daily_corrections,
                        "task_switching": 100.0 * daily_volatility,
                        "incompletion": 100.0 * daily_incompletion,
                        "throughput_deficit": 100.0 * daily_throughput_deficit
                    }
                    
                    burnout_score = sum(norm_weights[k] * components[k] for k in components.keys())
                    all_daily_burnouts.append(burnout_score)
            
            # Average across all days and replications
            burnout_by_role[role] = np.mean(all_daily_burnouts) if all_daily_burnouts else 0.0
    else:
        burnout_by_role = {r: 0.0 for r in active_roles}
    
    # Build the table data
    table_data = {
        "Metric (daily)": ["Re-routes", "Missing info", "Workload % utilization", "Burnout", "Items completed", "Backlog"]
    }
    
    # Add columns for each role
    for role in active_roles:
        col_name = role.replace("Administrative staff", "Administrative staff").replace("Doctors", "Doctors")
        
        reroutes_mean = np.mean(reroutes_by_role[role])
        reroutes_std = np.std(reroutes_by_role[role], ddof=1) if len(reroutes_by_role[role]) > 1 else 0.0
        
        missing_mean = np.mean(missing_info_by_role[role])
        missing_std = np.std(missing_info_by_role[role], ddof=1) if len(missing_info_by_role[role]) > 1 else 0.0
        
        util_mean = np.mean(util_by_role[role]) if util_by_role[role] else 0.0
        util_std = np.std(util_by_role[role], ddof=1) if len(util_by_role[role]) > 1 else 0.0
        
        burnout_val = burnout_by_role[role]
        
        # NEW CODE STARTS HERE ============================================
        # Items completed per day BY ROLE
        role_prefix_map = {
            "Administrative staff": "AD",
            "Nurse": "NU",
            "Doctors": "DO",
            "Other staff": "OT"
        }
        prefix = role_prefix_map.get(role, "")
        
        num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
        completed_per_day_list = []
        
        for metrics in all_metrics:
            for d in range(num_days):
                day_start = d * DAY_MIN
                day_end = day_start + p["open_minutes"]
                
                completed_today = sum(1 for task_id, ct in metrics.task_completion_time.items()
                                     if day_start <= ct < day_end and task_id.startswith(prefix))
                completed_per_day_list.append(completed_today)
        
        completed_mean = np.mean(completed_per_day_list) if completed_per_day_list else 0.0
        completed_std = np.std(completed_per_day_list, ddof=1) if len(completed_per_day_list) > 1 else 0.0
        
        # Backlog at end of each day BY ROLE
        backlog_per_day_list = []
        
        for metrics in all_metrics:
            for d in range(num_days):
                day_end_time = d * DAY_MIN + p["open_minutes"]
                
                if len(metrics.time_stamps) > 0:
                    closest_idx = min(range(len(metrics.time_stamps)), 
                                    key=lambda i: abs(metrics.time_stamps[i] - day_end_time))
                    backlog_per_day_list.append(metrics.queues[role][closest_idx])
                else:
                    backlog_per_day_list.append(0)
        
        backlog_mean = np.mean(backlog_per_day_list) if backlog_per_day_list else 0.0
        backlog_std = np.std(backlog_per_day_list, ddof=1) if len(backlog_per_day_list) > 1 else 0.0
        # NEW CODE ENDS HERE ==============================================
        
        table_data[col_name] = [
            f"{reroutes_mean:.1f} ± {reroutes_std:.1f}",
            f"{missing_mean:.1f} ± {missing_std:.1f}",
            f"{util_mean:.1f}% ± {util_std:.1f}%",
            f"{burnout_val:.1f}",
            f"{completed_mean:.1f} ± {completed_std:.1f}",
            f"{backlog_mean:.1f} ± {backlog_std:.1f}"
        ]
    
    df = pd.DataFrame(table_data)
    return df

def _excel_engine():
    try:
        import xlsxwriter
        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl
            return "openpyxl"
        except Exception:
            return None

def create_excel_download(all_metrics: List[Metrics], p: Dict) -> BytesIO:
    """
    Create an Excel file with the event log ONLY.
    """
    output = BytesIO()
    
    engine = _excel_engine()
    if engine is None:
        st.error("No Excel engine available. Install xlsxwriter or openpyxl.")
        return None
    
    all_events_data = []
    for rep_idx, metrics in enumerate(all_metrics):
        for t, name, step, note, arr in metrics.events:
            all_events_data.append({
                "Replication": rep_idx + 1,
                "Time (min)": float(t),
                "Day": int(t // DAY_MIN) + 1,
                "Task ID": name,
                "Step Code": step,
                "Step Description": pretty_step(step),
                "Note": note,
                "Arrival Time (min)": (float(arr) if arr is not None else None),
            })
    
    events_df = pd.DataFrame(all_events_data)
    
    with pd.ExcelWriter(output, engine=engine) as writer:
        events_df.to_excel(writer, sheet_name='Event Log', index=False)
    
    output.seek(0)
    return output


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Community Health Center Workflow Model", layout="wide")
st.title("Community Health Center Workflow Model")
st.subheader("Healthcare Systems Engineering Institute (AHRQ grant #R01HS028458)")

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "results" not in st.session_state:
    st.session_state["results"] = None
if "design" not in st.session_state:
    st.session_state["design"] = None
if "design_saved" not in st.session_state:
    st.session_state.design_saved = False
if "ran" not in st.session_state:
    st.session_state.ran = False
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

def go_next():
    st.session_state.wizard_step = min(2, st.session_state.wizard_step + 1)

def go_back():
    st.session_state.wizard_step = max(1, st.session_state.wizard_step - 1)

def _init_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def prob_input(label: str, key: str, default: float = 0.0, help: str | None = None, disabled: bool = False) -> float:
    if key not in st.session_state:
        st.session_state[key] = f"{float(default):.2f}"
    raw = st.text_input(label, value=st.session_state[key], key=key, help=help, disabled=disabled)
    try:
        val = float(str(raw).replace(",", "."))
    except ValueError:
        val = float(default)
    val = max(0.0, min(1.0, val))
    st.caption(f"{val:.2f}")
    return val

# -------- STEP 1: DESIGN --------
if st.session_state.wizard_step == 1:
    
    with st.expander("ℹ️ About This Model", expanded=False):
        st.markdown("""
        This model simulates the workflow of CHC paperwork processes to help analyze system performance, work burden, and 
        associated burnout and to help evaluate potential process improvements and interventions.
        
        **How tool works:**
        - Patient-initiated paperwork, calls, or portal messages are received by various staff (nurses, doctors, staff)
        - Staff process these based on availability and processing times
        - Items received by the wrong type of personnel are routed appropriately
        - The model tracks daily workload, inefficiency, response delays, and contribution to burnout for each type of personnel
        
        **How to use:**
        1. **Define your clinic** below by setting staffing levels, routing logic, and processing times
        2. **Click "Save"** to store your scenario
        3. **Click "Run Simulation"** to see results including burnout scores, utilization, and bottlenecks
        4. **Try different scenarios** to test interventions (e.g. more staff, reduce volumes, improve workflows)
        
        **Tips:**
        - Start with inputs that define the current process
        - Change one thing at a time to understand its impact
        - Pay attention to roles showing high utilization (>75%) or high burnout (>50)
        - Use results to quickly compare different scenarios
        """)
    
    st.markdown("---")
    
    def route_row_ui(from_role: str, defaults: Dict[str, float], disabled_source: bool = False, 
                     fd_cap_val: int = 0, nu_cap_val: int = 0, pr_cap_val: int = 0, bo_cap_val: int = 0) -> Dict[str, float]:
        current_cap_map = {"Administrative staff": fd_cap_val, "Nurse": nu_cap_val, "Doctors": pr_cap_val, "Other staff": bo_cap_val}
        st.markdown(f"**{from_role} →**")
        targets = [r for r in ROLES if r != from_role] + [DONE]
        cols = st.columns(len(targets))
        row: Dict[str, float] = {}
        for i, tgt in enumerate(targets):
            tgt_disabled = disabled_source or (tgt in ROLES and current_cap_map[tgt] == 0)
            if tgt == DONE:
                label_name = "Done"
            elif tgt == "Nurse":
                label_name = "Send to Nurses"
            elif tgt == "Doctors":
                label_name = "Send to Doctors"
            elif tgt == "Administrative staff":
                label_name = "Send to Administrative Staff"
            elif tgt == "Other staff":
                label_name = "Send to Other Staff"
            else:
                label_name = tgt
            key_name = f"r_{from_role}_{'done' if tgt==DONE else label_name.replace(' ','_').lower()}"
            default_val = float(defaults.get(tgt, 0.0))
            with cols[i]:
                val = prob_input(f"{label_name}", key=key_name, 
                               default=(0.0 if tgt_disabled else default_val), disabled=tgt_disabled)
                if tgt_disabled:
                    val = 0.0
            row[tgt] = val
        return row
                         
    with st.form("design_form", clear_on_submit=False):
        with st.expander("Simulation Horizon", expanded=False):
            sim_days = st.number_input("Days to simulate", 1, 30, _init_ss("sim_days", 5), 1, "%d",
                                   help="Number of clinic operating days to simulate")
            open_hours = st.number_input("Hours open per day", 1, 24, _init_ss("open_hours", 8), 1, "%d",
                                      help="Number of hours the clinic is open each day")
    
        seed = 42  # Fixed seed for reproducibility

        with st.expander("Roles", expanded=False):            
            st.markdown("##### Administrative staff")
            cFD1, cFD2, cFD3 = st.columns(3)
            with cFD1:
                fd_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("fd_cap", 3), 1, "%d", key="fd_cap_input",
                                               help="Number of Administrative staff")
            with cFD2:
                arr_fd = st.number_input("Volume items per day", 0, 5000, _init_ss("arr_fd", 32), 1, "%d", disabled=(fd_cap_form==0), key="arr_fd_input",
                         help="Average number of tasks per day")
            with cFD3:
                avail_fd = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_fd", 240), 1, "%d", disabled=(fd_cap_form==0), key="avail_fd_input",
                               help="Minutes per day available for work (max = hours open × 60)")
            
            st.markdown("---")
            
            st.markdown("##### Nurses")
            cNU1, cNU2, cNU3 = st.columns(3)
            with cNU1:
                nu_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("nurse_cap", 3), 1, "%d", key="nurse_cap_input",
                                                  help="Number of nurses or medical assistants")
            with cNU2:
                arr_nu = st.number_input("Volume items per day", 0, 5000, _init_ss("arr_nu", 24), 1, "%d", disabled=(nu_cap_form==0), key="arr_nu_input",
                         help="Average number of tasks per day")
            with cNU3:
                avail_nu = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_nu", 120), 1, "%d", disabled=(nu_cap_form==0), key="avail_nu_input",
                               help="Minutes per day available for work (max = hours open × 60)")
            
            st.markdown("---")
            
            st.markdown("##### Doctors")
            cPR1, cPR2, cPR3 = st.columns(3)
            with cPR1:
                pr_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("provider_cap", 2), 1, "%d", key="provider_cap_input",
                                                     help="Number of Doctors")
            with cPR2:
                arr_pr = st.number_input("Volume items per day", 0, 5000, _init_ss("arr_pr", 16), 1, "%d", disabled=(pr_cap_form==0), key="arr_pr_input",
                         help="Average number of tasks per day")
            with cPR3:
                avail_pr = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_pr", 60), 1, "%d", disabled=(pr_cap_form==0), key="avail_pr_input",
                               help="Minutes per day available for work (max = hours open × 60)")
            
            st.markdown("---")
            
            st.markdown("##### Other staff")
            cBO1, cBO2, cBO3 = st.columns(3)
            with cBO1:
                bo_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("backoffice_cap", 2), 1, "%d", key="bo_cap_input",
                                               help="Number of Other staff")
            with cBO2:
                 arr_bo = st.number_input("Volume items per day", 0, 5000, _init_ss("arr_bo", 16), 1, "%d", disabled=(bo_cap_form==0), key="arr_bo_input",
                         help="Average number of tasks per day")
            with cBO3:
                avail_bo = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_bo", 180), 1, "%d", disabled=(bo_cap_form==0), key="avail_bo_input",
                               help="Minutes per day available for work (max = hours open × 60)")

        st.markdown("  Simulation Settings")
        st.caption("Configure variability and number of simulation runs")
    
        cv_speed_label = st.select_slider(
            "Task speed variability",
            options=["Very Low", "Low", "Moderate", "High", "Very High"],
            value=_init_ss("cv_speed_label", "Moderate"),
            help="Variation in task completion times"
        )
    
        cv_speed_map = {
            "Very Low": 0.1,
            "Low": 0.2,
            "Moderate": 0.3,
            "High": 0.5,
            "Very High": 0.7
        }
        cv_speed = cv_speed_map[cv_speed_label]

        num_replications = st.number_input("Number of replications", 1, 1000, _init_ss("num_replications", 30), 1, "%d", 
                                      help="Number of independent simulation runs")

        with st.expander("Processing times & routing", expanded=False):
            with st.expander("Administrative staff", expanded=False):
                st.markdown("**Processing time**")
                svc_frontdesk = st.slider("Mean processing time (minutes)", 0.0, 30.0, _init_ss("svc_frontdesk", 3.0), 0.5, disabled=(fd_cap_form==0),
                                      help="Average time to complete a task")
            
                st.markdown("**Correction Loops**")            
                st.markdown("*Missing Information (patient-side):*")
                cFDL1, cFDL2 = st.columns(2)
                with cFDL1:
                    p_fd_insuff = st.slider("Percent with insufficient information", 0.0, 1.0, _init_ss("p_fd_insuff", 0.25), 0.01, disabled=(fd_cap_form==0), key="fd_p_insuff")
                with cFDL2:
                    fd_insuff_delay = st.slider("Delay to obtain information (min)", 0.0, 2400.0, _init_ss("fd_insuff_delay", 240.0), 1.0, disabled=(fd_cap_form==0))
            
                st.markdown("*Corrections (internal errors):*")
                cFDL3, cFDL4 = st.columns(2)
                with cFDL3:
                    p_fd_corrections = st.slider("Percent with corrections needed", 0.0, 1.0, _init_ss("p_fd_corrections", 0.10), 0.01, disabled=(fd_cap_form==0), key="fd_p_corrections")
                with cFDL4:
                    fd_corrections_delay = st.slider("Delay to correct (min)", 0.0, 240.0, _init_ss("fd_corrections_delay", 60.0), 1.0, disabled=(fd_cap_form==0), key="fd_corrections_delay")
            
                max_fd_loops = st.number_input("Maximum number of loops (both types combined)", 0, 10, _init_ss("max_fd_loops", 3), 1, "%d", disabled=(fd_cap_form==0), key="fd_max_loops")
    
                st.markdown("**Disposition**")
                fd_route_defaults = {
                    "Nurse": float(st.session_state.get("saved_r_Administrative staff_to_nurse", 
                                   st.session_state.get("r_Administrative staff_to_nurse", "0.50")).replace(",", ".")),
                    "Doctors": float(st.session_state.get("saved_r_Administrative staff_to_doctors",
                                     st.session_state.get("r_Administrative staff_to_doctors", "0.10")).replace(",", ".")),
                    "Other staff": float(st.session_state.get("saved_r_Administrative staff_to_other_staff",
                                         st.session_state.get("r_Administrative staff_to_other_staff", "0.10")).replace(",", ".")),
                    DONE: float(st.session_state.get("saved_r_Administrative staff_done",
                                st.session_state.get("r_Administrative staff_done", "0.30")).replace(",", "."))
                }
                fd_route = route_row_ui("Administrative staff", fd_route_defaults, 
                                   disabled_source=(fd_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            with st.expander("Nurses", expanded=False):
                st.markdown("**Processing times**")
                cNS1, cNS2 = st.columns(2)
                with cNS1:
                    svc_nurse_protocol = st.slider("Standard process time (minutes)", 0.0, 30.0, _init_ss("svc_nurse_protocol", 2.0), 0.5, disabled=(nu_cap_form==0))
                    p_protocol = st.slider("Percent follow standard process", 0.0, 1.0, _init_ss("p_protocol", 0.30), 0.05, disabled=(nu_cap_form==0))
                with cNS2:
                    svc_nurse = st.slider("Non-standard process time (minutes)", 0.0, 40.0, _init_ss("svc_nurse", 5.0), 0.5, disabled=(nu_cap_form==0))
                
                st.markdown("**Correction Loops**")
                st.caption("Insufficient information = patient-side delays. corrections = internal errors.")
                
                st.markdown("*Missing Information (patient-side):*")
                cNUL1, cNUL2 = st.columns(2)
                with cNUL1:
                    p_nurse_insuff = st.slider("Percent with insufficient information", 0.0, 1.0, _init_ss("p_nurse_insuff", 0.20), 0.01, disabled=(nu_cap_form==0), key="nu_p_insuff")
                with cNUL2:
                    nurse_insuff_delay = st.slider("Delay to obtain information (min)", 0.0, 2400.0, _init_ss("nurse_insuff_delay", 240.0), 1.0, disabled=(nu_cap_form==0), key="nu_insuff_delay")
                
                st.markdown("*Corrections (internal errors):*")
                cNUL3, cNUL4 = st.columns(2)
                with cNUL3:
                    p_nurse_corrections = st.slider("Percent with corrections needed", 0.0, 1.0, _init_ss("p_nurse_corrections", 0.10), 0.01, disabled=(nu_cap_form==0), key="nu_p_corrections")
                with cNUL4:
                    nurse_corrections_delay = st.slider("Delay to correct (min)", 0.0, 240.0, _init_ss("nurse_corrections_delay", 60.0), 1.0, disabled=(nu_cap_form==0), key="nu_corrections_delay")
                
                max_nurse_loops = st.number_input("Maximum number of loops (both types combined)", 0, 10, _init_ss("max_nurse_loops", 3), 1, "%d", disabled=(nu_cap_form==0), key="nu_max_loops")
                
                st.markdown("**Disposition**")
                nu_route_defaults = {
                    "Doctors": float(st.session_state.get("saved_r_Nurse_to_doctors",
                                     st.session_state.get("r_Nurse_to_doctors", "0.40")).replace(",", ".")),
                    "Other staff": float(st.session_state.get("saved_r_Nurse_to_other_staff",
                                         st.session_state.get("r_Nurse_to_other_staff", "0.20")).replace(",", ".")),
                    DONE: float(st.session_state.get("saved_r_Nurse_done",
                                st.session_state.get("r_Nurse_done", "0.40")).replace(",", "."))
                }
                nu_route = route_row_ui("Nurse", nu_route_defaults, 
                                   disabled_source=(nu_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)

            
            with st.expander("Doctors", expanded=False):
                st.markdown("**Processing time**")
                svc_provider = st.slider("Mean Processing time (minutes)", 0.0, 480.0, _init_ss("svc_provider", 7.0), 0.5, disabled=(pr_cap_form==0))
            
                st.markdown("**Correction Loops**")
                st.caption("Insufficient information = patient-side delays. corrections = internal errors.")
                
                st.markdown("*Missing Information (patient-side):*")
                cPRL1, cPRL2 = st.columns(2)
                with cPRL1:
                    p_provider_insuff = st.slider("Percent with insufficient information", 0.0, 1.0, _init_ss("p_provider_insuff", 0.15), 0.01, disabled=(pr_cap_form==0), key="pr_p_insuff")
                with cPRL2:
                    provider_insuff_delay = st.slider("Delay to obtain information (min)", 0.0, 2400.0, _init_ss("provider_insuff_delay", 300.0), 1.0, disabled=(pr_cap_form==0), key="pr_insuff_delay")
                
                st.markdown("*Corrections (internal errors):*")
                cPRL3, cPRL4 = st.columns(2)
                with cPRL3:
                    p_provider_corrections = st.slider("Percent with corrections needed", 0.0, 1.0, _init_ss("p_provider_corrections", 0.10), 0.01, disabled=(pr_cap_form==0), key="pr_p_corrections")
                with cPRL4:
                    provider_corrections_delay = st.slider("Delay to correct (min)", 0.0, 240.0, _init_ss("provider_corrections_delay", 60.0), 1.0, disabled=(pr_cap_form==0), key="pr_corrections_delay")
                
                max_provider_loops = st.number_input("Maximum number of loops (both types combined)", 0, 10, _init_ss("max_provider_loops", 3), 1, "%d", disabled=(pr_cap_form==0), key="pr_max_loops")

                st.markdown("**Disposition**")
                pr_route_defaults = {
                    "Other staff": float(st.session_state.get("saved_r_Doctors_to_other_staff",
                                         st.session_state.get("r_Doctors_to_other_staff", "0.30")).replace(",", ".")),
                    DONE: float(st.session_state.get("saved_r_Doctors_done",
                                st.session_state.get("r_Doctors_done", "0.70")).replace(",", "."))
                }
                pr_route = route_row_ui("Doctors", pr_route_defaults, 
                                   disabled_source=(pr_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            with st.expander("Other staff", expanded=False):
                st.markdown("**Processing time**")
                svc_backoffice = st.slider("Mean Processing time (minutes)", 0.0, 480.0, _init_ss("svc_backoffice", 5.0), 0.5, disabled=(bo_cap_form==0))
            
                st.markdown("**Correction Loops**")
                st.caption("Insufficient information = patient-side delays. corrections = internal errors.")
                
                st.markdown("*Missing Information (patient-side):*")
                cBOL1, cBOL2 = st.columns(2)
                with cBOL1:
                    p_backoffice_insuff = st.slider("Percent with insufficient information", 0.0, 1.0, _init_ss("p_backoffice_insuff", 0.18), 0.01, disabled=(bo_cap_form==0), key="bo_p_insuff")
                with cBOL2:
                    backoffice_insuff_delay = st.slider("Delay to obtain information (min)", 0.0, 2400.0, _init_ss("backoffice_insuff_delay", 180.0), 1.0, disabled=(bo_cap_form==0), key="bo_insuff_delay")
                
                st.markdown("*Corrections (internal errors):*")
                cBOL3, cBOL4 = st.columns(2)
                with cBOL3:
                    p_backoffice_corrections = st.slider("Percent with corrections needed", 0.0, 1.0, _init_ss("p_backoffice_corrections", 0.10), 0.01, disabled=(bo_cap_form==0), key="bo_p_corrections")
                with cBOL4:
                    backoffice_corrections_delay = st.slider("Delay to correct (min)", 0.0, 240.0, _init_ss("backoffice_corrections_delay", 60.0), 1.0, disabled=(bo_cap_form==0), key="bo_corrections_delay")
                
                max_backoffice_loops = st.number_input("Maximum number of loops (both types combined)", 0, 10, _init_ss("max_backoffice_loops", 3), 1, "%d", disabled=(bo_cap_form==0), key="bo_max_loops")

                st.markdown("**Disposition**")
                bo_route_defaults = {
                    "Administrative staff": float(st.session_state.get("saved_r_Other staff_to_administrative_staff",
                                                  st.session_state.get("r_Other staff_to_administrative_staff", "0.10")).replace(",", ".")),
                    "Nurse": float(st.session_state.get("saved_r_Other staff_to_nurse",
                                   st.session_state.get("r_Other staff_to_nurse", "0.10")).replace(",", ".")),
                    "Doctors": float(st.session_state.get("saved_r_Other staff_to_doctors",
                                     st.session_state.get("r_Other staff_to_doctors", "0.10")).replace(",", ".")),
                    DONE: float(st.session_state.get("saved_r_Other staff_done",
                                st.session_state.get("r_Other staff_done", "0.70")).replace(",", "."))
                }
                bo_route = route_row_ui("Other staff", bo_route_defaults, 
                                   disabled_source=(bo_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            route: Dict[str, Dict[str, float]] = {}
            route["Administrative staff"] = fd_route
            route["Nurse"] = nu_route
            route["Doctors"] = pr_route
            route["Other staff"] = bo_route

            with st.expander("Contributors to Burnout - Relative Weights", expanded=False):
                st.caption("Assign each factor a weight between 0 and 10 (0 = no contribution, 10 = maximum contribution)")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Emotional Exhaustion Contributors:**")
                    w_utilization = st.slider("Utilization", 0, 10, _init_ss("w_utilization", 8), 1, 
                                  help="How much does high utilization contribute to burnout?")
                    w_availability_stress = st.slider("Availability Stress", 0, 10, _init_ss("w_availability_stress", 1), 1,
                                         help="How much does limited availability contribute to burnout?")
                
                    st.markdown("**Depersonalization Contributors:**")
                    w_corrections = st.slider("corrections Percentage", 0, 10, _init_ss("w_corrections", 8), 1,
                            help="How much does corrections contribute to burnout?")
                    w_task_switching = st.slider("Task Switching (Queue Volatility)", 0, 10, _init_ss("w_task_switching", 1), 1,
                                    help="How much does unpredictable workload contribute to burnout?")
                
                with col2:
                    st.markdown("**Reduced Accomplishment Contributors:**")
                    w_incompletion = st.slider("Incomplete Tasks", 0, 10, _init_ss("w_incompletion", 1), 1,
                                   help="How much do incomplete tasks contribute to burnout?")
                    w_throughput_deficit = st.slider("Throughput Deficit", 0, 10, _init_ss("w_throughput_deficit", 1), 1,
                                        help="How much does falling behind expected throughput contribute to burnout?")
                
                # Calculate and display normalized weights
                total_weight = w_utilization + w_availability_stress + w_corrections + w_task_switching + w_incompletion + w_throughput_deficit
                if total_weight > 0:
                    st.info(f"**Total weight: {total_weight}** — All scores will be normalized to 0-100 scale")
                else:
                    st.warning("All weights are 0 - burnout scores will be 0")

        saved = st.form_submit_button("Save", type="primary")

        if saved:
            # Save ALL form values back to session state
            st.session_state.sim_days = sim_days
            st.session_state.open_hours = open_hours
            st.session_state.cv_speed_label = cv_speed_label
            st.session_state.num_replications = num_replications

            # Staff capacities
            st.session_state.fd_cap = fd_cap_form
            st.session_state.nurse_cap = nu_cap_form
            st.session_state.provider_cap = pr_cap_form
            st.session_state.backoffice_cap = bo_cap_form

            # Arrivals
            st.session_state.arr_fd = arr_fd
            st.session_state.arr_nu = arr_nu
            st.session_state.arr_pr = arr_pr
            st.session_state.arr_bo = arr_bo

            # Availability
            st.session_state.avail_fd = avail_fd
            st.session_state.avail_nu = avail_nu
            st.session_state.avail_pr = avail_pr
            st.session_state.avail_bo = avail_bo
    
            # Burnout weights
            st.session_state.w_utilization = w_utilization
            st.session_state.w_availability_stress = w_availability_stress
            st.session_state.w_corrections = w_corrections
            st.session_state.w_task_switching = w_task_switching
            st.session_state.w_incompletion = w_incompletion
            st.session_state.w_throughput_deficit = w_throughput_deficit
    
            # Service times
            st.session_state.svc_frontdesk = svc_frontdesk
            st.session_state.svc_nurse_protocol = svc_nurse_protocol
            st.session_state.svc_nurse = svc_nurse
            st.session_state.p_protocol = p_protocol
            st.session_state.svc_provider = svc_provider
            st.session_state.svc_backoffice = svc_backoffice

            # Loop parameters - Administrative staff
            st.session_state.p_fd_insuff = p_fd_insuff
            #st.session_state.fd_insuff_delay = fd_insuff_delay
            #st.session_state.p_fd_corrections = p_fd_corrections
            #st.session_state.fd_corrections_delay = fd_corrections_delay
            st.session_state.max_fd_loops = max_fd_loops
            
            # Loop parameters - Nurse
            st.session_state.p_nurse_insuff = p_nurse_insuff
            st.session_state.nurse_insuff_delay = nurse_insuff_delay
            st.session_state.p_nurse_corrections = p_nurse_corrections
            st.session_state.nurse_corrections_delay = nurse_corrections_delay
            st.session_state.max_nurse_loops = max_nurse_loops
            
            # Loop parameters - Provider
            st.session_state.p_provider_insuff = p_provider_insuff
            st.session_state.provider_insuff_delay = provider_insuff_delay
            st.session_state.p_provider_corrections = p_provider_corrections
            st.session_state.provider_corrections_delay = provider_corrections_delay
            st.session_state.max_provider_loops = max_provider_loops
            
            # Loop parameters - Back office
            st.session_state.p_backoffice_insuff = p_backoffice_insuff
            st.session_state.backoffice_insuff_delay = backoffice_insuff_delay
            st.session_state.p_backoffice_corrections = p_backoffice_corrections
            st.session_state.backoffice_corrections_delay = backoffice_corrections_delay
            st.session_state.max_backoffice_loops = max_backoffice_loops
    
            # NEW: Save routing values explicitly
            # Administrative staff routing
            for target in ["nurse", "doctors", "other_staff", "done"]:
                key = f"r_Administrative staff_to_{target}" if target != "done" else "r_Administrative staff_done"
                if key in st.session_state:
                    st.session_state[f"saved_{key}"] = st.session_state[key]
    
            # Nurse routing
            for target in ["doctors", "other_staff", "done"]:
                key = f"r_Nurse_to_{target}" if target != "done" else "r_Nurse_done"
                if key in st.session_state:
                    st.session_state[f"saved_{key}"] = st.session_state[key]
    
            # Doctors routing
            for target in ["other_staff", "done"]:
                key = f"r_Doctors_to_{target}" if target != "done" else "r_Doctors_done"
                if key in st.session_state:
                    st.session_state[f"saved_{key}"] = st.session_state[key]
    
            # Other staff routing
            for target in ["administrative_staff", "nurse", "doctors", "done"]:
                key = f"r_Other staff_to_{target}" if target != "done" else "r_Other staff_done"
                if key in st.session_state:
                    st.session_state[f"saved_{key}"] = st.session_state[key]
        
            # ... rest of the save logic ...
    
            # Calculate derived values
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            # Clean up routes
            for r in ROLES:
                if r in route:
                    route[r].pop(r, None)
            for r in ROLES:
                if r in route:
                    for tgt in list(route[r].keys()):
                        if tgt in ROLES and {"Administrative staff": fd_cap_form, "Nurse": nu_cap_form, "Doctors": pr_cap_form, "Other staff": bo_cap_form}[tgt] == 0:
                            route[r][tgt] = 0.0

            # Save design configuration
            st.session_state["design"] = dict(
                sim_minutes=sim_minutes, open_minutes=open_minutes,
                seed=seed, num_replications=num_replications,
                frontdesk_cap=fd_cap_form, nurse_cap=nu_cap_form,
                provider_cap=pr_cap_form, backoffice_cap=bo_cap_form,
                arrivals_per_hour_by_role={
                    "Administrative staff": int(arr_fd) / int(open_hours), 
                    "Nurse": int(arr_nu) / int(open_hours), 
                    "Doctors": int(arr_pr) / int(open_hours), 
                    "Other staff": int(arr_bo) / int(open_hours)
                },
                availability_per_day={"Administrative staff": int(avail_fd), "Nurse": int(avail_nu),
                      "Doctors": int(avail_pr), "Other staff": int(avail_bo)},
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider, svc_backoffice=svc_backoffice,
                dist_role={"Administrative staff": "normal", "NurseProtocol": "normal", "Nurse": "exponential",
                  "Doctors": "exponential", "Other staff": "exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Administrative staff": 0.5, "Nurse": 0.5, "NurseProtocol": 0.5, "Doctors": 0.5, "Other staff": 0.5},
                burnout_weights={
                    "utilization": w_utilization,
                    "availability_stress": w_availability_stress,
                    "corrections": w_corrections,
                    "task_switching": w_task_switching,
                    "incompletion": w_incompletion,
                    "throughput_deficit": w_throughput_deficit
                },
                p_fd_insuff=p_fd_insuff, p_fd_corrections=p_fd_corrections, 
                fd_insuff_delay=fd_insuff_delay, fd_corrections_delay=fd_corrections_delay, max_fd_loops=max_fd_loops,
                p_nurse_insuff=p_nurse_insuff, p_nurse_corrections=p_nurse_corrections,
                nurse_corrections_delay=nurse_corrections_delay, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, p_provider_corrections=p_provider_corrections, 
                provider_insuff_delay=provider_insuff_delay, provider_corrections_delay=provider_corrections_delay, max_provider_loops=max_provider_loops,
                p_backoffice_insuff=p_backoffice_insuff, p_backoffice_corrections=p_backoffice_corrections, 
                backoffice_insuff_delay=backoffice_insuff_delay, backoffice_corrections_delay=backoffice_corrections_delay, max_backoffice_loops=max_backoffice_loops, 
                p_protocol=p_protocol, route_matrix=route
            )
            st.session_state.design_saved = True
            st.success("Configuration saved successfully")

    if st.session_state.design_saved:
        if st.button("Run Simulation", type="primary", width="stretch"):
            st.session_state.wizard_step = 2
            st.rerun()
        
# -------- STEP 2: RUN & RESULTS --------
elif st.session_state.wizard_step == 2:
    st.markdown("## Simulation Run")
    st.button("← Back to Design", on_click=go_back)

    if not st.session_state["design"]:
        st.info("Use Save on Step 1 first.")
        st.session_state.wizard_step = 1
        st.rerun()

    p = st.session_state["design"]
    seed = p.get("seed", 42)
    num_replications = p.get("num_replications", 30)
        
    active_roles_caps = [("Doctors", p["provider_cap"]), ("Administrative staff", p["frontdesk_cap"]),
                        ("Nurse", p["nurse_cap"]), ("Other staff", p["backoffice_cap"])]
    active_roles = [r for r, cap in active_roles_caps if cap > 0]
    
    all_metrics = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for rep in range(num_replications):
        status_text.text(f"Running replication {rep + 1} of {num_replications}...")
        metrics = run_single_replication(p, seed + rep)
        all_metrics.append(metrics)
        progress_bar.progress((rep + 1) / num_replications)
    
    status_text.text(f"Completed {num_replications} replications")
    progress_bar.empty()
    
    agg_results = aggregate_replications(p, all_metrics, active_roles)
    
    flow_df = agg_results["flow_df"]
    time_at_role_df = agg_results["time_at_role_df"]
    queue_df = agg_results["queue_df"]
    corrections_overview_df = agg_results["corrections_overview_df"]
    loop_origin_df = agg_results["loop_origin_df"]
    throughput_full_df = agg_results["throughput_full_df"]
    util_df = agg_results["util_df"]
    summary_df = agg_results["summary_df"]
    
    burnout_data = calculate_burnout(all_metrics, p, active_roles)
    
    all_events_data = []
    for rep_idx, metrics in enumerate(all_metrics):
        for t, name, step, note, arr in metrics.events:
            all_events_data.append({
                "Replication": rep_idx + 1, "Time (min)": float(t), "Task": name,
                "Step": step, "Step label": pretty_step(step), "Note": note,
                "Arrival time (min)": (float(arr) if arr is not None else None),
                "Day": int(t // DAY_MIN)
            })
    events_df = pd.DataFrame(all_events_data)
    
    st.markdown(f"## Results")
    
    # Summary Table
    with st.expander("Summary Table", expanded=True):
        summary_df = create_summary_table(all_metrics, p, burnout_data, active_roles)
        st.dataframe(summary_df, width="stretch", hide_index=True)

    # System Performance - Collapsible
    with st.expander("System Performance", expanded=False):
        
        col1, col2 = st.columns(2)
        with col1:
            fig_throughput = plot_daily_throughput(all_metrics, p, active_roles)
            st.pyplot(fig_throughput)
            plt.close(fig_throughput)
        
        with col2:
            fig_queue = plot_queue_over_time(all_metrics, p, active_roles)
            st.pyplot(fig_queue)
            plt.close(fig_queue)
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** Counts tasks completed each day across replications (mean ± SD). "
                 "**Interpretation:** Declining = falling behind; stable/increasing = keeping up. "
                 "Shaded area shows ±1 standard deviation across replications.",
                 title="How is Number Processed calculated?")
        with col2:
            help_icon("**Calculation:** Tracks tasks waiting in each queue every minute (mean ± SD). "
                 "**Interpretation:** Persistent high queues = bottlenecks.",
                 title="How is Backlog graph calculated?")

        st.markdown("---")
        
        # Add daily utilization graph
        col1, col2 = st.columns(2)
        with col1:
            fig_daily_util = plot_daily_utilization(all_metrics, p, active_roles)
            st.pyplot(fig_daily_util)
            plt.close(fig_daily_util)
        
        with col2:
            pass  # Empty column for balance
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** For each role per day: (estimated work time / available capacity time) × 100. "
                 "Work time estimated as: tasks served × average service time.\n\n"
                 "**Interpretation:** Shows what % of staff time is spent actively working. "
                 "75-90% = healthy utilization. >90% = high stress. "
                 "This metric directly feeds into the Utilization burnout component.",
                 title="How is Daily Staff Utilization calculated?")
        with col2:
            pass  # Empty column for balance

    # Response Times (Patient Care) - Collapsible
    with st.expander("Completion Times", expanded=False):
        
        col1, col2 = st.columns(2)
        with col1:
            fig_response_dist = plot_response_time_distribution(all_metrics, p)
            st.pyplot(fig_response_dist)
            plt.close(fig_response_dist)
        
        with col2:
            fig_completion_days = plot_completion_by_day(all_metrics, p)
            st.pyplot(fig_completion_days)
            plt.close(fig_completion_days)
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** Groups all completed tasks into 3-hour bins (0-3hrs, 3-6hrs, etc.) up to 48 hours. "
                 "Shows mean count ± SD across replications.\n\n"
                 "**Interpretation:** Peak near 0 hours = fast response times. Long tail = delays. "
                 "Shaded area shows variability across simulation runs.",
                 title="How is Response Time Distribution calculated?")
        with col2:
            help_icon("**Calculation:** Counts tasks by completion time relative to arrival day:\n"
                 "• Same Day = completed same day\n"
                 "• +1 Day = completed 1 day later\n"
                 "• +2/+3 Days = 2-3 days later\n"
                 "• +4+ Days = 4 or more days later\n\n"
                 "**Interpretation:** More green (same day) = better patient care. "
                 "Red bars (+3/+4 days) indicate significant delays.",
                 title="How is Task Completion Timeline calculated?")
        
        st.markdown("---")
        
        # Add the new daily completion rate graph
        col1, col2 = st.columns(2)
        with col1:
            fig_daily_completion = plot_daily_completion_rate(all_metrics, p, active_roles)
            st.pyplot(fig_daily_completion)
            plt.close(fig_daily_completion)
        
        with col2:
            pass  # Empty column for balance
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** For each day, calculates: (tasks completed same day / tasks arrived that day) × 100\n\n"
                 "**Interpretation:** Higher completion rate = better same-day resolution. "
                 "Lower rate = more tasks carried over to next day. "
                 "This metric directly feeds into the Incompletion burnout component (Incompletion = 100 - Completion Rate).",
                 title="How is Same-Day Task Completion calculated?")
        with col2:
            pass  # Empty column for balance

    # Workload - Collapsible
    with st.expander("Workload burden and burnout", expanded=False):
        
        # First row: Daily Workload and Burnout Over Days
        col1, col2 = st.columns(2)
        with col1:
            fig_daily_workload = plot_daily_workload(all_metrics, p, active_roles)
            st.pyplot(fig_daily_workload)
            plt.close(fig_daily_workload)
        
        with col2:
            fig_burnout_days = plot_burnout_over_days(all_metrics, p, active_roles)
            st.pyplot(fig_burnout_days)
            plt.close(fig_burnout_days)
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** Counts task arrivals to each role per day (mean ± SD). "
                 "Threshold lines show approximate workload levels that correspond to 75% and 90% utilization.\n\n"
                 "**Interpretation:** Lines approaching/exceeding thresholds indicate high workload. "
                 "Consistent high workload leads to burnout.",
                 title="How is Daily Workload calculated?")
        with col2:
            help_icon("**Calculation:** Calculates daily burnout score using that day's metrics: "
                 "utilization, availability stress, corrections, task switching, incompletion, and throughput deficit. "
                 "Weighted by your custom burnout weights.\n\n"
                 "**Interpretation:** Scores above 50 (orange line) = moderate burnout. "
                 "Scores above 75 (red line) = high burnout risk.",
                 title="How is Burnout calculated?")
        
        st.markdown("---")
        
        # Second row: Rerouting and Missing Information
        col1, col2 = st.columns(2)
        with col1:
            fig_rerouting = plot_rerouting_by_day(all_metrics, p, active_roles)
            st.pyplot(fig_rerouting)
            plt.close(fig_rerouting)
        
        with col2:
            fig_missing_info = plot_missing_info_by_day(all_metrics, p, active_roles)
            st.pyplot(fig_missing_info)
            plt.close(fig_missing_info)
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** Counts tasks that arrived at a role that did not originally receive them "
                 "(inappropriate initial routing). Tracked per day.\n\n"
                 "**Interpretation:** High rerouting indicates poor initial task assignment, "
                 "causing inefficiency and delays.",
                 title="How is Rerouting (Inappropriate Receipt) calculated?")
        with col2:
            help_icon("**Calculation:** Counts 'INSUFF' events (insufficient information) per role per day. "
                 "These trigger correction loops where staff must follow up for missing information.\n\n"
                 "**Interpretation:** High missing information rates indicate communication gaps, "
                 "incomplete documentation, or unclear processes.",
                 title="How is Missing Information or Corrections calculated?")
        
        st.markdown("---")
        
        # Third row: Overtime Needed
        col1, col2 = st.columns(2)
        with col1:
            fig_overtime = plot_overtime_needed(all_metrics, p, active_roles)
            st.pyplot(fig_overtime)
            plt.close(fig_overtime)
        
        with col2:
            pass  # Empty column for balance
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** (Total work needed - Available capacity) ÷ (Days × Staff count)\n\n"
                 "Measures additional hours per person per day needed to finish all tasks.\n\n"
                 "**Interpretation:**\n"
                 "• 0 hours = Keeping up with workload\n"
                 "• 0.5 hours = 30min overtime daily\n"
                 "• 1+ hours = Serious capacity shortage\n"
                 "• 2+ hours = Critical understaffing",
                 title="How is Overtime Needed calculated?")
        
        with col2:
            pass  # Empty column for balance
    
    # Excel download - at the very end
    st.markdown("---")
    st.markdown("### Export Results")
    
    excel_file = create_excel_download(all_metrics, p)
    
    if excel_file:
        st.download_button(
            label="📥 Download Event Log as Excel",
            data=excel_file,
            file_name=f"CHC_Event_Log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )
