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
import matplotlib.dates as mdates
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
        self.loop_nurse_insufficient = 0
        self.loop_provider_insufficient = 0
        self.loop_backoffice_insufficient = 0
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
    "PR_INSUFF": "Doctors: rework needed", "PR_RECHECK_QUEUE": "Doctors: recheck queued",
    "PR_RECHECK_DONE": "Doctors: recheck done", "BO_QUEUE": "Other staff: queued", "BO_DONE": "Other staff: completed",
    "BO_INSUFF": "Other staff: rework needed", "BO_RECHECK_QUEUE": "Other staff: recheck queued",
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
            
            if not is_open(self.env.now, open_minutes):
                yield self.env.timeout(minutes_until_open(self.env.now, open_minutes))
                continue
            
            if len(available_set) > 0 and current_min not in available_set:
                yield self.env.timeout(1)
                continue
            
            window = minutes_until_close(self.env.now, open_minutes)
            
            if len(available_set) > 0:
                avail_window = 1
                check_min = current_min + 1
                while check_min in available_set and avail_window < window and avail_window < remaining:
                    avail_window += 1
                    check_min += 1
                work_chunk = min(remaining, window, avail_window)
            else:
                work_chunk = min(remaining, window)
            
            with resource.request() as req:
                t_req = self.env.now
                yield req
                self.m.waits[role_account].append(self.env.now - t_req)
                self.m.taps[role_account] += 1
                yield self.env.timeout(work_chunk)
                self.m.service_time_sum[role_account] += work_chunk
            remaining -= work_chunk

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
            while (fd_loops < s.p["max_fd_loops"]) and (random.random() < s.p["p_fd_insuff"]):
                fd_loops += 1
                s.m.loop_fd_insufficient += 1
                s.m.log(env.now, task_id, "FD_INSUFF", f"Missing info loop #{fd_loops}")
                yield env.timeout(s.p["fd_loop_delay"])
                s.m.log(env.now, task_id, "FD_RETRY_QUEUE", f"Loop #{fd_loops}")
                yield from s.scheduled_service(res, "Administrative staff", s.p["svc_frontdesk"])
                s.m.log(env.now, task_id, "FD_RETRY_DONE", f"Loop #{fd_loops}")

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
            provider_loops = 0
            while (provider_loops < s.p["max_provider_loops"]) and (random.random() < s.p["p_provider_insuff"]):
                provider_loops += 1
                s.m.loop_provider_insufficient += 1
                s.m.log(env.now, task_id, "PR_INSUFF", f"Doctors rework loop #{provider_loops}")
                yield env.timeout(s.p["provider_loop_delay"])
                s.m.log(env.now, task_id, "PR_RECHECK_QUEUE", f"Loop #{provider_loops}")
                yield from s.scheduled_service(res, "Doctors", max(0.0, 0.5 * s.p["svc_provider"]))
                s.m.log(env.now, task_id, "PR_RECHECK_DONE", f"Loop #{provider_loops}")

    elif role == "Other staff":
        if res is not None:
            s.m.log(env.now, task_id, "BO_QUEUE", "")
            yield from s.scheduled_service(res, "Other staff", s.p["svc_backoffice"])
            s.m.log(env.now, task_id, "BO_DONE", "")
            bo_loops = 0
            while (bo_loops < s.p["max_backoffice_loops"]) and (random.random() < s.p["p_backoffice_insuff"]):
                bo_loops += 1
                s.m.loop_backoffice_insufficient += 1
                s.m.log(env.now, task_id, "BO_INSUFF", f"Other staff rework loop #{bo_loops}")
                yield env.timeout(s.p["backoffice_loop_delay"])
                s.m.log(env.now, task_id, "BO_RECHECK_QUEUE", f"Loop #{bo_loops}")
                yield from s.scheduled_service(res, "Other staff", max(0.0, 0.5 * s.p["svc_backoffice"]))
                s.m.log(env.now, task_id, "BO_RECHECK_DONE", f"Loop #{bo_loops}")

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
    while True:
        inter = random.expovariate(lam) if lam > 0 else 999999999
        yield env.timeout(inter)
        i += 1
        task_id = f"{role_name[:2].upper()}-{i:05d}"
        env.process(task_lifecycle(env, task_id, s, initial_role=role_name))

def monitor(env, s: CHCSystem):
    while True:
        s.m.time_stamps.append(env.now)
        for r in ROLES:
            res = s.role_to_res[r]
            self_q = len(res.queue) if res is not None else 0
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

    env.process(monitor(env, system))
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
      - Rework Percentage
      - Task Switching (queue volatility)
      - Incompletion Rate
      - Throughput Deficit
    """
    weights = p.get("burnout_weights", {
        "utilization": 7, "availability_stress": 3,
        "rework": 6, "task_switching": 4,
        "incompletion": 5, "throughput_deficit": 5
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
        rework_pct_list = []
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

            # ReworkPct (0–1)
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
            estimated_rework = loops * max(0.0, svc_time) * 0.5
            rework_pct = (estimated_rework / max(1, total_service)) if total_service > 0 else 0.0
            rework_pct_list.append(min(1.0, rework_pct))

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
        avg_rework = float(np.mean(rework_pct_list)) if rework_pct_list else 0.0
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
        rework_transformed = min(1.0, avg_rework * 2.5)  # Linear scaling, amplified 2.5x
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
            "rework": 100.0 * rework_transformed,
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
def plot_utilization_by_role(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Bar chart showing utilization by role (mean ± SD across replications).
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    
    util_lists = {r: [] for r in active_roles}
    
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
                open_minutes_per_day = p["open_minutes"]
                avail_minutes_per_day = p.get("availability_per_day", {}).get(role, open_minutes_per_day)
                num_days = p["sim_minutes"] / DAY_MIN
                available_capacity = capacity * num_days * avail_minutes_per_day
                util = min(1.0, total_service / max(1, available_capacity))
                util_lists[role].append(util)
    
    means = [np.mean(util_lists[r]) * 100 for r in active_roles]
    stds = [np.std(util_lists[r]) * 100 for r in active_roles]
    
    colors = []
    for mean_util in means:
        if mean_util < 50:
            colors.append('#2ecc71')
        elif mean_util < 75:
            colors.append('#f39c12')
        elif mean_util < 90:
            colors.append('#e67e22')
        else:
            colors.append('#e74c3c')
    
    x = np.arange(len(active_roles))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6)
    
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='75% threshold')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='90% critical')
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Utilization (%)', fontsize=10)
    ax.set_title('Staff Utilization by Role', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        # Calculate label position - place inside bar if utilization is high, above if low
        if height > 85:
            # Place inside the bar for high utilization
            label_y = height - 10
            label_color = 'white'
        else:
            # Place above the bar for lower utilization
            label_y = height + std + 3
            label_color = 'black'
    
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{mean:.0f}%\n±{std:.0f}%',
                ha='center', va='top' if height > 85 else 'bottom', 
                fontsize=8, fontweight='bold', color=label_color)
    
    plt.tight_layout()
    return fig

def plot_queue_over_time(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurse': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
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
            ax.plot(x, mean_daily, color=colors.get(role, '#333333'), 
                   linewidth=2.5, marker='o', markersize=7, label=role, alpha=0.9)
            
            # Add confidence band
            ax.fill_between(x, mean_daily - std_daily, mean_daily + std_daily,
                          color=colors.get(role, '#333333'), alpha=0.15)
    
    ax.set_xlabel('Operational Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Queue Length (end of day)', fontsize=11, fontweight='bold')
    ax.set_title('Queue Backlog Trends by Role', fontsize=12, fontweight='bold')
    
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
    Line graph showing total daily throughput (tasks completed) over time with SD shading.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes = p["open_minutes"]
    
    daily_completed_lists = [[] for _ in range(num_days)]
    
    for metrics in all_metrics:
        for d in range(num_days):
            start_t = d * DAY_MIN
            end_t = start_t + open_minutes
            
            # Count tasks completed during this operational day
            completed = sum(1 for ct in metrics.task_completion_time.values() if start_t <= ct < end_t)
            daily_completed_lists[d].append(completed)
    
    # Calculate mean and std for each day
    daily_means = [np.mean(daily_completed_lists[d]) for d in range(num_days)]
    daily_stds = [np.std(daily_completed_lists[d]) for d in range(num_days)]
    
    x = np.arange(1, num_days + 1)
    
    # Plot line with markers
    ax.plot(x, daily_means, color='#2ca02c', linewidth=2.5, marker='o', 
            markersize=7, label='Mean Daily Throughput', alpha=0.9)
    
    # Add confidence band (±1 SD)
    upper_bound = [daily_means[i] + daily_stds[i] for i in range(num_days)]
    lower_bound = [max(0, daily_means[i] - daily_stds[i]) for i in range(num_days)]
    ax.fill_between(x, lower_bound, upper_bound, color='#2ca02c', alpha=0.15)
    
    ax.set_xlabel('Operational Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasks Completed', fontsize=11, fontweight='bold')
    ax.set_title('Daily Throughput Over Time', fontsize=12, fontweight='bold')
    
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
    Line graph showing distribution of task completion times in 3-hour bins up to 48 hours.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    # Define bins: 0-3, 3-6, 6-9, ..., 45-48 hours
    bin_edges_hours = np.arange(0, 51, 3)  # 0, 3, 6, 9, ..., 48
    bin_edges_minutes = bin_edges_hours * 60
    num_bins = len(bin_edges_hours) - 1
    
    # Collect histogram data from each replication
    bin_counts_per_rep = []
    
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        if len(done_ids) > 0:
            turnaround_times = [comp_times[k] - arr_times.get(k, comp_times[k]) for k in done_ids]
            counts, _ = np.histogram(turnaround_times, bins=bin_edges_minutes)
            bin_counts_per_rep.append(counts)
        else:
            bin_counts_per_rep.append(np.zeros(num_bins))
    
    # Calculate mean and std for each bin
    bin_counts_array = np.array(bin_counts_per_rep)
    mean_counts = np.mean(bin_counts_array, axis=0)
    std_counts = np.std(bin_counts_array, axis=0)
    
    # Create x-axis positions (center of each bin)
    bin_centers = (bin_edges_hours[:-1] + bin_edges_hours[1:]) / 2
    
    # Plot line with markers
    ax.plot(bin_centers, mean_counts, color='#1f77b4', linewidth=2.5, 
            marker='o', markersize=6, label='Mean Task Count', alpha=0.9)
    
    # Add confidence band (±1 SD)
    upper_bound = mean_counts + std_counts
    lower_bound = np.maximum(0, mean_counts - std_counts)
    ax.fill_between(bin_centers, lower_bound, upper_bound, color='#1f77b4', alpha=0.15)
    
    ax.set_xlabel('Response Time (hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Task Completion Times', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 48)
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    return fig

def plot_completion_by_day(all_metrics: List[Metrics], p: Dict):
    """
    Bar chart showing number of tasks completed same day, +1 day, +2 days, +3 days.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    
    categories = ['Same Day', '+1 Day', '+2 Days', '+3 Days', '+4+ Days']
    
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
            elif days_diff == 1:
                category_counts['+1 Day'] += 1
            elif days_diff == 2:
                category_counts['+2 Days'] += 1
            elif days_diff == 3:
                category_counts['+3 Days'] += 1
            else:
                category_counts['+4+ Days'] += 1
        
        for cat in categories:
            counts_per_rep[cat].append(category_counts[cat])
    
    # Calculate mean and std for each category
    means = [np.mean(counts_per_rep[cat]) for cat in categories]
    stds = [np.std(counts_per_rep[cat]) for cat in categories]
    
    # Color gradient from green (good) to red (bad)
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    x = np.arange(len(categories))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    
    # Add error bars
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6)
    
    ax.set_xlabel('Time to Completion', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax.set_title('Task Completion Timeline', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    f'{mean:.0f}±{std:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_daily_workload(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line graph showing daily task arrivals by role with utilization-based burnout thresholds.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurse': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
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
        ax.plot(x, means, color=colors.get(role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label=role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    # Add utilization-based thresholds
    # Calculate capacity-based thresholds (tasks that would lead to 75% and 90% utilization)
    open_minutes = p["open_minutes"]
    
    # Add threshold lines (these are approximate based on average service times)
    y_max = ax.get_ylim()[1]
    ax.axhline(y=y_max * 0.75, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='75% load threshold')
    ax.axhline(y=y_max * 0.90, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='90% load threshold')
    
    ax.set_xlabel('Operational Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasks Arriving', fontsize=11, fontweight='bold')
    ax.set_title('Daily Workload by Role', fontsize=12, fontweight='bold')
    
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
    Line graph showing burnout score by role over days (cumulative metrics up to each day).
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    colors = {'Administrative staff': '#1f77b4', 'Nurse': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes_per_day = p["open_minutes"]
    
    weights = p.get("burnout_weights", {
        "utilization": 7, "availability_stress": 3,
        "rework": 6, "task_switching": 4,
        "incompletion": 5, "throughput_deficit": 5
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
        available_capacity_per_day = capacity * avail_minutes_per_day
        
        for metrics in all_metrics:
            # Track cumulative metrics up to each day
            cumulative_service_time = 0.0
            cumulative_loops = 0
            cumulative_tasks_completed = 0
            cumulative_same_day = 0
            
            # We need to process events chronologically to build up daily metrics
            loop_counts_map = {
                "Administrative staff": "loop_fd_insufficient",
                "Nurse": "loop_nurse_insufficient",
                "Doctors": "loop_provider_insufficient",
                "Other staff": "loop_backoffice_insufficient"
            }
            
            for d in range(num_days):
                day_start = d * DAY_MIN
                day_end = (d + 1) * DAY_MIN
                
                # Accumulate service time for this role up to this day
                # (Approximation: divide total by days and multiply by current day)
                cumulative_service_time = metrics.service_time_sum[role] * (d + 1) / num_days
                
                # Accumulate loops up to this day
                total_loops = getattr(metrics, loop_counts_map[role])
                cumulative_loops = total_loops * (d + 1) / num_days
                
                # Count tasks completed up to end of this day
                for task_id, comp_time in metrics.task_completion_time.items():
                    if comp_time < day_end:
                        cumulative_tasks_completed += 1
                        # Check if same day
                        arr_time = metrics.task_arrival_time.get(task_id, comp_time)
                        if int(arr_time // DAY_MIN) == int(comp_time // DAY_MIN):
                            cumulative_same_day += 1
                
                # Calculate utilization (cumulative)
                days_so_far = d + 1
                total_capacity = available_capacity_per_day * days_so_far
                util = min(1.0, cumulative_service_time / max(1, total_capacity))
                
                # Availability stress (constant)
                avail_stress = (open_minutes_per_day - float(avail_minutes_per_day)) / open_minutes_per_day
                avail_stress = min(max(avail_stress, 0.0), 1.0)
                
                # Rework percentage
                svc_time = {
                    "Administrative staff": p["svc_frontdesk"],
                    "Nurse": p["svc_nurse"],
                    "Doctors": p["svc_provider"],
                    "Other staff": p["svc_backoffice"]
                }[role]
                estimated_rework = cumulative_loops * max(0.0, svc_time) * 0.5
                rework_pct = min(1.0, (estimated_rework / max(1, cumulative_service_time)) if cumulative_service_time > 0 else 0.0)
                
                # Queue volatility (use overall - can't split easily)
                queue_lengths = metrics.queues[role]
                if len(queue_lengths) > 1:
                    q_mean = np.mean(queue_lengths)
                    q_std = np.std(queue_lengths)
                    q_cv = (q_std / max(1e-6, q_mean)) if q_mean > 0 else 0.0
                    queue_volatility = min(1.0, q_cv)
                else:
                    queue_volatility = 0.0
                
                # Completion rate
                completion_rate = (cumulative_same_day / max(1, cumulative_tasks_completed)) if cumulative_tasks_completed > 0 else 0.0
                
                # Throughput
                tasks_per_day = cumulative_tasks_completed / days_so_far
                expected_throughput = p["arrivals_per_hour_by_role"].get(role, 1) * open_minutes_per_day / 60.0
                throughput_ratio = tasks_per_day / max(1e-6, expected_throughput)
                throughput_deficit = min(1.0, max(0.0, 1.0 - throughput_ratio))
                
                # Apply transformations
                def transform_utilization(u):
                    if u <= 0.75:
                        return u / 0.75 * 0.5
                    else:
                        excess = (u - 0.75) / 0.25
                        return 0.5 + 0.5 * (np.exp(2 * excess) - 1) / (np.exp(2) - 1)
                
                util_transformed = transform_utilization(util)
                rework_transformed = rework_pct ** 1.5
                volatility_transformed = np.sqrt(queue_volatility)
                incompletion = 1.0 - completion_rate
                incompletion_transformed = incompletion ** 0.7
                
                # Component scores
                components = {
                    "utilization": 100.0 * util_transformed,
                    "availability_stress": 100.0 * avail_stress,
                    "rework": 100.0 * rework_transformed,
                    "task_switching": 100.0 * volatility_transformed,
                    "incompletion": 100.0 * incompletion_transformed,
                    "throughput_deficit": 100.0 * throughput_deficit
                }
                
                # Calculate burnout
                burnout_score = sum(norm_weights[k] * components[k] for k in components.keys())
                daily_burnout_per_rep[d].append(burnout_score)
                
                # Reset cumulative counters for next day's calculation
                cumulative_tasks_completed = 0
                cumulative_same_day = 0
        
        # Calculate mean and std
        means = [np.mean(daily_burnout_per_rep[d]) if daily_burnout_per_rep[d] else 0 
                for d in range(num_days)]
        stds = [np.std(daily_burnout_per_rep[d]) if len(daily_burnout_per_rep[d]) > 1 else 0 
               for d in range(num_days)]
        
        x = np.arange(1, num_days + 1)
        
        # Plot line
        ax.plot(x, means, color=colors.get(role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label=role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    # Add burnout threshold lines
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='Moderate burnout (50)')
    ax.axhline(y=75, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='High burnout (75)')
    
    ax.set_xlabel('Operational Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Burnout Score (0-100)', fontsize=11, fontweight='bold')
    ax.set_title('Burnout Progression by Role', fontsize=12, fontweight='bold')
    
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
    colors = {'Administrative staff': '#1f77b4', 'Nurse': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
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
        ax.plot(x, means, color=colors.get(role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label=role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    ax.set_xlabel('Operational Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Reroutes', fontsize=11, fontweight='bold')
    ax.set_title('Rerouting (Inappropriate Receipt) by Role', fontsize=12, fontweight='bold')
    
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
    colors = {'Administrative staff': '#1f77b4', 'Nurse': '#ff7f0e', 'Doctors': '#2ca02c', 'Other staff': '#d62728'}
    
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
        ax.plot(x, means, color=colors.get(role, '#333333'), 
               linewidth=2.5, marker='o', markersize=6, label=role, alpha=0.9)
        
        # Add confidence band
        upper = [means[i] + stds[i] for i in range(num_days)]
        lower = [max(0, means[i] - stds[i]) for i in range(num_days)]
        ax.fill_between(x, lower, upper, color=colors.get(role, '#333333'), alpha=0.1)
    
    ax.set_xlabel('Operational Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Missing Info Events', fontsize=11, fontweight='bold')
    ax.set_title('Missing Info (Call Backs) by Role', fontsize=12, fontweight='bold')
    
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
    
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='0.5 hr/day')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='1.0 hr/day')
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Additional Hours per Day per Person', fontsize=10)
    ax.set_title('Overtime Needed to Clear Backlog', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles, fontsize=9, rotation=15, ha='right')
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

def create_kpi_banner(all_metrics: List[Metrics], p: Dict, burnout_data: Dict, active_roles: List[str]):
    """
    Create a simple one-line banner showing key burnout metrics.
    """
    # Calculate average component scores across all roles
    all_utilization = []
    all_rework = []
    all_task_switching = []
    all_incompletion = []
    
    for r in active_roles:
        if r in burnout_data["by_role"]:
            components = burnout_data["by_role"][r].get("components", {})
            all_utilization.append(components.get("utilization", 0.0))
            all_rework.append(components.get("rework", 0.0))
            all_task_switching.append(components.get("task_switching", 0.0))
            all_incompletion.append(components.get("incompletion", 0.0))
    
    avg_utilization = np.mean(all_utilization) if all_utilization else 0.0
    avg_rework = np.mean(all_rework) if all_rework else 0.0
    avg_task_switching = np.mean(all_task_switching) if all_task_switching else 0.0
    avg_incompletion = np.mean(all_incompletion) if all_incompletion else 0.0
    
    overall_burnout = burnout_data["overall_clinic"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Burnout", f"{overall_burnout:.1f}")
    
    with col2:
        st.metric("Utilization Stress", f"{avg_utilization:.1f}")
    
    with col3:
        st.metric("Rework Stress", f"{avg_rework:.1f}")
    
    with col4:
        st.metric("Task Switching", f"{avg_task_switching:.1f}")

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
    
    rework_pct_list = []
    loop_counts_lists = {"Administrative staff": [], "Nurse": [], "Doctors": [], "Other staff": []}
    
    for metrics in all_metrics:
        rework_tasks = set()
        for t, name, step, note, _arr in metrics.events:
            if step.endswith("INSUFF") or "RECHECK" in step:
                rework_tasks.add(name)
        
        done_ids = set(metrics.task_completion_time.keys())
        rework_pct_list.append(100.0 * len(rework_tasks & done_ids) / max(1, len(done_ids)))
        
        loop_counts_lists["Administrative staff"].append(metrics.loop_fd_insufficient)
        loop_counts_lists["Nurse"].append(metrics.loop_nurse_insufficient)
        loop_counts_lists["Doctors"].append(metrics.loop_provider_insufficient)
        loop_counts_lists["Other staff"].append(metrics.loop_backoffice_insufficient)
    
    rework_overview_df = pd.DataFrame([
        {"Metric": "% tasks with any rework", "Value": fmt_mean_std_pct(rework_pct_list)}
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
        "Rework (% of completed)": np.mean(rework_pct_list),
        "Utilization overall (%)": np.mean(util_overall_list),
    }
    summary_df = pd.DataFrame([summary_row])
    
    return {
        "flow_df": flow_df, "time_at_role_df": time_at_role_df, "queue_df": queue_df,
        "rework_overview_df": rework_overview_df, "loop_origin_df": loop_origin_df,
        "throughput_full_df": throughput_full_df, "util_df": util_df, "summary_df": summary_df
    }

def create_summary_table(all_metrics: List[Metrics], p: Dict, burnout_data: Dict, active_roles: List[str]):
    """
    Create a comprehensive summary table showing key metrics across all categories.
    """
    num_reps = len(all_metrics)
    
    # Calculate Patient Care metrics
    turnaround_times_list = []
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        if len(done_ids) > 0:
            tt = [comp_times[k] - arr_times.get(k, comp_times[k]) for k in done_ids]
            avg_tt = np.mean(tt)
            turnaround_times_list.append(avg_tt)
    
    mean_turnaround = np.mean(turnaround_times_list) if turnaround_times_list else 0.0
    std_turnaround = np.std(turnaround_times_list, ddof=1) if len(turnaround_times_list) > 1 else 0.0
    
    # Calculate Inefficiency metrics
    total_reroutes_list = []
    missing_info_pct_list = []
    
    for metrics in all_metrics:
        # Count re-routes (tasks that went through loops)
        reroutes = (metrics.loop_fd_insufficient + 
                   metrics.loop_nurse_insufficient + 
                   metrics.loop_provider_insufficient + 
                   metrics.loop_backoffice_insufficient)
        total_reroutes_list.append(reroutes)
        
        # Calculate % of tasks with missing info
        tasks_with_rework = set()
        for t, name, step, note, _arr in metrics.events:
            if step.endswith("INSUFF") or "RECHECK" in step:
                tasks_with_rework.add(name)
        
        done_ids = set(metrics.task_completion_time.keys())
        if len(done_ids) > 0:
            missing_info_pct = 100.0 * len(tasks_with_rework & done_ids) / len(done_ids)
            missing_info_pct_list.append(missing_info_pct)
        else:
            missing_info_pct_list.append(0.0)
    
    mean_reroutes = np.mean(total_reroutes_list)
    std_reroutes = np.std(total_reroutes_list, ddof=1) if len(total_reroutes_list) > 1 else 0.0
    mean_missing_info = np.mean(missing_info_pct_list)
    std_missing_info = np.std(missing_info_pct_list, ddof=1) if len(missing_info_pct_list) > 1 else 0.0
    
    # Calculate Utilization metrics
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
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
    
    # Get Burnout metrics
    burnout_by_role = {r: burnout_data["by_role"][r]["overall"] for r in active_roles}
    
    # Build the table data
    table_data = []
    
    # Patient Care
    table_data.append({
        "Focus": "Patient care",
        "Measure": "Mean response time",
        "Result": f"{mean_turnaround:.1f} ± {std_turnaround:.1f} min ({mean_turnaround/60:.1f} ± {std_turnaround/60:.1f} hrs)"
    })
    
    # Inefficiency
    table_data.append({
        "Focus": "Inefficiency",
        "Measure": "Re-routes",
        "Result": f"{mean_reroutes:.1f} ± {std_reroutes:.1f}"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Missing info",
        "Result": f"{mean_missing_info:.1f}% ± {std_missing_info:.1f}%"
    })
    
    # Utilization
    table_data.append({
        "Focus": "Utilization",
        "Measure": "Administrative staff",
        "Result": f"{np.mean(util_by_role['Administrative staff']):.1f}% ± {np.std(util_by_role['Administrative staff'], ddof=1):.1f}%" if util_by_role['Administrative staff'] else "N/A"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Nurses",
        "Result": f"{np.mean(util_by_role['Nurse']):.1f}% ± {np.std(util_by_role['Nurse'], ddof=1):.1f}%" if util_by_role['Nurse'] else "N/A"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Doctors",
        "Result": f"{np.mean(util_by_role['Doctors']):.1f}% ± {np.std(util_by_role['Doctors'], ddof=1):.1f}%" if util_by_role['Doctors'] else "N/A"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Other staff",
        "Result": f"{np.mean(util_by_role['Other staff']):.1f}% ± {np.std(util_by_role['Other staff'], ddof=1):.1f}%" if util_by_role['Other staff'] else "N/A"
    })
    
    # Burnout
    table_data.append({
        "Focus": "Burnout",
        "Measure": "Administrative staff",
        "Result": f"{burnout_by_role['Administrative staff']:.1f}" if 'Administrative staff' in burnout_by_role else "N/A"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Nurses",
        "Result": f"{burnout_by_role['Nurse']:.1f}" if 'Nurse' in burnout_by_role else "N/A"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Doctors",
        "Result": f"{burnout_by_role['Doctors']:.1f}" if 'Doctors' in burnout_by_role else "N/A"
    })
    table_data.append({
        "Focus": "",
        "Measure": "Other staff",
        "Result": f"{burnout_by_role['Other staff']:.1f}" if 'Other staff' in burnout_by_role else "N/A"
    })
    
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

def create_excel_download(all_metrics: List[Metrics], p: Dict, burnout_data: Dict, 
                         active_roles: List[str], agg_results: Dict) -> BytesIO:
    """
    Create an Excel file with all simulation results across multiple sheets.
    """
    output = BytesIO()
    
    engine = _excel_engine()
    if engine is None:
        st.error("No Excel engine available. Install xlsxwriter or openpyxl.")
        return None
    
    with pd.ExcelWriter(output, engine=engine) as writer:
        # Summary sheet
        summary_df = create_summary_table(all_metrics, p, burnout_data, active_roles)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Flow time metrics
        agg_results["flow_df"].to_excel(writer, sheet_name='Flow Times', index=False)
        
        # Time at role
        agg_results["time_at_role_df"].to_excel(writer, sheet_name='Time at Role', index=False)
        
        # Queue metrics
        agg_results["queue_df"].to_excel(writer, sheet_name='Queues', index=False)
        
        # Utilization
        agg_results["util_df"].to_excel(writer, sheet_name='Utilization', index=False)
        
        # Rework metrics
        agg_results["rework_overview_df"].to_excel(writer, sheet_name='Rework Overview', index=False)
        agg_results["loop_origin_df"].to_excel(writer, sheet_name='Loop Origins', index=False)
        
        # Daily throughput
        agg_results["throughput_full_df"].to_excel(writer, sheet_name='Daily Throughput', index=False)
        
        # Burnout scores
        burnout_rows = []
        for role in active_roles:
            if role in burnout_data["by_role"]:
                components = burnout_data["by_role"][role].get("components", {})
                burnout_rows.append({
                    "Role": role,
                    "Overall Burnout": burnout_data["by_role"][role]["overall"],
                    "Utilization": components.get("utilization", 0.0),
                    "Availability Stress": components.get("availability_stress", 0.0),
                    "Rework": components.get("rework", 0.0),
                    "Task Switching": components.get("task_switching", 0.0),
                    "Incompletion": components.get("incompletion", 0.0),
                    "Throughput Deficit": components.get("throughput_deficit", 0.0)
                })
        burnout_rows.append({
            "Role": "Clinic Average",
            "Overall Burnout": burnout_data["overall_clinic"],
            "Utilization": None,
            "Availability Stress": None,
            "Rework": None,
            "Task Switching": None,
            "Incompletion": None,
            "Throughput Deficit": None
        })
        burnout_df = pd.DataFrame(burnout_rows)
        burnout_df.to_excel(writer, sheet_name='Burnout Scores', index=False)
        
        # Configuration parameters
        config_data = []
        config_data.append({"Parameter": "Simulation Days", "Value": p["sim_minutes"] / DAY_MIN})
        config_data.append({"Parameter": "Hours Open per Day", "Value": p["open_minutes"] / 60})
        config_data.append({"Parameter": "Number of Replications", "Value": p["num_replications"]})
        config_data.append({"Parameter": "", "Value": ""})
        
        config_data.append({"Parameter": "Administrative Staff Count", "Value": p["frontdesk_cap"]})
        config_data.append({"Parameter": "Administrative Staff Arrivals/hr", "Value": p["arrivals_per_hour_by_role"]["Administrative staff"]})
        config_data.append({"Parameter": "Administrative Staff Availability (min/day)", "Value": p["availability_per_day"]["Administrative staff"]})
        config_data.append({"Parameter": "", "Value": ""})
        
        config_data.append({"Parameter": "Nurse Count", "Value": p["nurse_cap"]})
        config_data.append({"Parameter": "Nurse Arrivals/hr", "Value": p["arrivals_per_hour_by_role"]["Nurse"]})
        config_data.append({"Parameter": "Nurse Availability (min/day)", "Value": p["availability_per_day"]["Nurse"]})
        config_data.append({"Parameter": "", "Value": ""})
        
        config_data.append({"Parameter": "Doctor Count", "Value": p["provider_cap"]})
        config_data.append({"Parameter": "Doctor Arrivals/hr", "Value": p["arrivals_per_hour_by_role"]["Doctors"]})
        config_data.append({"Parameter": "Doctor Availability (min/day)", "Value": p["availability_per_day"]["Doctors"]})
        config_data.append({"Parameter": "", "Value": ""})
        
        config_data.append({"Parameter": "Other Staff Count", "Value": p["backoffice_cap"]})
        config_data.append({"Parameter": "Other Staff Arrivals/hr", "Value": p["arrivals_per_hour_by_role"]["Other staff"]})
        config_data.append({"Parameter": "Other Staff Availability (min/day)", "Value": p["availability_per_day"]["Other staff"]})
        
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    output.seek(0)
    return output

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Community Health Center Workflow Model", layout="wide")
st.title("Community Health Center Workflow Model")
st.caption("By Ines Sereno")

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
        - Patient-initiated paperwork, calls, or postal messages are received by various staff (nurses, Doctors, staff)
        - Staff process these based on availability and Processing times
        - Items received by the wrong type of personnel are routed appropriately
        The model tracks daily workload, inefficiency, response delays, and contribution to burnout for each type of role
        - The simulation tracks queues, wait times, and completion rates over multiple days
        
        **How to use:**
        1. **Define your clinic** below by setting staffing levels, routing logic, and processing times
        2. **Click "Save"** to store your configuration
        3. **Click "Run Simulation"** to see results including burnout scores, utilization, and bottlenecks
        4. **Try different scenarios** to test interventions (e.g. more staff, reduce volumes, improve workflows)
        
        **Tips:**
        - Start with inputs that define the current process
        - Change one thing at a time to understand its impact
        - Pay attention to roles showing high utilization (>75%) or high burnout (>50)
        - Use the KPI banner in results to quickly compare different scenarios
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
            label_name = "Done" if tgt == DONE else tgt
            key_name = f"r_{from_role}_{'done' if tgt==DONE else label_name.replace(' ','_').lower()}"
            default_val = float(defaults.get(tgt, 0.0))
            with cols[i]:
                val = prob_input(f"to {label_name} ({from_role})", key=key_name, 
                               default=(0.0 if tgt_disabled else default_val), disabled=tgt_disabled)
                if tgt_disabled:
                    val = 0.0
            row[tgt] = val
        return row
                         
    with st.form("design_form", clear_on_submit=False):
        st.markdown("  Simulation horizon")
        sim_days = st.number_input("Days to simulate", 1, 30, _init_ss("sim_days", 5), 1, "%d",
                               help="Number of clinic operating days to simulate")
        open_hours = st.number_input("Hours open per day", 1, 24, _init_ss("open_hours", 8), 1, "%d",
                                  help="Number of hours the clinic is open each day")
    
        seed = 42  # Fixed seed for reproducibility

        st.markdown("  Roles")
        st.caption("Configure staffing, arrivals, and availability for each role")
    
        with st.expander("Administrative staff", expanded=False):
            cFD1, cFD2, cFD3 = st.columns(3)
            with cFD1:
                fd_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("fd_cap", 3), 1, "%d", key="fd_cap_input",
                                               help="Number of Administrative staff staff")
            with cFD2:
                arr_fd = st.number_input("Volume", 0, 500, _init_ss("arr_fd", 4), 1, "%d", disabled=(fd_cap_form==0), key="arr_fd_input",
                             help="Average number of tasks per hour")
            with cFD3:
                avail_fd = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_fd", 240), 1, "%d", disabled=(fd_cap_form==0), key="avail_fd_input",
                               help="Minutes per day available for work (max = hours open × 60)")

        with st.expander("Nurses", expanded=False):
            cNU1, cNU2, cNU3 = st.columns(3)
            with cNU1:
                nu_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("nurse_cap", 3), 1, "%d", key="nurse_cap_input",
                                                  help="Number of nurses or medical assistants")
            with cNU2:
                arr_nu = st.number_input("Volume", 0, 500, _init_ss("arr_nu", 3), 1, "%d", disabled=(nu_cap_form==0), key="arr_nu_input",
                             help="Average number of tasks per hour")
            with cNU3:
                avail_nu = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_nu", 120), 1, "%d", disabled=(nu_cap_form==0), key="avail_nu_input",
                               help="Minutes per day available for work (max = hours open × 60)")

        with st.expander("Doctors", expanded=False):
            cPR1, cPR2, cPR3 = st.columns(3)
            with cPR1:
                pr_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("provider_cap", 2), 1, "%d", key="provider_cap_input",
                                                     help="Number of Doctors")
            with cPR2:
                arr_pr = st.number_input("Volume", 0, 500, _init_ss("arr_pr", 2), 1, "%d", disabled=(pr_cap_form==0), key="arr_pr_input",
                             help="Average number of tasks per hour")
            with cPR3:
                avail_pr = st.number_input("Availability (min/day)", 0, 480, _init_ss("avail_pr", 60), 1, "%d", disabled=(pr_cap_form==0), key="avail_pr_input",
                               help="Minutes per day available for work (max = hours open × 60)")

        with st.expander("Other staff", expanded=False):
            cBO1, cBO2, cBO3 = st.columns(3)
            with cBO1:
                bo_cap_form = st.number_input("Number working per day", 0, 50, _init_ss("backoffice_cap", 2), 1, "%d", key="bo_cap_input",
                                               help="Number of Other staff staff")
            with cBO2:
                 arr_bo = st.number_input("Volume", 0, 500, _init_ss("arr_bo", 2), 1, "%d", disabled=(bo_cap_form==0), key="arr_bo_input",
                             help="Average number of tasks per hour")
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

        with st.expander("Advanced Settings – Processing times, loops & routing", expanded=False):
        
            st.markdown("  Contributors to Burnout-Relative Weights")
            st.caption("Assign each factor a weight between 0 and 10 (0 = no contribution, 10 = maximum contribution)")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Emotional Exhaustion Contributors:**")
                w_utilization = st.slider("Utilization", 0, 10, _init_ss("w_utilization", 7), 1, 
                              help="How much does high utilization contribute to burnout?")
                w_availability_stress = st.slider("Availability Stress", 0, 10, _init_ss("w_availability_stress", 3), 1,
                                     help="How much does limited availability contribute to burnout?")
    
                st.markdown("**Depersonalization Contributors:**")
                w_rework = st.slider("Rework Percentage", 0, 10, _init_ss("w_rework", 6), 1,
                        help="How much does rework contribute to burnout?")
                w_task_switching = st.slider("Task Switching (Queue Volatility)", 0, 10, _init_ss("w_task_switching", 4), 1,
                                help="How much does unpredictable workload contribute to burnout?")

            with col2:
                st.markdown("**Reduced Accomplishment Contributors:**")
                w_incompletion = st.slider("Incomplete Tasks", 0, 10, _init_ss("w_incompletion", 5), 1,
                               help="How much do incomplete tasks contribute to burnout?")
                w_throughput_deficit = st.slider("Throughput Deficit", 0, 10, _init_ss("w_throughput_deficit", 5), 1,
                                    help="How much does falling behind expected throughput contribute to burnout?")

            # Calculate and display normalized weights
            total_weight = w_utilization + w_availability_stress + w_rework + w_task_switching + w_incompletion + w_throughput_deficit
            if total_weight > 0:
                st.info(f"**Total weight: {total_weight}** — All scores will be normalized to 0-100 scale")
            else:
                st.warning("⚠️ All weights are 0 - burnout scores will be 0")
        
            with st.expander("Administrative staff", expanded=False):
                st.markdown("**Processing time**")
                svc_frontdesk = st.slider("Mean Processing time (minutes)", 0.0, 30.0, _init_ss("svc_frontdesk", 3.0), 0.5, disabled=(fd_cap_form==0),
                                      help="Average time to complete a task")
            
                st.markdown("**Rework Loops**")
                cFDL1, cFDL2, cFDL3 = st.columns(3)
                with cFDL1:
                    p_fd_insuff = st.slider("Percent with insufficient info", 0.0, 1.0, _init_ss("p_fd_insuff", 0.25), 0.01, disabled=(fd_cap_form==0), key="fd_p_insuff")
                with cFDL2:
                    max_fd_loops = st.number_input("Maximum number of loops", 0, 10, _init_ss("max_fd_loops", 3), 1, "%d", disabled=(fd_cap_form==0), key="fd_max_loops")
                with cFDL3:
                    fd_loop_delay = st.slider("Delay to obtain", 0.0, 480.0, _init_ss("fd_loop_delay", 240.0), 0.5, disabled=(fd_cap_form==0), key="fd_delay")
            
                st.markdown("**Disposition or routing Administrative staff**")
                fd_route = route_row_ui("Administrative staff", {"Nurse": 0.50, "Doctors": 0.10, "Other staff": 0.10, DONE: 0.30}, 
                                   disabled_source=(fd_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            with st.expander("Nurses", expanded=False):
                st.markdown("**Processing times**")
                cNS1, cNS2 = st.columns(2)
                with cNS1:
                    svc_nurse_protocol = st.slider("Protocol Processing time (minutes)", 0.0, 30.0, _init_ss("svc_nurse_protocol", 2.0), 0.5, disabled=(nu_cap_form==0))
                    p_protocol = st.slider("Probability of using protocol", 0.0, 1.0, _init_ss("p_protocol", 0.30), 0.05, disabled=(nu_cap_form==0))
                with cNS2:
                    svc_nurse = st.slider("Non-protocol Processing time (minutes)", 0.0, 40.0, _init_ss("svc_nurse", 5.0), 0.5, disabled=(nu_cap_form==0))
                
                st.markdown("**Rework Loops**")
                cNUL1, cNUL2 = st.columns(2)
                with cNUL1:
                    p_nurse_insuff = st.slider("Percent with insufficient info", 0.0, 1.0, _init_ss("p_nurse_insuff", 0.20), 0.01, disabled=(nu_cap_form==0), key="nu_p_insuff")
                with cNUL2:
                    max_nurse_loops = st.number_input("Maximum number of loops", 0, 10, _init_ss("max_nurse_loops", 3), 1, "%d", disabled=(nu_cap_form==0), key="nu_max_loops")
            
                st.markdown("**Disposition or routing Nurse**")
                nu_route = route_row_ui("Nurse", {"Doctors": 0.40, "Other staff": 0.20, DONE: 0.40}, 
                                   disabled_source=(nu_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            with st.expander("Doctors", expanded=False):
                st.markdown("**Processing time**")
                svc_provider = st.slider("Mean Processing time (minutes)", 0.0, 480.0, _init_ss("svc_provider", 7.0), 0.5, disabled=(pr_cap_form==0))
            
                st.markdown("**Rework Loops**")
                cPRL1, cPRL2, cPRL3 = st.columns(3)
                with cPRL1:
                    p_provider_insuff = st.slider("Probability of rework needed", 0.0, 1.0, _init_ss("p_provider_insuff", 0.15), 0.01, disabled=(pr_cap_form==0), key="pr_p_insuff")
                with cPRL2:
                    max_provider_loops = st.number_input("Maximum number of loops", 0, 10, _init_ss("max_provider_loops", 3), 1, "%d", disabled=(pr_cap_form==0), key="pr_max_loops")
                with cPRL3:
                    provider_loop_delay = st.slider("Delay to obtain", 0.0, 480.0, _init_ss("provider_loop_delay", 300.0), 0.5, disabled=(pr_cap_form==0), key="pr_delay")
            
                st.markdown("**Disposition or routing Doctors**")
                pr_route = route_row_ui("Doctors", {"Other staff": 0.30, DONE: 0.70}, 
                                   disabled_source=(pr_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            with st.expander("Other staff", expanded=False):
                st.markdown("**Processing time**")
                svc_backoffice = st.slider("Mean Processing time (minutes)", 0.0, 480.0, _init_ss("svc_backoffice", 5.0), 0.5, disabled=(bo_cap_form==0))
            
                st.markdown("**Rework Loops**")
                cBOL1, cBOL2, cBOL3 = st.columns(3)
                with cBOL1:
                    p_backoffice_insuff = st.slider("Probability of rework needed", 0.0, 1.0, _init_ss("p_backoffice_insuff", 0.18), 0.01, disabled=(bo_cap_form==0), key="bo_p_insuff")
                with cBOL2:
                    max_backoffice_loops = st.number_input("Maximum number of loops", 0, 10, _init_ss("max_backoffice_loops", 3), 1, "%d", disabled=(bo_cap_form==0), key="bo_max_loops")
                with cBOL3:
                    backoffice_loop_delay = st.slider("Delay to obtain", 0.0, 480.0, _init_ss("backoffice_loop_delay", 180.0), 0.5, disabled=(bo_cap_form==0), key="bo_delay")
            
                st.markdown("**Disposition or routing Other staff**")
                bo_route = route_row_ui("Other staff", {"Administrative staff": 0.10, "Nurse": 0.10, "Doctors": 0.10, DONE: 0.70}, 
                                   disabled_source=(bo_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                   pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
        
            route: Dict[str, Dict[str, float]] = {}
            route["Administrative staff"] = fd_route
            route["Nurse"] = nu_route
            route["Doctors"] = pr_route
            route["Other staff"] = bo_route

        saved = st.form_submit_button("Save", type="primary")

        if saved:
            st.session_state.fd_cap = fd_cap_form
            st.session_state.nurse_cap = nu_cap_form
            st.session_state.provider_cap = pr_cap_form
            st.session_state.bo_cap = bo_cap_form
            
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            for r in ROLES:
                if r in route:
                    route[r].pop(r, None)
            for r in ROLES:
                if r in route:
                    for tgt in list(route[r].keys()):
                        if tgt in ROLES and {"Administrative staff": fd_cap_form, "Nurse": nu_cap_form, "Doctors": pr_cap_form, "Other staff": bo_cap_form}[tgt] == 0:
                            route[r][tgt] = 0.0

            st.session_state["design"] = dict(
                sim_minutes=sim_minutes, open_minutes=open_minutes,
                seed=seed, num_replications=num_replications,
                frontdesk_cap=fd_cap_form, nurse_cap=nu_cap_form,
                provider_cap=pr_cap_form, backoffice_cap=bo_cap_form,
                arrivals_per_hour_by_role={"Administrative staff": int(arr_fd), "Nurse": int(arr_nu), 
                                          "Doctors": int(arr_pr), "Other staff": int(arr_bo)},
                availability_per_day={"Administrative staff": int(avail_fd), "Nurse": int(avail_nu),
                      "Doctors": int(avail_pr), "Other staff": int(avail_bo)},
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider, svc_backoffice=svc_backoffice,
                dist_role={"Administrative staff": "normal", "NurseProtocol": "normal", "Nurse": "exponential",
                          "Doctors": "exponential", "Other staff": "exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Administrative staff": 0.5, "Nurse": 0.5, "NurseProtocol": 0.5, "Doctors": 0.5, "Other staff": 0.5},
                burnout_weights={
                    # Emotional Exhaustion contributors
                    "utilization": w_utilization,                    # Weight for workload intensity (0-10)
                    "availability_stress": w_availability_stress,    # Weight for limited work time (0-10)
            
                    # Depersonalization contributors
                    "rework": w_rework,                             # Weight for correction/loop time (0-10)
                    "task_switching": w_task_switching,             # Weight for queue volatility (0-10)
            
                    # Reduced Accomplishment contributors
                    "incompletion": w_incompletion,                 # Weight for incomplete tasks (0-10)
                     "throughput_deficit": w_throughput_deficit      # Weight for falling behind (0-10)
                },
                p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
                p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, max_provider_loops=max_provider_loops, provider_loop_delay=provider_loop_delay,
                p_backoffice_insuff=p_backoffice_insuff, max_backoffice_loops=max_backoffice_loops, backoffice_loop_delay=backoffice_loop_delay,
                p_protocol=p_protocol, route_matrix=route
            )
            st.session_state.design_saved = True
            st.success("Configuration saved successfully")

    if st.session_state.design_saved:
        if st.button("Run Simulation", type="primary", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
        
# -------- STEP 2: RUN & RESULTS --------
elif st.session_state.wizard_step == 2:
    st.markdown("  Simulation Run")
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
    rework_overview_df = agg_results["rework_overview_df"]
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
    
    st.markdown(f"  Results")

    # KPI Banner (first, always visible)
    create_kpi_banner(all_metrics, p, burnout_data, active_roles)

    help_icon(
        "**Overall Burnout (0-100):**\n"
        "Clinic-wide burnout score averaged across all roles. Scale: 0-25 Low, 25-50 Moderate, 50-75 High, 75-100 Severe.\n\n"
    
        "**Component Scores (0-100 each, averaged across roles):**\n\n"
    
        "• **Utilization Stress (0-100)** - Measures workload intensity based on how much available time is consumed. "
        "Uses non-linear scaling where utilization >75% creates exponentially higher stress (a staff member at 80% utilization "
        "experiences much more than 5% more stress than at 75%).\n\n"
    
        "• **Rework Stress (0-100)** - Quantifies time wasted on corrections, loops, and re-processing tasks. "
        "Calculated from the percentage of work time spent redoing work due to missing information or errors. "
        "Uses quadratic penalty (rework² × 100) so doubling rework quadruples the stress.\n\n"
    
        "• **Task Switching (0-100)** - Measures unpredictability and context switching caused by volatile queue lengths. "
        "Calculated from queue volatility (standard deviation ÷ mean). High scores indicate constantly changing priorities "
        "and frequent interruptions.\n\n"
    
        "**How your custom weights affect these scores:**\n"
        "You assigned weights (0-10) to six underlying factors: utilization, availability stress, rework, task switching, "
        "incompletion, and throughput deficit. The component scores shown here are pre-weighted averages. The Overall Burnout "
        "score combines all six weighted factors, normalized to 0-100.",
        title="How are the burnout metrics calculated?"
    )

    st.markdown("---")

    # Summary Table (second, always visible)
    st.markdown("  Summary")
    summary_df = create_summary_table(all_metrics, p, burnout_data, active_roles)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Summary Table (second, always visible)
    st.markdown("  Summary")
    summary_df = create_summary_table(all_metrics, p, burnout_data, active_roles)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ADD THIS SECTION HERE:
    st.markdown("---")
    st.markdown("  Export Results")
    
    excel_file = create_excel_download(all_metrics, p, burnout_data, active_roles, agg_results)
    
    if excel_file:
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_file,
            file_name=f"CHC_Simulation_Results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True

    # System Performance - Collapsible
    with st.expander("  System Performance", expanded=False):
        st.caption("How well is the clinic handling incoming work?")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_throughput = plot_daily_throughput(all_metrics, p, active_roles)
            st.pyplot(fig_throughput, use_container_width=False)
            plt.close(fig_throughput)
        
        with col2:
            fig_queue = plot_queue_over_time(all_metrics, p, active_roles)
            st.pyplot(fig_queue, use_container_width=False)
            plt.close(fig_queue)
        
        col1, col2 = st.columns(2)
        with col1:
            help_icon("**Calculation:** Counts tasks completed each day across replications (mean ± SD). "
                 "**Interpretation:** Declining = falling behind; stable/increasing = keeping up. "
                 "Shaded area shows ±1 standard deviation across replications.",
                 title="How is Daily Throughput calculated?")
        with col2:
            help_icon("**Calculation:** Tracks tasks waiting in each queue every minute (mean ± SD). "
                 "**Interpretation:** Persistent high queues = bottlenecks.",
                 title="How is Queue Backlog Trend graph calculated?")

    # Response Times (Patient Care) - Collapsible
    with st.expander("  Response Times (Patient Care)", expanded=False):
        st.caption("How quickly are tasks being completed?")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_response_dist = plot_response_time_distribution(all_metrics, p)
            st.pyplot(fig_response_dist, use_container_width=False)
            plt.close(fig_response_dist)
        
        with col2:
            fig_completion_days = plot_completion_by_day(all_metrics, p)
            st.pyplot(fig_completion_days, use_container_width=False)
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
                 "• Same Day = completed same operational day\n"
                 "• +1 Day = completed 1 operational day later\n"
                 "• +2/+3 Days = 2-3 days later\n"
                 "• +4+ Days = 4 or more days later\n\n"
                 "**Interpretation:** More green (same day) = better patient care. "
                 "Red bars (+3/+4 days) indicate significant delays.",
                 title="How is Task Completion Timeline calculated?")

    # Workload - Collapsible
    with st.expander("  Workload", expanded=False):
        st.caption("How is workload distributed and evolving over time?")
        
        # First row: Daily Workload and Burnout Over Days
        col1, col2 = st.columns(2)
        with col1:
            fig_daily_workload = plot_daily_workload(all_metrics, p, active_roles)
            st.pyplot(fig_daily_workload, use_container_width=False)
            plt.close(fig_daily_workload)
        
        with col2:
            fig_burnout_days = plot_burnout_over_days(all_metrics, p, active_roles)
            st.pyplot(fig_burnout_days, use_container_width=False)
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
                 "utilization, availability stress, rework, task switching, incompletion, and throughput deficit. "
                 "Weighted by your custom burnout weights.\n\n"
                 "**Interpretation:** Scores above 50 (orange line) = moderate burnout. "
                 "Scores above 75 (red line) = high burnout risk.",
                 title="How is Burnout Progression calculated?")
        
        st.markdown("---")
        
        # Second row: Rerouting and Missing Info
        col1, col2 = st.columns(2)
        with col1:
            fig_rerouting = plot_rerouting_by_day(all_metrics, p, active_roles)
            st.pyplot(fig_rerouting, use_container_width=False)
            plt.close(fig_rerouting)
        
        with col2:
            fig_missing_info = plot_missing_info_by_day(all_metrics, p, active_roles)
            st.pyplot(fig_missing_info, use_container_width=False)
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
                 "These trigger rework loops where staff must follow up for missing information.\n\n"
                 "**Interpretation:** High missing info rates indicate communication gaps, "
                 "incomplete documentation, or unclear processes.",
                 title="How is Missing Info (Call Backs) calculated?")
        
        st.markdown("---")
        
        # Third row: Overtime Needed
        st.markdown("  Capacity Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_overtime = plot_overtime_needed(all_metrics, p, active_roles)
            st.pyplot(fig_overtime, use_container_width=False)
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

        )
    
    st.markdown("---")
