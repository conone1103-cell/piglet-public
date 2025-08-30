from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict
import glob, os, sys,time,json

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)




#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################

# Performance tuning constants and helpers
WAIT_PENALTY = 0.35
TURN_COST = 0.05
LOCAL_HORIZON_PADDING = 128
# New tunables
GOAL_HOLD = 8              # reserve goal cell for K ticks after arrival
HEUR_W = 1.08              # weighted A* factor (1.05~1.15)
USE_CORRIDOR_PENALTY = True

# Global transition cache keyed by rail id
_TRANSITION_CACHE = {}

def get_transitions_cached(rail: GridTransitionMap, x: int, y: int, d: int):
    key = (id(rail), x, y, d)
    v = _TRANSITION_CACHE.get(key)
    if v is None:
        v = rail.get_transitions(x, y, d)
        _TRANSITION_CACHE[key] = v
    return v

try:
    DIR_DELTAS = {
        Directions.NORTH: (-1, 0),
        Directions.EAST: (0, 1),
        Directions.SOUTH: (1, 0),
        Directions.WEST: (0, -1),
    }
except Exception:
    DIR_DELTAS = {}

def step(x: int, y: int, action: int):
    dx, dy = DIR_DELTAS.get(action, (0, 0))
    return x + dx, y + dy

# --- helper: pad paths to a given timeline length ---
def _pad_paths_to(paths: List[List[Tuple[int,int]]], target_len: int) -> List[List[Tuple[int,int]]]:
    if target_len is None or target_len <= 0:
        return paths
    for i, p in enumerate(paths):
        if not p:
            continue
        last = p[-1]
        # make length exactly target_len (0-based timeline, so +1 for count)
        need = target_len + 1 - len(p)
        if need > 0:
            paths[i] = p + [last] * need
    return paths


# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(agents: List[EnvAgent],rail: GridTransitionMap, max_timestep: int):
    """Prioritized planning with weighted A*, corridor penalty, edge-swap block,
    and goal-hold reservations. Non-invasive: returns list of (x,y) paths.
    """
    from heapq import heappush, heappop

    num_agents = len(agents)

    def manhattan(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority: earliest deadline slack first, then longer distance
    priorities = list(range(num_agents))
    def slack_for(i: int) -> int:
        ddl = getattr(agents[i], 'deadline', None)
        h0 = manhattan(agents[i].initial_position, agents[i].target)
        if ddl is None:
            return 1 << 30
        return ddl - h0
    priorities.sort(key=lambda i: (slack_for(i), -manhattan(agents[i].initial_position, agents[i].target)))

    # Horizon taken from evaluator (do not over-pad)
    horizon = max_timestep if max_timestep and max_timestep > 0 else 10**9

    # Reservation tables: time -> set of occupied nodes/edges
    node_reserved: Dict[int, set] = {}
    edge_reserved: Dict[int, set] = {}

    # Corridor penalty (degree==2 cells)
    corridor_penalty: Dict[Tuple[int,int], float] = {}
    try:
        H, W = rail.grid.shape  # (rows, cols)
        neighbor_cache: Dict[Tuple[int,int], set] = {}
        for x in range(H):
            for y in range(W):
                nbrs = set()
                for d in (Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST):
                    try:
                        valid = rail.get_transitions(x, y, d)
                    except Exception:
                        valid = None
                    if not valid:
                        continue
                    for action in range(0, len(valid)):
                        if valid[action]:
                            nx, ny = step(x, y, action)
                            if 0 <= nx < H and 0 <= ny < W:
                                nbrs.add((nx, ny))
                neighbor_cache[(x, y)] = nbrs
                if len(nbrs) == 2:
                    corridor_penalty[(x, y)] = 0.2
    except Exception:
        corridor_penalty = {}

    def reserve_path(path: list):
        # Reserve traversed nodes/edges + hold goal for a few ticks
        for t in range(0, len(path)):
            node_reserved.setdefault(t, set()).add(path[t])
            if t + 1 < len(path):
                edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))
        if path:
            gcell = path[-1]
            t_goal = len(path) - 1
            for k in range(1, GOAL_HOLD + 1):
                node_reserved.setdefault(t_goal + k, set()).add(gcell)

    def is_conflict(curr: tuple, nxt: tuple, t_next: int) -> bool:
        # Node occupancy or head-on swap
        if nxt in node_reserved.get(t_next, set()):
            return True
        if (nxt, curr) in edge_reserved.get(t_next, set()):
            return True
        return False

    def heuristic(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    transition_cache = {}
    def get_valid(x: int, y: int, d: int):
        key = (x, y, d)
        v = transition_cache.get(key)
        if v is None:
            v = rail.get_transitions(x, y, d)
            transition_cache[key] = v
        return v

    def plan_single(start: tuple, start_dir: int, goal: tuple, deadline: int) -> list:
        # Already at goal → no padding (avoid SOC blowup)
        if start == goal:
            return [start]

        open_heap = []
        h0 = heuristic(start, goal)
        late0 = 1 if (0 + h0 > deadline) else 0 if deadline is not None else 0
        # (f, late_flag, h, turn_bias, g, state)
        heappush(open_heap, (HEUR_W * h0, late0, h0, 0, 0.0, (start[0], start[1], start_dir, 0)))
        best_g = {(start[0], start[1], start_dir, 0): 0.0}
        parent = {(start[0], start[1], start_dir, 0): None}
        goal_state = None

        while open_heap:
            f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
            if t > horizon:
                break
            if (x, y) == goal:
                goal_state = (x, y, d, t)
                break

            # 1) wait
            nt = t + 1
            if nt <= horizon and not is_conflict((x, y), (x, y), nt):
                s = (x, y, d, nt)
                # small wait cost + corridor penalty at current cell
                step_cost = 1.0 + WAIT_PENALTY + (corridor_penalty.get((x, y), 0.0) if USE_CORRIDOR_PENALTY else 0.0)
                ng = g + step_cost
                nh = heuristic((x, y), goal)
                if ng < best_g.get(s, 1e18):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
                    heappush(open_heap, (ng + HEUR_W * nh, late, nh, 2, ng, s))

            # 2) move
            valid = get_valid(x, y, d)
            for action in range(0, len(valid)):
                if not valid[action]:
                    continue
                nx, ny = x, y
                if action == Directions.NORTH:
                    nx -= 1
                elif action == Directions.EAST:
                    ny += 1
                elif action == Directions.SOUTH:
                    nx += 1
                elif action == Directions.WEST:
                    ny -= 1
                if is_conflict((x, y), (nx, ny), t + 1):
                    continue
                s = (nx, ny, action, t + 1)
                base = 1.0 + (corridor_penalty.get((nx, ny), 0.0) if USE_CORRIDOR_PENALTY else 0.0)
                turn_extra = 0.0 if action == d else TURN_COST
                step_cost = base + turn_extra
                ng = g + step_cost
                nh = heuristic((nx, ny), goal)
                turn_bias = 0 if action == d else 1
                if ng < best_g.get(s, 1e18):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
                    heappush(open_heap, (ng + HEUR_W * nh, late, nh, turn_bias, ng, s))

        if goal_state is None:
            return []
        rev = []
        s = goal_state
        while s is not None:
            rev.append((s[0], s[1]))
            s = parent[s]
        rev.reverse()
        return rev

    # --- initial prioritized plan ---
    path_all = [[] for _ in range(num_agents)]
    for aid in priorities:
        ddl = getattr(agents[aid], 'deadline', None)
        p = plan_single(agents[aid].initial_position, agents[aid].initial_direction, agents[aid].target, ddl if ddl is not None else 1 << 30)
        if len(p) == 0:
            p = [agents[aid].initial_position]
        path_all[aid] = p
        reserve_path(p)

    # --- LNS-lite post-plan repair (conflict groups up to REPAIR_GROUP_LIMIT) ---
    def detect_conflict_groups(paths: List[List[Tuple[int,int]]]):
        # build union-find
        parent = list(range(len(paths)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra!=rb: parent[rb]=ra
        # simulate timeline
        T = max((len(p) for p in paths if p), default=0)
        def pos_at(p,t):
            if not p: return None
            if t < len(p): return p[t]
            return p[-1]
        for t in range(T-1):
            # vertex conflicts
            occ: Dict[Tuple[int,int], List[int]] = {}
            for i,p in enumerate(paths):
                if not p: continue
                c = pos_at(p,t)
                occ.setdefault(c, []).append(i)
            for cell, ids in occ.items():
                if len(ids)>1:
                    for i in range(len(ids)-1):
                        union(ids[i], ids[i+1])
            # edge swap
            edge: Dict[Tuple[Tuple[int,int],Tuple[int,int]], int] = {}
            for i,p in enumerate(paths):
                if not p or t+1>=len(p):
                    continue
                u = pos_at(p,t); v = pos_at(p,t+1)
                edge[(u,v)] = i
            for (u,v), i in list(edge.items()):
                j = edge.get((v,u))
                if j is not None and j != i:
                    union(i,j)
        # build groups
        groups: Dict[int,List[int]] = {}
        for i in range(len(paths)):
            if not paths[i]:
                continue
            r = find(i)
            groups.setdefault(r, []).append(i)
        return [sorted(g) for g in groups.values() if len(g) >= 2]

    def plan_group_with_res(paths: List[List[Tuple[int,int]]], group: List[int]) -> List[List[Tuple[int,int]]]:
        # rebuild reservations from others
        nr: Dict[int,set] = {}
        er: Dict[int,set] = {}
        def reserve_from_path(path: list):
            for t in range(len(path)):
                nr.setdefault(t,set()).add(path[t])
                if t+1 < len(path):
                    er.setdefault(t+1,set()).add((path[t], path[t+1]))
            if path:
                g = path[-1]; tg = len(path)-1
                for k in range(1, GOAL_HOLD+1):
                    nr.setdefault(tg+k,set()).add(g)
        for aid,p in enumerate(paths):
            if aid not in group:
                reserve_from_path(p)
        # local planners reading these reservations
        def local_is_conflict(curr,nxt,t_next):
            if nxt in nr.get(t_next,set()): return True
            if (nxt,curr) in er.get(t_next,set()): return True
            return False
        def local_plan_single(aid: int) -> list:
            start = agents[aid].initial_position
            goal = agents[aid].target
            start_dir = agents[aid].initial_direction
            ddl = getattr(agents[aid], 'deadline', None)
            ddl_use = ddl if ddl is not None else 1<<30
            if start == goal: return [start]
            open_heap = []
            h0 = manhattan(start, goal)
            heappush(open_heap, (HEUR_W*h0, 0, h0, 0, 0.0, (start[0], start[1], start_dir, 0)))
            best_g = {(start[0], start[1], start_dir, 0): 0.0}
            parent = {(start[0], start[1], start_dir, 0): None}
            goal_state=None
            while open_heap:
                f,late,h,tb,g,(x,y,d,t)=heappop(open_heap)
                if t>horizon: break
                if (x,y)==goal:
                    goal_state=(x,y,d,t); break
                # wait
                nt=t+1
                if nt<=horizon and not local_is_conflict((x,y),(x,y),nt):
                    s=(x,y,d,nt)
                    step=1.0 + WAIT_PENALTY + (corridor_penalty.get((x,y),0.0) if USE_CORRIDOR_PENALTY else 0.0)
                    ng=g+step; nh=manhattan((x,y),goal)
                    if ng < best_g.get(s,1e18):
                        best_g[s]=ng; parent[s]=(x,y,d,t)
                        heappush(open_heap,(ng+HEUR_W*nh, 1 if (nt+nh>ddl_use) else 0, nh, 2, ng, s))
                # move
                valid=get_valid(x,y,d)
                for action in range(len(valid)):
                    if not valid[action]:
                        continue
                    nx,ny=x,y
                    if action==Directions.NORTH: nx-=1
                    elif action==Directions.EAST: ny+=1
                    elif action==Directions.SOUTH: nx+=1
                    elif action==Directions.WEST: ny-=1
                    if local_is_conflict((x,y),(nx,ny),t+1):
                        continue
                    s=(nx,ny,action,t+1)
                    base=1.0 + (corridor_penalty.get((nx,ny),0.0) if USE_CORRIDOR_PENALTY else 0.0)
                    turn_extra=0.0 if action==d else TURN_COST
                    ng=g+base+turn_extra; nh=manhattan((nx,ny),goal)
                    if ng < best_g.get(s,1e18):
                        best_g[s]=ng; parent[s]=(x,y,d,t)
                        heappush(open_heap,(ng+HEUR_W*nh, 1 if ((t+1)+nh>ddl_use) else 0, nh, 0 if action==d else 1, ng, s))
            if goal_state is None:
                return []
            rev=[]; s=goal_state
            while s is not None:
                rev.append((s[0],s[1])); s=parent[s]
            rev.reverse(); return rev
        # plan group in slack-first order
        order = sorted(group, key=lambda i: (slack_for(i), -manhattan(agents[i].initial_position, agents[i].target)))
        new_paths = {aid: None for aid in group}
        ok=True
        for aid in order:
            p = local_plan_single(aid)
            if not p:
                ok=False; break
            new_paths[aid]=p
            # commit to reservations
            for t in range(len(p)):
                nr.setdefault(t,set()).add(p[t])
                if t+1 < len(p):
                    er.setdefault(t+1,set()).add((p[t], p[t+1]))
            if p:
                g=p[-1]; tg=len(p)-1
                for k in range(1, GOAL_HOLD+1):
                    nr.setdefault(tg+k,set()).add(g)
        if not ok:
            return paths
        # merge
        for aid,p in new_paths.items():
            paths[aid]=p
        return paths

    # run a few rounds
    for _ in range(REPAIR_ROUNDS):
        groups = detect_conflict_groups(path_all)
        # keep only small groups for speed
        groups = [g for g in groups if len(g) <= REPAIR_GROUP_LIMIT]
        if not groups:
            break
        for g in groups:
            path_all = plan_group_with_res(path_all, g)

    # Return minimal (arrival-time) paths only; evaluator computes makespan from these.
    return path_all


# This function return a list of location tuple as the solution.
# @param rail The flatland railway GridTransitionMap
# @param agents A list of EnvAgent.
# @param current_timestep The timestep that malfunction/collision happens .
# @param existing_paths The existing paths from previous get_plan or replan.
# @param max_timestep The max timestep of this episode.
# @param new_malfunction_agents  The id of agents have new malfunction happened at current time step (Does not include agents already have malfunciton in past timesteps)
# @param failed_agents  The id of agents failed to reach the location on its path at current timestep.
# @return path_all  Return paths that locaitons from current_timestp is updated to handle malfunctions and failed execuations.
def replan(agents: List[EnvAgent],rail: GridTransitionMap,  current_timestep: int, existing_paths: List[Tuple], max_timestep:int, new_malfunction_agents: List[int], failed_agents: List[int]):
    """Local repair with goal-hold and corridor-aware weighted A*.
    Only replan affected agents; keep others reserved from now on.
    """
    from heapq import heappush, heappop

    horizon = max_timestep if max_timestep and max_timestep > 0 else 10**9

    # Build reservations from current time onward
    node_reserved: Dict[int, set] = {}
    edge_reserved: Dict[int, set] = {}

    def reserve_path(path: list, from_t: int):
        if not path:
            return
        for t in range(from_t, len(path)):
            node_reserved.setdefault(t, set()).add(path[t])
            if t + 1 < len(path):
                edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))
        # hold the final cell to prevent rear-end hits
        last_t = len(path) - 1
        if last_t >= from_t:
            gcell = path[-1]
            for k in range(1, GOAL_HOLD + 1):
                node_reserved.setdefault(last_t + k, set()).add(gcell)

    # Precompute corridor penalty
    corridor_penalty: Dict[Tuple[int,int], float] = {}
    try:
        H, W = rail.grid.shape
        for x in range(H):
            for y in range(W):
                nbrs = set()
                for d in (Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST):
                    try:
                        valid = rail.get_transitions(x, y, d)
                    except Exception:
                        valid = None
                    if not valid:
                        continue
                    for action in range(0, len(valid)):
                        if valid[action]:
                            nx, ny = step(x, y, action)
                            if 0 <= nx < H and 0 <= ny < W:
                                nbrs.add((nx, ny))
                if len(nbrs) == 2:
                    corridor_penalty[(x, y)] = 0.2
    except Exception:
        corridor_penalty = {}

    # Replan set
    num_agents = len(agents)
    replan_set = set(new_malfunction_agents) | set(failed_agents)

    # Reserve all unaffected agents beyond current time
    for aid in range(num_agents):
        if aid not in replan_set:
            reserve_path(existing_paths[aid], current_timestep)

    # Boost safety around failed/blocked moves
    K_HOLD = 3
    for aid in replan_set:
        cur_pos = agents[aid].position
        if cur_pos is None:
            if current_timestep < len(existing_paths[aid]):
                cur_pos = existing_paths[aid][current_timestep]
            else:
                cur_pos = agents[aid].initial_position
        for tt in range(current_timestep + 1, min(horizon, current_timestep + K_HOLD) + 1):
            node_reserved.setdefault(tt, set()).add(cur_pos)

    TABOO_TICKS = 5
    for aid in failed_agents:
        nxt_t = current_timestep + 1
        if nxt_t < len(existing_paths[aid]):
            nxt_cell = existing_paths[aid][nxt_t]
            curr_cell = existing_paths[aid][current_timestep] if current_timestep < len(existing_paths[aid]) else agents[aid].initial_position
            for tt in range(current_timestep + 1, min(horizon, current_timestep + TABOO_TICKS) + 1):
                node_reserved.setdefault(tt, set()).add(nxt_cell)
            edge_reserved.setdefault(nxt_t, set()).add((curr_cell, nxt_cell))

    def is_conflict(curr: tuple, nxt: tuple, t_next: int) -> bool:
        if nxt in node_reserved.get(t_next, set()):
            return True
        if (nxt, curr) in edge_reserved.get(t_next, set()):
            return True
        return False

    def heuristic(a: tuple, b: tuple) -> int:
        if a is None or b is None:
            return 1 << 30
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_valid(x: int, y: int, d: int):
        return get_transitions_cached(rail, x, y, d)

    def plan_from(start: tuple, start_dir: int, start_time: int, goal: tuple, deadline: int) -> list:
        open_heap = []
        h0 = heuristic(start, goal)
        local_horizon = min(horizon, start_time + h0 + LOCAL_HORIZON_PADDING)
        late0 = 1 if (start_time + h0 > deadline) else 0 if deadline is not None else 0
        heappush(open_heap, (start_time + HEUR_W * h0, late0, h0, 0, 0.0, (start[0], start[1], start_dir, start_time)))
        best_g = {(start[0], start[1], start_dir, start_time): 0.0}
        parent = {(start[0], start[1], start_dir, start_time): None}
        goal_state = None
        while open_heap:
            f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
            if t > local_horizon:
                break
            if (x, y) == goal:
                goal_state = (x, y, d, t)
                break
            # wait
            nt = t + 1
            if nt <= local_horizon and not is_conflict((x, y), (x, y), nt):
                s = (x, y, d, nt)
                step_cost = 1.0 + WAIT_PENALTY + (corridor_penalty.get((x, y), 0.0) if USE_CORRIDOR_PENALTY else 0.0)
                ng = g + step_cost
                nh = heuristic((x, y), goal)
                if ng < best_g.get(s, 1e18):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
                    heappush(open_heap, (ng + HEUR_W * nh, late, nh, 2, ng, s))
            # move
            valid = get_valid(x, y, d)
            for action in range(0, len(valid)):
                if not valid[action]:
                    continue
                nx, ny = step(x, y, action)
                if is_conflict((x, y), (nx, ny), t + 1):
                    continue
                s = (nx, ny, action, t + 1)
                base = 1.0 + (corridor_penalty.get((nx, ny), 0.0) if USE_CORRIDOR_PENALTY else 0.0)
                turn_extra = 0.0 if action == d else TURN_COST
                step_cost = base + turn_extra
                ng = g + step_cost
                nh = heuristic((nx, ny), goal)
                turn_bias = 0 if action == d else 1
                if ng < best_g.get(s, 1e18):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
                    heappush(open_heap, (ng + HEUR_W * nh, late, nh, turn_bias, ng, s))
        if goal_state is None:
            return []
        rev = []
        s = goal_state
        while s is not None:
            rev.append((s[0], s[1]))
            s = parent[s]
        rev.reverse()
        return rev

    # Replan order: smallest slack first (deterministic)
    order = list(replan_set)
    def current_slack(aid: int) -> int:
        ddl = getattr(agents[aid], 'deadline', None)
        agent_pos = agents[aid].position
        if agent_pos is None:
            if current_timestep < len(existing_paths[aid]):
                agent_pos = existing_paths[aid][current_timestep]
            else:
                agent_pos = agents[aid].initial_position
        mal = agents[aid].malfunction_data["malfunction"]
        start_time = current_timestep + max(0, mal)
        h0 = heuristic(agent_pos, agents[aid].target)
        if ddl is None:
            return 1 << 30
        return ddl - (start_time - current_timestep + h0)
    order.sort(key=lambda i: current_slack(i))

    for aid in order:
        agent_pos = agents[aid].position
        if agent_pos is None:
            if current_timestep < len(existing_paths[aid]):
                agent_pos = existing_paths[aid][current_timestep]
            else:
                agent_pos = agents[aid].initial_position

        agent_dir = agents[aid].direction
        if agent_dir is None:
            agent_dir = agents[aid].initial_direction

        # Forced wait due to malfunction
        mal = agents[aid].malfunction_data["malfunction"]
        wait_steps = max(0, mal)
        for t in range(current_timestep + 1, min(horizon, current_timestep + wait_steps) + 1):
            node_reserved.setdefault(t, set()).add(agent_pos)

        # Keep prefix consistent up to current time
        base = list(existing_paths[aid][:current_timestep])
        if len(base) <= current_timestep:
            pad_pos = base[-1] if base else agent_pos
            while len(base) <= current_timestep:
                base.append(pad_pos)
        base[current_timestep] = agent_pos

        start_time = current_timestep + wait_steps
        ddl = getattr(agents[aid], 'deadline', None)
        ddl_use = ddl if ddl is not None else 1 << 30

        if agent_pos is None or agents[aid].target is None:
            suffix = []
        else:
            suffix = plan_from(agent_pos, agent_dir, start_time, agents[aid].target, ddl_use)

        if len(suffix) > 0 and suffix[0] == agent_pos:
            suffix = suffix[1:]
        new_path = base[:current_timestep+1] + suffix

        existing_paths[aid] = new_path
        reserve_path(new_path, current_timestep)

    # Return minimal (arrival-time) paths only to avoid inflating makespan.
    return existing_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)


