"""
This is the python script for question 2. In this script, you are required to implement a single agent path-finding algorithm
that avoids conflicts with existing paths by planning on a time-expanded graph.
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys


#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!", e)
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
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
# The path should avoid conflicts with existing paths.
#########################

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths. The index of each location is the time that
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, agent_id: int, existing_paths: list, max_timestep: int):
    # Time-expanded A* with reservation table (vertex/edge/goal occupancy) to avoid conflicts with existing paths
    from heapq import heappush, heappop

    # Build reservation tables
    node_reserved = {}
    edge_reserved = {}
    stay_reserved_earliest = {}  # node -> earliest time from which the node stays occupied
    horizon = max_timestep if max_timestep and max_timestep > 0 else 10**9
    for p in existing_paths:
        for t in range(0, len(p)):
            node_reserved.setdefault(t, set()).add(p[t])
            if t + 1 < len(p):
                edge_reserved.setdefault(t + 1, set()).add((p[t], p[t + 1]))
        if len(p) > 0:
            last = p[-1]
            arrive_t = len(p) - 1
            prev = stay_reserved_earliest.get(last)
            if prev is None or arrive_t < prev:
                stay_reserved_earliest[last] = arrive_t

    def is_conflict(curr: tuple, nxt: tuple, t_next: int) -> bool:
        # Vertex conflict
        if nxt in node_reserved.get(t_next, set()):
            return True
        # Edge swap conflict
        if (nxt, curr) in edge_reserved.get(t_next, set()):
            return True
        # Extended goal occupancy conflict
        earliest = stay_reserved_earliest.get(nxt)
        if earliest is not None and t_next >= earliest:
            return True
        return False

    def heuristic(x: int, y: int) -> int:
        return abs(x - goal[0]) + abs(y - goal[1])

    # Handle start==goal: wait in place until earliest safe time
    if start == goal:
        t = 0
        path = [start]
        while t + 1 <= horizon and is_conflict(start, start, t + 1):
            path.append(start)
            t += 1
        return path

    start_state = (start[0], start[1], start_direction, 0)
    open_heap = []
    h0 = heuristic(start[0], start[1])
    heappush(open_heap, (h0, h0, 0, 0, start_state))  # (f, h, turn_bias, g, state)
    best_g = {start_state: 0}
    parent = {start_state: None}

    goal_state = None
    while open_heap:
        f, h, tb, g, (x, y, d, t) = heappop(open_heap)
        if t > horizon:
            break
        if (x, y) == goal:
            goal_state = (x, y, d, t)
            break

        # 1) Wait (lower priority via turn_bias=2)
        nt = t + 1
        if nt <= horizon and not is_conflict((x, y), (x, y), nt):
            s = (x, y, d, nt)
            ng = g + 1
            nh = heuristic(x, y)
            if ng < best_g.get(s, 1 << 60):
                best_g[s] = ng
                parent[s] = (x, y, d, t)
                heappush(open_heap, (ng + nh, nh, 2, ng, s))

        # 2) Move
        valid = rail.get_transitions(x, y, d)
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
            ng = g + 1
            nh = heuristic(nx, ny)
            turn_bias = 0 if action == d else 1
            if ng < best_g.get(s, 1 << 60):
                best_g[s] = ng
                parent[s] = (x, y, d, t)
                heappush(open_heap, (ng + nh, nh, turn_bias, ng, s))

    if goal_state is None:
        return []

    # Reconstruct path as a time-ordered list of (x, y)
    rev = []
    s = goal_state
    while s is not None:
        rev.append((s[0], s[1]))
        s = parent[s]
    rev.reverse()
    return rev


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,2)


















