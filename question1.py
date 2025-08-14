"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""
from lib_piglet.utils.tools import eprint
import glob, os, sys

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
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
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
#########################


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
    # 단일 에이전트 최단 경로: 방향을 상태에 포함한 A* 탐색
    from heapq import heappush, heappop

    if start == goal:
        return [start]

    def heuristic(x: int, y: int) -> int:
        return abs(x - goal[0]) + abs(y - goal[1])

    # 우선순위 큐: (f, g, (x, y, dir))
    open_heap = []
    start_state = (start[0], start[1], start_direction)
    heappush(open_heap, (heuristic(start[0], start[1]), 0, start_state))

    # 최단 g 기록과 부모 추적
    best_g = {start_state: 0}
    parent = {start_state: None}

    goal_state = None

    # 방문 상한: 필요 시 max_timestep를 안전 장치로 사용
    search_limit = max_timestep if max_timestep and max_timestep > 0 else 10**9

    while open_heap:
        f, g, (x, y, d) = heappop(open_heap)
        if g > search_limit:
            break
        # 위치만 목표에 도달하면 종료(방향 무관)
        if (x, y) == goal:
            goal_state = (x, y, d)
            break

        # 가능한 전이 확장
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

            nd = action
            ng = g + 1
            state = (nx, ny, nd)
            if ng < best_g.get(state, 1 << 60):
                best_g[state] = ng
                parent[state] = (x, y, d)
                heappush(open_heap, (ng + heuristic(nx, ny), ng, state))

    # 경로 복원
    if goal_state is None:
        return []

    rev_path = []
    s = goal_state
    while s is not None:
        rev_path.append((s[0], s[1]))
        s = parent[s]
    rev_path.reverse()

    return rev_path


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,1)



















