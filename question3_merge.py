from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import Directions, evaluator, remote_evaluator
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
# 일관 코스트 정책 (대기 억제/진동 방지/혼잡 제어)
WAIT_PENALTY = 0.32
TURN_COST = 0.05
LOCAL_HORIZON_PADDING = 192
K_HOLD = 2
TABOO_TICKS = 3
MAX_COST = 1 << 30
LARGE_G = 1 << 60
INFINITE_HORIZON = 10**9
MAX_NODES = 100000  # 기본 확장 캡(소규모)
CONGESTION_COST = 0.0  # 0이면 혼잡 비용 계산 생략
CONGESTION_WINDOW = 6  # 혼잡도 계산 시간 창

# (reserved for future adaptive constraints)

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

# 동적 로컬 지평선 패딩 계산: 거리/혼잡도(에이전트 수)에 따라 조정
def compute_dynamic_padding(estimated_distance: int, num_agents: int) -> int:
    base_padding = 96 + int(estimated_distance * 0.5)
    # 기본 클램프
    base_padding = max(96, min(base_padding, 192))
    # 대규모 인스턴스 보호: 과도한 확장 방지
    if num_agents >= 50:
        base_padding = min(base_padding, 112)
    return base_padding

# Common helper functions
def heuristic(a: tuple, b: tuple) -> int:
    """Manhattan distance heuristic with None handling"""
    if a is None or b is None:
        return MAX_COST
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def create_reservation_system():
    """Create reservation tables for path planning"""
    return {}, {}  # node_reserved, edge_reserved

def reserve_path_in_tables(path: list, node_reserved: dict, edge_reserved: dict, from_t: int = 0):
    """Reserve a path in the reservation tables"""
    for t in range(from_t, len(path)):
        node_reserved.setdefault(t, set()).add(path[t])
        if t + 1 < len(path):
            edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))
    
    # 목표 셀 soft-hold: 도착 직후 1틱 추가 점유 (꼬리물기 방지)
    if len(path) > 0:
        goal_pos = path[-1]
        arrival_time = len(path) - 1 + from_t
        soft_hold_time = arrival_time + 1
        node_reserved.setdefault(soft_hold_time, set()).add(goal_pos)

def is_conflict_in_tables(curr: tuple, nxt: tuple, t_next: int, node_reserved: dict, edge_reserved: dict, node_capacity: dict = None) -> bool:
    """Check if movement conflicts with reservations or capacity limits"""
    if nxt in node_reserved.get(t_next, set()):
        return True
    if (nxt, curr) in edge_reserved.get(t_next, set()):
        return True
    
    # 교차로 용량 제한 체크 (노드별 용량이 1 초과일 때만 의미가 있음)
    if node_capacity is not None:
        max_capacity = node_capacity.get(nxt, 1)
        if max_capacity > 1:
            occupied = 1 if nxt in node_reserved.get(t_next, set()) else 0
            if occupied >= max_capacity:
                return True
    
    return False

def calculate_congestion_cost(pos: tuple, t: int, node_reserved: dict, edge_reserved: dict) -> float:
    """계산: 향후 CONGESTION_WINDOW 틱 동안의 혼잡도"""
    if CONGESTION_COST <= 0.0:
        return 0.0
    congestion_score = 0
    for future_t in range(t, min(t + CONGESTION_WINDOW, t + 20)):  # 상한 20틱
        congestion_score += len(node_reserved.get(future_t, set()))
        if pos in node_reserved.get(future_t, set()):
            congestion_score += 2  # 직접 충돌 가중
    return congestion_score * CONGESTION_COST / CONGESTION_WINDOW

def calculate_merge_risk(start: tuple, goal: tuple, rail: GridTransitionMap) -> int:
    """합류 위험도 계산: 경로상 교차/합류 노드 개수 추정"""
    if start is None or goal is None:
        return 0
    
    # 간단한 맨해튼 경로상의 잠재 교차점 개수 추정
    dx = abs(goal[0] - start[0])
    dy = abs(goal[1] - start[1])
    
    # 교차/합류 위험 추정: 긴 경로일수록, 대각선 이동이 많을수록 위험
    merge_risk = min(dx, dy) * 2 + abs(dx - dy)  # 대각선 + 직선 부분
    
    # 전체 거리 대비 정규화 (0~10 범위)
    total_distance = dx + dy
    if total_distance > 0:
        merge_risk = min(10, int(merge_risk * 10 / total_distance))
    
    return merge_risk


# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(agents: List[EnvAgent],rail: GridTransitionMap, max_timestep: int):
    # 우선순위 기반 다중 에이전트 계획: 예약 테이블 + 시간-공간 A*, DDL 우선순위 반영
    from heapq import heappush, heappop

    # 전역 캐시/상태 초기화(에피소드마다)
    try:
        if _TRANSITION_CACHE:
            _TRANSITION_CACHE.clear()
    except Exception:
        pass

    num_agents = len(agents)
    # DDL 기반 우선순위: slack = deadline - 최단거리(맨해튼). slack이 작은(급한) 에이전트 먼저
    def manhattan(a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    priorities = list(range(num_agents))
    def slack_for(i: int) -> int:
        ddl = getattr(agents[i], 'deadline', None)
        h0 = manhattan(agents[i].initial_position, agents[i].target)
        if ddl is None:
            return MAX_COST
        return ddl - h0
    # slack 오름차순, tie-breaker: 합류위험도, 거리 내림차순, malfunction 잔여시간 오름차순
    def sort_key(i: int):
        distance = manhattan(agents[i].initial_position, agents[i].target)
        merge_risk = calculate_merge_risk(agents[i].initial_position, agents[i].target, rail)
        mal = 0
        try:
            mal = int(agents[i].malfunction_data.get("malfunction", 0))
        except Exception:
            mal = 0
        return (slack_for(i), merge_risk, -distance, mal)
    priorities.sort(key=sort_key)

    horizon = max_timestep if max_timestep and max_timestep > 0 else INFINITE_HORIZON
    node_reserved, edge_reserved = create_reservation_system()
    
    # 교차로 용량 시스템: 모든 노드 기본 용량 1 (동시 진입 방지)
    node_capacity = {}  # 필요시 특정 노드만 용량 설정

    def get_valid(x: int, y: int, d: int):
        return get_transitions_cached(rail, x, y, d)

    def plan_single(start: tuple, start_dir: int, goal: tuple, deadline: int) -> list:
        if start == goal:
            # 시작=목표인 경우 단일 위치만 반환 (패딩 제거로 성능 향상)
            return [start]
        open_heap = []
        h0 = heuristic(start, goal)
        # 동적 로컬 지평선 계산 (대규모 인스턴스 보호 포함)
        local_padding = compute_dynamic_padding(h0, num_agents)
        local_horizon = min(horizon, h0 + local_padding)
        # (f, late_flag, h, turn_bias, g, state)
        late0 = 1 if (0 + h0 > deadline) else 0 if deadline is not None else 0
        heappush(open_heap, (h0, late0, h0, 0, 0, (start[0], start[1], start_dir, 0)))
        best_g = {(start[0], start[1], start_dir, 0): 0}
        parent = {(start[0], start[1], start_dir, 0): None}
        goal_state = None

        expanded = 0
        while open_heap and expanded < MAX_NODES:
            f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
            expanded += 1
            if t > local_horizon:
                break
            if (x, y) == goal:
                goal_state = (x, y, d, t)
                break
            # wait: 낮은 우선순위(턴 바이어스 2) + 대기 패널티
            nt = t + 1
            if nt <= local_horizon and not is_conflict_in_tables((x, y), (x, y), nt, node_reserved, edge_reserved):
                s = (x, y, d, nt)
                ng = g + 1
                nh = heuristic((x, y), goal)
                if ng < best_g.get(s, LARGE_G):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
                    # 데드라인 압박 반영(일관 정책)
                    if deadline is not None:
                        slack = deadline - (nt + nh)
                        late_boost = 2.2 if slack <= 0 else (1.5 if slack <= 5 else 1.0)
                    else:
                        late_boost = 1.0
                    heappush(open_heap, (ng + nh + WAIT_PENALTY * late_boost, late, nh, 2, ng, s))

            # move
            valid = get_valid(x, y, d)
            for action in range(0, len(valid)):
                if not valid[action]:
                    continue
                nx, ny = step(x, y, action)
                if is_conflict_in_tables((x, y), (nx, ny), t + 1, node_reserved, edge_reserved):
                    continue
                s = (nx, ny, action, t + 1)
                ng = g + 1
                nh = heuristic((nx, ny), goal)
                turn_bias = 0 if action == d else 1
                if ng < best_g.get(s, LARGE_G):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
                    extra = 0.0 if action == d else TURN_COST
                    # 혼잡 비용 (활성화시에만 계산)
                    if CONGESTION_COST > 0.0:
                        congestion = calculate_congestion_cost((nx, ny), t + 1, node_reserved, edge_reserved)
                    else:
                        congestion = 0.0
                    heappush(open_heap, (ng + nh + extra + congestion, late, nh, turn_bias, ng, s))
        if goal_state is None:
            return []
        rev = []
        s = goal_state
        while s is not None:
            rev.append((s[0], s[1]))
            s = parent[s]
        rev.reverse()
        # 도착까지만 반환(패딩 금지) → SIC 증가 방지
        return rev

    path_all = [[] for _ in range(num_agents)]
    for aid in priorities:
        ddl = getattr(agents[aid], 'deadline', None)

        p = plan_single(agents[aid].initial_position, agents[aid].initial_direction, agents[aid].target, ddl if ddl is not None else MAX_COST)
        if len(p) == 0:
            p = [agents[aid].initial_position]
        path_all[aid] = p
            
        reserve_path_in_tables(path_all[aid], node_reserved, edge_reserved)
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
    # 재계획: 고장/실패 에이전트와 영향 받은 에이전트만 부분 재탐색
    from heapq import heappush, heappop

    horizon = max_timestep if max_timestep and max_timestep > 0 else INFINITE_HORIZON

    # 예약 테이블: 현재 시점 이후 고려
    node_reserved = {}
    edge_reserved = {}

    # 전역 예약 함수와 동일 동작으로 통일
    # reserve_path 래퍼 삭제 예정: 호출부에서 전역 함수를 직접 사용
    def reserve_path(path: list, from_t: int):
        reserve_path_in_tables(path, node_reserved, edge_reserved, from_t)

    # 재계획 대상 외 에이전트는 경로 고정(현재 시점 이후 예약)
    num_agents = len(agents)
    replan_set = set(new_malfunction_agents) | set(failed_agents)
    for aid in range(num_agents):
        if aid not in replan_set:
            reserve_path_in_tables(existing_paths[aid], node_reserved, edge_reserved, current_timestep)

    # 실패/고장 대응 보강: 동시 돌진 방지 및 데드락 완화
    for aid in replan_set:
        cur_pos = agents[aid].position
        if cur_pos is None:
            if current_timestep < len(existing_paths[aid]):
                cur_pos = existing_paths[aid][current_timestep]
            else:
                cur_pos = agents[aid].initial_position
        for tt in range(current_timestep + 1, min(horizon, current_timestep + K_HOLD) + 1):
            node_reserved.setdefault(tt, set()).add(cur_pos)


    for aid in failed_agents:
        nxt_t = current_timestep + 1
        if nxt_t < len(existing_paths[aid]):
            nxt_cell = existing_paths[aid][nxt_t]
            curr_cell = existing_paths[aid][current_timestep] if current_timestep < len(existing_paths[aid]) else agents[aid].initial_position
            for tt in range(current_timestep + 1, min(horizon, current_timestep + TABOO_TICKS) + 1):
                node_reserved.setdefault(tt, set()).add(nxt_cell)
            edge_reserved.setdefault(nxt_t, set()).add((curr_cell, nxt_cell))

    # 충돌 판정은 전역 is_conflict_in_tables를 사용합니다.

    # heuristic은 전역 함수를 사용

    def get_valid(x: int, y: int, d: int):
        return get_transitions_cached(rail, x, y, d)

    def plan_from(start: tuple, start_dir: int, start_time: int, goal: tuple, deadline: int) -> list:
        open_heap = []
        h0 = heuristic(start, goal)
        # 동적 로컬 지평선 계산 (대규모 인스턴스 보호 포함)
        local_padding = compute_dynamic_padding(h0, num_agents)
        local_horizon = min(horizon, start_time + h0 + local_padding)
        late0 = 1 if (start_time + h0 > deadline) else 0 if deadline is not None else 0
        heappush(open_heap, (h0 + start_time, late0, h0, 0, start_time - current_timestep, (start[0], start[1], start_dir, start_time)))  # (f, late, h, turn_bias, g, state)
        best_g = {(start[0], start[1], start_dir, start_time): start_time - current_timestep}
        parent = {(start[0], start[1], start_dir, start_time): None}
        goal_state = None
        expanded = 0
        while open_heap and expanded < MAX_NODES:
            f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
            expanded += 1
            if t > local_horizon:
                break
            if (x, y) == goal:
                goal_state = (x, y, d, t)
                break
            # wait
            nt = t + 1
            if nt <= local_horizon and not is_conflict_in_tables((x, y), (x, y), nt, node_reserved, edge_reserved):
                s = (x, y, d, nt)
                ng = g + 1
                nh = heuristic((x, y), goal)
                if ng < best_g.get(s, LARGE_G):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
                    late_boost = 1.5 if late == 1 else 1.0
                    heappush(open_heap, (ng + nh + WAIT_PENALTY * late_boost, late, nh, 2, ng, s))
            # move
            valid = get_valid(x, y, d)
            for action in range(0, len(valid)):
                if not valid[action]:
                    continue
                nx, ny = step(x, y, action)
                if is_conflict_in_tables((x, y), (nx, ny), t + 1, node_reserved, edge_reserved):
                    continue
                s = (nx, ny, action, t + 1)
                ng = g + 1
                nh = heuristic((nx, ny), goal)
                turn_bias = 0 if action == d else 1
                if ng < best_g.get(s, LARGE_G):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
                    extra = 0.0 if action == d else TURN_COST
                    heappush(open_heap, (ng + nh + extra, late, nh, turn_bias, ng, s))
        if goal_state is None:
            return []
        rev = []
        s = goal_state
        while s is not None:
            rev.append((s[0], s[1]))
            s = parent[s]
        rev.reverse()
        # 도착까지만 반환(패딩 금지)
        return rev

    # 재계획 순서: 현재 시점 기준 slack 오름차순 (결정적 정렬)
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
            return MAX_COST
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

        # 고장 강제 대기 반영
        mal = agents[aid].malfunction_data["malfunction"]
        wait_steps = max(0, mal)
        for t in range(current_timestep + 1, min(horizon, current_timestep + wait_steps) + 1):
            node_reserved.setdefault(t, set()).add(agent_pos)

        # 접두사 정합성: current_timestep까지는 보존하되, 현 위치로 동기화
        base = list(existing_paths[aid][:current_timestep])
        if len(base) <= current_timestep:
            pad_pos = base[-1] if base else agent_pos
            while len(base) <= current_timestep:
                base.append(pad_pos)
        base[current_timestep] = agent_pos
        
        start_time = current_timestep + wait_steps
        ddl = getattr(agents[aid], 'deadline', None)
        ddl_use = ddl if ddl is not None else MAX_COST
        
        if agent_pos is None or agents[aid].target is None:
            suffix = []
        else:
            suffix = plan_from(agent_pos, agent_dir, start_time, agents[aid].target, ddl_use)

        if len(suffix) > 0 and suffix[0] == agent_pos:
            suffix = suffix[1:]
        new_path = base[:current_timestep+1] + suffix

        existing_paths[aid] = new_path
        reserve_path_in_tables(new_path, node_reserved, edge_reserved, current_timestep)

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
