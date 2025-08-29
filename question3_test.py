from lib_piglet.utils.tools import eprint
from typing import List, Tuple
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

#debug = True
#visualizer = True

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
#test_single_instance = True
#level = 1
#test = 5

#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################


# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(agents: List[EnvAgent],rail: GridTransitionMap, max_timestep: int):
	# 우선순위 기반 다중 에이전트 계획: 예약 테이블 + 시간-공간 A*, DDL 우선순위 반영
	from heapq import heappush, heappop

	num_agents = len(agents)
	# DDL 기반 우선순위: slack = deadline - 최단거리(맨해튼). slack이 작은(급한) 에이전트 먼저
	def manhattan(a: tuple, b: tuple) -> int:
		return abs(a[0] - b[0]) + abs(a[1] - b[1])
	priorities = list(range(num_agents))
	def slack_for(i: int) -> int:
		ddl = getattr(agents[i], 'deadline', None)
		h0 = manhattan(agents[i].initial_position, agents[i].target)
		if ddl is None:
			return 1 << 30
		return ddl - h0
	# slack 오름차순, slack이 같으면 거리가 먼 순(내림차순)으로 정렬
	priorities.sort(key=lambda i: (slack_for(i), -manhattan(agents[i].initial_position, agents[i].target)))

	horizon = max_timestep if max_timestep and max_timestep > 0 else 10**9
	node_reserved = {}
	edge_reserved = {}
	stay_reserved_earliest = {}

	def reserve_path(path: list):
		for t in range(0, len(path)):
			node_reserved.setdefault(t, set()).add(path[t])
			if t + 1 < len(path):
				edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))
		# 과거 호환을 위한 키만 유지(실제 충돌 판정에 사용하지 않음)
		if len(path) > 0:
			last = path[-1]
			arrive_t = len(path) - 1
			prev = stay_reserved_earliest.get(last)
			if prev is None or arrive_t < prev:
				stay_reserved_earliest[last] = arrive_t

	def is_conflict(curr: tuple, nxt: tuple, t_next: int) -> bool:
		if nxt in node_reserved.get(t_next, set()):
			return True
		if (nxt, curr) in edge_reserved.get(t_next, set()):
			return True
		# 목표의 장기 점유 제약은 사용하지 않음
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
		if start == goal:
			# 시작=목표인 경우 패딩(시각화용) — 필요시 제거 가능
			rev = [start]
			while len(rev) <= horizon:
				rev.append(start)
			return rev
		open_heap = []
		h0 = heuristic(start, goal)
		# (f, late_flag, h, turn_bias, g, state)
		late0 = 1 if (0 + h0 > deadline) else 0 if deadline is not None else 0
		heappush(open_heap, (h0, late0, h0, 0, 0, (start[0], start[1], start_dir, 0)))
		best_g = {(start[0], start[1], start_dir, 0): 0}
		parent = {(start[0], start[1], start_dir, 0): None}
		goal_state = None
		WAIT_PENALTY = 0.35
		TURN_COST = 0.05
		while open_heap:
			f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
			if t > horizon:
				break
			if (x, y) == goal:
				goal_state = (x, y, d, t)
				break
			# wait
			nt = t + 1
			if nt <= horizon and not is_conflict((x, y), (x, y), nt):
				s = (x, y, d, nt)
				ng = g + 1
				nh = heuristic((x, y), goal)
				if ng < best_g.get(s, 1 << 60):
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
				nh = heuristic((nx, ny), goal)
				turn_bias = 0 if action == d else 1
				if ng < best_g.get(s, 1 << 60):
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

	path_all = [[] for _ in range(num_agents)]
	for aid in priorities:
		ddl = getattr(agents[aid], 'deadline', None)
		p = plan_single(agents[aid].initial_position, agents[aid].initial_direction, agents[aid].target, ddl if ddl is not None else 1 << 30)
		if len(p) == 0:
			p = [agents[aid].initial_position]
		# 패딩 금지: 도착까지만 사용
		path_all[aid] = p
		reserve_path(p)
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

	horizon = max_timestep if max_timestep and max_timestep > 0 else 10**9

	# 예약 테이블: 현재 시점 이후 고려
	node_reserved = {}
	edge_reserved = {}
	stay_reserved_earliest = {}

	def reserve_path(path: list, from_t: int):
		for t in range(from_t, len(path)):
			node_reserved.setdefault(t, set()).add(path[t])
			if t + 1 < len(path):
				edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))
		# 목표 영구 점유는 사용하지 않음(과거 호환 키만 유지)
		if len(path) > 0:
			last = path[-1]
			arrive_t = len(path) - 1
			prev = stay_reserved_earliest.get(last)
			if prev is None or arrive_t < prev:
				stay_reserved_earliest[last] = arrive_t

	# 재계획 대상 외 에이전트는 경로 고정(현재 시점 이후 예약)
	num_agents = len(agents)
	replan_set = set(new_malfunction_agents) | set(failed_agents)
	for aid in range(num_agents):
		if aid not in replan_set:
			reserve_path(existing_paths[aid], current_timestep)

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

	transition_cache = {}
	def get_valid(x: int, y: int, d: int):
		key = (x, y, d)
		v = transition_cache.get(key)
		if v is None:
			v = rail.get_transitions(x, y, d)
			transition_cache[key] = v
		return v

	def plan_from(start: tuple, start_dir: int, start_time: int, goal: tuple, deadline: int) -> list:
		open_heap = []
		h0 = heuristic(start, goal)
		late0 = 1 if (start_time + h0 > deadline) else 0 if deadline is not None else 0
		heappush(open_heap, (h0 + start_time, late0, h0, 0, start_time - current_timestep, (start[0], start[1], start_dir, start_time)))
		best_g = {(start[0], start[1], start_dir, start_time): start_time - current_timestep}
		parent = {(start[0], start[1], start_dir, start_time): None}
		goal_state = None
		while open_heap:
			f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
			if t > horizon:
				break
			if (x, y) == goal:
				goal_state = (x, y, d, t)
				break
			# wait
			nt = t + 1
			if nt <= horizon and not is_conflict((x, y), (x, y), nt):
				s = (x, y, d, nt)
				ng = g + 1
				nh = heuristic((x, y), goal)
				if ng < best_g.get(s, 1 << 60):
					best_g[s] = ng
					parent[s] = (x, y, d, t)
					late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
					heappush(open_heap, (ng + nh, late, nh, 2, ng, s))
			# move
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
				ng = g + 1
				nh = heuristic((nx, ny), goal)
				turn_bias = 0 if action == d else 1
				if ng < best_g.get(s, 1 << 60):
					best_g[s] = ng
					parent[s] = (x, y, d, t)
					late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
					heappush(open_heap, (ng + nh, late, nh, turn_bias, ng, s))
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

	# 재계획 순서: 현재 시점 기준 slack 오름차순
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
		# 고장 강제 대기 반영
		mal = agents[aid].malfunction_data["malfunction"]
		wait_steps = max(0, mal)
		for t in range(current_timestep + 1, min(horizon, current_timestep + wait_steps) + 1):
			node_reserved.setdefault(t, set()).add(agent_pos)
		# 접두사 정합성
		base = list(existing_paths[aid][:current_timestep])
		if len(base) <= current_timestep:
			pad_pos = base[-1] if base else agent_pos
			while len(base) <= current_timestep:
				base.append(pad_pos)
		base[current_timestep] = agent_pos
		start_time = current_timestep + wait_steps
		ddl = getattr(agents[aid], 'deadline', None)
		ddl_use = ddl if ddl is not None else 1 << 30
		suffix = [] if agent_pos is None or agents[aid].target is None else plan_from(agent_pos, agent_dir, start_time, agents[aid].target, ddl_use)
		if len(suffix) > 0 and suffix[0] == agent_pos:
			suffix = suffix[1:]
		new_path = base[:current_timestep+1] + suffix
		existing_paths[aid] = new_path
		reserve_path(new_path, current_timestep)
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