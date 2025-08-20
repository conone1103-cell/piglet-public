import os
import sys

try:
    from flatland.utils.controller import evaluator
except Exception as e:
    print("Cannot load flatland modules!", e)
    sys.exit(1)

# question3_merge의 계획기와 재계획기를 불러오고, 이 맵 전용으로 상수값을 미세 조정합니다.
import question3_merge as q3
from flatland.utils.controller import Directions
import random


def override_constants_for_level3_test_5():
    # 이 맵 전용 강한 오버라이드 (더 공격적으로 조정)
    # - 대기패널티 완화(대기 허용해 충돌 회피), 회전비용 약화(우회 허용)
    # - K_HOLD/TABOO 상향으로 동시 돌진 억제
    # - 탐색 한도/로컬 지평선 확대
    # - 혼잡 비용 강화로 병목 회피 유도
    q3.WAIT_PENALTY = 0.14
    q3.TURN_COST = 0.03
    q3.K_HOLD = 3
    q3.TABOO_TICKS = 4
    q3.MAX_NODES = 800000
    q3.CONGESTION_COST = 0.10

    # 동적 패딩 강제 상향(모든 거리/규모에서 충분히 멀리 보게)
    def _padding_override(estimated_distance: int, num_agents: int) -> int:
        return max(512, estimated_distance + 320)
    q3.compute_dynamic_padding = _padding_override

    # 정렬의 합류 위험도 영향 제거(0으로 고정)
    def _no_merge_risk(*args, **kwargs):
        return 0
    q3.calculate_merge_risk = _no_merge_risk

    # Soft-hold 제거: 도착 직후 1틱 추가 점유를 끄면 꼬리물기 위험은 늘지만 처리량은 증가
    def _reserve_path_no_soft_hold(path: list, node_reserved: dict, edge_reserved: dict, from_t: int = 0):
        for t in range(from_t, len(path)):
            node_reserved.setdefault(t, set()).add(path[t])
            if t + 1 < len(path):
                edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))
    q3.reserve_path_in_tables = _reserve_path_no_soft_hold


def main():
    # 디버그/시각화는 기본 끔
    debug = False
    visualizer = False

    # 상수 오버라이드
    override_constants_for_level3_test_5()

    # 단일 케이스만 실행
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_case = os.path.join(script_dir, "multi_test_case", "level3_test_5.pkl")
    ddl_file = test_case.replace(".pkl", ".ddl")

    if not os.path.exists(test_case):
        print(f"Test case not found: {test_case}")
        sys.exit(1)
    if not os.path.exists(ddl_file):
        print(f"DDL file not found: {ddl_file}")
        sys.exit(1)

    # 여러 프로파일을 순차 실행하여 최적 조합 탐색
    profiles = []

    # 공통: 합류위험도 영향 제거, 패딩/예약 오버라이드 등은 override에서 수행

    def schedule_none(aid, agents):
        return 0

    def schedule_parity_by_dir(aid, agents):
        d = agents[aid].initial_direction
        # 수직계열(N,S)만 1틱 지연 → 동시 교차 감소
        return 1 if d in (Directions.NORTH, Directions.SOUTH) else 0

    def schedule_distance_mod3(aid, agents):
        s = agents[aid].initial_position
        g = agents[aid].target
        if s is None or g is None:
            return 0
        dist = abs(s[0] - g[0]) + abs(s[1] - g[1])
        return dist % 3

    def schedule_short_first_delay(aid, agents, median_dist_cache={}):
        # 짧은 경로 에이전트에게 2틱 지연 → 장거리 먼저 빠져나가게
        if 'median' not in median_dist_cache:
            dists = []
            for i, ag in enumerate(agents):
                s = ag.initial_position
                g = ag.target
                if s is None or g is None:
                    dists.append(0)
                else:
                    dists.append(abs(s[0] - g[0]) + abs(s[1] - g[1]))
            sd = sorted(dists)
            median = sd[len(sd)//2] if sd else 0
            median_dist_cache['median'] = median
            median_dist_cache['dists'] = dists
        dist = median_dist_cache['dists'][aid]
        return 2 if dist <= median_dist_cache['median'] else 0

    def make_get_path_with_schedule(schedule_fn):
        def _get_path(agents, rail, max_timestep):
            base = q3.get_path(agents, rail, max_timestep)
            new_paths = []
            for aid in range(len(agents)):
                delay = schedule_fn(aid, agents) or 0
                delay = int(delay)
                if delay <= 0:
                    new_paths.append(base[aid])
                    continue
                start = agents[aid].initial_position
                if start is None:
                    new_paths.append(base[aid])
                else:
                    new_paths.append([start]*delay + base[aid])
            return new_paths
        return _get_path

    # 프로파일 정의
    def profile_A():
        override_constants_for_level3_test_5()
        return make_get_path_with_schedule(schedule_none)

    def profile_B():
        override_constants_for_level3_test_5()
        # 보수 강화: 대기/회전 완화, 혼잡 강함, TABOO↑
        q3.WAIT_PENALTY = 0.12
        q3.TURN_COST = 0.02
        q3.TABOO_TICKS = 5
        q3.CONGESTION_COST = 0.12
        return make_get_path_with_schedule(schedule_parity_by_dir)

    def profile_C():
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.18
        q3.TURN_COST = 0.02
        q3.CONGESTION_COST = 0.08
        return make_get_path_with_schedule(schedule_distance_mod3)

    def profile_D():
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.16
        q3.TURN_COST = 0.04
        q3.CONGESTION_COST = 0.10
        q3.K_HOLD = 4
        q3.TABOO_TICKS = 5
        return make_get_path_with_schedule(schedule_short_first_delay)

    def profile_B2():
        # Profile-B 강화판: 더 강한 스태깅 + 넉넉한 탐색/패딩
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.10
        q3.TURN_COST = 0.02
        q3.K_HOLD = 3
        q3.TABOO_TICKS = 6
        q3.CONGESTION_COST = 0.12
        q3.MAX_NODES = 1000000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(720, estimated_distance + 400)
        q3.compute_dynamic_padding = _padding_override

        # 방향 기반 + 근접 혼잡 + 위치 지터 스케줄
        def build_schedule(agents):
            n = len(agents)
            starts = [ag.initial_position for ag in agents]
            dirs = [ag.initial_direction for ag in agents]
            crowd = [0]*n
            for i in range(n):
                si = starts[i]
                if si is None:
                    continue
                c = 0
                for j in range(n):
                    if i==j: continue
                    sj = starts[j]
                    if sj is None: continue
                    if si[0]==sj[0] and abs(si[1]-sj[1])<=4:
                        c += 1
                    if si[1]==sj[1] and abs(si[0]-sj[0])<=4:
                        c += 1
                crowd[i] = c
            delays = [0]*n
            for i in range(n):
                base = 1 if dirs[i] in (Directions.NORTH, Directions.SOUTH) else 0
                near = min(2, crowd[i]//3)
                s = starts[i]
                jitter = ((s[0]*5 + s[1]*11) % 2) if s else 0
                delay = base + near + jitter
                if delay > 6:
                    delay = 6
                delays[i] = delay
            def _schedule(aid, _agents):
                return delays[aid]
            return _schedule

        def _get_path_with_b2(agents, rail, max_timestep):
            sched = build_schedule(agents)
            base = q3.get_path(agents, rail, max_timestep)
            new_paths = []
            for aid in range(len(agents)):
                delay = int(sched(aid, agents) or 0)
                if delay <= 0:
                    new_paths.append(base[aid])
                    continue
                start = agents[aid].initial_position
                new_paths.append([start]*delay + base[aid] if start else base[aid])
            return new_paths
        return _get_path_with_b2

    def profile_B3():
        # Profile-B 변형: 거리 기반 가벼운 변조 추가
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.12
        q3.TURN_COST = 0.02
        q3.K_HOLD = 3
        q3.TABOO_TICKS = 6
        q3.CONGESTION_COST = 0.15
        q3.MAX_NODES = 900000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(640, estimated_distance + 360)
        q3.compute_dynamic_padding = _padding_override

        def build_schedule(agents):
            n = len(agents)
            starts = [ag.initial_position for ag in agents]
            goals = [ag.target for ag in agents]
            dirs = [ag.initial_direction for ag in agents]
            dists = []
            for i in range(n):
                s, g = starts[i], goals[i]
                if s is None or g is None:
                    dists.append(0)
                else:
                    dists.append(abs(s[0]-g[0]) + abs(s[1]-g[1]))
            delays = [0]*n
            for i in range(n):
                base = 1 if dirs[i] in (Directions.NORTH, Directions.SOUTH) else 0
                mod = dists[i] % 2
                s = starts[i]
                jitter = ((s[0]*3 + s[1]*7) % 2) if s else 0
                delay = base + mod + jitter
                if delay > 5:
                    delay = 5
                delays[i] = delay
            def _schedule(aid, _agents):
                return delays[aid]
            return _schedule

        def _get_path_with_b3(agents, rail, max_timestep):
            sched = build_schedule(agents)
            base = q3.get_path(agents, rail, max_timestep)
            new_paths = []
            for aid in range(len(agents)):
                delay = int(sched(aid, agents) or 0)
                if delay <= 0:
                    new_paths.append(base[aid])
                    continue
                start = agents[aid].initial_position
                new_paths.append([start]*delay + base[aid] if start else base[aid])
            return new_paths
        return _get_path_with_b3

    def profile_B4():
        # 행/열 기반 스태깅: 같은 행(수평)·열(수직) 그룹 내부에서 단계적 지연 부여
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.10
        q3.TURN_COST = 0.02
        q3.K_HOLD = 3
        q3.TABOO_TICKS = 6
        q3.CONGESTION_COST = 0.12
        q3.MAX_NODES = 1100000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(700, estimated_distance + 380)
        q3.compute_dynamic_padding = _padding_override

        def build_schedule(agents):
            from collections import defaultdict
            n = len(agents)
            starts = [ag.initial_position for ag in agents]
            dirs = [ag.initial_direction for ag in agents]
            row_groups = defaultdict(list)
            col_groups = defaultdict(list)
            for i in range(n):
                s = starts[i]
                d = dirs[i]
                if s is None or d is None:
                    continue
                if d in (Directions.EAST, Directions.WEST):
                    row_groups[s[0]].append(i)
                else:
                    col_groups[s[1]].append(i)
            delays = [0]*n
            # 행 그룹: 좌->우 정렬 후 0,1,2,... 순차 지연(최대 6)
            for r, idxs in row_groups.items():
                idxs.sort(key=lambda i: starts[i][1])
                for k, i in enumerate(idxs):
                    delays[i] = min(6, k)
            # 열 그룹: 상->하 정렬 후 0,1,2,... 순차 지연(최대 6)
            for c, idxs in col_groups.items():
                idxs.sort(key=lambda i: starts[i][0])
                for k, i in enumerate(idxs):
                    # 이미 행에서 지정된 경우 더 큰 값 사용(더 강한 분산)
                    delays[i] = max(delays[i], min(6, k))
            return lambda aid, _agents: delays[aid]

        def _get_path_with_b4(agents, rail, max_timestep):
            sched = build_schedule(agents)
            base = q3.get_path(agents, rail, max_timestep)
            new_paths = []
            for aid in range(len(agents)):
                delay = int(sched(aid, agents) or 0)
                if delay <= 0:
                    new_paths.append(base[aid])
                    continue
                start = agents[aid].initial_position
                new_paths.append([start]*delay + base[aid] if start else base[aid])
            return new_paths
        return _get_path_with_b4
    def profile_E():
        # 가장 강한 스태깅 + 넉넉한 탐색
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.12
        q3.TURN_COST = 0.02
        q3.K_HOLD = 3
        q3.TABOO_TICKS = 5
        q3.CONGESTION_COST = 0.12
        q3.MAX_NODES = 1200000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(640, estimated_distance + 400)
        q3.compute_dynamic_padding = _padding_override

        # 사전 계산 기반 혼잡-방향-거리 혼합 스케줄
        def build_schedule(agents):
            n = len(agents)
            starts = [ag.initial_position for ag in agents]
            goals = [ag.target for ag in agents]
            dirs = [ag.initial_direction for ag in agents]
            dists = []
            for i in range(n):
                s, g = starts[i], goals[i]
                if s is None or g is None:
                    dists.append(0)
                else:
                    dists.append(abs(s[0]-g[0]) + abs(s[1]-g[1]))
            sd = sorted(dists)
            median = sd[n//2] if n>0 else 0

            # 근접 혼잡도(맨해튼 반경 5 내 같은 행/열 근접 수)
            crowd = [0]*n
            for i in range(n):
                si = starts[i]
                if si is None:
                    continue
                cx = cy = 0
                for j in range(n):
                    if i==j: continue
                    sj = starts[j]
                    if sj is None: continue
                    if si[0]==sj[0] and abs(si[1]-sj[1])<=5:
                        cy += 1
                    if si[1]==sj[1] and abs(si[0]-sj[0])<=5:
                        cx += 1
                crowd[i] = cx + cy

            delays = [0]*n
            for i in range(n):
                base = 1 if dirs[i] in (Directions.NORTH, Directions.SOUTH) else 0
                near = crowd[i]//3
                short = 1 if dists[i] <= median else 0
                s = starts[i]
                jitter = ((s[0]*7 + s[1]*13) % 2) if s else 0
                delay = base + near + short + jitter
                if delay > 6: delay = 6
                delays[i] = delay

            def _schedule(aid, _agents):
                return delays[aid]
            return _schedule

        def _get_path_with_adv_schedule(agents, rail, max_timestep):
            sched = build_schedule(agents)
            base = q3.get_path(agents, rail, max_timestep)
            new_paths = []
            for aid in range(len(agents)):
                delay = int(sched(aid, agents) or 0)
                if delay <= 0:
                    new_paths.append(base[aid])
                    continue
                start = agents[aid].initial_position
                if start is None:
                    new_paths.append(base[aid])
                else:
                    new_paths.append([start]*delay + base[aid])
            return new_paths

        return _get_path_with_adv_schedule

    def profile_B2X():
        # 2-패스: 1차 경로로 혼잡 노드/시간을 추출 → 해당 시점 노드에 일시적 금지(traffic light) 후 재계획
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.10
        q3.TURN_COST = 0.02
        q3.K_HOLD = 3
        q3.TABOO_TICKS = 6
        q3.CONGESTION_COST = 0.12
        q3.MAX_NODES = 1200000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(720, estimated_distance + 420)
        q3.compute_dynamic_padding = _padding_override

        def _get_path_with_b2x(agents, rail, max_timestep):
            # 1차: 기본 경로 산출(가벼운 랜덤 지터 스케줄)
            rnd = random.Random(42)
            def sched(aid, _agents):
                d = _agents[aid].initial_direction
                base = 1 if d in (Directions.NORTH, Directions.SOUTH) else 0
                j = rnd.randint(0, 1)
                return base + j
            base_paths = make_get_path_with_schedule(sched)(agents, rail, max_timestep)

            # 혼잡 노드 감지
            node_load = {}
            for aid, p in enumerate(base_paths):
                for t, pos in enumerate(p):
                    node_load.setdefault(t, {})[pos] = node_load.get(t, {}).get(pos, 0) + 1

            # traffic light 예약 집합 생성: base에서 동시 점유(>=2) 발생한 노드를 동일/다음 틱에 교대로 차단
            traffic_reserved = {}
            for t, counts in node_load.items():
                for pos, c in counts.items():
                    if c >= 2:
                        # 해당 시점과 다음 시점 중 하나만 차단(짝/홀 교대)
                        target_t = t + ( (pos[0] + pos[1]) & 1 )
                        s = traffic_reserved.setdefault(target_t, set())
                        s.add(pos)

            # is_conflict를 임시 패치
            orig_is_conflict = q3.is_conflict_in_tables
            def patched_is_conflict(curr, nxt, t_next, node_reserved, edge_reserved, node_capacity=None):
                if nxt in traffic_reserved.get(t_next, set()):
                    return True
                return orig_is_conflict(curr, nxt, t_next, node_reserved, edge_reserved)
            q3.is_conflict_in_tables = patched_is_conflict

            try:
                # 2차: 차단 반영하여 재계획
                final_paths = q3.get_path(agents, rail, max_timestep)
            finally:
                # 원복
                q3.is_conflict_in_tables = orig_is_conflict
            return final_paths

        return _get_path_with_b2x

    def profile_B2LNS():
        # B2 기반 + LNS(부분 재계획 반복)로 충돌/지각 상위 에이전트만 선택 재계획
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.10
        q3.TURN_COST = 0.02
        q3.K_HOLD = 3
        q3.TABOO_TICKS = 6
        q3.CONGESTION_COST = 0.12
        q3.MAX_NODES = 1200000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(720, estimated_distance + 420)
        q3.compute_dynamic_padding = _padding_override

        def build_schedule(agents):
            n = len(agents)
            starts = [ag.initial_position for ag in agents]
            dirs = [ag.initial_direction for ag in agents]
            crowd = [0]*n
            for i in range(n):
                si = starts[i]
                if si is None:
                    continue
                c = 0
                for j in range(n):
                    if i==j: continue
                    sj = starts[j]
                    if sj is None: continue
                    if si[0]==sj[0] and abs(si[1]-sj[1])<=4:
                        c += 1
                    if si[1]==sj[1] and abs(si[0]-sj[0])<=4:
                        c += 1
                crowd[i] = c
            delays = [0]*n
            for i in range(n):
                base = 1 if dirs[i] in (Directions.NORTH, Directions.SOUTH) else 0
                near = min(2, crowd[i]//3)
                delay = base + near
                if delay > 6:
                    delay = 6
                delays[i] = delay
            return lambda aid, _agents: delays[aid]

        def _count_scores(agents, paths):
            node_occ = {}
            edge_occ = {}
            scores = [0]*len(paths)
            for aid, p in enumerate(paths):
                for t in range(len(p)):
                    pos = p[t]
                    node_occ.setdefault(t, {}).setdefault(pos, []).append(aid)
                    if t+1 < len(p):
                        e = (p[t], p[t+1])
                        edge_occ.setdefault(t+1, {}).setdefault(e, []).append(aid)
            for t, occ in node_occ.items():
                for pos, lst in occ.items():
                    if len(lst) > 1:
                        for aid in lst:
                            scores[aid] += len(lst)-1
            for t, occ in edge_occ.items():
                for (a,b), lst in occ.items():
                    rev = (b,a)
                    if rev in occ:
                        both = set(lst) | set(occ[rev])
                        for aid in both:
                            scores[aid] += 1
            for aid in range(len(agents)):
                ddl = getattr(agents[aid], 'deadline', None)
                if ddl is not None:
                    lateness = max(0, len(paths[aid])-1 - ddl)
                    scores[aid] += 2*lateness
            return scores

        def _get_path_with_b2lns(agents, rail, max_timestep):
            sched = build_schedule(agents)
            paths = make_get_path_with_schedule(sched)(agents, rail, max_timestep)
            for _ in range(3):
                scores = _count_scores(agents, paths)
                top = sorted(range(len(agents)), key=lambda i: scores[i], reverse=True)[:min(10, len(agents))]
                if all(scores[i] == 0 for i in top):
                    break
                paths = q3.replan(agents, rail, 0, paths, max_timestep, [], top)
            return paths

        return _get_path_with_b2lns

    def profile_B5():
        # 더 강한 스태깅(최대 10틱) + K_HOLD/TABOO 강화 + 탐색 확대 + 제한적 재계획(1회)
        override_constants_for_level3_test_5()
        q3.WAIT_PENALTY = 0.08
        q3.TURN_COST = 0.02
        q3.K_HOLD = 4
        q3.TABOO_TICKS = 8
        q3.CONGESTION_COST = 0.12
        q3.MAX_NODES = 1500000
        def _padding_override(estimated_distance: int, num_agents: int) -> int:
            return max(800, estimated_distance + 480)
        q3.compute_dynamic_padding = _padding_override

        from collections import defaultdict

        def build_schedule(agents):
            n = len(agents)
            starts = [ag.initial_position for ag in agents]
            goals = [ag.target for ag in agents]
            dirs = [ag.initial_direction for ag in agents]

            # 방향 수량 비율에 따라 우선 파 생성(수평 우선 또는 수직 우선)
            horiz = sum(1 for d in dirs if d in (Directions.EAST, Directions.WEST))
            vert = n - horiz
            horiz_first = horiz >= vert

            # 행/열 그룹 수집
            row_groups = defaultdict(list)
            col_groups = defaultdict(list)
            for i, s in enumerate(starts):
                if s is None or dirs[i] is None:
                    continue
                if dirs[i] in (Directions.EAST, Directions.WEST):
                    row_groups[s[0]].append(i)
                else:
                    col_groups[s[1]].append(i)

            # 기본 지연: 방향 파동 + 로컬 군집 지연
            delays = [0]*n
            for i in range(n):
                d = dirs[i]
                base = 0
                if d in (Directions.EAST, Directions.WEST):
                    base = 0 if horiz_first else 2
                else:
                    base = 0 if not horiz_first else 2
                delays[i] = base

            # 행 그룹에서 좌->우 순차 지연, 열 그룹에서 상->하 순차 지연
            for r, idxs in row_groups.items():
                idxs.sort(key=lambda i: starts[i][1])
                for k, i in enumerate(idxs):
                    delays[i] = max(delays[i], min(10, k))
            for c, idxs in col_groups.items():
                idxs.sort(key=lambda i: starts[i][0])
                for k, i in enumerate(idxs):
                    delays[i] = max(delays[i], min(10, k))

            # 거리 기반 미세 조정(혼잡 완화): 짧은 경로는 추가 +1, 중간 0, 긴 경로 0
            for i in range(n):
                s, g = starts[i], goals[i]
                if s is None or g is None:
                    continue
                dist = abs(s[0]-g[0]) + abs(s[1]-g[1])
                if dist <= 8:
                    delays[i] = min(10, delays[i] + 1)

            return lambda aid, _agents: delays[aid]

        def _count_scores(agents, paths):
            node_occ = {}
            edge_occ = {}
            scores = [0]*len(paths)
            for aid, p in enumerate(paths):
                for t in range(len(p)):
                    pos = p[t]
                    node_occ.setdefault(t, {}).setdefault(pos, []).append(aid)
                    if t+1 < len(p):
                        e = (p[t], p[t+1])
                        edge_occ.setdefault(t+1, {}).setdefault(e, []).append(aid)
            for t, occ in node_occ.items():
                for pos, lst in occ.items():
                    if len(lst) > 1:
                        for aid in lst:
                            scores[aid] += len(lst)-1
            for t, occ in edge_occ.items():
                for (a,b), lst in occ.items():
                    rev = (b,a)
                    if rev in occ:
                        both = set(lst) | set(occ[rev])
                        for aid in both:
                            scores[aid] += 1
            for aid in range(len(agents)):
                ddl = getattr(agents[aid], 'deadline', None)
                if ddl is not None:
                    lateness = max(0, len(paths[aid])-1 - ddl)
                    scores[aid] += 2*lateness
            return scores

        def _get_path_with_b5(agents, rail, max_timestep):
            sched = build_schedule(agents)
            paths = make_get_path_with_schedule(sched)(agents, rail, max_timestep)
            # 상위 문제 에이전트 소수만 1회 재계획
            scores = _count_scores(agents, paths)
            top = sorted(range(len(agents)), key=lambda i: scores[i], reverse=True)[:min(8, len(agents))]
            if any(scores[i] > 0 for i in top):
                paths = q3.replan(agents, rail, 0, paths, max_timestep, [], top)
            return paths

        return _get_path_with_b5

    profiles = [
        ("Profile-B5", profile_B5),
        ("Profile-B2", profile_B2),
        ("Profile-B3", profile_B3),
    ]

    for name, builder in profiles:
        print(f"\n=== Running {name} ===")
        get_path_fn = builder()
        evaluator(get_path_fn, [test_case], debug, visualizer, 3, [ddl_file], replan=q3.replan)


if __name__ == "__main__":
    main()


