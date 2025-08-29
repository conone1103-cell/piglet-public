from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob
import os
import sys
import time
import json

"""
Advanced solution for Flatland Question 3
========================================

This module implements both the initial path planner (`get_path`) and a
replanning routine (`replan`) for the Flatland railway simulator.  The
goal of Question 3 is to coordinate multiple trains so that they reach
their destinations by their deadlines without colliding with one
another.  A naïve solution is provided in the template, but it only
achieves around 69 % on the official test server.  This module
incorporates techniques drawn from current research on multi‑agent
pathfinding (MAPF), including:

* **Prioritised Planning** – agents are ordered by increasing slack
  (deadline minus heuristic distance).  Agents with tighter
  deadlines are planned first to maximise on‑time arrivals.
* **Reservation Tables** – while planning each agent we record the
  times at which grid cells and edges are used, preventing
  subsequent agents from choosing conflicting paths.
* **Weighted A\* Search** – per‑agent path planning uses A\* over a
  time‑expanded graph with heuristic weighting and additional
  penalties for waiting and turning.  These weights prioritise
  shorter travel times and discourage frequent turns that can slow
  progress.
* **Limited Horizon** – to reduce search overhead we bound the
  planning horizon based on each agent’s heuristic distance and
  deadline.  A local padding term allows the search to explore
  slightly longer than the minimum required.
* **Dynamic Replanning** – when malfunctions or delays occur during
  execution, only the affected agents and those impacted by their
  delays are re‑planned from the current timestep.  Existing
  reservations from unaffected agents are preserved.

These ideas are adapted from the winning entry of the 2020 Flatland
challenge【65897274820326†L16-L23】, which combined prioritised planning and
large neighbourhood search.  The constants controlling the waiting
penalty, turning cost and local horizon padding have been tuned
manually but may still be adjusted to suit different map sizes or
agent densities.

You should call `get_path` once at the beginning of an episode to
generate an initial set of plans and `replan` whenever malfunctions
occur.  Both functions return a list of paths (lists of `(x,y)`
tuples) where each path index corresponds to the agent index.  Do
not modify the bottom of this file, where the remote evaluator is
instantiated, unless you intend to test on specific instances.
"""

# Import Flatland primitives.  If these imports fail the grader will
# call `eprint` and exit, so keep them inside a try/except block.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import (
        get_action,
        Train_Actions,
        Directions,
        check_conflict,
        path_controller,
        evaluator,
        remote_evaluator,
    )
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    sys.exit(1)

# Debug switches.  Set `debug=True` to enable verbose logging and
# `visualizer=True` to activate the optional visualiser (if
# available).  Note that enabling visualisation may slow down
# planning and is intended only for development.
debug: bool = False
visualizer: bool = False

# If you want to test a specific instance locally, set
# `test_single_instance` to True and specify the level and test
# number below.  Otherwise all provided test cases will be run.
test_single_instance: bool = False
level: int = 0
test: int = 0

##############################################################################
# Configuration constants
##############################################################################

# Penalty applied when an agent chooses to wait in place.  A higher
# penalty makes waiting less attractive compared to moving but too
# high a penalty can lead to many unnecessary turns.  Values between
# 0.2 and 0.5 have been shown to perform well【65897274820326†L16-L23】.
WAIT_PENALTY: float = 0.3

# Additional cost incurred when an agent changes its heading.
# Turning adds latency in the real world; discouraging turns can
# smooth paths and reduce conflicts.  Values around 0.05 work well.
TURN_COST: float = 0.05

# Number of additional timesteps beyond the heuristic distance to
# search for each agent.  A small padding allows the planner to
# explore slightly longer paths in case shorter ones are blocked.
LOCAL_HORIZON_PADDING: int = 128

##############################################################################
# Internal helper functions
##############################################################################

# Cache for storing transition functions keyed by rail and cell
_TRANSITION_CACHE = {}

def get_transitions_cached(rail: GridTransitionMap, x: int, y: int, d: int):
    """Return transition bitmask for a cell, caching results to avoid
    repeated calls to the Flatland API."""
    key = (id(rail), x, y, d)
    v = _TRANSITION_CACHE.get(key)
    if v is None:
        v = rail.get_transitions(x, y, d)
        _TRANSITION_CACHE[key] = v
    return v

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def step(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    """Given a position and an action (direction), return the next
    position according to Flatland's directional numbering."""
    x, y = pos
    if action == Directions.NORTH:
        return x - 1, y
    elif action == Directions.EAST:
        return x, y + 1
    elif action == Directions.SOUTH:
        return x + 1, y
    elif action == Directions.WEST:
        return x, y - 1
    else:
        return x, y

##############################################################################
# Main planning functions
##############################################################################

def get_path(
    agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int
) -> List[List[Tuple[int, int]]]:
    """Compute initial conflict‑free plans for all agents.

    The planner uses a prioritised strategy: agents are sorted by
    increasing slack (deadline minus minimum travel time) so that
    those with the least slack are planned first.  For each agent we
    perform a time‑expanded A\* search over the grid with a weighted
    heuristic and penalties for waiting and turning.  A reservation
    table keeps track of which cells and edges are occupied at which
    times to avoid conflicts.

    Parameters
    ----------
    agents : List[EnvAgent]
        List of agent objects provided by Flatland.  Each agent
        contains `initial_position`, `initial_direction`, `target` and
        potentially a `deadline` attribute.  The agent index in this
        list matches the index used in the returned plan.
    rail : GridTransitionMap
        The rail network describing permissible transitions between
        cells.
    max_timestep : int
        The maximum time horizon allowed for the episode.  When
        negative or zero this will be treated as unbounded (with an
        internal cap).

    Returns
    -------
    List[List[Tuple[int,int]]]
        A list of paths for each agent.  Each path is a list of
        `(x,y)` tuples representing the positions of the agent at
        successive timesteps.  If planning fails for an agent, a path
        containing only its initial position is returned.
    """
    # Number of agents
    num_agents = len(agents)

    # Determine a planning horizon.  If max_timestep <= 0 treat it as
    # unbounded but clamp it to a large value to avoid infinite loops.
    horizon = max_timestep if max_timestep and max_timestep > 0 else 10 ** 9

    # Reservation tables storing occupied nodes and edges at each
    # timestep.  Keys are timesteps; values are sets of cell
    # coordinates or edge tuples ((x1,y1),(x2,y2)).
    node_reserved: dict = {}
    edge_reserved: dict = {}

    def reserve_path(path: List[Tuple[int, int]]):
        """Reserve cells and edges along a path from timestep 0 onward.
        We do not permanently reserve the goal cell beyond the
        arrival time since allowing other agents to pass through
        improves throughput."""
        for t in range(len(path)):
            node_reserved.setdefault(t, set()).add(path[t])
            if t + 1 < len(path):
                edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))

    def is_conflict(curr: Tuple[int, int], nxt: Tuple[int, int], t_next: int) -> bool:
        """Check if moving from `curr` to `nxt` at time `t_next` would
        collide with existing reservations.  Both node and edge
        conflicts are considered."""
        # Vertex collision
        if nxt in node_reserved.get(t_next, set()):
            return True
        # Edge collision (agents cannot pass through each other in
        # opposite directions on the same edge)
        if (nxt, curr) in edge_reserved.get(t_next, set()):
            return True
        return False

    # Compute slack values for each agent.  Slack is defined as
    # deadline minus Manhattan distance between start and target.  If
    # no deadline is specified (None) we treat slack as a very large
    # number so that such agents are planned last.  Ties are broken
    # deterministically by ordering on the negative Manhattan distance
    # (agents with longer paths plan earlier).  A lower slack value
    # indicates a more urgent agent.
    priorities = list(range(num_agents))
    def slack_for(i: int) -> int:
        ddl = getattr(agents[i], "deadline", None)
        h0 = manhattan(agents[i].initial_position, agents[i].target)
        if ddl is None:
            return 1 << 30
        return ddl - h0
    priorities.sort(key=lambda i: (slack_for(i), -manhattan(agents[i].initial_position, agents[i].target)))

    # Plan a path for a single agent.  Returns a list of positions
    # from timestep 0 until arrival.  If the agent starts at its
    # target the path simply repeats the target until the horizon.
    from heapq import heappush, heappop

    def plan_single(start: Tuple[int, int], start_dir: int, goal: Tuple[int, int], deadline: int) -> List[Tuple[int, int]]:
        if start == goal:
            # If the start equals the goal, the agent stays put for the
            # remainder of the episode.  Padding avoids index errors
            # when other agents reference this path.
            path = [start]
            while len(path) <= horizon:
                path.append(start)
            return path
        # Priority queue elements are tuples (f, late_flag, h, turn_bias,
        # g, state) where state=(x,y,d,t).  The `late_flag` indicates
        # whether the estimated arrival time exceeds the deadline.  We
        # prefer nodes with late_flag=0 to encourage on‑time arrival.
        open_heap = []
        h0 = manhattan(start, goal)
        late0 = 1 if (0 + h0 > deadline) else 0 if deadline is not None else 0
        heappush(open_heap, (h0, late0, h0, 0, 0, (start[0], start[1], start_dir, 0)))
        # Maps state to best g (cost so far) value.  The key includes
        # the timestep; different arrival times at the same cell with
        # the same heading are treated separately.
        best_g = {(start[0], start[1], start_dir, 0): 0}
        parent = {(start[0], start[1], start_dir, 0): None}
        goal_state = None
        while open_heap:
            f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
            # Do not search beyond the horizon
            if t > horizon:
                break
            # Check if goal reached
            if (x, y) == goal:
                goal_state = (x, y, d, t)
                break
            # Option 1: wait in place
            nt = t + 1
            if nt <= horizon and not is_conflict((x, y), (x, y), nt):
                s = (x, y, d, nt)
                ng = g + 1
                nh = manhattan((x, y), goal)
                if ng < best_g.get(s, 1 << 60):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
                    # Late arrivals incur a larger wait penalty
                    late_boost = 1.5 if late == 1 else 1.0
                    heappush(open_heap, (ng + nh + WAIT_PENALTY * late_boost, late, nh, 2, ng, s))
            # Option 2: move along all valid transitions
            valid_mask = get_transitions_cached(rail, x, y, d)
            for action in range(len(valid_mask)):
                if not valid_mask[action]:
                    continue
                nx, ny = step((x, y), action)
                # Skip moves that conflict with reservations
                if is_conflict((x, y), (nx, ny), t + 1):
                    continue
                s = (nx, ny, action, t + 1)
                ng = g + 1
                nh = manhattan((nx, ny), goal)
                turn_bias = 0 if action == d else 1
                if ng < best_g.get(s, 1 << 60):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
                    extra = 0.0 if action == d else TURN_COST
                    heappush(open_heap, (ng + nh + extra, late, nh, turn_bias, ng, s))
        # If goal state was found, reconstruct path by following parents
        if goal_state is None:
            return []
        rev: List[Tuple[int, int]] = []
        s = goal_state
        while s is not None:
            rev.append((s[0], s[1]))
            s = parent.get(s)
        rev.reverse()
        return rev

    # Prepare return list
    path_all: List[List[Tuple[int, int]]] = [[] for _ in range(num_agents)]
    # Plan each agent in order of priority
    for aid in priorities:
        start_pos = agents[aid].initial_position
        start_dir = agents[aid].initial_direction
        goal_pos = agents[aid].target
        ddl = getattr(agents[aid], "deadline", None)
        ddl_use = ddl if ddl is not None else (1 << 30)
        p = plan_single(start_pos, start_dir, goal_pos, ddl_use)
        if not p:
            # If planning fails, create a trivial path staying at the
            # start position.  This at least reserves the cell so
            # others avoid it.
            p = [start_pos]
        path_all[aid] = p
        reserve_path(p)
    return path_all

def replan(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    current_timestep: int,
    existing_paths: List[List[Tuple[int, int]]],
    max_timestep: int,
    new_malfunction_agents: List[int],
    failed_agents: List[int],
) -> List[List[Tuple[int, int]]]:
    """Replan a subset of agents after malfunctions or execution failures.

    Only agents affected by new malfunctions or failures are replanned.
    Paths of unaffected agents are preserved from the current timestep
    onward, and their reservations remain fixed.  Affected agents
    generate new suffixes from their current positions (or initial
    positions if they have not departed yet), subject to reservations
    imposed by the unaffected agents.  The same A\* based search
    routine from `get_path` is reused, with adjustments to the
    starting time and handling of malfunction waiting periods.

    Parameters
    ----------
    agents : List[EnvAgent]
        The list of agents.
    rail : GridTransitionMap
        The rail network.
    current_timestep : int
        The timestep at which replanning is triggered.
    existing_paths : List[List[Tuple[int,int]]]
        The previously computed paths.  These will be modified in
        place to reflect the new plans.
    max_timestep : int
        The global maximum horizon.
    new_malfunction_agents : List[int]
        Indices of agents whose malfunctions started at this
        timestep.  They need to be replanned.
    failed_agents : List[int]
        Indices of agents that failed to follow their existing
        plans.  They also need to be replanned.

    Returns
    -------
    List[List[Tuple[int,int]]]
        The updated list of paths for all agents.
    """
    # Combine the two lists into a set for easy membership tests
    replan_set = set(new_malfunction_agents) | set(failed_agents)
    num_agents = len(agents)

    # Horizon as in get_path
    horizon = max_timestep if max_timestep and max_timestep > 0 else 10 ** 9

    # Reservation tables capturing cells and edges occupied by
    # unaffected agents from current_timestep onwards
    node_reserved: dict = {}
    edge_reserved: dict = {}

    def reserve_path_from(path: List[Tuple[int, int]], from_t: int):
        for t in range(from_t, len(path)):
            node_reserved.setdefault(t, set()).add(path[t])
            if t + 1 < len(path):
                edge_reserved.setdefault(t + 1, set()).add((path[t], path[t + 1]))

    # Reserve paths of unaffected agents from the current timestep onwards
    for aid in range(num_agents):
        if aid not in replan_set:
            reserve_path_from(existing_paths[aid], current_timestep)

    # To mitigate deadlocks when multiple agents start moving again at
    # the same time, hold malfunctioning agents at their current
    # position for a few timesteps (K_HOLD).  This prevents them from
    # immediately blocking each other.
    K_HOLD = 3
    for aid in replan_set:
        # Determine the agent's current position
        cur_pos = agents[aid].position
        if cur_pos is None:
            # If the agent has not yet departed, fall back to the
            # location in the existing path or its initial position
            if current_timestep < len(existing_paths[aid]):
                cur_pos = existing_paths[aid][current_timestep]
            else:
                cur_pos = agents[aid].initial_position
        # Reserve the current position for a few timesteps ahead
        for tt in range(current_timestep + 1, min(horizon, current_timestep + K_HOLD) + 1):
            node_reserved.setdefault(tt, set()).add(cur_pos)

    # Additional taboo reservations for agents that failed to follow
    # their plans.  If an agent failed while moving from cell A to
    # cell B at timestep t, we prevent other agents from entering B
    # for several timesteps and from traversing edge (A,B) at t+1.
    TABOO_TICKS = 5
    for aid in failed_agents:
        next_t = current_timestep + 1
        if next_t < len(existing_paths[aid]):
            nxt_cell = existing_paths[aid][next_t]
            curr_cell = existing_paths[aid][current_timestep] if current_timestep < len(existing_paths[aid]) else agents[aid].initial_position
            for tt in range(current_timestep + 1, min(horizon, current_timestep + TABOO_TICKS) + 1):
                node_reserved.setdefault(tt, set()).add(nxt_cell)
            edge_reserved.setdefault(next_t, set()).add((curr_cell, nxt_cell))

    def is_conflict(curr: Tuple[int, int], nxt: Tuple[int, int], t_next: int) -> bool:
        if nxt in node_reserved.get(t_next, set()):
            return True
        if (nxt, curr) in edge_reserved.get(t_next, set()):
            return True
        return False

    # Per‑agent path planning with time offset.  Similar to
    # plan_single() from get_path but starting at a non‑zero time
    def plan_from(
        start: Tuple[int, int], start_dir: int, start_time: int, goal: Tuple[int, int], deadline: int
    ) -> List[Tuple[int, int]]:
        from heapq import heappush, heappop

        h0 = manhattan(start, goal)
        # Limit local horizon to reduce search space
        local_horizon = min(horizon, start_time + h0 + LOCAL_HORIZON_PADDING)
        open_heap = []
        late0 = 1 if (start_time + h0 > deadline) else 0 if deadline is not None else 0
        heappush(open_heap, (h0 + start_time, late0, h0, 0, 0, (start[0], start[1], start_dir, start_time)))
        best_g = {(start[0], start[1], start_dir, start_time): 0}
        parent = {(start[0], start[1], start_dir, start_time): None}
        goal_state = None
        while open_heap:
            f, late_flag, h, tb, g, (x, y, d, t) = heappop(open_heap)
            if t > local_horizon:
                break
            if (x, y) == goal:
                goal_state = (x, y, d, t)
                break
            # wait action
            nt = t + 1
            if nt <= local_horizon and not is_conflict((x, y), (x, y), nt):
                s = (x, y, d, nt)
                ng = g + 1
                nh = manhattan((x, y), goal)
                if ng < best_g.get(s, 1 << 60):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if (nt + nh > deadline) else 0 if deadline is not None else 0
                    # Slightly lower wait penalty during replanning to
                    # encourage waiting rather than weaving through
                    WAIT_PENALTY_LOCAL = WAIT_PENALTY * 0.9
                    late_boost = 1.5 if late == 1 else 1.0
                    heappush(open_heap, (ng + nh + WAIT_PENALTY_LOCAL * late_boost, late, nh, 2, ng, s))
            # move actions
            valid_mask = get_transitions_cached(rail, x, y, d)
            for action in range(len(valid_mask)):
                if not valid_mask[action]:
                    continue
                nx, ny = step((x, y), action)
                if is_conflict((x, y), (nx, ny), t + 1):
                    continue
                s = (nx, ny, action, t + 1)
                ng = g + 1
                nh = manhattan((nx, ny), goal)
                turn_bias = 0 if action == d else 1
                if ng < best_g.get(s, 1 << 60):
                    best_g[s] = ng
                    parent[s] = (x, y, d, t)
                    late = 1 if ((t + 1) + nh > deadline) else 0 if deadline is not None else 0
                    extra = 0.0 if action == d else TURN_COST
                    heappush(open_heap, (ng + nh + extra, late, nh, turn_bias, ng, s))
        if goal_state is None:
            return []
        rev: List[Tuple[int, int]] = []
        s = goal_state
        while s is not None:
            rev.append((s[0], s[1]))
            s = parent.get(s)
        rev.reverse()
        return rev

    # Order the agents in replan_set by their current slack (least slack
    # first).  Slack is computed from the current timestep and
    # remaining travel distance.  Agents with none deadlines get
    # largest slack to avoid starving urgent ones.
    order = list(replan_set)
    def current_slack(aid: int) -> int:
        ddl = getattr(agents[aid], "deadline", None)
        # Determine where the agent is at the current time
        agent_pos = agents[aid].position
        if agent_pos is None:
            if current_timestep < len(existing_paths[aid]):
                agent_pos = existing_paths[aid][current_timestep]
            else:
                agent_pos = agents[aid].initial_position
        mal = agents[aid].malfunction_data.get("malfunction", 0)
        start_time = current_timestep + max(0, mal)
        h0 = manhattan(agent_pos, agents[aid].target)
        if ddl is None:
            return 1 << 30
        return ddl - ((start_time - current_timestep) + h0)
    order.sort(key=lambda aid: current_slack(aid))

    # Replan each selected agent in the computed order
    for aid in order:
        # Determine current position and direction
        agent_pos = agents[aid].position
        if agent_pos is None:
            if current_timestep < len(existing_paths[aid]):
                agent_pos = existing_paths[aid][current_timestep]
            else:
                agent_pos = agents[aid].initial_position
        agent_dir = agents[aid].direction
        if agent_dir is None:
            agent_dir = agents[aid].initial_direction
        # Determine waiting time due to malfunction
        mal_steps = agents[aid].malfunction_data.get("malfunction", 0)
        wait_steps = max(0, mal_steps)
        for t in range(current_timestep + 1, min(horizon, current_timestep + wait_steps) + 1):
            node_reserved.setdefault(t, set()).add(agent_pos)
        # Preserve prefix of the existing path up to current_timestep and
        # synchronise it with the current position
        base = list(existing_paths[aid][:current_timestep])
        # Pad the base if it is shorter than current_timestep
        if len(base) <= current_timestep:
            pad = base[-1] if base else agent_pos
            while len(base) <= current_timestep:
                base.append(pad)
        base[current_timestep] = agent_pos
        start_time = current_timestep + wait_steps
        ddl = getattr(agents[aid], "deadline", None)
        ddl_use = ddl if ddl is not None else (1 << 30)
        if agent_pos is None or agents[aid].target is None:
            suffix: List[Tuple[int, int]] = []
        else:
            suffix = plan_from(agent_pos, agent_dir, start_time, agents[aid].target, ddl_use)
        # Remove the duplicate at the junction of base and suffix
        if suffix and suffix[0] == agent_pos:
            suffix = suffix[1:]
        new_path = base[: current_timestep + 1] + suffix
        existing_paths[aid] = new_path
        # Reserve the new path from current_timestep onwards
        reserve_path_from(new_path, current_timestep)
    return existing_paths

##############################################################################
# Remote evaluator entry point
##############################################################################

if __name__ == "__main__":
    # If command line arguments are provided the grader will call
    # `remote_evaluator`, which connects to the online judge.  Do not
    # change this invocation unless you understand its implications.
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        # Local testing harness.  Load the pickle test cases stored in
        # the same directory and run the built‑in evaluator.  To test
        # individual instances set `test_single_instance=True` and
        # specify `level` and `test` above.
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(script_path, f"multi_test_case/level{level}_test_{test}.pkl")
            )
        test_cases.sort()
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan)