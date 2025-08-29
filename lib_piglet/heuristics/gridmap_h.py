# heuristics/gridmap_h.py
#
# Heuristics for gridmap.
#
# @author: mike
# @created: 2020-07-22
#

import math

def piglet_heuristic(domain, current_state, goal_state):
    # 2단계: A*가 Octile Heuristic을 사용하도록 변경합니다.
    return octile_heuristic(current_state, goal_state)

def piglet_multi_agent_heuristic(domain, current_state, goal_state):
    h = 0
    for agent, loc in current_state.agent_locations_.items():
        h += manhattan_heuristic(loc, goal_state.agent_locations_[agent])
    return h

def manhattan_heuristic(current_state, goal_state):
    """
    Calculates the Manhattan distance (D1).
    Suitable for 4-connected grids.
    """
    dx = abs(current_state[0] - goal_state[0])
    dy = abs(current_state[1] - goal_state[1])
    return dx + dy

def straight_heuristic(current_state, goal_state):
    """
    Calculates the Straight-line (Euclidean) distance.
    Admissible for any grid type.
    """
    dx = abs(current_state[0] - goal_state[0])
    dy = abs(current_state[1] - goal_state[1])
    return math.sqrt(dx**2 + dy**2)

def octile_heuristic(current_state, goal_state):
    """
    Calculates the Octile distance.
    A more accurate heuristic for 8-connected grids.
    """
    dx = abs(current_state[0] - goal_state[0])
    dy = abs(current_state[1] - goal_state[1])
    
    # Cost for cardinal moves (D1) and diagonal moves (D2)
    D1 = 1
    D2 = math.sqrt(2) # or 1.41
    
    # The heuristic formula
    return D2 * min(dx, dy) + D1 * (max(dx, dy) - min(dx, dy))

def differential_heuristic(domain, current_state, goal_state):
    # This heuristic is not required for this part of the assignment.
    return NotImplementedError