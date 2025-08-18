import pickle
import glob
import os

def analyze_testcases():
    """테스트 케이스들을 분석하여 맵 구조와 난이도를 파악"""
    test_cases = glob.glob('multi_test_case/level*_test_*.pkl')
    test_cases.sort()
    
    print("=== Test Case Analysis ===")
    print("File\t\t\t| Grid Size | Agents | Avg Dist | Max Dist | Density")
    print("-" * 80)
    
    for test_file in test_cases:
        try:
            with open(test_file, 'rb') as f:
                data = pickle.load(f)
            
            # 환경 정보 추출
            env = data['env']
            rail = env.rail
            agents = env.agents
            
            # 에이전트들의 거리 분석
            distances = []
            for agent in agents:
                dist = abs(agent.initial_position[0] - agent.target[0]) + abs(agent.initial_position[1] - agent.target[1])
                distances.append(dist)
            
            avg_dist = sum(distances) / len(distances) if distances else 0
            max_dist = max(distances) if distances else 0
            
            # 밀도 계산 (에이전트 수 / 맵 크기)
            density = len(agents) / (rail.height * rail.width)
            
            print(f"{test_file:<25} | {rail.height:2d}x{rail.width:<2d}   | {len(agents):6d} | {avg_dist:8.1f} | {max_dist:8d} | {density:.4f}")
            
        except Exception as e:
            print(f"{test_file:<25} | Error: {e}")

if __name__ == "__main__":
    analyze_testcases()
