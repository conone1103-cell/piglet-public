import os
import sys
import glob
import argparse


def ensure_flatland_importable():
    try:
        import flatland  # noqa: F401
        return
    except Exception:
        # Fallback to local repository copy
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        local_flatland_root = os.path.join(repo_root, 'flatland')
        if os.path.isdir(local_flatland_root) and local_flatland_root not in sys.path:
            sys.path.insert(0, local_flatland_root)


def inspect_case(pkl_path: str):
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import rail_from_file
    from flatland.envs.schedule_generators import schedule_from_file
    from flatland.utils.controller import malfunction_from_file

    ddl_path = pkl_path.replace('.pkl', '.ddl')
    env = RailEnv(
        width=1,
        height=1,
        rail_generator=rail_from_file(pkl_path),
        schedule_generator=schedule_from_file(pkl_path),
        malfunction_generator_and_process_data=malfunction_from_file(pkl_path),
        remove_agents_at_target=True,
    )
    env.reset()

    deadlines = env.read_deadlines(ddl_path) if os.path.exists(ddl_path) else []
    mp = env.malfunction_process_data

    summary = {
        'case': os.path.basename(pkl_path),
        'agents': len(env.agents),
        'size': f"{env.width}x{env.height}",
        'malfunc_rate': getattr(mp, 'malfunction_rate', None),
        'malfunc_dur': (getattr(mp, 'min_duration', None), getattr(mp, 'max_duration', None)),
        'ddl_min': min(deadlines) if deadlines else None,
        'ddl_max': max(deadlines) if deadlines else None,
    }
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, help='Path to a single .pkl case')
    parser.add_argument('--all', action='store_true', help='Inspect all multi_test_case/*.pkl')
    args = parser.parse_args()

    ensure_flatland_importable()

    if args.case:
        inspect_case(args.case)
        return

    if args.all:
        cases = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'multi_test_case', 'level*_test_*.pkl')))
        for p in cases:
            try:
                inspect_case(p)
            except Exception as e:
                print({'case': os.path.basename(p), 'error': str(e)})
        return

    print('Usage:')
    print('  python piglet-public/inspect_cases.py --case piglet-public/multi_test_case/level1_test_5.pkl')
    print('  python piglet-public/inspect_cases.py --all')


if __name__ == '__main__':
    main()


