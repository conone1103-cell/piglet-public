import sys
import pickle

def main(path: str):
    obj = None
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print('Loaded:', type(obj))
    # Try dict-like payload
    if isinstance(obj, dict):
        print('Keys:', list(obj.keys()))
        env = obj.get('rail_env') or obj.get('env') or obj.get('rail') or obj.get('state')
    else:
        env = getattr(obj, 'rail_env', None)
        if env is None:
            env = getattr(obj, 'env', None)
    print('Env type:', type(env))
    # Try to reach env.rail
    rail = None
    if env is not None:
        rail = getattr(env, 'rail', None)
        print('Rail type:', type(rail))
    else:
        # maybe obj is directly the rail
        rail = getattr(obj, 'rail', None)
    if rail is None:
        print('No rail found in object')
        return
    H = getattr(rail, 'height', None)
    W = getattr(rail, 'width', None)
    grid = getattr(rail, 'grid', None)
    print('height:', H, 'width:', W)
    if grid is not None:
        try:
            print('grid shape:', grid.shape)
        except Exception:
            pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/inspect_pickle.py <path.pkl>')
        sys.exit(2)
    main(sys.argv[1])

