import pickle, sys, os
from collections import Counter

try:
    from flatland.utils.controller import Directions
except Exception:
    print("Flatland not available in this interpreter.")
    sys.exit(1)

def analyze(path):
    env = pickle.load(open(path, 'rb'))
    rail = env.rail
    H, W = getattr(rail, 'height', 0), getattr(rail, 'width', 0)
    def deg(x,y):
        nbr=set()
        for d in [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]:
            vt = rail.get_transitions(x,y,d)
            for a,v in enumerate(vt):
                if v:
                    nx,ny=x,y
                    if a==Directions.NORTH: nx-=1
                    elif a==Directions.EAST: ny+=1
                    elif a==Directions.SOUTH: nx+=1
                    elif a==Directions.WEST: ny-=1
                    if 0<=nx<H and 0<=ny<W:
                        nbr.add((nx,ny))
        return len(nbr)
    cnt=Counter()
    for x in range(H):
        for y in range(W):
            if rail.grid[x,y] != 0:
                cnt[deg(x,y)] += 1
    n_cells = sum(cnt.values()) or 1
    print(os.path.basename(path), 'agents', len(env.agents), 'size', H, W)
    print('deg dist:', {k: round(v/n_cells,3) for k,v in sorted(cnt.items())})

if __name__=='__main__':
    for p in sys.argv[1:]:
        analyze(p)
