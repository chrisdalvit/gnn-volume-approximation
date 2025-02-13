import argparse
import json
from scipy.spatial import ConvexHull
import numpy as np

parser = argparse.ArgumentParser(
    prog='Dataset Generator',
    description='Script that generates the dataset',
    epilog=''
)
parser.add_argument("--size", type=int) # Size of the dataset
parser.add_argument("--npoints", type=int) # Number of points
parser.add_argument("--dim", type=int, default=2) # Dimension of the points
parser.add_argument("--val_max", type=int, default=100) # Maximum value of the points
parser.add_argument("--output", type=str, default="dataset.json")

def main():
    args = parser.parse_args()
    points = np.random.uniform(low=0.0, high=args.val_max, size=(args.size, args.npoints, args.dim))
    data = []
    for p in points:
        entry = dict()
        entry['points'] = p.tolist()
        hull = ConvexHull(p)
        entry['vertices'] = hull.vertices.tolist()
        entry['volume'] = hull.volume
        entry['area'] = hull.area
        data.append(entry)
        
    with open(args.output, 'w') as f:
        f.write(json.dumps(data))
    
if __name__ == "__main__":
    main()