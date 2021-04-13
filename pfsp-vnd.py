import sys
import argparse
from pfspfunctions import *
    
parser = argparse.ArgumentParser(description="PSFP Iterative Improvement Algorithm - Variable Neighbourhood Descent")
parser.add_argument('filename', type=str, help="instance file name")
group_solution = parser.add_mutually_exclusive_group(required=False)
group_solution.add_argument('-y', "--yes", action='store_true', help="Returns solution as a vector instead of the sum")
group_method = parser.add_mutually_exclusive_group(required=True)
group_method.add_argument('-i', "--tie", action='store_true', help="Transpose, Insert, Exchange")
group_method.add_argument('-e', "--tei", action='store_true', help="Transpose, Exchange, Insert")
args = parser.parse_args()

if args.tei:
    method = 'tei'
elif args.tie:
    method = 'tie'
    

if method == 'tie':
    method_list = ['transpose','insert','exchange']
else:
    method_list = ['transpose','exchange','insert']

m,w,s = initialSolution(args.filename,'srz')
i = 0
while i < 3:
    m,w,summ = calculateFirstNeighbour(m,w,method_list[i])
    if summ == s:
        i += 1
    s = summ    

if args.yes:
    print((w[:,0].astype(int)))
else:
    print(s)
