import sys
import argparse
from pfspfunctions import *
    
parser = argparse.ArgumentParser(description="PSFP Iterative Improvement Algorithm")
parser.add_argument('filename', type=str, help="instance file name")
group_solution = parser.add_mutually_exclusive_group(required=False)
group_solution.add_argument('-y', "--yes", action='store_true', help="Returns solution as a vector instead of the sum")
group_initialization = parser.add_mutually_exclusive_group(required=True)
group_initialization.add_argument('-r', "--random-init", action='store_true', help="Random initial solution")
group_initialization.add_argument('-s', "--srz", action='store_true', help="Simplified RZ heuristic initial solution")
group_neighborhood = parser.add_mutually_exclusive_group(required=True)
group_neighborhood.add_argument('-t', "--transpose", action='store_true', help="Transpose Neighborhood")
group_neighborhood.add_argument('-e', "--exchange", action='store_true', help="Exchange Neighborhood")
group_neighborhood.add_argument('-i', "--insert", action='store_true', help="Insert Neighborhood")
group_improvement = parser.add_mutually_exclusive_group(required=True)
group_improvement.add_argument('-b', "--best", action='store_true', help="Best improvement")
group_improvement.add_argument('-f', "--first", action='store_true', help="First improvement")
args = parser.parse_args()

if args.transpose:
    nrule = 'transpose'
elif args.exchange:
    nrule = 'exchange'
elif args.insert:
    nrule = 'insert'
    
if args.random_init:
    irule = 'ri'
elif args.srz:
    irule = 'srz'

mat, weights, summ = initialSolution(args.filename,irule)

if args.best:
    while True:
        mat2, weights2, summ2 = calculateBestNeighbour(mat,weights,nrule)
        if summ2 == summ:
            break
        summ = summ2
        weights = weights2
        mat = mat2

elif args.first:
    while True:
        mat2, weights2, summ2 = calculateFirstNeighbour(mat,weights,nrule)
        if summ2 == summ:
            break
        summ = summ2
        weights = weights2
        mat = mat2
else:
    print('Pivoting rule not recognized. Please choose between ''first'' and ''best''.')
    exit()
    

if args.yes:
    print((weights2[:,0].astype(int)))
else:
    print(summ2)