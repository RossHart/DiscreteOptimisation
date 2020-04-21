#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from julia import Julia


def solve_it(input_data):
    """
    Run appropriate jl script to solve the
    knapsack problem in Julia.
    """
    jl = Julia()
    jl.include("knapsack.jl")
    output_data = jl.eval(f'optimise_knapsack("{input_data}", timeout=60)')
    
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

