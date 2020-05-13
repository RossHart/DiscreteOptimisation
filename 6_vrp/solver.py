#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import subprocess

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def save_data_to_file(input_data):
    with open("data/problem.txt", "w", newline="\n") as text_file:
        text_file.write(input_data)


def solve_it(input_data):
    """
    Call the appropriate set of Julia functions to solve 
    the graph colouring problem.
    """
    # save to a file for Julia to read
    save_data_to_file(input_data)
    # solve using Julia
    print("Data loaded. Sent to julia to solve...")
    subprocess.call("julia solve.jl", shell = True)
    # read output file
    with open("data/result.txt", "r") as myfile:
          output_data = myfile.read()
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

