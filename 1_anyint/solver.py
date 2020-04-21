#!/usr/bin/python
# -*- coding: utf-8 -*-
from julia import Julia

def solve_it(input_data):
    """
    Run appropriate jl script to generate an integer.
    """
    jl = Julia()
    jl.include("any_integer.jl")
    result = jl.eval(f'generate_any_integer("{input_data}")')
    return result
    

if __name__ == '__main__':
    print('This script submits the integer: %s\n' % solve_it(''))

