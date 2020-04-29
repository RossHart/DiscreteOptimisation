include("travelling_salesman.jl")

# read in the text file
input_data = read("data/problem.txt", String)
result, D_t = solve_tsp_metropolis(input_data)
println(result)
write("data/result.txt", result)