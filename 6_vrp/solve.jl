include("vehicle_routing.jl")

# read in the text file
input_data = read("data/problem.txt", String)
result = optimise_vrp(input_data, timeout=60)
write("data/result.txt", result)