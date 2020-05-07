include("facility_location.jl")

# read in the text file
input_data = read("data/problem.txt", String)
result = solve_facility_location(input_data)
write("data/result.txt", result)