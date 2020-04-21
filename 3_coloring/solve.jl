include("graph_colouring.jl")

# read in the text file
input_data = read("data/problem.txt", String)
result = colour_graph(input_data, timeout=60)
write("data/result.txt", result)