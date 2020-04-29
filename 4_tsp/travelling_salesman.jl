using Printf
using Random
using Dates

function read_input_data(input_data)
    """
    Read in the set of cities from 
    an input string.
    
    Parameters
    ----------
    input_data: problem data for travelling
    salesman.
    
    Returns
    -------
    N: number of locations in the problem.
    
    xy: x, y co-ordinate of each location.
    """

    input_lines = split(input_data, "\n")
    N = parse(Int64, input_lines[1])
    xy = []

    for line in input_lines[2:end-1]
        x_i, y_i = split(line, " ")
        x_i = parse(Float64, x_i)
        y_i = parse(Float64, y_i)
        push!(xy, [x_i, y_i])
    end
    return N, xy
end


mutable struct Location
    number::Int64
    x::Float64
    y::Float64
end

    
function create_nodes(xy)
        
    nodes = []
    
    for (i, xy_i) in enumerate(xy)
        x, y = xy_i
        xy_location = Location(i, x, y)
        push!(nodes, xy_location)
    end
    
    return nodes
end


function calculate_distance(node_i, node_j)
    """
    Calculate the distance between two nodes
    i and j.
    """
    D = ((node_i.x - node_j.x) ^ 2
         + (node_i.y - node_j.y) ^ 2) ^ (1/2)
    return D
end


function calculate_total_distance(nodes, N)
    
    D = 0
    
    for i in (1:N-1)
        D += calculate_distance(nodes[i], nodes[i+1])
    end
    
    # add the final trip
    D += calculate_distance(nodes[N], nodes[1])
end


function print_path(nodes, N)
    path = ""
    for i in 1:N
        path *= "$(nodes[i].number)-"
    end
    path *= "$(nodes[1].number)"
end


function format_result(nodes, N)
    
    D = calculate_total_distance(nodes, N)
    result = @sprintf("%.1f", D)
    result *= " 0 \n"
    
    for i in 1:N
        result *= "$(nodes[i].number - 1) "
    end
    
    return result
end


function two_opt(nodes, i, j)
    """
    Perform a two-opt swap of the list of nodes.
    """
    @assert j > i
    nodes_2 = vcat(nodes[1:i-1], reverse(nodes[i:j]), nodes[j+1:end])
    @assert length(nodes_2) == length(nodes)
    return nodes_2
end


function select_i_and_j(N)
    """
    Function to select which value of i to use. 
    The function can be tuned to preferentially
    select neighbours with p_neighbour.
    """
    i = rand(1:N-1)
    j = rand(i+1:N)
    return i, j
end


function take_move_metropolis(nodes, nodes_mod, N, temp)
    
    # calculate how good the move is
    D = calculate_total_distance(nodes, N)
    D_mod = calculate_total_distance(nodes_mod, N)

    # calculate whether the move improves 
    improvement = D - D_mod#(D - D_mod) / D
    probability = exp(improvement / temp)
    take_move = rand() < probability
    
    if take_move == true
        return nodes_mod
    else
        return nodes
    end
end


function fill_greedy(nodes, N; i = nothing, verbose = true)
    
    if i == nothing
        i = rand(1:N)
    end
    if verbose == true
        println("Creating route greedily starting with $i...")
    end
    node_i = nodes[i]
    nodes_greedy = [node_i]
    nodes = [node for node in nodes if node.number != node_i.number]

    for n in 1:N-1
        D_ij = [calculate_distance(node_i, node_j)  for node_j in nodes]
        node_j = nodes[argmin(D_ij)]
        push!(nodes_greedy, node_j)
        nodes = [node for node in nodes if node.number != node_j.number]
        node_i = node_j
        if verbose == true                        
            if (n % 10000 == 0)
                println("Created path for $n nodes.")
            end
        end
    end
    D = calculate_total_distance(nodes_greedy, N)
    if verbose == true
        println("Greedy route completed. D = ", @sprintf("%.1f", D))
    end
    return nodes_greedy
end
                            
                            
function plot_improvement(D_values)
    
    n = length(D_values)
    x = 1:n
    y = D_values
    plot(x, y, xlabel = "number of iterations", ylabel = "shortest path distance")
end


function solve_tsp_metropolis(input_data; N_swaps = 100000, starting_temp = nothing, 
        temp_gradient = 1e-3, timeout = 180, n_restarts_min = 1, n_restarts_max = 1000, 
        verbose = false)
    
    """
    Use a metropolis algorithm to solve the TSP. 
    
    Parameters
    ----------
    N_swaps: number of swaps to consider.
    
    starting_temp: starting temperature for the metropolis
    algorithm.
    
    temp_gradient: how much to decrease the temperature by
    at each time step.
    
    timeout: maximum time to allow the algorithm to run for.
                                
    n_starts_min: minimum number of starts we need to have 
    completed (overrides timeout setting).
    
    n_starts_max: maximum number of restarts to perform.
    """
    
    Random.seed!(1)

    # read in the problem data
    N, xy = read_input_data(input_data)
    nodes = create_nodes(xy)
    # use a greedy algo. to start
    nodes = fill_greedy(nodes, N, i = 1, verbose = true)
                                
    # make algo. more greedy if data is larger if specified
    # (these temps are worked out by hand effectively!)
    if starting_temp == nothing
        # use the mean edge length as the startign temperature
        starting_temp = calculate_total_distance(nodes, N) / N
        # make algo. more greedy if large (to stop premature end
        # when we time out).
        if N > 100
            starting_temp /= 10
        elseif N >= 1000
            starting_temp /= 100
        end
    end
    
    # store the greedy result to start
    D_best = calculate_total_distance(nodes, N)
    D_values = [D_best]
    nodes_best = deepcopy(nodes)

    # set up some counters so we can keep track
    n_starts_completed = 0
    t0 = Dates.now()
    t1 = Dates.now()
    timeout_ms = Dates.Millisecond(1000 * timeout)
    timed_out = false
    
    while (timed_out == false) & (n_starts_completed < n_restarts_max)
        
        # get the starting nodes and shuffle them
        nodes_start = fill_greedy(nodes, N, verbose = false)
        D_values_start = push!(D_values, calculate_total_distance(nodes_start, N))

        # set the starting temp as the user input
        temp = starting_temp

        # perform potential swaps N_swaps_start times
        for iter in 1:N_swaps

            # generate an edge at random
            i, j = select_i_and_j(N)

            # perform the swap
            nodes_start_mod = two_opt(nodes_start, i, j)
            nodes_start = take_move_metropolis(nodes_start, nodes_start_mod, N, temp)
            
            # decrease the temp and increase the neighbour probability
            #temp *= (1-temp_gradient)
            temp -= (starting_temp / N_swaps)
                                        
            # calculate the improvement after one iteration
            D_iter = calculate_total_distance(nodes_start, N)
            push!(D_values_start, D_iter)
                          
            # update the best result if it is better than what we have
            if D_iter < D_best
                println("Updating best result to ", @sprintf("%.1f", D_iter))
                D_best = D_iter
                nodes = nodes_start
                D_values = D_values_start
            end
        end             
        
        # update the status of the starts
        t1 = Dates.now()
        n_starts_completed += 1
        timed_out = (t1 - t0 >= timeout_ms) & (n_starts_completed >= n_restarts_min)             
                            
        if verbose == true
            println("Performed $(n_starts_completed) restarts.")
        end
        
    end
                                
    result = format_result(nodes, N)
    
    return result, D_best
end                