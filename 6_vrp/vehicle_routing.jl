using Dates
using Printf
using Random
using StatsBase


function read_input(input_data)

    lines = split(input_data, "\n")
    N, V, c = split(lines[1], " ") # no of locations, no of vehicles + capacity of each
    N = parse(Int64, N)
    V = parse(Int64, V)
    c = parse(Int64, c)

    dxy = Array{Float64, 2}(undef, 0, 3)

    for i in 1:N
        d_i, x_i, y_i = split(lines[i+1], " ")
        d_i = parse(Float64, d_i)
        x_i = parse(Float64, x_i)
        y_i = parse(Float64, y_i)
        dxy = vcat(dxy, transpose([d_i, x_i, y_i]))
    end  
    
    return N, V, c, dxy
end


function calculate_distance(dxy, i, j)
    x_i, y_i = dxy[i, 2:end]
    x_j, y_j = dxy[j, 2:end]
    D_ij = ((x_i - x_j ) ^ 2 + (y_i - y_j) ^ 2) ^ (1/2)
end


function calculate_clarke_wright_savings(dxy, N)
    # savings matrix of i, j & saving
    savings = Array{Int64, 2}(undef, 0, 3)

    # get a matrix of "savings"
    for i in 2:N
        for j in i+1:N
            D_i0 = calculate_distance(dxy, i, 1)
            D_j0 = calculate_distance(dxy, j, 1)
            D_ij = calculate_distance(dxy, i, j)
            saving = D_i0 + D_j0 - D_ij
            savings = vcat(savings, transpose([i, j, saving]))
        end
    end

    # sort the savings by size
    savings = savings[sortperm(savings[:, 3], rev=true), :]
    return savings
end  


function calculate_capacity_left(route, dxy, c)
    capacity_used = 0
    for i in route
        capacity_used += dxy[i, 1]
    end
    capacity_left = c - capacity_used
    return capacity_left
end        


function check_interior(route, i)
    
    r_i = findfirst(x -> x == i, route)
    route_length = length(route)
    is_first = r_i == 2
    is_last = r_i == route_length - 1
    is_interior = (is_first) | (is_last)
    return is_interior, is_first
end
    

function build_clarke_wright_routes(dxy, N, c)
    routes = Dict()
    i_visited = []

    savings = calculate_clarke_wright_savings(dxy, N)
    N_savings, _ = size(savings)

    routes = Dict()
    for n in 2:N
        routes[n] = [1, n, 1] # initialise a new route starting at the depot
    end

    for s in 1:N_savings

        i, j = savings[s, 1:2]
        n_i = nothing
        n_j = nothing

        # find the routes associated with i & j
        for (n, route) in routes
            if i in route
                n_i = n
            end

            if j in route
                n_j = n
            end
        end

        # check that we have two different interior routes
        routes_differ = !(n_i == n_j)
        is_interior_i, is_first_i = check_interior(routes[n_i], i)
        is_interior_j, is_first_j = check_interior(routes[n_j], j)

        # if we have two interior routes, merge them
        if (routes_differ) & (is_interior_i) & (is_interior_j)
            # reverse routes so that i is visited last and j first
            if is_first_i == true
                routes[n_i] = routes[n_i][end:-1:1]
            end
            if is_first_j == false
                routes[n_j] = routes[n_j][end:-1:1]
            end

            # suggest new route
            new_route = vcat(routes[n_i][1:end-1], routes[n_j][2:end])

            # check the capacity of the new route. Iff so, merge them.
            c_left = calculate_capacity_left(new_route, dxy, c)
            if c_left >= 0
                routes[n_i] = new_route
                delete!(routes, n_j)
            end 
        end   
    end

    return routes
end


function force_V_routes(routes, V, c, dxy)
    """
    Greedily ensure that we only have V routes for the optimisation.
    """
    n_routes = length(routes)
    final_routes = Dict()

    to_reassign = []
    routes_by_length = sort(collect(routes), by = x -> length(x[2]), rev=true)

    for v in 1:n_routes
        if v <= V
        # add the biggest route to the current list
            final_routes[v] = routes_by_length[v][2]
        else
        # otherwise add them to the 'reassignment list'
            to_reassign = vcat(to_reassign, routes_by_length[v][2][2:end-1])
        end
    end
    
    # initialise an empty set of routes for any excess vehicles
    if n_routes < V
        for v in n_routes+1:V
            final_routes[v] = [1, 1]
        end
    end
    
    # reassign any leftover nodes to wherever they can fit
    for i in to_reassign
        best_D = Inf
        best_v = 1
        best_j = 1
        
        for (v, route) in final_routes
            # look through all the routes and find a neighbour
            c_v = dxy[i, 1]
            c_left = calculate_capacity_left(route, dxy, c)

            # find the best route to insert our location into.
            # Must be a route that can take the location _and_
            # use find the closest node.
            if c_left >= c_v
                for j in route
                    D_ij = calculate_distance(dxy, i, j)
                    if D_ij < best_D
                        best_D = D_ij
                        best_v = v
                        best_j = j
                    end
                end
            end
        end
            
        # insert i into the best found location...
        r_j = findfirst(x -> x == best_j, final_routes[best_v])
        final_routes[best_v] = vcat(
        final_routes[best_v][1:r_j], [i], final_routes[best_v][r_j+1:end])
        
    end

    return final_routes         
end


function calculate_exceeded_capacity(routes, dxy, c)
    
    exceeded_capacity = 0
    for (v, route) in routes
        c_left = minimum([calculate_capacity_left(route, dxy, c), 0])
        exceeded_capacity -= c_left
    end
    return exceeded_capacity
end      


# quick tools to calculate overall distances
function calculate_distance_route(route, dxy)
    D = 0
    for (i, j) in zip(route[1:end-1], route[2:end])
        D_ij = calculate_distance(dxy, i, j)
        D += D_ij
    end
    return D
end


function calculate_total_distance(routes, dxy)
    D = 0
    for (v, route) in routes
        D_v = calculate_distance_route(route, dxy)
        D += D_v
    end
    return D
end
    

function format_result(routes, dxy)
    D = calculate_total_distance(routes, dxy)
    result = @sprintf("%.1f 0 \n", D)
    
    for (v, route) in routes
        for i in route
            result *= "$(i-1) "
        end
        result = result[1:end]
        result *= " \n"
    end
    
    result = result[1:end-3]
    return result
end 


function check_feasibility(routes, V, N, dxy, c)
    
    # check all of our routes do not exceed capacity limit
    @assert calculate_exceeded_capacity(routes, dxy, c) == 0
    @assert length(routes) == V
    
    # check all nodes are visited
    all_visited = []
    for (v, route) in routes
        route_start = route[1]
        route_end = route[end]
        @assert route_start == 1
        @assert route_end == 1
        # append all visited
        all_visited = vcat(all_visited, route[2:end-1])
    end
    
    for i in 2:N
        @assert sum(all_visited .== i) == 1
    end
        
end


# 3 sets of possible local moves:
function swap_customers(routes, N, V)

    new_routes = deepcopy(routes)
    
    v_i = rand(1:V-1)
    v_j = rand(v_i+1:V)
    
    # need to ensure we aren't using unitialised routes
    L_v_i = length(new_routes[v_i])
    L_v_j = length(new_routes[v_j])
    
    if (L_v_i > 2) & (L_v_j > 2)
    
        i = rand(new_routes[v_i][2:end-1])
        j = rand(new_routes[v_j][2:end-1])
        # make the simple swap
        new_routes[v_i][new_routes[v_i] .== i] .= j
        new_routes[v_j][new_routes[v_j] .== j] .= i
    end
    return new_routes
end
    

function move_customer(routes, dxy, N, V)
    
    new_routes = deepcopy(routes)
    v_i = rand(1:V-1)
    v_j = rand(v_i+1:V)
    
    # need to ensure we aren't using unitialised routes
    L_v_i = length(new_routes[v_i])
    L_v_j = length(new_routes[v_j])
    
    if L_v_i > 2
        i = rand(new_routes[v_i][2:end-1])
        #println("Moving customer $i from $(new_routes[v_i]) to $(new_routes[v_j])...")
        new_routes[v_i] = new_routes[v_i][new_routes[v_i] .!== i]
        
        # greedily put our customer into the other route (after their nearest neighbour)
        Ds = [calculate_distance(dxy, i, j) for j in new_routes[v_j][1:end-1]]
        insert_index = findfirst(x -> x .== minimum(Ds), Ds)
        splice!(new_routes[v_j], insert_index+1:insert_index, i)
    end
    
    return new_routes
end
        

function swap_nodes_in_route(routes, V)

    new_routes = deepcopy(routes)
    
    v_ij = rand(1:V)
    s_v = length(new_routes[v_ij])
    if s_v > 4
        # select part of the route to swap
        r_i = rand(2:s_v-2)
        r_j = rand(r_i+1:s_v-1)
        
        #println("Performing TSP on $(new_routes[v_ij])")

        # make the simple swap
        new_routes[v_ij] = vcat(
            new_routes[v_ij][1:r_i-1], 
            reverse(new_routes[v_ij][r_i:r_j]),
            new_routes[v_ij][r_j+1:end])
    end
    return new_routes
end


function generate_local_move(routes, dxy, N, V; weights = [0.05, 0.1, 0.85])
    choices = ["swap", "move", "tsp"]
    weights = Weights(weights)
    local_move = sample(choices, weights)

    if local_move == "swap"
        new_routes = swap_customers(routes, N, V)
    elseif local_move == "move"
        new_routes = move_customer(routes, dxy, N, V)
    else
        new_routes = swap_nodes_in_route(routes, V)
    end
    
    return new_routes
end


function take_move_metropolis(routes, dxy, N, V, c, T; 
        weights = [0.05, 0.15, 0.8], objective = "distance")
    new_routes = generate_local_move(routes, dxy, N, V; weights = weights)
    
    # calculate the improvement (if any)
    if objective == "distance"
        D = calculate_total_distance(routes, dxy)
        D_new = calculate_total_distance(new_routes, dxy)
    else
        # when packing, use where we exceed capacity as the measure
        D = calculate_exceeded_capacity(routes, dxy, c)
        D_new = calculate_exceeded_capacity(new_routes, dxy, c)
    end
    
    p_move = minimum([exp(-(D_new - D) / T), 1])
    
    # calculate whether the new move is valid
    for (v, route) in new_routes
        # if any route exceeds capacity, discard this move (unless we are trying to optimise capacity)
        capacity_exceeded = calculate_exceeded_capacity(new_routes, dxy, c)
        if (capacity_exceeded > 0) & (objective == "distance")
            p_move = 0.0 # if we have no cap. left, don't make the move
        end
    end 
    
    take_move = rand() < p_move
    if take_move == true
        return new_routes
    else
        return routes
    end
end


function perform_simulated_annealing(routes, N, V, dxy, c, T0; N_iter = 1000, objective = "distance")
    
    # set the start temp
    T = deepcopy(T0)
    
    best_routes = deepcopy(routes)
    
    if objective == "distance"
        # calculate the starting distance
        D_best = calculate_total_distance(routes, dxy)
        
        for iter in 1:N_iter
            routes = take_move_metropolis(routes, dxy, N, V, c, T,
                objective = objective) # move with metropolis probability
            T -= T0 / N_iter # reduce T -> 0 over time
            D = calculate_total_distance(routes, dxy)
            if D < D_best
                best_routes = deepcopy(routes)
            end
        end
    # Algo. to make local swaps until we reach a valid solution
    else
        # calculate the starting capacity breach
        D_best = calculate_exceeded_capacity(routes, dxy, c)
        # keep cycling until we find a valid solution
        while D_best > 0
            routes = take_move_metropolis(routes, dxy, N, V, c, T; 
                weights=[0.5, 0.5, 0], objective = objective)
            D = calculate_exceeded_capacity(routes, dxy, c)
            T *= 0.99 # reduce T -> 0 over time
            if D < D_best
                best_routes = deepcopy(routes)
                D_best = D
                println("Capacity excess = $D")
            end
        end
    end
    
    return best_routes, D_best
end         
        

function optimise_vrp(input_data; N_iter = 5000, N_restarts = 100, N_restarts_min = 5, timeout = 60)
    Random.seed!(0)
    N, V, c, dxy = read_input(input_data)

    # find a set of feasible starting routes using the CW heuristic
    println("Building routes using CW heuristic...")
    routes = build_clarke_wright_routes(dxy, N, c)
    routes = force_V_routes(routes, V, c, dxy)
    
    # check the feasibility of the routes. Perform local search until 
    # we reach a feasible solution
    c_exceeded = calculate_exceeded_capacity(routes, dxy, c)
    if c_exceeded > 0
        println("Capacity exceeded by $(c_exceeded). Making local swaps...")
        routes, D = perform_simulated_annealing(routes, N, V, dxy, c, 1,
            objective = "capacity")
    end

    # run a quick feasibility check on the solution at this stage
    D_best = calculate_total_distance(routes, dxy)
    println("Built CW heuristic with D = " * @sprintf("%.1f", D_best) * ".")
    println("Performing local search to find a better solution...")
    
    # use ~mean travel distance as the starting temp
    T0 = D_best / (N + (2*V)) * 2
    
    # setup a timer
    t0 = Dates.now()
    t1 = Dates.now()
    time_limit = Dates.Millisecond(timeout * 1000)
    timed_out = false
    n = 0
    
    while (n < N_restarts) & (timed_out == false)
        n += 1
        routes, D = perform_simulated_annealing(routes, N, V, dxy, c, T0, 
                N_iter = N_iter, objective = "distance")
        if D < D_best
            println("Found solution with D = " * @sprintf("%.1f", D_best) * ".")
            D_best = D
        end
        
        t1 = Dates.now()
        # update "timeout" status if we have enough iterations
        if (t1 - t0 > time_limit) & (n > N_restarts_min)
            timed_out = true
        end
            
    end
    
    # return the result in string format
    # run feasibility check on result and return
    check_feasibility(routes, V, N, dxy, c)
    result = format_result(routes, dxy)
    return result
    
end