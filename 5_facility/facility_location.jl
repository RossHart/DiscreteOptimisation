"""
Split the facility location problem into clusters and send
the results to Cbc 'one at a time'.
"""

# base jl libraries
using Dates
using Printf

# optimisation/ML libraries
using ScikitLearn
using Cbc
using GLPK
using JuMP

using ScikitLearn: fit!, predict
using JuMP: optimize!
@sk_import cluster: KMeans

function read_input(input_data)
    """
    Read in the input data from string to 
    numeric values.
    
    Parameters
    ----------
    input_data: string input data.
    
    Returns
    -------
    N: number of warehouses
    
    M: number of customers
    
    warehouses: array of setup cost (s_i), capacity (cap_i),
    and location co-ordinates (x_i, y_i) for each warehouse.
    
    customers: array of demand (d_i) and location (x_i, y_i)
    for each customer
    """
    # read in the string to separate lines
    lines = split(input_data, "\n")
    NM = split(lines[1], " ")
    N = parse(Int64, NM[1])
    M = parse(Int64, NM[2])
    
    # create empty array for warehouses + customers
    warehouses = Array{Float64}(undef, 0, 4)
    customers = Array{Float64}(undef, 0, 3)

    # add warehouse details to array
    for line in lines[2:N+1]
        #s_i, cap_i, x_i, y_i = split(line, " ")
        vars = split(line, " ")
        s_i = parse(Float64, vars[1])
        cap_i = parse(Float64, vars[2])
        x_i = parse(Float64, vars[3])
        y_i = parse(Float64, vars[4])
        warehouses = vcat(warehouses, transpose([s_i, cap_i, x_i, y_i]))
    end
    # add column of w numbers
    warehouses = hcat(1:N, warehouses)

    # add customer details to array
    for line in lines[N+2:N+M+1]
        vars = split(line, " ")
        d_i = parse(Float64, vars[1])
        x_i = parse(Float64, vars[2])
        y_i = parse(Float64, vars[3])
        customers = vcat(customers, transpose([d_i, x_i, y_i]))
    end
    
    # add column of j numbers
    customers = hcat(1:M, customers)

    # ensure we have the correct length
    @assert size(warehouses)[1] == N
    @assert size(customers)[1] == M
    
    return N, M, warehouses, customers
end


function birch_cluster(N, M, warehouses, customers; cluster_size = 100, verbose = false)

    if N <= cluster_size
        warehouse_clusters = zeros(Int, N)
        customer_clusters = zeros(Int, M)
        n_clusters = 1
    else
        n_clusters = trunc(Int, N / cluster_size) + 1
        clustering_model = KMeans(n_clusters = n_clusters)
        fit!(clustering_model, warehouses[:, 3:4])
        warehouse_clusters = predict(clustering_model, warehouses[:, 4:5])
        customer_clusters = predict(clustering_model, customers[:, 3:4])
    end
    
    if verbose == true
        println("Divided warehouses and customers into $(n_clusters) clusters.")
    end
    
    @assert minimum(warehouse_clusters) >= 0
    @assert minimum(customer_clusters) >= 0
    N_clusters = maximum(warehouse_clusters) + 1
    
    return warehouse_clusters, customer_clusters, N_clusters
end


function check_clustering(warehouses, customers, 
        warehouse_clusters, customer_clusters, N_clusters; excess = .25)
    """
    Check that the clustering will allow for a feasible result.
    Specify a necessary excess to make sure we can safely get a 
    result.
    """
    for cluster in 0:N_clusters-1
        is_cluster_warehouses = warehouse_clusters .== cluster
        is_cluster_customers = customer_clusters .== cluster
        N_cluster = sum(is_cluster_warehouses)
        M_cluster = sum(is_cluster_customers)

        cluster_capacity = sum(warehouses[is_cluster_warehouses, 3])
        cluster_demand = sum(customers[is_cluster_customers, 2])

        if cluster_capacity < (1+excess) * cluster_demand
            return false
        end
    end
    return true
end


function cluster(N, M, warehouses, customers, start_size = 100; verbose = false)
    
    pass = false
    cluster_size = start_size
    
    warehouse_clusters = nothing
    customer_clusters = nothing
    N_clusters = nothing
    
    # keep increasing the cluster size until the demand is less than
    # the capacity for each
    
    while pass == false
        if verbose == true
            println("Trying clusters of size $(cluster_size)...")
        end
        
        warehouse_clusters, customer_clusters, N_clusters = birch_cluster(
            N, M, warehouses, customers, 
            verbose = false, cluster_size = cluster_size)

        pass = check_clustering(warehouses, customers, 
                        warehouse_clusters, customer_clusters, N_clusters)

        cluster_size += 10
    end
    
    if verbose == true
        println("Divided problem into $(N_clusters) clusters")
    end
    
    return warehouse_clusters, customer_clusters, N_clusters
end


function initialise_MIP_model(N, M, warehouses, customers)
    """
    Setup the base LP model for the problem.
    
    Parameters
    ----------
    N: number of warehouses in the problem.
    
    M: number of customers in the problem.
    
    customers: customer statistics (locations and distances
    to warehouses).
    
    warehouses: warehouse statistics (locations and setup
    costs).
    
    Returns
    -------
    lp_model: LP model for solving the facility location
    problem. Contains base constraints for all warehouses
    and customers, where each customer is served by a single
    warehouse, and no warehouse can exceed its capacity.
    """
    # set up the initial MIP model
    mip_model = Model(
        optimizer_with_attributes(
            Cbc.Optimizer, "ratioGap" => 0.01, "seconds" => 300, "threads" => 3))
    set_optimizer_attribute(mip_model, MOI.Silent(), true)

    # set up all of our decision vars. Opening all the warehouses
    # should give us our first feasible solution
    x = @variable(mip_model, [w=1:N], base_name = "x", 
        binary = true)
    # use MOI library to set the start value
    for x_ in x
        MOI.set(mip_model, MOI.VariablePrimalStart(), x_, 1)
    end
    y = @variable(mip_model, [w = 1:N, c = 1:M], base_name = "y", 
        binary = true)

    # number of customers at each x, y location
    n_customers = customers[:, 2]

    # ensure that:
    # 1) a customer can only be served by an open warehouse (y_w,c <= x_w)
    # 2) a warehouse cannot exceed its capacity (sum(y_w,c over all C) <= cap_w for w in W)
    for w in 1:N
        # customer only served if warehouse is open
        @constraint(mip_model, y[w, :] .<= x[w], 
            base_name = "warehouse_open_constraint_$w")

        # capacity constraint
        cap_w = warehouses[w, 3]
        @constraint(mip_model, sum(n_customers .* y[w, :]) .<= cap_w, 
            base_name = "warehouse_capacity_constraint_$w")
    end

    # ensure that each customer is fully served by a warehouse
    # (sum(y_w,c = 1 over all W) for c in C)
    for c in 1:M
        # ensure each customer is actually served
        @constraint(mip_model, sum(y[:, c]) .== 1, 
                    base_name = "customer_served_constraint_$c")
    end

    setup_costs = warehouses[:, 2]
    distance_costs = transpose(customers[:, 5:end])
    @objective(mip_model, Min, sum(setup_costs .* x) + sum(distance_costs .* y))
    
    return mip_model, x, y

end


function get_euclidian_distances(N, M, warehouses, customers)
    """
    Add columns to customers array to show the distance
    between each customer and their specific warehouse.
    """
    for w in 1:N
        x_w, y_w = warehouses[w, 4:5]
        x_C = customers[:, 3]
        y_C = customers[:, 4]
        # calculate distance between warehouses
        d_Cw = ((x_C .- x_w) .^ 2 .+ (y_C .- y_w) .^ 2) .^ (1/2)
        customers = hcat(customers, d_Cw)
    end
    return warehouses, customers
end


function get_warehouses(y, warehouses)
    
    y_values = value.(y)
    warehouses_selected = findall(x -> x > 0.5, y_values)
    warehouses_selected = [w[1] for w in warehouses_selected]
    actual_warehouses_selected = zeros(length(warehouses_selected))

    for (i, w) in enumerate(warehouses[:, 1])
        actual_warehouses_selected[warehouses_selected .== i] .= w
    end
    
    return actual_warehouses_selected
end


function solve_MIP(N, M, warehouses, customers)
    # calculate euclidian disctances for all clusters
    warehouses, customers = get_euclidian_distances(
        N, M, warehouses, customers)

    # initialise + solve the MIP model
    mip_model, x, y = initialise_MIP_model(
        N, M, warehouses, customers)
    optimize!(mip_model)
    obj_value = objective_value(mip_model)
    y_values = get_warehouses(y, warehouses)
    
    println(termination_status(mip_model))
    
    # check if solution solved to optimality
    if termination_status(mip_model) == MOI.OPTIMAL
        optimal = 1
    else
        optimal = 0
    end
    
    return obj_value, y_values, optimal
end

function format_result(obj_value, optimal, facilities_chosen)
    
    result = @sprintf("%.1f", obj_value)
    result *= " $(optimal) \n"
    
    for facility in facilities_chosen
        facility = trunc(Int, facility)
        result *= "$(facility) "
    end
    
    result = result[1:end-1]
    return result
end


function solve_facility_location(input_data; verbose = true)
    """
    Divide the space into small clusters, and solve. 
    """
    
    N, M, warehouses, customers = read_input(input_data)

    # cluster data and calculate time to spend on each cluster
    warehouse_clusters, customer_clusters, N_clusters = cluster(
        N, M, warehouses, customers, verbose = verbose)

    # set up "results store"
    obj_value = 0
    facilities_chosen = zeros(M)

    if N_clusters == 1
        optimal = 1
    else
        optimal = 0
    end

    for cluster_ in 0:N_clusters-1
        is_cluster_warehouses = warehouse_clusters .== cluster_
        is_cluster_customers = customer_clusters .== cluster_
        N_cluster = sum(is_cluster_warehouses)
        M_cluster = sum(is_cluster_customers)
        
        if verbose == true
            println("Solving cluster = $(cluster_) (N = $N_cluster, M = $M_cluster)")
        end
        
        warehouses_cluster = warehouses[is_cluster_warehouses, :]
        customers_cluster = customers[is_cluster_customers, :]

        obj_cluster, y_cluster, opt_cluster = solve_MIP(
            N_cluster, M_cluster, warehouses_cluster, customers_cluster)
        
        optimal = minimum([optimal, opt_cluster])
        obj_value += obj_cluster
        facilities_chosen[is_cluster_customers] = y_cluster .- 1

        if verbose == true
            println("Solved MIP for $(cluster_ + 1)/$(N_clusters) clusters.")
        end
    end 

    result = format_result(obj_value, optimal, facilities_chosen)
    return result
end