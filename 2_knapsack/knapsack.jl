using Dates

mutable struct KnapsackNode
    """
    A simple node structure to store all of the choices
    made by the knapsack so far.
    """
    choices # items that have/have not been picked
    serial_number # string format list of choices for the node
    weight_left # room left in the knapsack
    value # linearly-relaxed est. of value of the node
end


function read_knapsack_file(input_data)
    """
    Read in a knapsack file from disk.
    """
    file_strings = split(input_data, "\n")
    n, K = split(file_strings[1], " ")
    n = parse(Int64, n)
    K = parse(Int64, K)

    # array of weights, values and densities
    vs_and_ws = zeros(n, 3)

    for (i, string_) in enumerate(file_strings[2:2+n-1])
        v, w = split(string_, " ")
        v = parse(Int64, v)
        w = parse(Int64, w)
        vs_and_ws[i, 1] = v
        vs_and_ws[i, 2] = w
        vs_and_ws[i, 3] = v / w
    end
    # return knapsack parameters (number of items, weight limit 
    # + value/weight array)
    return n, K, vs_and_ws
end


function sort_by_density(vs_and_ws)
    """
    Sort the items by value density.
    """
    order = sortperm(vs_and_ws[:, 3], rev = true)
    vs_and_ws = vs_and_ws[order, :]
    return vs_and_ws, order
end


function initialise_starting_node(K::Int, vs_and_ws::Array)
    """
    Initialise the node which forms the base of the node tree.
    
    Parameters
    ----------
    K: weight capacity of the bag.
    
    items: array of items with their associated weights and values.
    
    Returns
    -------
    node_dict: structure to keep trac of B+B nodes. Has only the 
    first node in it after this function is called.
    """
    node = KnapsackNode([],
                        join(string.([])),
                        K,
                        0)
    
    # calculate linearly-relaxed weight of sack 
    update_node_value(node, vs_and_ws)
    
    # create a dictionary to store all of the node values
    node_dict = Dict{String, Any}()
    node_dict[node.serial_number] = node
    
    return node_dict
end


function update_node_value(node, vs_and_ws::Array)
    """
    Given a knapsack node, calculate an estimate
    of the maximum value we could possibly put in 
    the knapsack in future (using linear relaxation).
    
    Parameters
    ----------
    node: node we want to consider.
    
    items: array of items where first column is weight and second
    column is value to fill up the sack.
    """
    
    weight_left = node.weight_left
    depth = length(node.choices)
    
    # calculate value accrued to date
    for (i, choice) in enumerate(node.choices)
        node.value += (choice * vs_and_ws[i, 1])
    end
        
    if weight_left > 0
        
        # fill the knapsack with the densest items in turn
        values = vs_and_ws[depth+1:end, 1]
        weights = vs_and_ws[depth+1:end, 2]  
        
        # cumulatively count the weight to fill the sack
        cumulative_weight = [0]
        append!(cumulative_weight, cumsum(weights)[1:end-1])
        fraction_of_items = (weight_left .- cumulative_weight) ./ weights
        fraction_of_items[fraction_of_items .> 1] .= 1
        fraction_of_items[fraction_of_items .< 0] .= 0
        
        # use the linearly relaxed weight
        node.value += sum(fraction_of_items .* values)
    elseif weight_left < 0
        node.value = -999
    end  
end


function create_child_node(node_dict, serial_number, vs_and_ws; child="left")
    """
    Given a node dictionary plus a serial number, expand the node
    to get it's child node.
    
    Parameters
    ----------
    node_dict: a dictionary containing all of the nodes.
    
    serial_number: number of the node we want to expand.
    
    vs_and_ws: values and weights for each item we want to
    use to fill up the sack.
    
    child: if 'left', then the child node will be the one
    where no item is selected. If 'right', then the child
    node will be the one where the next item in turn is
    selected.
    """
    node = node_dict[serial_number]
    depth = length(node.choices)
    
    # get the weight of the next item to be added
    item_weight = vs_and_ws[depth+1, 2]
    
    if child == "left"
        choices = vcat(node.choices, 0)
        K = node.weight_left
    else
        choices = vcat(node.choices, 1)
        K = node.weight_left - item_weight
    end
    
    child_node = KnapsackNode(choices,
                              join(string.(choices)),
                              K,
                              0)
    # update node's value
    update_node_value(child_node, vs_and_ws)
    # add the new child node to the dictionary of nodes
    node_dict[child_node.serial_number] = child_node
end


# fill the sack using a greedy heuristic
function fill_sack_by_density(K, vs_and_ws)
    """
    Fill up the sack by taking the densest item
    first until the sack is full. This forms the
    initial starting heuristic for filling the 
    sack.
    
    Parameters
    ----------
    K: weight capacity of the sack.
    
    vs_and_ws: values and weights of the items to
    put into the sack.
    """
    
    # initialise the first node of the sack
    node_dict = initialise_starting_node(K, vs_and_ws)
    current_node_number = ""
    n_items, _ = size(vs_and_ws)
    
    for i in 1:n_items
        node = node_dict[current_node_number]
        item_weight = vs_and_ws[i, 2]
        if item_weight < node.weight_left
            create_child_node(node_dict, current_node_number, vs_and_ws, child = "right")
            current_node_number *= "1"
        else
            create_child_node(node_dict, current_node_number, vs_and_ws, child = "left") 
            current_node_number *= "0"
        end
    end
    
    return node_dict
end


function check_terminality(node, vs_and_ws)
    """
    Check whether a node is terminal (i.e.) 
    cannot have any more child nodes.
    """
    n_items, _ = size(vs_and_ws)
    if length(node.choices) == n_items
        return true
    else
        return false
    end
end


function clip_nodes(node_dict, vs_and_ws; verbose=true)
    """
    Clip out any nodes which have a lower 'best'
    value than the 
    """
    # clip any nodes which we know cannot be better than the best node
    
    # return the node dict plus the current solution node
    best_value = -999
    solution_node = nothing
    
    # first find the best terminal node
    for (serial_number, node) in node_dict
        is_terminal = check_terminality(node, vs_and_ws)
        if is_terminal == true
            if node.value > best_value
                best_value = node.value
                solution_node = node
            end
        end
    end
    
    # count the number of nodes we clip
    n_clipped = 0
    
    # clip any nodes which have been fully expanded already
    for (serial_number, node) in node_dict
            # remove fully expanded node from the dictionary
        left_child_serial_number = serial_number * "0"
        right_child_serial_number = serial_number * "1"
        has_left_child = left_child_serial_number in keys(node_dict)
        has_right_child = right_child_serial_number in keys(node_dict)
        if (has_left_child) & (has_right_child)
            delete!(node_dict, serial_number)
            n_clipped += 1
        end
    end
    
    # clip out any nodes which have values below the best solution
    for (serial_number, node) in node_dict
        if node.value < best_value
            delete!(node_dict, serial_number)
            n_clipped += 1
        end
    end
    
    if verbose == true
        println(n_clipped, " nodes have been clipped.")
        println("Nodes left = $(keys(node_dict))")
    end

    
    return node_dict, solution_node
end


function branch_and_bound(node_dict, solution_node, vs_and_ws; verbose=true)
    """
    Perform depth-first branch and bound. Expand the 
    deepest available node into a left and right child, 
    and delete the fully expanded node.
    """
    
    node_to_explore = nothing
    best_value = -999
    deepest = -999
    n_items, _ = size(vs_and_ws)
    
    for (serial_number, node) in node_dict
        node_depth = length(serial_number)
        is_terminal = check_terminality(node, vs_and_ws)
        if (is_terminal == false) & (node_depth > deepest)
            deepest = node_depth
            node_to_explore = node
        end
    end
    
    if node_to_explore != nothing
        if verbose == true
            println("Expanding node ", node_to_explore.serial_number)
        end
        # create children for both the left and right for the new node
        create_child_node(node_dict, node_to_explore.serial_number, vs_and_ws, 
            child = "left")
        create_child_node(node_dict, node_to_explore.serial_number, vs_and_ws, 
            child = "right")
        delete!(node_dict, node_to_explore.serial_number)
    end
    
end


function format_result(node_dict, solution_node, n, K, order; verbose = true)
    """
    Format the result to print out the found
    solution to the correct format for result submission.
    """

    optimal = 1
    lp_gap = 0
    value = convert(Int64, solution_node.value)
    println(value)

    # first check for optimality
    if length(node_dict) == 1
        optimal = 1
        lp_gap = 0
    else
        optimal = 0
        max_value = -999
        for (serial_number, node) in node_dict
            if node.value > max_value
                max_value = node.value
            end
        end
        println(max_value)
        lp_gap = (max_value / value) - 1
    end

    # now produce a string with the correct result format
    output_string = "$(value) $(optimal) \n"
    reorder = sortperm(order)
    for i in reorder
        output_string *= string(solution_node.choices[i]) * " "
    end
    
    if verbose == true
        println("Solution found. Parameters:")
        println("---------------------------")
        println("optimal = $optimal")
        println("value = $value")
        println("LP gap = $lp_gap")
        println("---------------------------")
    end
    
    return output_string
end


function optimise_knapsack(input_data; iteration_limit = Inf, 
                           timeout = 10, verbose = false)
    
    # read in our sack and sort by density
    n, K, vs_and_ws = read_knapsack_file(input_data)
    vs_and_ws, order = sort_by_density(vs_and_ws)
    println("Initialising knapsack with n = $n and K = $K")
    
    # fill sack using the simple heuristic
    node_dict = fill_sack_by_density(K, vs_and_ws)
    node_dict, current_solution_node = clip_nodes(node_dict, vs_and_ws, verbose = verbose)
    
    t0 = Dates.now() # get the current time
    t1 = Dates.now() # variable to record the iteration time
    time_limit = Dates.Millisecond(timeout * 1000)
    counter = 0
    
    while (length(node_dict) > 1) & (counter < iteration_limit) & (t1 - t0 < time_limit)
        branch_and_bound(node_dict, current_solution_node, vs_and_ws, verbose = verbose)
        node_dict, current_solution_node = clip_nodes(node_dict, vs_and_ws, verbose = verbose)
        counter += 1
        t1 = Dates.now() # record time at end of each loop
    end
    
    println("Knapsack optimised. t = $(t1-t0); n_iter = $(counter)")
    result = format_result(node_dict, current_solution_node, n, K, order, 
        verbose = true)
    return result
end