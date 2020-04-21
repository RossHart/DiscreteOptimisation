using Dates
using Random
using StatsBase

function read_input(input_data)
    """
    Read in the input data to a value for N, E
    and an array of edges.
    """

    input_strings = split(input_data, "\n")
    N, E = split(input_strings[1], " ")
    N = parse(Int64, N)
    E = parse(Int64, E)
    edges = Array{Int64}(undef, 0, 2)
    new_edges = []
    # loop through and generate a complete list of edges
    for i in 2:E+1
        edge_start, edge_end = split(input_strings[i], " ")
        edge_start = parse(Int64, edge_start)
        edge_end = parse(Int64, edge_end)
        edges = vcat(edges, transpose([edge_start, edge_end]))
    end
    
    return N, E, edges
end


mutable struct Node
    """
    Node to store which node has which 
    neighbours.
    """
    node_number::Int64
    edges::Array
    colour::Int64
end


function generate_nodes(N::Int64, E::Int64, edges::Array)
    """
    Generate a complete list of nodes.
    """
    nodes = Dict{Int64, Any}()
    for i in 0:(N-1)
        nodes[i] = Node(i, [], -1)
    end

    # set up the connections between the nodes
    for i in 1:E
        edge_start, edge_end = edges[i, :]
        push!(nodes[edge_start].edges, edge_end)
        push!(nodes[edge_end].edges, edge_start)
    end
    return nodes
end


function find_domain(node, nodes, N_colours = 2)
    """
    Find the domain of a node. This should be 
    all of the colours our node can take plus
    one extra colour which we may choose.
    """
    domain = collect(1:N_colours+1)
    
    for neighbour_number in node.edges
        filter!(x -> x != nodes[neighbour_number].colour, domain)
    end     
    
    return domain
end


function find_N_colours(nodes)
    """
    Find whichever is the max colour that was
    used in the graph colouring.
    """
    max_colour = 0
    
    for (key, node) in nodes
        if node.colour > max_colour
            max_colour = node.colour
        end
    end
    return max_colour
end


function format_result(result, N, optimal, N_colours)
    
    result_string = "$(N_colours) $(optimal) \n"
    for i in 0:N-1
        result_string *= " $(result[i].colour)"
    end
    return result_string
end


function order_nodes_welsh_powell(nodes)
    ordering = sort(collect(nodes), by = x -> length(x[2].edges), rev = true)
    ordering = [order[1] for order in ordering]
    return ordering
end


function fill_greedily(nodes, ordering; verbose = true)
    # fill up the nodes greedily
    N_colours = 1
    choice_cache = []
    
    for node_number in ordering
        node = nodes[node_number]
        domain = find_domain(node, nodes, N_colours)
        node.colour = domain[1]
        # re-calculate the domain size
        N_colours = find_N_colours(nodes)
    end
    
    if verbose == true
        N_colours = find_N_colours(nodes)
        println("Graph filled. Result has $(N_colours) colours.")
    end
    
    return nodes
end


function order_by_colour(nodes, colour_order)

    N_colours = find_N_colours(nodes)
    colours = 1:N_colours

    nodes_by_colour = Dict()
    for colour in colours
        nodes_by_colour[colour] = Dict()
    end

    for (node_number, node) in nodes
        for colour in colours
            if node.colour == colour
                nodes_by_colour[colour][node.node_number] = node
            end
        end
    end

    # order our set of colours by whichever order we supplied
    if colour_order == "reverse"
        colour_ordering = N_colours:-1:1
    elseif colour_order == "decreasing"
        colour_ordering = sort(collect(nodes_by_colour), by = x -> length(x))
        colour_ordering = [colour[1] for colour in colour_ordering]
    elseif colour_order == "increasing" 
        colour_ordering = sort(collect(nodes_by_colour), by = x -> length(x))
        colour_ordering = reverse([colour[1] for colour in colour_ordering])
    else # random ordering
        colour_ordering = shuffle!(collect(N_colours:-1:1))
    end

    final_ordering = []

    for colour in colour_ordering
        ordering_by_colour = order_nodes_welsh_powell(nodes_by_colour[colour])
        final_ordering = vcat(final_ordering, ordering_by_colour)
    end
    
    return final_ordering
end

function reset_nodes(nodes)
    """
    Reset all of the nodes to be uncoloured.
    """
    
    for (key, node) in nodes
        node.colour = -1
    end
end


function generate_colour_order()
    """
    Generate a colour order by the frequency suggested in 
    https://pdfs.semanticscholar.org/0535/997d80cc4d1dbd7e02e02a57fe7d82e6fda1.pdf
    ?_ga=2.125705482.1414436374.1587467710-1386486068.1587467710
    """
    colour_orders = ["decreasing", "reverse", "increasing", "random"]
    weights = [70, 50, 10, 30]
    colour_order = sample(colour_orders, Weights(weights))
    return colour_order
end


function colour_graph(input_data; 
        random_seed = 0, iteration_limit = Inf, timeout = 10)
    
    Random.seed!(random_seed)
    
    # use Welsh-Powell to get the first set of colours
    N, E, edges = read_input(input_data)
    nodes = generate_nodes(N, E, edges)
    ordering = order_nodes_welsh_powell(nodes)
    nodes = fill_greedily(nodes, ordering, verbose = true)
    best_N = find_N_colours(nodes)
    
    # set up iterators to set up in the loop
    i = 1
    t0 = Dates.now()
    t1 = Dates.now()
    timeout_ms = Dates.Millisecond(1000 * timeout)
    
    while (i < iteration_limit) & (t1 - t0 < timeout_ms)
        colour_order = generate_colour_order()
        ordering = order_by_colour(nodes, colour_order)
        reset_nodes(nodes)
        fill_greedily(nodes, ordering, verbose = false)
        
        N_colours = find_N_colours(nodes)
        if N_colours < best_N
            println("New solution found with $(N_colours) colours.")
            best_N = N_colours
        end
        
        # add to the iteration counter
        i += 1
        t1 = Dates.now()
    end
    
    # output final string result
    N_colours = find_N_colours(nodes)
    result = format_result(nodes, N, 0, N_colours)
    return result
end