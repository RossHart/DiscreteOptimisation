# Discrete Optimisation

Solutions to the 6 problems in the [Discrete Optimisation](https://www.coursera.org/learn/discrete-optimization) course on Coursera. All problems are solved to the point of passing the assignment in a reasonable time (usually less than 1 hour to solve all of the problems in a single assignment). All code is written in [Julia](https://julialang.org/), with the help of its [JuMP](https://www.juliaopt.org/JuMP.jl/stable/) and [Cbc](https://github.com/JuliaOpt/Cbc.jl) libraries. 

There are 6 problems in total, and the approaches I took to solve them are listed here:

### 1. AnyInt 
A simple test assignment

### 2. Knapsack 

Solved using a simple branch and bound algorithm, relaxing the integrality constraint on branching. Note that this is effectively the simplest possible MIP program.

### 3. Graph colouring 

This was solved using an iterative greedy approach. If we greedily colour, then re-order so that groups are together, then we can get the same or better colouring at each iteration. This is a very easy scalable approach to the problem (outlined [here](https://pdfs.semanticscholar.org/0535/997d80cc4d1dbd7e02e02a57fe7d82e6fda1.pdf?_ga=2.32457529.686261839.1589381418-1386486068.1587467710).

### 4. Travelling salesman 

Solved using a simple 2-opt simulated annealing algorithm. This takes some time to run, but does generate adequate solutions after a few minutes of CPU time on a small machine. 

### 5. Facility location 

I used a [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering algorithm to break the problem down into small neighbourhoods (the assumption being that far-away warehouse/customer pairs are unlikely to be good). Each neighbourhood is then solved  and using the open source [Cbc](https://github.com/coin-or/Cbc) optimiser. Breaking the problem down into smaller problems is vital for solving this problem in a reasonable time (good solutions are generated in a couple of minutes of CPU time on my small machine for each problem).

### 6. Vehicle routing problem 

I first used the Clarke-Wright savings heuristic (outlined [here](http://courses.ieor.berkeley.edu/ieor151/lecture_notes/ieor151_lec18.pdf)) to get a good initial guess at the solution. In the case that this produced too many vehicles, then the shortest routes were reassigned to their nearest nesighbours. A short random search swapping neighbours and moving neighbours between routes is then used to form the first feasible solution. Local search with three options (to move customers between routes, swap customers or to optimise individual routes as TSP problems) are then used to improve the solution using simulated annealing. The algorithm produces good solutions to all the problems in only a few seconds of CPU time on my small machine.

