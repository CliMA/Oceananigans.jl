using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: active_cells_per_column

abstract type BalancingStrategy end

# x and y partitioning optimised together to produce balanced loads
struct GeneralizedBlockDistribution <: BalancingStrategy end

# x and y partitioning balanced separately
struct SimplifiedGeneralizedBlockDistribution <: BalancingStrategy end

# x partitioning balanced first, then the y partitioning of every column of ranks separately
struct JaggedBlockDistribution <: BalancingStrategy end

create_balanced_partition(strategy, partition, cost_map) = partition

"""
    partition_1d(costs, R)

Split `costs` into `R` contiguous nonempty ranges, minimizing the cost of the most expensive one.
"""
function partition_1d(costs, R)
    N = length(costs)

    # Bisect on the cost of the most expensive range, counting the ranges of a greedy split below that cost
    lower, upper = maximum(costs), sum(costs)
    for bisection in 1:64
        maximum_cost = (lower + upper) / 2
        count = 1
        cost = zero(eltype(costs))
        for c in costs
            if cost + c > maximum_cost
                count += 1
                cost = zero(cost)
            end
            cost += c
        end
        count ≤ R ? (upper = maximum_cost) : (lower = maximum_cost)
    end

    # Split greedily below that cost, leaving at least one entry to every range still to come
    ranges = UnitRange{Int}[]
    first_index = 1
    cost = zero(eltype(costs))
    for n in 1:N
        ranges_to_come = R - length(ranges) - 1
        entries_to_come = N - n + 1
        if n > first_index && ranges_to_come > 0 && (cost + costs[n] > upper || entries_to_come ≤ ranges_to_come)
            push!(ranges, first_index:n-1)
            first_index = n
            cost = zero(cost)
        end
        cost += costs[n]
    end
    push!(ranges, first_index:N)

    return ranges
end

function create_balanced_partition(strategy::SimplifiedGeneralizedBlockDistribution, unbalanced_partition, cost_map)
    Rx, Ry, _ = ranks(unbalanced_partition)
    costs = on_architecture(CPU(), cost_map)

    # Partition each direction independently
    x = partition_1d(vec(sum(costs; dims=(2, 3))), Rx)
    y = partition_1d(vec(sum(costs; dims=(1, 3))), Ry)

    return Partition(; x=Sizes(length.(x)...), y=Sizes(length.(y)...))
end

function create_balanced_partition(strategy::GeneralizedBlockDistribution, unbalanced_partition, cost_map)
    # Iterative algorithm based on "Manne, F., Sørevik, T. (1996). Partitioning an array onto a mesh of processors"
    Rx, Ry, _ = ranks(unbalanced_partition)
    costs = on_architecture(CPU(), cost_map)

    x_costs(y) = vec(maximum([sum(costs[i, j, :]) for i in axes(costs, 1), j in y]; dims=2))
    y_costs(x) = vec(maximum([sum(costs[i, j, :]) for i in x, j in axes(costs, 2)]; dims=1))

    x = partition_1d(vec(sum(costs; dims=(2, 3))), Rx)
    y = partition_1d(y_costs(x), Ry)

    # Alternate until a partition repeats: a (local) optimum, or a cycle between partitions
    visited = Set([(x, y)])
    while true
        x = partition_1d(x_costs(y), Rx)
        y = partition_1d(y_costs(x), Ry)
        (x, y) ∈ visited && break
        push!(visited, (x, y))
    end

    return Partition(; x=Sizes(length.(x)...), y=Sizes(length.(y)...))
end

function create_balanced_partition(strategy::JaggedBlockDistribution, unbalanced_partition, cost_map)
    Rx, Ry, _ = ranks(unbalanced_partition)
    costs = on_architecture(CPU(), cost_map)

    column_costs(i) = vec(sum(costs[i, :, :]; dims=(1, 3)))

    x = partition_1d(vec(sum(costs; dims=(2, 3))), Rx)
    y = Tuple(Sizes(length.(partition_1d(column_costs(i), Ry))...) for i in x)

    return Partition(; x=Sizes(length.(x)...), y)
end

function create_cost_map(grid, ib)
    materialized_ib = on_architecture(architecture(grid), materialize_immersed_boundary(grid, ib))
    return active_cells_per_column(grid, materialized_ib)
end
