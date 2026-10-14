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

# Number of contiguous parts of `costs` that cost at most `B` each
function count_parts(costs, B)
    parts = 1
    part_cost = zero(eltype(costs))
    for c in costs
        if part_cost + c > B
            parts += 1
            part_cost = zero(part_cost)
        end
        part_cost += c
    end
    return parts
end

"""
    balanced_ranges(costs, R)

Split `costs` into `R` contiguous nonempty ranges, minimizing the cost of the most expensive one.
"""
function balanced_ranges(costs, R)
    lower, upper = maximum(costs), sum(costs)
    for bisection in 1:64
        middle = (lower + upper) / 2
        count_parts(costs, middle) ≤ R ? (upper = middle) : (lower = middle)
    end

    ranges = UnitRange{Int}[]
    first_index = 1
    part_cost = zero(eltype(costs))
    N = length(costs)
    for n in 1:N
        open_parts = R - length(ranges)
        if n > first_index && open_parts > 1 && (part_cost + costs[n] > upper || N - n < open_parts - 1)
            push!(ranges, first_index:n-1)
            first_index = n
            part_cost = zero(part_cost)
        end
        part_cost += costs[n]
    end
    push!(ranges, first_index:N)

    return ranges
end

function create_balanced_partition(strategy::SimplifiedGeneralizedBlockDistribution, unbalanced_partition, cost_map)
    Rx, Ry, _ = ranks(unbalanced_partition)
    costs = on_architecture(CPU(), cost_map)

    # Partition each direction independently
    x = balanced_ranges(vec(sum(costs; dims=(2, 3))), Rx)
    y = balanced_ranges(vec(sum(costs; dims=(1, 3))), Ry)

    return Partition(; x=Sizes(length.(x)...), y=Sizes(length.(y)...))
end

function create_balanced_partition(strategy::GeneralizedBlockDistribution, unbalanced_partition, cost_map)
    # Iterative algorithm based on "Manne, F., Sørevik, T. (1996). Partitioning an array onto a mesh of processors"
    Rx, Ry, _ = ranks(unbalanced_partition)
    costs = on_architecture(CPU(), cost_map)

    x_costs(y) = vec(maximum([sum(costs[i, j, :]) for i in axes(costs, 1), j in y]; dims=2))
    y_costs(x) = vec(maximum([sum(costs[i, j, :]) for i in x, j in axes(costs, 2)]; dims=1))

    x = balanced_ranges(vec(sum(costs; dims=(2, 3))), Rx)
    y = balanced_ranges(y_costs(x), Ry)

    while true
        new_x = balanced_ranges(x_costs(y), Rx)
        new_y = balanced_ranges(y_costs(new_x), Ry)

        # If the partition stays the same, we have reached a (local) optimum
        (new_x, new_y) == (x, y) && break
        x, y = new_x, new_y
    end

    return Partition(; x=Sizes(length.(x)...), y=Sizes(length.(y)...))
end

function create_balanced_partition(strategy::JaggedBlockDistribution, unbalanced_partition, cost_map)
    Rx, Ry, _ = ranks(unbalanced_partition)
    costs = on_architecture(CPU(), cost_map)

    column_costs(i) = vec(sum(costs[i, :, :]; dims=(1, 3)))

    x = balanced_ranges(vec(sum(costs; dims=(2, 3))), Rx)
    y = Tuple(Sizes(length.(balanced_ranges(column_costs(i), Ry))...) for i in x)

    return Partition(; x=Sizes(length.(x)...), y)
end

function create_cost_map(grid, ib)
    materialized_ib = on_architecture(architecture(grid), materialize_immersed_boundary(grid, ib))
    return active_cells_per_column(grid, materialized_ib)
end
