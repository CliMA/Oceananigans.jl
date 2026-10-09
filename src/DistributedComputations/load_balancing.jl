using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: active_cells_per_column

abstract type BalancingStrategy end

# x and y partitioning optimised together to produce balanced loads
struct GeneralizedBlockDistribution <: BalancingStrategy end

# x and y partitioning balanced separately
struct SimplifiedGeneralizedBlockDistribution <: BalancingStrategy end

create_balanced_partition(strategy, partition, cost_map) = partition

function partition_1d(costs, ranks)
  csum = cumsum(costs)
  total = csum[end]
  # Optimal cost of each partition
  optimal_cost = total / ranks
  # Indices of ends of partitions
  left = [searchsortedfirst(csum, optimal_cost * i) for i in 0:ranks-1]
  right = [searchsortedlast(csum, optimal_cost * i) for i in 1:ranks]

  return zip(left, right)
end

function ends_to_sizes(ends)
  sizes = [r-l+1 for (l,r) in ends]
  return sizes
end

function create_balanced_partition(strategy::SimplifiedGeneralizedBlockDistribution, unbalanced_partition, cost_map)
  Rx, Ry, _ = ranks(unbalanced_partition)
  costs = on_architecture(CPU(), cost_map)

  # Partition each direction independently
  x_costs = Iterators.flatten(sum(costs; dims=(2,3)))
  x_ends = partition_1d(x_costs, Rx)
  x_sizes = ends_to_sizes(x_ends)

  y_costs = Iterators.flatten(sum(costs; dims=(1,3)))
  y_ends = partition_1d(y_costs, Ry)
  y_sizes = ends_to_sizes(y_ends)

  return Partition(; x=Sizes(x_sizes...), y=Sizes(y_sizes...))

end

function create_balanced_partition(strategy::GeneralizedBlockDistribution, unbalanced_partition, cost_map)
  # Iterative algorithm based on "Manne, F., Sørevik, T. (1996). Partitioning an array onto a mesh of processors"
  Rx, Ry, _ = ranks(unbalanced_partition)
  costs = on_architecture(CPU(), cost_map)

  # Partition x first
  x_costs = Iterators.flatten(sum(costs; dims=(2,3)))
  x_ends = partition_1d(x_costs, Rx)

  # Reduce map from mxn to pxn
  y_costs = Iterators.flatten(maximum([sum(costs[l:r,j]) for (l,r) in x_ends, j in axes(costs, 2)]; dims=1))
  y_ends = partition_1d(y_costs, Ry)

  x_sizes = ends_to_sizes(x_ends)
  y_sizes = ends_to_sizes(y_ends)

  old_x_sizes = x_sizes
  old_y_sizes = y_sizes

  optimized = false

  while !optimized
    x_costs = Iterators.flatten(maximum([sum(costs[i,l:r]) for i in axes(costs, 1), (l,r) in y_ends]; dims=2))
    x_ends = partition_1d(x_costs, Rx)

    y_costs = Iterators.flatten(maximum([sum(costs[l:r,j]) for (l,r) in x_ends, j in axes(costs, 2)]; dims=1))
    y_ends = partition_1d(y_costs, Ry)

    x_sizes = ends_to_sizes(x_ends)
    y_sizes = ends_to_sizes(y_ends)

    # If the partition stays the same, we have reached a (local) optimum
    if x_sizes == old_x_sizes && y_sizes == old_y_sizes
      optimized = true
    else
      old_x_sizes = x_sizes
      old_y_sizes = y_sizes
    end
  end

  return Partition(; x=Sizes(x_sizes...), y=Sizes(y_sizes...))
end

function create_cost_map(grid, ib)
  materialized_ib = on_architecture(architecture(grid), materialize_immersed_boundary(grid, ib))
  return active_cells_per_column(grid, materialized_ib)
end
