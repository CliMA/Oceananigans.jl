using Oceananigans.Architectures: on_architecture

abstract type BalancingStrategy end

# x and y partitioning optimised together to produce balanced loads
struct GeneralisedBlockDistribution <: BalancingStrategy end

# x and y partitioning balanced separately
struct SimplifiedGeneralisedBlockDistribution <: BalancingStrategy end

create_balanced_partition(strategy, partition, weight_map) = partition

function partition_1d(weights, ranks)
  csum = cumsum(weights)
  total = csum[end]
  # Optimal weight of each partition
  optimal_weight = total / ranks
  # Indices of ends of partitions
  left = [searchsortedfirst(csum, optimal_weight * i) for i in 0:ranks-1]
  right = [searchsortedfirst(csum, optimal_weight * i) for i in 1:ranks]

  return zip(left, right)
end

function ends_to_sizes(ends)
  sizes = [r-l for (l,r) in ends]
  return sizes
end

function create_balanced_partition(strategy::SimplifiedGeneralisedBlockDistribution, ranks, weight_map)
  weights = on_architecture(CPU(), interior(weight_map))

  # Partition each direction independently
  x_weights = Iterators.flatten(sum(weights; dims=(2,3)))
  x_ends = partition_1d(x_weights, ranks.x)
  x_sizes = ends_to_sizes(x_ends)

  y_weights = Iterators.flatten(sum(weights; dims=(1,3)))
  y_ends = partition_1d(y_weights, ranks.y)
  y_sizes = ends_to_sizes(y_ends)

  return Partition(; x=Sizes(x_sizes...), y=Sizes(y_sizes...))

end

function create_balanced_partition(strategy::GeneralisedBlockDistribution, ranks, weight_map)
  # Iterative algorithm based on "Manne, F., Sørevik, T. (1996). Partitioning an array onto a mesh of processors"
  weights = on_architecture(CPU(), interior(weight_map))

  # Partition x first
  x_weights = Iterators.flatten(sum(weights; dims=(2,3)))
  x_ends = partition_1d(x_weights, ranks.x)

  # Reduce map from mxn to pxn
  y_weights = Iterators.flatten(maximum([sum(weights[l:r,j]) for (l,r) in x_ends, j in axes(weights, 2)]; dims=1))
  y_ends = partition_1d(y_weights, ranks.y)

  x_sizes = ends_to_sizes(x_ends)
  y_sizes = ends_to_sizes(y_ends)

  old_x_sizes = x_sizes
  old_y_sizes = y_sizes

  optimized = false

  while !optimized
    x_weights = Iterators.flatten(maximum([sum(weights[i,l:r]) for i in axes(weights, 1), (l,r) in y_ends]; dims=2))
    x_ends = partition_1d(x_weights, ranks.x)

    y_weights = Iterators.flatten(maximum([sum(weights[l:r,j]) for (l,r) in x_ends, j in axes(weights, 2)]; dims=1))
    y_ends = partition_1d(y_weights, ranks.y)

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

function create_weight_map(grid, ib)
  materialized_ib = materialize_immersed_boundary(grid, ib)
  return active_cells_per_column(grid, materialized_ib)
end
