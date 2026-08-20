abstract type BalancingStrategy end

# x and y partitioning optimised together to produce balanced loads
struct GeneralisedBlockDistribution <: BalancingStrategy end

# x and y partitioning balanced separately
struct SimplifiedGeneralisedBlockDistribution <: BalancingStrategy end

create_balanced_partition(strategy, partition, weight_map) = partition

function create_balanced_partition(strategy::SimplifiedGeneralisedBlockDistribution, ranks, weight_map)
  weights = interior(weight_map).data

  x_cumsum = cumsum(weights; dims=1)
  x_total = x_cumsum[end]
  # Optimal weight of each x partition
  x_part_weight = x_total / ranks.x
  # Indices of ends of partitions
  x_ends = [ searchsortedfirst(x_cumsum, x_part_weight * i) for i in 1:ranks.x]
  x_sizes = accumulate((a,b) -> b-a, x_ends; init=1)

  y_cumsum = cumsum(weights; dims=2)
  y_total = y_cumsum[end]
  # Optimal weight of each y partition
  y_part_weight = y_total / ranks.y
  # Indices of ends of partitions
  y_ends = [ searchsortedfirst(y_cumsum, y_part_weight * i) for i in 1:ranks.y]
  y_sizes = accumulate((a,b) -> b-a, y_ends; init=1)

  return Partition(; x=Sizes(x_sizes...), y=Sizes(y_sizes...))

end

function create_weight_map(grid, ib)
  materialized_ib = materialize_immersed_boundary(grid, ib)
  return active_cells_per_column(grid, materialized_ib)
end
