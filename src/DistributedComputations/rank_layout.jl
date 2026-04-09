"""
    RankLayout{P}

Container describing how a `Partition` is mapped to MPI ranks for a load-balanced
distributed immersed grid. Empty tiles (no active 3D cells) are not assigned a rank.

$(TYPEDFIELDS)
"""
struct RankLayout{P}
    partition    :: P
    tile_to_rank :: Matrix{Int}
    rank_to_tile :: Vector{Tuple{Int, Int}}
end

n_active_tiles(layout::RankLayout) = length(layout.rank_to_tile)
tile_shape(layout::RankLayout) = size(layout.tile_to_rank)
is_active_tile(layout::RankLayout, ix::Integer, iy::Integer) = layout.tile_to_rank[ix, iy] >= 0

function save_rank_layout(path::AbstractString, layout::RankLayout)
    jldopen(path, "w") do file
        file["partition"]    = layout.partition
        file["tile_to_rank"] = layout.tile_to_rank
        file["rank_to_tile"] = layout.rank_to_tile
    end
    return nothing
end

function load_rank_layout(path::AbstractString)
    jldopen(path, "r") do file
        return RankLayout(file["partition"], file["tile_to_rank"], file["rank_to_tile"])
    end
end
