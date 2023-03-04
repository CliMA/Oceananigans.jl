using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices

using DocStringExtensions

struct CubedSpherePartition{M, P} <: AbstractPartition
    div :: Int
     Rx :: M
     Ry :: P

    CubedSpherePartition(div, Rx::M, Ry::P) where {M, P} = new{M, P}(div, Rx, Ry)
end

""""
    CubedSpherePartition(; R = 1)

Return a cubed sphere partition with `R` partitions in each dimension of the panel
of the sphere.
"""
function CubedSpherePartition(; R = 1)
    # at the moment only CubedSpherePartitions with Rx = Ry are supported
    Rx = Ry = R

    if Rx isa Number 
        if Ry isa Number
            Rx != Ry && 
                throw(ArgumentError("Regular cubed sphere partiction must have Rx == Ry!"))
            div = 6 * Rx * Ry
        else
            div = sum(Ry .* Rx)
        end
    else
        div = sum(Ry .* Rx)
    end

    div < 6 && throw(ArgumentError("Cubed sphere requires at least 6 regions!"))

    return CubedSpherePartition(div, Rx, Ry)
end

const RegularCubedSpherePartition  = CubedSpherePartition{<:Number, <:Number}
const XRegularCubedSpherePartition = CubedSpherePartition{<:Number}
const YRegularCubedSpherePartition = CubedSpherePartition{<:Any, <:Number}

Base.length(p::CubedSpherePartition) = p.div

"""
utilities to get the index of the panel the index within the panel and the global index
"""
@inline div_per_panel(panel_idx, partition::RegularCubedSpherePartition)  = partition.Rx            * partition.Ry  
@inline div_per_panel(panel_idx, partition::XRegularCubedSpherePartition) = partition.Rx            * partition.Ry[panel_idx]
@inline div_per_panel(panel_idx, partition::YRegularCubedSpherePartition) = partition.Rx[panel_idx] * partition.Ry

@inline Rx(panel_idx, partition::RegularCubedSpherePartition)  = partition.Rx    
@inline Rx(panel_idx, partition::XRegularCubedSpherePartition) = partition.Rx    
@inline Rx(panel_idx, partition::CubedSpherePartition)         = partition.Rx[panel_idx] 

@inline Ry(panel_idx, partition::RegularCubedSpherePartition)  = partition.Ry    
@inline Ry(panel_idx, partition::YRegularCubedSpherePartition) = partition.Ry    
@inline Ry(panel_idx, partition::CubedSpherePartition)         = partition.Ry[panel_idx] 

@inline panel_index(r, partition)         = (r - 1) ÷ div_per_panel(r, partition) + 1
@inline intra_panel_index(r, partition)   = mod(r - 1, div_per_panel(r, partition)) + 1
@inline intra_panel_index_x(r, partition) = mod(intra_panel_index(r, partition) - 1, Rx(r, partition)) + 1
@inline intra_panel_index_y(r, partition) = (intra_panel_index(r, partition) - 1) ÷ Rx(r, partition) + 1

@inline rank_from_panel_idx(pᵢ, pⱼ, panel_idx, partition::CubedSpherePartition) =
            (panel_idx - 1) * div_per_panel(panel_idx, partition) + Rx(panel_idx, partition) * (pⱼ - 1) + pᵢ

@inline function region_corners(r, p::CubedSpherePartition)
    pᵢ = intra_panel_index_x(r, p)
    pⱼ = intra_panel_index_y(r, p)

    bottom_left  = pᵢ == 1              && pⱼ == 1              ? true : false
    bottom_right = pᵢ == p.div_per_side && pⱼ == 1              ? true : false
    top_left     = pᵢ == 1              && pⱼ == p.div_per_side ? true : false
    top_right    = pᵢ == p.div_per_side && pⱼ == p.div_per_side ? true : false

    return (; bottom_left, bottom_right, top_left, top_right)
end

@inline function region_edge(r, p::CubedSpherePartition)
    pᵢ = intra_panel_index_x(r, p)
    pⱼ = intra_panel_index_y(r, p)

    west  = pᵢ == 1              ? true : false
    east  = pᵢ == p.div_per_side ? true : false
    south = pⱼ == 1              ? true : false
    north = pⱼ == p.div_per_side ? true : false

    return (; west, east, south, north)
end

# Adopted from figure 8.4 of https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-6tile
# The configuration of the panels for the cubed sphere. Each panel is partitioned in two parts YPartition(2)
#
#                              ponel P5      panel P6
#                           + ---------- + ---------- +
#                           |     ↑↑     |     ↑↑     |
#                           |     1W     |     1S     |
#                           |←3N      6W→|←5E      2S→|
#                           |------------|------------|
#                           |←3N      6W→|←5E      2S→|
#                           |     4N     |     4E     |
#                 panel P3  |     ↓↓     |     ↓↓     |
#              + ---------- +------------+------------+
#              |     ↑↑     |     ↑↑     | 
#              |     5W     |     5S     | 
#              |←1N      4W→|←3E      6S→| 
#              |------------|------------| 
#              |←1N      4W→|←3E      6S→| 
#              |     2N     |     2E     | 
#              |     ↓↓     |     ↓↓     | 
# + -----------+------------+------------+ 
# |     ↑↑     |     ↑↑     |  panel P4
# |     3W     |     3S     |
# |←5N      2W→|←1E      4S→|
# |------------|------------|
# |←5N      2W→|←1E      4S→|
# |     6N     |     6E     |
# |     ↓↓     |     ↓↓     |
# + -----------+------------+
#    panel P1     panel P2

#####
##### Boundary-specific Utils
#####

abstract type AbstractCubedSphereConnectivity end

"""
    struct CubedSphereConnectivity <: AbstractCubedSphereConnectivity 

The connectivity among various regions for a cubed sphere grid.

$(TYPEDFIELDS)

Example
=======

A connectivity that implies that the boundary condition for the
north side of region 1 comes from the west side of region 3 is:

```julia
julia> CubedSphereConnectivity(1, 3, :north, :west)
CubedSphereConnectivity(1, 3, :north, :west)
```
"""
struct CubedSphereConnectivity <: AbstractCubedSphereConnectivity 
    "the current region rank"
         rank :: Int
    "the region from which boundary condition comes from"
    from_rank :: Int
    "the current region side"
         side :: Symbol
    "the side of the region from which boundary condition comes from"
    from_side :: Symbol
end

function inject_west_boundary(region, p::CubedSpherePartition, global_bc)
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pᵢ == 1
        if mod(pidx, 2) == 0
            from_side = :east
            from_panel = pidx - 1
            from_pᵢ    = Rx(from_panel, p)
            from_pⱼ    = pⱼ
        else    
            from_side  = :north
            from_panel = mod(pidx + 3, 6) + 1
            from_pᵢ    = Rx(from_panel, p) - pⱼ + 1
            from_pⱼ    = Ry(from_panel, p)
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = :east
        from_rank = rank_from_panel_idx(pᵢ - 1, pⱼ, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :west, from_side))
end

function inject_east_boundary(region, p::CubedSpherePartition, global_bc) 
 
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pᵢ == p.Rx
        if mod(pidx, 2) != 0
            from_side  = :west
            from_panel = pidx + 1
            from_pᵢ    = 1
            from_pⱼ    = pⱼ
        else    
            from_side  = :south
            from_panel = mod(pidx + 1, 6) + 1
            from_pᵢ    = Rx(from_panel, p) - pⱼ + 1
            from_pⱼ    = 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = :west
        from_rank = rank_from_panel_idx(pᵢ + 1, pⱼ, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :east, from_side))
end

function inject_south_boundary(region, p::CubedSpherePartition, global_bc)
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pⱼ == 1
        if mod(pidx, 2) != 0
            from_side  = :north
            from_panel = mod(pidx + 4, 6) + 1
            from_pᵢ    = pᵢ 
            from_pⱼ    = Ry(from_panel, p)
        else    
            from_side  = :east
            from_panel = mod(pidx + 3, 6) + 1
            from_pᵢ    = Rx(from_panel, p)
            from_pⱼ    = Ry(from_panel, p) - pᵢ + 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = :north
        from_rank = rank_from_panel_idx(pᵢ, pⱼ - 1, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))
end

function inject_north_boundary(region, p::CubedSpherePartition, global_bc)
    pᵢ = intra_panel_index_x(region, p)
    pⱼ = intra_panel_index_y(region, p)

    pidx = panel_index(region, p)

    if pⱼ == p.Ry
        if mod(pidx, 2) == 0
            from_side  = :south
            from_panel = mod(pidx, 6) + 1
            from_pᵢ    = pᵢ 
            from_pⱼ    = 1
        else    
            from_side  = :west
            from_panel = mod(pidx + 1, 6) + 1
            from_pᵢ    = 1
            from_pⱼ    = Rx(from_panel, p) - pᵢ + 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, p)
    else
        from_side = :south
        from_rank = rank_from_panel_idx(pᵢ, pⱼ + 1, pidx, p)
    end

    return MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))
end

function Base.summary(p::CubedSpherePartition)
    region_str = p.Rx * p.Ry > 1 ? "regions" : "region"

    return "CubedSpherePartition with ($(p.Rx * p.Ry) $(region_str) in each panel)"
end

Base.show(io::IO, p::CubedSpherePartition) =
    print(io, summary(p), "\n",
          "├── Rx: ", p.Rx, "\n",
          "├── Ry: ", p.Ry, "\n",
          "└── div: ", p.div)
