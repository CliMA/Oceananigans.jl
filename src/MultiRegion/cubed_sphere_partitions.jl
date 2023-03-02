using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices

struct CubedSpherePartition{M, P} <: AbstractPartition
    div :: Int
     Rx :: M
     Ry :: P

    CubedSpherePartition(div, Rx::M, Ry::P) where {M, P} = new{M, P}(div, Rx, Ry)
end

""""
    CubedSpherePartition(; Rx = 1, Ry = Rx)

Return a cubed sphere partition.

* `Rx`: number of ``x``-division of each panel. Can be a number (i.e., all panels are divided in
        the same way) or a vector of length 6.
* `Ry`: number of ``y``-division of each panel. Can be a number (i.e., all panels are divided in
        the same way) or a vector of length 6.
"""
function CubedSpherePartition(; Rx = 1, Ry = Rx)
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
#   panel P1   panel P2

#####
##### Boundary-specific Utils
#####

abstract type AbstractCubedSphereConnectivity end

struct CubedSphereConnectivity <: AbstractCubedSphereConnectivity 
         rank :: Int
    from_rank :: Int
         side :: Symbol
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

    bc = MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :west, from_side))

    return bc
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

    bc = MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :east, from_side))

    return bc
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

    bc = MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))

    return bc
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

    bc = MultiRegionCommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))

    return bc
end

function Base.summary(p::CubedSpherePartition)
    region_str = "region"
    if p.Rx * p.Ry >1
        region_str = "regions"
    end

    return "CubedSpherePartition with ($(p.Rx * p.Ry) $(region_str) in each panel)"
end

"""
Partition with 4 regions per panel (2 divisions in x and 2 in y)

part = CubedSpherePartition(Rx = 2, Ry = 2)

testing the correct injection of `west` boundary conditions

for i in 1:24
    @show i, inject_west_boundary(i, part, 1).condition
end
(i, (inject_west_boundary(i, part2, 1)).condition) = (1, Oceananigans.MultiRegion.CubedSphereConnectivity(1, 20, :west, :north))
(i, (inject_west_boundary(i, part2, 1)).condition) = (2, Oceananigans.MultiRegion.CubedSphereConnectivity(2, 1, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (3, Oceananigans.MultiRegion.CubedSphereConnectivity(3, 19, :west, :north))
(i, (inject_west_boundary(i, part2, 1)).condition) = (4, Oceananigans.MultiRegion.CubedSphereConnectivity(4, 3, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (5, Oceananigans.MultiRegion.CubedSphereConnectivity(5, 2, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (6, Oceananigans.MultiRegion.CubedSphereConnectivity(6, 5, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (7, Oceananigans.MultiRegion.CubedSphereConnectivity(7, 4, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (8, Oceananigans.MultiRegion.CubedSphereConnectivity(8, 7, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (9, Oceananigans.MultiRegion.CubedSphereConnectivity(9, 4, :west, :north))
(i, (inject_west_boundary(i, part2, 1)).condition) = (10, Oceananigans.MultiRegion.CubedSphereConnectivity(10, 9, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (11, Oceananigans.MultiRegion.CubedSphereConnectivity(11, 3, :west, :north))
(i, (inject_west_boundary(i, part2, 1)).condition) = (12, Oceananigans.MultiRegion.CubedSphereConnectivity(12, 11, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (13, Oceananigans.MultiRegion.CubedSphereConnectivity(13, 10, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (14, Oceananigans.MultiRegion.CubedSphereConnectivity(14, 13, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (15, Oceananigans.MultiRegion.CubedSphereConnectivity(15, 12, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (16, Oceananigans.MultiRegion.CubedSphereConnectivity(16, 15, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (17, Oceananigans.MultiRegion.CubedSphereConnectivity(17, 12, :west, :north))
(i, (inject_west_boundary(i, part2, 1)).condition) = (18, Oceananigans.MultiRegion.CubedSphereConnectivity(18, 17, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (19, Oceananigans.MultiRegion.CubedSphereConnectivity(19, 11, :west, :north))
(i, (inject_west_boundary(i, part2, 1)).condition) = (20, Oceananigans.MultiRegion.CubedSphereConnectivity(20, 19, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (21, Oceananigans.MultiRegion.CubedSphereConnectivity(21, 18, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (22, Oceananigans.MultiRegion.CubedSphereConnectivity(22, 21, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (23, Oceananigans.MultiRegion.CubedSphereConnectivity(23, 20, :west, :east))
(i, (inject_west_boundary(i, part2, 1)).condition) = (24, Oceananigans.MultiRegion.CubedSphereConnectivity(24, 23, :west, :east))
"""
