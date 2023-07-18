using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices

using Rotations
using DocStringExtensions

import Oceananigans.Fields: replace_horizontal_velocity_halos!


"""
    rotation_from_panel_index(idx)

Return the rotation of each panel for the connectivity described in [`ConformalCubedSphereGrid`](@ref).
"""
rotation_from_panel_index(idx) = idx == 1 ? RotX(π/2) * RotY(π/2) :
                                 idx == 2 ? RotY(π) * RotX(-π/2) :
                                 idx == 3 ? RotZ(π) :
                                 idx == 4 ? RotX(π) * RotY(-π/2) :
                                 idx == 5 ? RotY(π/2) * RotX(π/2) :
                                 idx == 6 ? RotZ(π/2) * RotX(π) :
                                 error("invalid panel index")

default_rotations = (RotX(π/2) * RotY(π/2),
                     RotY(π) * RotX(-π/2),
                     RotZ(π),
                     RotX(π) * RotY(-π/2),
                     RotY(π/2) * RotX(π/2),
                     RotZ(π/2) * RotX(π))

struct CubedSphereConnectivity{C, R}
    connections :: C
    rotations :: R
end

function CubedSphereConnectivity(devices, partition::CubedSpherePartition, rotations::Tuple = default_rotations)
    regions = MultiRegionObject(Tuple(1:length(devices)), devices)
    rotations = MultiRegionObject(rotations, devices)
    @apply_regionally connectivity = find_regional_connectivities(regions, partition)

    return CubedSphereConnectivity(connectivity, rotations)
end


@inline getregion(connectivity::CubedSphereConnectivity, r) = _getregion(connectivity.connections, r)
@inline _getregion(connectivity::CubedSphereConnectivity, r) = getregion(connectivity.connections, r)

"""
    struct CubedSphereRegionalConnectivity{S, FS}

The connectivity among various regions for a cubed sphere grid. Parameters
`S` and `FS` denote the sides of the current region and the region from which
the boundary condition is coming from respectively.

$(TYPEDFIELDS)
"""
struct CubedSphereRegionalConnectivity{S <: AbstractRegionSide, FS <: AbstractRegionSide} <: AbstractConnectivity 
    "the current region rank"
            rank :: Int
    "the region from which boundary condition comes from"
       from_rank :: Int
    "the current region side"
            side :: S
    "the side of the region from which boundary condition comes from"
       from_side :: FS

    @doc """
        CubedSphereRegionalConnectivity(rank, from_rank, side, from_side)

    Return a `CubedSphereRegionalConnectivity`: `from_rank :: Int` → `rank :: Int` and
    `from_side :: AbstractRegionSide` → `side :: AbstractRegionSide`.

    Example
    =======

    A connectivity that implies that the boundary condition for the
    east side of region 1 comes from the west side of region 2 is:

    ```jldoctest cubedsphereconnectivity
    julia> using Oceananigans

    julia> using Oceananigans.MultiRegion: CubedSphereRegionalConnectivity, East, West, North, South

    julia> CubedSphereRegionalConnectivity(1, 2, East(), West())
    CubedSphereRegionalConnectivity{East, West}(1, 2, East(), West())
    ```

    A connectivity that implies that the boundary condition for the
    north side of region 1 comes from the east side of region 3 is 

    ```jldoctest cubedsphereconnectivity
    julia> CubedSphereRegionalConnectivity(1, 3, North(), East())
    CubedSphereRegionalConnectivity{North, East}(1, 3, North(), East())
    ```
    """
    CubedSphereRegionalConnectivity(rank, from_rank, side, from_side) = new{typeof(side), typeof(from_side)}(rank, from_rank, side, from_side)
end

"""
                                                         [5][6]
connectivity for a cubed sphere with configuration    [3][4]    or subdisions of this config.
                                                   [1][2]
"""

function find_west_connectivity(region, partition::CubedSpherePartition)
    pᵢ = intra_panel_index_x(region, partition)
    pⱼ = intra_panel_index_y(region, partition)

    pidx = panel_index(region, partition)

    if pᵢ == 1
        if mod(pidx, 2) == 0
            from_side  = East()
            from_panel = pidx - 1
            from_pᵢ    = Rx(from_panel, partition)
            from_pⱼ    = pⱼ
        else
            from_side  = North()
            from_panel = mod(pidx + 3, 6) + 1
            from_pᵢ    = Rx(from_panel, partition) - pⱼ + 1
            from_pⱼ    = Ry(from_panel, partition)
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, partition)
    else
        from_side = East()
        from_rank = rank_from_panel_idx(pᵢ - 1, pⱼ, pidx, partition)
    end

    return CubedSphereRegionalConnectivity(region, from_rank, West(), from_side)
end

function find_east_connectivity(region, partition::CubedSpherePartition)
    pᵢ = intra_panel_index_x(region, partition)
    pⱼ = intra_panel_index_y(region, partition)

    pidx = panel_index(region, partition)

    if pᵢ == partition.Rx
        if mod(pidx, 2) != 0
            from_side  = West()
            from_panel = pidx + 1
            from_pᵢ    = 1
            from_pⱼ    = pⱼ
        else
            from_side  = South()
            from_panel = mod(pidx + 1, 6) + 1
            from_pᵢ    = Rx(from_panel, partition) - pⱼ + 1
            from_pⱼ    = 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, partition)
    else
        from_side = West()
        from_rank = rank_from_panel_idx(pᵢ + 1, pⱼ, pidx, partition)
    end

    return CubedSphereRegionalConnectivity(region, from_rank, East(), from_side)
end

function find_south_connectivity(region, partition::CubedSpherePartition)
    pᵢ = intra_panel_index_x(region, partition)
    pⱼ = intra_panel_index_y(region, partition)

    pidx = panel_index(region, partition)

    if pⱼ == 1
        if mod(pidx, 2) != 0
            from_side  = North()
            from_panel = mod(pidx + 4, 6) + 1
            from_pᵢ    = pᵢ
            from_pⱼ    = Ry(from_panel, partition)
        else
            from_side  = East()
            from_panel = mod(pidx + 3, 6) + 1
            from_pᵢ    = Rx(from_panel, partition)
            from_pⱼ    = Ry(from_panel, partition) - pᵢ + 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, partition)
    else
        from_side = North()
        from_rank = rank_from_panel_idx(pᵢ, pⱼ - 1, pidx, partition)
    end

    return CubedSphereRegionalConnectivity(region, from_rank, South(), from_side)
end

function find_north_connectivity(region, partition::CubedSpherePartition)
    pᵢ = intra_panel_index_x(region, partition)
    pⱼ = intra_panel_index_y(region, partition)

    pidx = panel_index(region, partition)

    if pⱼ == partition.Ry
        if mod(pidx, 2) == 0
            from_side  = South()
            from_panel = mod(pidx, 6) + 1
            from_pᵢ    = pᵢ
            from_pⱼ    = 1
        else
            from_side  = West()
            from_panel = mod(pidx + 1, 6) + 1
            from_pᵢ    = 1
            from_pⱼ    = Rx(from_panel, partition) - pᵢ + 1
        end
        from_rank = rank_from_panel_idx(from_pᵢ, from_pⱼ, from_panel, partition)
    else
        from_side = South()
        from_rank = rank_from_panel_idx(pᵢ, pⱼ + 1, pidx, partition)
    end

    return CubedSphereRegionalConnectivity(region, from_rank, North(), from_side)
end

function find_regional_connectivities(region, partition::CubedSpherePartition)
    west =  find_west_connectivity(region, partition)
    east =  find_east_connectivity(region, partition)
    north = find_north_connectivity(region, partition)
    south = find_south_connectivity(region, partition)

    return (; west, east, north, south)
end

Base.summary(::CubedSphereConnectivity) = "CubedSphereConnectivity"

#####
##### Boundary-specific Utils
#####

"Trivial connectivities are East ↔ West, North ↔ South. Anything else is referred to as non-trivial."
const NonTrivialConnectivity = Union{CubedSphereRegionalConnectivity{East, South}, CubedSphereRegionalConnectivity{East, North},
                                     CubedSphereRegionalConnectivity{West, South}, CubedSphereRegionalConnectivity{West, North},
                                     CubedSphereRegionalConnectivity{South, East}, CubedSphereRegionalConnectivity{South, West},
                                     CubedSphereRegionalConnectivity{North, East}, CubedSphereRegionalConnectivity{North, West}}

@inline flip_west_and_east_indices(buff, conn) = buff
@inline flip_west_and_east_indices(buff, ::NonTrivialConnectivity) = reverse(permutedims(buff, (2, 1, 3)), dims = 2)

@inline flip_south_and_north_indices(buff, conn) = buff
@inline flip_south_and_north_indices(buff, ::NonTrivialConnectivity) = reverse(permutedims(buff, (2, 1, 3)), dims = 1)
