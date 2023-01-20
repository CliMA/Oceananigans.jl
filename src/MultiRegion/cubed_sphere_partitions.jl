using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: CBC, PBC

struct CubedSpherePartition{M, P} <: AbstractPartition
    div :: Int
     Rx :: M
     Ry :: P

    CubedSpherePartition(div, Rx::M, Ry::P) where {M, P} = new{M, P}(div, Rx, Ry)
end


""""
    CubedSpherePartition(; Rx = 1, Ry = 1)

Return a cubed sphere partition.

* `Rx`: number of ``x``-division of each panel. Can be a number (i.e., all panels are divided in
        the same way) or a vector of length 6.
* `Ry`: number of ``y``-division of each panel. Can be a number (i.e., all panels are divided in
        the same way) or a vector of length 6.
"""
function CubedSpherePartition(; Rx = 1, Ry = 1)
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
@inline div_per_panel(panel_idx, p::RegularCubedSpherePartition)  = p.Rx           * p.Ry  
@inline div_per_panel(panel_idx, p::XRegularCubedSpherePartition) = p.Rx           * p.Ry[panel_idx]
@inline div_per_panel(panel_idx, p::YRegularCubedSpherePartition) = p.Rx[panel_idx] * p.Ry

@inline Rx(panel_idx, p::RegularCubedSpherePartition)  = p.Rx    
@inline Rx(panel_idx, p::XRegularCubedSpherePartition) = p.Rx    
@inline Rx(panel_idx, p::YRegularCubedSpherePartition) = p.Rx[panel_idx] 

@inline panel_index(r, p)         = r ÷ div_per_panel(r, p) + 1
@inline intra_panel_index(r, p)   = mod(r, div_per_panel(r, p)) + 1
@inline intra_panel_index_x(r, p) = mod(intra_panel_index(r, p), Rx(r, p)) + 1
@inline intra_panel_index_y(r, p) = intra_panel_index(r, p) ÷ Rx(r, p) + 1 

@inline rank_from_panel_idx(fi, fj, panel_idx, p::CubedSpherePartition) = panel_idx * div_per_panel(panel_idx, p) + Rx(panel_idx, p) * (fj - 1) + fi

@inline function region_corners(r, p::CubedSpherePartition)  

    fi = intra_panel_index_x(r, p)
    fj = intra_panel_index_y(r, p)

    bottom_left  = fi == 1              && fj == 1              ? true : false
    bottom_right = fi == p.div_per_side && fj == 1              ? true : false
    top_left     = fi == 1              && fj == p.div_per_side ? true : false
    top_right    = fi == p.div_per_side && fj == p.div_per_side ? true : false

    return (; bottom_left, bottom_right, top_left, top_right)
end

@inline function region_edge(r, p::CubedSpherePartition)  
     
    fi = intra_panel_index_x(r, p)
    fj = intra_panel_index_y(r, p)

    west  = fi == 1              ? true : false
    east  = fi == p.div_per_side ? true : false
    south = fj == 1              ? true : false
    north = fj == p.div_per_side ? true : false

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
#              |←3N      6W→|←3E      6S→| 
#              |------------|------------| 
#              |←3N      6W→|←3E      6S→| 
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
        
    fi = intra_panel_index_x(region, p)
    fj = intra_panel_index_y(region, p)

    panel_idx = panel_index(region, p)

    if fi == 1
        if mod(panel_idx, 2) == 0
            from_side = :east
            from_panel = panel_index - 1
            from_fi   = p.div_per_side
            from_fj   = fj
        else    
            from_side = :north
            from_panel = mod(panel_idx + 3, 6) + 1
            from_fi   = p.div_per_side - fj + 1
            from_fj   = p.div_per_side
        end
        from_rank = rank_from_panel_idx(from_fi, from_fj, from_panel, p)
    else
        from_side = :east
        from_rank = rank_from_panel_idx(fi - 1, fj, panel_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :west, from_side))

    return bc
end

function inject_east_boundary(region, p::CubedSpherePartition, global_bc) 
        
    fi = intra_panel_index_x(region, p)
    fj = intra_panel_index_y(region, p)

    panel_idx = panel_index(region, p)

    if fi == p.div_per_side
        if mod(panel_idx, 2) != 0
            from_side = :west
            from_panel = panel_index + 1
            from_fi   = 1
            from_fj   = fj
        else    
            from_side = :south
            from_panel = mod(panel_idx + 1, 6) + 1
            from_fi   = p.div_per_side - fj + 1
            from_fj   = 1
        end
        from_rank = rank_from_panel_idx(from_fi, from_fj, from_panel, p)
    else
        from_side = :west
        from_rank = rank_from_panel_idx(fi + 1, fj, panel_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :east, from_side))

    return bc
end

function inject_south_boundary(region, p::CubedSpherePartition, global_bc)
        
    fi = intra_panel_index_x(region, p)
    fj = intra_panel_index_y(region, p)

    panel_idx = panel_index(region, p)

    if fj == 1
        if mod(panel_idx, 2) != 0
            from_side = :north
            from_panel = mod(panel_index + 4, 6) + 1
            from_fi   = fi 
            from_fj   = p.div_per_side
        else    
            from_side = :east
            from_panel = mod(panel_idx + 3, 6) + 1
            from_fi   = p.div_per_side
            from_fj   = p.div_per_side - fi + 1
        end
        from_rank = rank_from_panel_idx(from_fi, from_fj, from_panel, p)
    else
        from_side = :north
        from_rank = rank_from_panel_idx(fi, fj - 1, panel_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))

    return bc
end

function inject_north_boundary(region, p::CubedSpherePartition, global_bc) 
        
    fi = intra_panel_index_x(region, p)
    fj = intra_panel_index_y(region, p)

    panel_idx = panel_index(region, p)

    if fj == p.div_per_side
        if mod(panel_idx, 2) == 0
            from_side = :south
            from_panel = mod(panel_index, 6) + 1
            from_fi   = fi 
            from_fj   = 1
        else    
            from_side = :west
            from_panel = mod(panel_idx + 1, 6) + 1
            from_fi   = 1
            from_fj   = p.div_per_side - fi + 1
        end
        from_rank = rank_from_panel_idx(from_fi, from_fj, from_panel, p)
    else
        from_side = :south
        from_rank = rank_from_panel_idx(fi, fj + 1, panel_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))

    return bc
end
