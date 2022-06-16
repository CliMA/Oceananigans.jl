using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: CBC, PBC

struct CubedSpherePartition{N, M, P} <: AbstractPartition
    div :: N
    div_per_face :: M
    div_per_side :: P
    function CubedSpherePartition(sizes) 
        if length(sizes) > 1 && all(y -> y == sizes[1], sizes)
            sizes = length(sizes)
        end
        div_per_face = (length(p) ÷ 6)
        div_per_side = √div_per_face
        return new{typeof(sizes), typeof(div_per_face)}(sizes, div_per_face)
    end
end

const RegularCubedSpherePartition = CubedSpherePartition{<:Number}

Base.length(p::CubedSpherePartition)        = length(p.div)
Base.length(p::RegularCubedSpherePartition) = p.div

@inline face_index(r, p::CubedSpherePartition)         = r ÷ p.div_per_face + 1
@inline intra_face_index(r, p::CubedSpherePartition)   = mod(r, p.div_per_face) + 1
@inline intra_face_index_x(r, p::CubedSpherePartition) = mod(intra_face_index(r, p), p.div_per_side) + 1
@inline intra_face_index_y(r, p::CubedSpherePartition) = intra_face_index(r, p) ÷ p.div_per_side + 1 

@inline rank_from_face_idx(fi, fj, face_idx, p::CubedSpherePartition) = face_idx * p.div_per_face + p.div_per_side * (fj  - 1) + fi

@inline function region_corners(r, p::CubedSpherePartition)  
 
    face_idx_x = intra_face_index_x(r, p)
    face_idx_y = intra_face_index_y(r, p)

    bottom_left  = face_idx_x == 1 && face_idx_y == 1 ? true : false
    bottom_right = face_idx_x == p.div_per_side && face_idx_y == 1 ? true : false
    top_left     = face_idx_x == 1 && face_idx_y == p.div_per_side ? true : false
    top_right    = face_idx_x == p.div_per_side && face_idx_y == p.div_per_side ? true : false

    return (; bottom_left, bottom_right, top_left, top_right)
end

@inline function region_edge(r, p::CubedSpherePartition)  
     
    fi = intra_face_index_x(r, p)
    fj = intra_face_index_y(r, p)

    west  = fi == 1              ? true : false
    east  = fi == p.div_per_side ? true : false
    south = fj == 1              ? true : false
    north = fj == p.div_per_side ? true : false

    return (; west, east, south, north)
end

### TO FIX FIGURE!

# See figure 8.4 of https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-6tile
#
#                         face  F5   face  F6
#                       +----------+----------+
#                       |          |          |
#                       |          |          |
#                       +----------+----------+
#                       |          |          |
#              face  F3 |          |          |
#            +----------+----------+----------+
#            |    ↑↑    |    ↑↑    |
#            |    5W    |    5S    |
#             ---------- ----------
#            |    2N    |    2E    |
#            |    ↓↓    |    ↓↓    |
# +----------+----------+----------+
# |    ↑↑    |    ↑↑    | face  F4
# |    3W    |    3S    |
# |←5N F1 2W→|←1E F2 4S→|
# |    6N    |    6E    |
# |    ↓↓    |    ↓↓    |
# +----------+----------+
#   face  F1   face  F2

#####
##### Boundary specific Utils
#####

struct CubedSphereConnectivity
    rank :: Int
    from_rank :: Int
    side :: Symbol
    from_side :: Symbol
end

function inject_west_boundary(region, p::CubedSpherePartition, global_bc) 
        
    fi = intra_face_index_x(r, p)
    fj = intra_face_index_y(r, p)

    face_idx = face_index(r, p)

    if fi == 1
        if mod(face_idx, 2) == 0
            from_side = :east
            from_face = face_index - 1
            from_fi   = p.div_per_side
            from_fj   = fj
        else    
            from_side = :north
            from_face = mod(face_idx + 3, 6) + 1
            from_fi   = p.div_per_side - fj + 1
            from_fj   = p.div_per_side
        end
        from_rank = rank_from_face_idx(from_fi, from_fj, from_face, p)
    else
        from_side = :east
        from_rank = rank_from_face_idx(fi - 1, fj, face_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :west, from_side))

    return bc
end

function inject_east_boundary(region, p::CubedSpherePartition, global_bc) 
        
    fi = intra_face_index_x(r, p)
    fj = intra_face_index_y(r, p)

    face_idx = face_index(r, p)

    if fi == p.div_per_side
        if mod(face_idx, 2) != 0
            from_side = :west
            from_face = face_index + 1
            from_fi   = 1
            from_fj   = fj
        else    
            from_side = :south
            from_face = mod(face_idx + 1, 6) + 1
            from_fi   = p.div_per_side - fj + 1
            from_fj   = 1
        end
        from_rank = rank_from_face_idx(from_fi, from_fj, from_face, p)
    else
        from_side = :west
        from_rank = rank_from_face_idx(fi + 1, fj, face_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :east, from_side))

    return bc
end

function inject_south_boundary(region, p::CubedSpherePartition, global_bc) 
        
    fi = intra_face_index_x(r, p)
    fj = intra_face_index_y(r, p)

    face_idx = face_index(r, p)

    if fj == 1
        if mod(face_idx, 2) != 0
            from_side = :north
            from_face = mod(face_index + 4, 6) + 1
            from_fi   = fi 
            from_fj   = p.div_per_side
        else    
            from_side = :east
            from_face = mod(face_idx + 3, 6) + 1
            from_fi   = p.div_per_side
            from_fj   = p.div_per_side - fi + 1
        end
        from_rank = rank_from_face_idx(from_fi, from_fj, from_face, p)
    else
        from_side = :north
        from_rank = rank_from_face_idx(fi, fj - 1, face_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))

    return bc
end

function inject_north_boundary(region, p::CubedSpherePartition, global_bc) 
        
    fi = intra_face_index_x(r, p)
    fj = intra_face_index_y(r, p)

    face_idx = face_index(r, p)

    if fj == p.div_per_side
        if mod(face_idx, 2) == 0
            from_side = :south
            from_face = mod(face_index, 6) + 1
            from_fi   = fi 
            from_fj   = 1
        else    
            from_side = :west
            from_face = mod(face_idx + 1, 6) + 1
            from_fi   = 1
            from_fj   = p.div_per_side - fi + 1
        end
        from_rank = rank_from_face_idx(from_fi, from_fj, from_face, p)
    else
        from_side = :south
        from_rank = rank_from_face_idx(fi, fj + 1, face_idx, p)
    end

    bc = CommunicationBoundaryCondition(CubedSphereConnectivity(region, from_rank, :south, from_side))

    return bc
end

####
#### Global index flattening
####

@inline function displaced_xy_index(i, j, grid, region, p::XPartition)
    i′ = i + grid.Nx * (region - 1) 
    t  = i′ + (j - 1) * grid.Nx * length(p)
    return t
end
