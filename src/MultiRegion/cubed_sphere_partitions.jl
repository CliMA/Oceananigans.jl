using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices
using Oceananigans.BoundaryConditions: CBC, PBC

struct CubedSpherePartition{M, P} <: AbstractPartition
               div :: Int
    div_per_side_x :: M
    div_per_side_y :: P
    function CubedSpherePartition(div, div_per_side_x::M, div_per_side_y::P) where {M, P}
        return new{M, P}(div, div_per_side_x, div_per_side_y)
    end
end

function CubedSpherePartition(; div_per_side_x = 1, div_per_side_y = 1) 

    if div_per_side_x isa Number 
        if div_per_side_y isa Number
            div_per_side_x != div_per_side_y && 
                    throw(ArgumentError("Regular cubed sphere must have div_per_side_x == div_per_side_y!!"))
            div = 6 * div_per_side_x * div_per_side_y
        else
            div = sum(div_per_side_y .* div_per_side_x)
        end
    else
        div = sum(div_per_side_y .* div_per_side_x)
    end
    
    div < 6 && throw(ArgumentError("Cubed sphere requires at least 6 regions!"))

    return CubedSpherePartition(div, div_per_side_x, div_per_side_y)
end

const RegularCubedSpherePartition  = CubedSpherePartition{<:Number, <:Number}
const XRegularCubedSpherePartition = CubedSpherePartition{<:Number}
const YRegularCubedSpherePartition = CubedSpherePartition{<:Any, <:Number}

Base.length(p::CubedSpherePartition) = p.div

@inline div_per_face(r, p::RegularCubedSpherePartition)  = p.div_per_side_x    * p.div_per_side_y  
@inline div_per_face(r, p::XRegularCubedSpherePartition) = p.div_per_side_x    * p.div_per_side_y[r]
@inline div_per_face(r, p::YRegularCubedSpherePartition) = p.div_per_side_x[r] * p.div_per_side_y

@inline div_per_side_x(r, p::RegularCubedSpherePartition)  = p.div_per_side_x    
@inline div_per_side_x(r, p::XRegularCubedSpherePartition) = p.div_per_side_x    
@inline div_per_side_x(r, p::YRegularCubedSpherePartition) = p.div_per_side_x[r] 

@inline face_index(r, p)         = r ÷ div_per_face(r, p) + 1
@inline intra_face_index(r, p)   = mod(r, div_per_face(r, p)) + 1
@inline intra_face_index_x(r, p) = mod(intra_face_index(r, p), div_per_side_x(r, p)) + 1
@inline intra_face_index_y(r, p) = intra_face_index(r, p) ÷ div_per_side_x(r, p) + 1 

@inline rank_from_face_idx(fi, fj, face_idx, p::CubedSpherePartition) = face_idx * div_per_face(r, p) + div_per_side_x(r, p) * (fj  - 1) + fi

@inline function region_corners(r, p::CubedSpherePartition)  
 
    fi = intra_face_index_x(r, p)
    fj = intra_face_index_y(r, p)

    bottom_left  = fi == 1              && fj == 1              ? true : false
    bottom_right = fi == p.div_per_side && fj == 1              ? true : false
    top_left     = fi == 1              && fj == p.div_per_side ? true : false
    top_right    = fi == p.div_per_side && fj == p.div_per_side ? true : false

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
        
    fi = intra_face_index_x(region, p)
    fj = intra_face_index_y(region, p)

    face_idx = face_index(region, p)

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
        
    fi = intra_face_index_x(region, p)
    fj = intra_face_index_y(region, p)

    face_idx = face_index(region, p)

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
        
    fi = intra_face_index_x(region, p)
    fj = intra_face_index_y(region, p)

    face_idx = face_index(region, p)

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
        
    fi = intra_face_index_x(region, p)
    fj = intra_face_index_y(region, p)

    face_idx = face_index(region, p)

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
