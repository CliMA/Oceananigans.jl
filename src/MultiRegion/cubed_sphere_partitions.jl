using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, default_indices

using DocStringExtensions

import Oceananigans.Fields: replace_horizontal_velocity_halos!

struct CubedSpherePartition{M, P} <: AbstractPartition
             div :: Int
              Rx :: M
              Ry :: P

    CubedSpherePartition(div, Rx::M, Ry::P) where {M, P} = new{M, P}(div, Rx, Ry)
end

"""
    CubedSpherePartition(; R = 1)

Return a cubed sphere partition with `R` partitions in each horizontal dimension of each
panel of the sphere.
"""
function CubedSpherePartition(; R = 1)
    # CubedSpherePartitions must have Rx = Ry
    Rx = Ry = R

    if R isa Number
        div = 6 * R^2
    else
        div = sum(R .* R)
    end

    @assert mod(div, 6) == 0 "Total number of regions (div = $div) must be a multiple of 6 for a cubed sphere partition."

    return CubedSpherePartition(div, Rx, Ry)
end

const RegularCubedSpherePartition  = CubedSpherePartition{<:Number, <:Number}
const XRegularCubedSpherePartition = CubedSpherePartition{<:Number}
const YRegularCubedSpherePartition = CubedSpherePartition{<:Any, <:Number}

Base.length(p::CubedSpherePartition) = p.div

"""
utilities to get the index of the panel, the index within the panel, and the global index
"""

@inline div_per_panel(panel_idx, partition::RegularCubedSpherePartition)  = partition.Rx            * partition.Ry
@inline div_per_panel(panel_idx, partition::XRegularCubedSpherePartition) = partition.Rx            * partition.Ry[panel_idx]
@inline div_per_panel(panel_idx, partition::YRegularCubedSpherePartition) = partition.Rx[panel_idx] * partition.Ry

@inline Rx(panel_idx, partition::XRegularCubedSpherePartition) = partition.Rx    
@inline Rx(panel_idx, partition::CubedSpherePartition)         = partition.Rx[panel_idx]

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

    bottom_left  = pᵢ == 1    && pⱼ == 1    ? true : false
    bottom_right = pᵢ == p.Rx && pⱼ == 1    ? true : false
    top_left     = pᵢ == 1    && pⱼ == p.Ry ? true : false
    top_right    = pᵢ == p.Rx && pⱼ == p.Ry ? true : false

    return (; bottom_left, bottom_right, top_left, top_right)
end

@inline function region_edge(r, p::CubedSpherePartition)
    pᵢ = intra_panel_index_x(r, p)
    pⱼ = intra_panel_index_y(r, p)

    west  = pᵢ == 1    ? true : false 
    east  = pᵢ == p.Rx ? true : false
    south = pⱼ == 1    ? true : false
    north = pⱼ == p.Ry ? true : false

    return (; west, east, south, north)
end

#####
##### Boundary-specific Utils
#####

replace_horizontal_velocity_halos!(::PrescribedVelocityFields, ::AbstractGrid; signed=true) = nothing

function replace_horizontal_velocity_halos!(velocities, grid::OrthogonalSphericalShellGrid{<:Any, FullyConnected, FullyConnected}; signed=true)
    u, v, _ = velocities

    ubuff = u.boundary_buffers
    vbuff = v.boundary_buffers

    conn_west  = u.boundary_conditions.west.condition.from_side
    conn_east  = u.boundary_conditions.east.condition.from_side
    conn_south = u.boundary_conditions.south.condition.from_side
    conn_north = u.boundary_conditions.north.condition.from_side

    Hx, Hy, _ = halo_size(u.grid)
    Nx, Ny, _ = size(grid)

     replace_west_u_halos!(parent(u), vbuff, Nx, Hx, conn_west; signed)
     replace_east_u_halos!(parent(u), vbuff, Nx, Hx, conn_east; signed)
    replace_south_u_halos!(parent(u), vbuff, Ny, Hy, conn_south; signed)
    replace_north_u_halos!(parent(u), vbuff, Ny, Hy, conn_north; signed)

     replace_west_v_halos!(parent(v), ubuff, Nx, Hx, conn_west; signed)
     replace_east_v_halos!(parent(v), ubuff, Nx, Hx, conn_east; signed)
    replace_south_v_halos!(parent(v), ubuff, Ny, Hy, conn_south; signed)
    replace_north_v_halos!(parent(v), ubuff, Ny, Hy, conn_north; signed)

    return nothing
end

for vel in (:u, :v), dir in (:east, :west, :north, :south)
    @eval $(Symbol(:replace_, dir, :_, vel, :_halos!))(u, buff, N, H, conn; signed=true) = nothing
end

function replace_west_u_halos!(u, vbuff, N, H, ::North; signed)
    view(u, 1:H, :, :) .= vbuff.west.recv
    return nothing
end

function replace_west_v_halos!(v, ubuff, N, H, ::North; signed)
    Nv = size(v, 2)
    view(v, 1:H, 2:Nv, :) .= view(ubuff.west.recv, :, 1:Nv-1, :)
    if signed
        view(v, 1:H, :, :) .*= -1
    end
    return nothing
end

function replace_east_u_halos!(u, vbuff, N, H, ::South; signed)
    view(u, N+1+H:N+2H, :, :) .= vbuff.east.recv
    return nothing
end

function replace_east_v_halos!(v, ubuff, N, H, ::South; signed)
    Nv = size(v, 2)
    view(v, N+1+H:N+2H, 2:Nv, :) .= view(ubuff.east.recv, :, 1:Nv-1, :)
    if signed
        view(v, N+1+H:N+2H, :, :) .*= -1
    end
    return nothing
end

function replace_south_u_halos!(u, vbuff, N, H, ::East; signed)
    Nu = size(u, 1)
    view(u, 2:Nu, 1:H, :) .= view(vbuff.south.recv, 1:Nu-1, :, :)
    if signed
       view(u, :, 1:H, :) .*= -1
    end
    return nothing
end

function replace_south_v_halos!(v, ubuff, N, H, ::East; signed)
    view(v, :, 1:H, :) .= + ubuff.south.recv
    return nothing
end

function replace_north_u_halos!(u, vbuff, N, H, ::West; signed)
    Nv = size(u, 1)
    view(u, 2:Nv, N+1+H:N+2H, :) .= view(vbuff.north.recv, 1:Nv-1, :, :)
    if signed
       view(u, :, N+1+H:N+2H, :) .*= -1
    end
    return nothing
end

function replace_north_v_halos!(v, ubuff, N, H, ::West; signed)
    view(v, :, N+1+H:N+2H, :) .= + ubuff.north.recv
    return nothing
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
