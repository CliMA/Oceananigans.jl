using MPI
using Oceananigans.Grids: topology, size, halo_size, architecture, pop_flat_elements
using Oceananigans.Grids: validate_rectilinear_domain, validate_topology
using Oceananigans.Grids: validate_rectilinear_grid_args, validate_lat_lon_grid_args
using Oceananigans.Grids: generate_coordinate, with_precomputed_metrics
using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z

import Oceananigans.Grids: RectilinearGrid, LatitudeLongitudeGrid, with_halo

const DistributedGrid{FT, TX, TY, TZ} = AbstractGrid{FT, TX, TY, TZ, <:MultiArch}

@inline get_local_coords(c::Tuple         , nc, R, index) = (c[1] + (index-1) * (c[2] - c[1]) / R,    c[1] + index * (c[2] - c[1]) / R)
@inline get_local_coords(c::AbstractVector, nc, R, index) = c[1 + (index-1) * nc : 1 + nc * index]

@inline get_global_coords(c::Tuple        , nc, R,  index, arch) = (c[2] - index * (c[2] - c[1]), c[2] - (index - R) * (c[2] - c[1]))

"""
    get_global_coords(c_local::AbstractVector, nc, R, index, arch) 

Build a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `nc`, number of ranks `R`, rank `index`,
and `arch`itecture.
"""
function get_global_coords(c_local::AbstractVector, nc, R, index, arch) 
    c_global = zeros(eltype(c_local), nc*R+1)
    c_global[1 + (index-1) * nc : nc * index] .= c[1:end-1]
    index == R && (c_global[end] = c[end])

    MPI.Allreduce!(c_global, +, arch.communicator)

    return c_global
end

"""
    RectilinearGrid(arch::MultiArch, FT=Float64; kw...)

Return the rank-local portion of `RectilinearGrid` on `arch`itecture.
"""
function RectilinearGrid(arch::MultiArch, FT = Float64;
                         size,
                         x = nothing,
                         y = nothing,
                         z = nothing,
                         halo = nothing,
                         extent = nothing,
                         topology = (Periodic, Periodic, Bounded))

    TX, TY, TZ, size, halo, x, y, z =
        validate_rectilinear_grid_args(topology, size, halo, FT, extent, x, y, z)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    ri, rj, rk = arch.local_index
    Rx, Ry, Rz = arch.ranks

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    # Local sizes are denoted with lowercase `n`
    nx, ny, nz = local_size = Nx÷Rx, Ny÷Ry, Nz÷Rz

    xl = get_local_coords(x, nx, Rx, ri)
    yl = get_local_coords(y, ny, Ry, rj)
    zl = get_local_coords(z, nz, Rz, rk)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology[1], nx, Hx, xl, child_architecture(arch))
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology[2], ny, Hy, yl, child_architecture(arch))
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology[3], nz, Hz, zl, child_architecture(arch))

    architecture = MultiArch(child_architecture(arch), topology = topology, ranks = arch.ranks, communicator = arch.communicator)

    return RectilinearGrid{TX, TY, TZ}(architecture,
                                       nx, ny, nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

function LatitudeLongitudeGrid(arch::MultiArch,
                               FT = Float64; 
                               precompute_metrics = false,
                               size,
                               latitude,
                               longitude,
                               z,                      
                               radius = R_Earth,
                               halo = (1, 1, 1))

    Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, topo =
        validate_lat_lon_grid_args(latitude, longitude, size, halo)

    i, j, k    = arch.local_index
    Rx, Ry, Rz = arch.ranks

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nλ / Rx)
    @assert isinteger(Nφ / Ry)
    @assert isinteger(Nz / Rz)

    nλ, nφ, nz = local_size = Nλ÷Rx, Nφ÷Ry, Nz÷Rz

    λl = get_local_coords(longitude, nx, Rx, i)
    φl = get_local_coords(latitude , ny, Ry, j)
    zl = get_local_coords(z,         nz, Rz, k)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topo[1], nλ, Hλ, λl, arch.child_architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topo[2], nφ, Hφ, φl, arch.child_architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topo[3], nz, Hz, zl, arch.child_architecture)

    architecture = MultiArch(child_architecture(arch), grid = grid, ranks = arch.ranks, communicator = arch.communicator)

    preliminary_grid = LatitudeLongitudeGrid(architecture,
                                             Nλ, Nφ, Nz,
                                             Hλ, Hφ, Hz,
                                             Lλ, Lφ, Lz,
                                             Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                             Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                             Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                             (nothing for i=1:10)..., radius)

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

function reconstruct_global_grid(grid::RectilinearGrid)

    arch    = grid.architecture
    i, j, k = arch.local_index

    Rx, Ry, Rz = R = arch.ranks

    nx, ny, nz = n = size(grid)
    Hx, Hy, Hz = H = halo_size(grid)
    Nx, Ny, Nz = n .* R

    TX, TY, TZ = topology(grid)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    xG = get_global_coords(x, nx, Rx, i, arch)
    yG = get_global_coords(y, ny, Ry, j, arch)
    zG = get_global_coords(z, nz, Rz, k, arch)

    architecture = child_architecture(arch)

    FT = eltype(grid)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, TX, Nx, Hx, xG, architecture)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, TY, Ny, Hy, yG, architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, Nz, Hz, zG, architecture)

    return RectilinearGrid{TX, TY, TZ}(architecture,
                                       Nx, Ny, Nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
        
end

function with_halo(new_halo, grid::DistributedGrid) 
    new_grid = with_halo(new_halo, reconstruct_global_grid(grid))
    return scatter_local_grids(architecture(grid), new_grid)
end

function scatter_grid_properties(global_grid)
    # Pull out face grid constructors
    x = cpu_face_constructor_x(global_grid)
    y = cpu_face_constructor_y(global_grid)
    z = cpu_face_constructor_z(global_grid)

    topo = topology(grid)
    sz   = pop_flat_elements(size(global_grid), topo)
    halo = pop_flat_elements(halo_size(global_grid), topo)

    return x, y, z, topo, sz, halo
end

function scatter_local_grids(arch::MultiArch, global_grid::RectilinearGrid)
    x, y, z, topo, sz, halo = scatter_grid_properties(global_grid)
    return RectilinearGrid(arch, eltype(grid); size=sz, x=x, y=y, z=z, halo=halo, topology=topo)
end

function scatter_local_grids(arch::MultiArch, global_grid::LatitudeLongitudeGrid)
    x, y, z, topo, sz, halo = scatter_grid_properties(global_grid)
    return LatitudeLongitudeGrid(arch, eltype(global_grid); size=sz, longitude=x, latitude=y, z=z, halo=halo)
end
