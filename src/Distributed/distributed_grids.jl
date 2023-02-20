using MPI
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: topology, size, halo_size, architecture, pop_flat_elements
using Oceananigans.Grids: validate_rectilinear_grid_args, validate_lat_lon_grid_args
using Oceananigans.Grids: generate_coordinate, with_precomputed_metrics
using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z
using Oceananigans.Grids: R_Earth, metrics_precomputed

using Oceananigans.ImmersedBoundaries

import Oceananigans.Grids: RectilinearGrid, LatitudeLongitudeGrid, with_halo

const DistributedGrid{FT, TX, TY, TZ} = AbstractGrid{FT, TX, TY, TZ, <:DistributedArch}
const DistributedRectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ} =
    RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, <:DistributedArch} where {FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ}
const DistributedLatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ} = 
    LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, <:DistributedArch} where {FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ}

const DistributedImmersedBoundaryGrid = ImmersedBoundaryGrid{FT, TX, TY, TZ, <:DistributedGrid, I, M, <:DistributedArch} where {FT, TX, TY, TZ, I, M}

"""
    RectilinearGrid(arch::DistributedArch, FT=Float64; kw...)

Return the rank-local portion of `RectilinearGrid` on `arch`itecture.
"""
function RectilinearGrid(arch::DistributedArch, 
                         FT::DataType = Float64;
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

    TX = insert_connected_topology(TX, Rx, ri)
    TY = insert_connected_topology(TY, Ry, rj)
    TZ = insert_connected_topology(TZ, Rz, rk)

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    # Local sizes are denoted with lowercase `n`
    nx, ny, nz = local_size = Nx÷Rx, Ny÷Ry, Nz÷Rz

    xl = partition(x, nx, Rx, ri)
    yl = partition(y, ny, Ry, rj)
    zl = partition(z, nz, Rz, rk)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology[1], nx, Hx, xl, child_architecture(arch))
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology[2], ny, Hy, yl, child_architecture(arch))
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology[3], nz, Hz, zl, child_architecture(arch))

    architecture = DistributedArch(child_architecture(arch), 
                                   topology = topology, 
                                   ranks = arch.ranks, 
                                   communicator = arch.communicator,
                                   use_buffers = using_buffered_communication(arch))

    return RectilinearGrid{TX, TY, TZ}(architecture,
                                       nx, ny, nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

"""
    LatitudeLongitudeGrid(arch::DistributedArch, FT=Float64; kw...)

Return the rank-local portion of `LatitudeLongitudeGrid` on `arch`itecture.
"""
function LatitudeLongitudeGrid(arch::DistributedArch,
                               FT::DataType = Float64; 
                               precompute_metrics = true,
                               size,
                               latitude,
                               longitude,
                               z,           
                               topology = nothing,           
                               radius = R_Earth,
                               halo = (1, 1, 1))

    Nλ, Nφ, Nz, Hλ, Hφ, Hz, latitude, longitude, z, topology, precompute_metrics =
        validate_lat_lon_grid_args(FT, latitude, longitude, z, size, halo, topology, precompute_metrics)
    
    ri, rj, rk = arch.local_index
    Rx, Ry, Rz = arch.ranks

    TX = insert_connected_topology(topology[1], Rx, ri)
    TY = insert_connected_topology(topology[2], Ry, rj)
    TZ = insert_connected_topology(topology[3], Rz, rk)

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nλ / Rx)
    @assert isinteger(Nφ / Ry)
    @assert isinteger(Nz / Rz)

    nλ, nφ, nz = local_size = Nλ÷Rx, Nφ÷Ry, Nz÷Rz

    λl = partition(longitude, nλ, Rx, ri)
    φl = partition(latitude,  nφ, Ry, rj)
    zl = partition(z,         nz, Rz, rk)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX, nλ, Hλ, λl, arch.child_architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, nz, Hz, zl, arch.child_architecture)
    # The Latitudinal direction is _special_ :
    # Preconmpute metrics assumes that `length(φᵃᶠᵃ) = length(φᵃᶜᵃ) + 1`, which is always the case in a 
    # serial grid because `LatitudeLongitudeGrid` should be always `Bounded`, but it is not true for a
    # partitioned `DistributedGrid` with Ry > 1 (one rank will hold a `RightConnected` topology)
    # But we need an extra point to precompute the Y direction in case of only one halo so we disregard the topology
    # when constructing the metrics!
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, Bounded, nφ, Hφ, φl, arch.child_architecture)

    architecture = DistributedArch(child_architecture(arch); 
                                   topology = topology, 
                                   ranks = arch.ranks, 
                                   communicator = arch.communicator,
                                   use_buffers = using_buffered_communication(arch))

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         nλ, nφ, nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                         (nothing for i=1:10)..., radius)

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

"""
    reconstruct_global_grid(grid::DistributedGrid)

Return the global grid on `child_architecture(grid)`
"""
function reconstruct_global_grid(grid::DistributedRectilinearGrid)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    Rx, Ry, Rz = R = arch.ranks

    nx, ny, nz = n = size(grid)
    Hx, Hy, Hz = H = halo_size(grid)
    Nx, Ny, Nz = n .* R

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, Rx, ri, rj, rk, arch.communicator)
    TY = reconstruct_global_topology(TY, Ry, rj, ri, rk, arch.communicator)
    TZ = reconstruct_global_topology(TZ, Rz, rk, ri, rj, arch.communicator)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    ## This will not work with 3D parallelizations!!
    xG = Rx == 1 ? x : assemble(x, nx, Rx, ri, rj, rk, arch.communicator)
    yG = Ry == 1 ? y : assemble(y, ny, Ry, rj, ri, rk, arch.communicator)
    zG = Rz == 1 ? z : assemble(z, nz, Rz, rk, ri, rj, arch.communicator)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, TX, Nx, Hx, xG, child_arch)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, TY, Ny, Hy, yG, child_arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, Nz, Hz, zG, child_arch)

    return RectilinearGrid{TX, TY, TZ}(child_arch,
                                       Nx, Ny, Nz,
                                       Hx, Hy, Hz,
                                       Lx, Ly, Lz,
                                       Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ,
                                       Δyᵃᶠᵃ, Δyᵃᶜᵃ, yᵃᶠᵃ, yᵃᶜᵃ,
                                       Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

function reconstruct_global_grid(grid::DistributedLatitudeLongitudeGrid)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    Rx, Ry, Rz = R = arch.ranks

    nλ, nφ, nz = n = size(grid)
    Hλ, Hφ, Hz = H = halo_size(grid)
    Nλ, Nφ, Nz = n .* R

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, Rx, ri, rj, rk, arch.communicator)
    TY = reconstruct_global_topology(TY, Ry, rj, ri, rk, arch.communicator)
    TZ = reconstruct_global_topology(TZ, Rz, rk, ri, rj, arch.communicator)

    λ = cpu_face_constructor_x(grid)
    φ = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    ## This will not work with 3D parallelizations!!
    λG = Rx == 1 ? λ : assemble(λ, nλ, Rx, ri, rj, rk, arch.communicator)
    φG = Ry == 1 ? φ : assemble(φ, nφ, Ry, rj, ri, rk, arch.communicator)
    zG = Rz == 1 ? z : assemble(z, nz, Rz, rk, ri, rj, arch.communicator)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, TX, Nλ, Hλ, λG, child_arch)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, TY, Nφ, Hφ, φG, child_arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, TZ, Nz, Hz, zG, child_arch)

    precompute_metrics = metrics_precomputed(grid)

    preliminary_grid = LatitudeLongitudeGrid{TX, TY, TZ}(child_arch,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                                                         (nothing for i=1:10)..., grid.radius)

    return !precompute_metrics ? preliminary_grid : with_precomputed_metrics(preliminary_grid)
end

function reconstruct_global_grid(grid::ImmersedBoundaryGrid)
    arch      = grid.architecture
    local_ib  = grid.immersed_boundary    
    global_ug = reconstruct_global_grid(grid.underlying_grid)
    global_ib = getnamewrapper(local_ib)(construct_global_array(arch, local_ib.bottom_height, size(grid)))
    return ImmersedBoundaryGrid(global_ug, global_ib)
end

# We _HAVE_ to dispatch individually for all grid types because
# `RectilinearGrid`, `LatitudeLongitudeGrid` and `ImmersedBoundaryGrid`
# take precedence on `DistributedGrid` 
function with_halo(new_halo, grid::DistributedRectilinearGrid) 
    new_grid = with_halo(new_halo, reconstruct_global_grid(grid))    
    return scatter_local_grids(architecture(grid), new_grid)
end

function with_halo(new_halo, grid::DistributedLatitudeLongitudeGrid) 
    new_grid = with_halo(new_halo, reconstruct_global_grid(grid))    
    return scatter_local_grids(architecture(grid), new_grid)
end

function with_halo(new_halo, grid::DistributedImmersedBoundaryGrid)
    global_immmersed_grid = reconstruct_global_grid(grid)
    immersed_boundary     = global_immmersed_grid.immersed_boundary
    underlying_grid       = with_halo(new_halo, global_immmersed_grid)
    immersed_boundary     = resize_immersed_boundary!(immersed_boundary, underlying_grid)
    new_grid              = ImmersedBoundaryGrid(underlying_grid, immersed_boundary)

    return scatter_local_grids(architecture(grid), new_grid)
end

resize_immersed_boundary!(ib::AbstractGridFittedBottom{<:Function}, grid) = ib

function resize_immersed_boundary!(ib::AbstractGridFittedBottom{<:OffsetArray}, grid)

    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    bottom_heigth_size = (Nx, Ny) .+ 2 .* (Hx, Hy)

    # Check that the size of a bottom field are 
    # consistent with the size of the field
    if any(size(ib.bottom_height) .!= bottom_heigth_size)
        @warn "Resizing the bottom field to match the grids' halos"
        bottom_field = Field((Center, Center, Nothing), grid)
        cpu_bottom   = arch_array(CPU(), ib.bottom_height)[1:Nx, 1:Ny] 
        set!(bottom_field, cpu_bottom)
        fill_halo_regions!(bottom_field)
        offset_bottom_array = dropdims(bottom_field.data, dims=3)

        return getnamewrapper(ib)(offset_bottom_array)
    end
    
    return ib
end

""" 
    scatter_grid_properties(global_grid)

returns individual `extent`, `topology`, `size` and `halo` of a `global_grid` 
"""
function scatter_grid_properties(global_grid)
    # Pull out face grid constructors
    x = cpu_face_constructor_x(global_grid)
    y = cpu_face_constructor_y(global_grid)
    z = cpu_face_constructor_z(global_grid)

    topo = topology(global_grid)
    sz   = pop_flat_elements(size(global_grid), topo)
    halo = pop_flat_elements(halo_size(global_grid), topo)

    return x, y, z, topo, sz, halo
end

function scatter_local_grids(arch::DistributedArch, global_grid::RectilinearGrid)
    x, y, z, topo, sz, halo = scatter_grid_properties(global_grid)
    return RectilinearGrid(arch, eltype(global_grid); size=sz, x=x, y=y, z=z, halo=halo, topology=topo)
end

function scatter_local_grids(arch::DistributedArch, global_grid::LatitudeLongitudeGrid)
    x, y, z, topo, sz, halo = scatter_grid_properties(global_grid)
    return LatitudeLongitudeGrid(arch, eltype(global_grid); size=sz, longitude=x, latitude=y, z=z, halo=halo, topology=topo)
end

function scatter_local_grids(arch::DistributedArch, global_grid::ImmersedBoundaryGrid)
    ib = global_grid.immersed_boundary
    ug = global_grid.underlying_grid

    local_ug = scatter_local_grids(arch, ug)
    local_ib = getnamewrapper(ib)(partition_global_array(arch, ib.bottom_height, size(global_grid)))
    
    return ImmersedBoundaryGrid(local_ug, local_ib)
end

""" 
    insert_connected_topology(T, R, r)

returns the local topology associated with the global topology `T`, the amount of ranks 
in `T` direction (`R`) and the local rank index `r` 
"""
insert_connected_topology(T, R, r) = T

insert_connected_topology(::Type{Bounded}, R, r) = ifelse(R == 1, Bounded,
                                                   ifelse(r == 1, RightConnected,
                                                   ifelse(r == R, LeftConnected,
                                                   FullyConnected)))

insert_connected_topology(::Type{Periodic}, R, r) = ifelse(R == 1, Periodic, FullyConnected)

""" 
    reconstruct_global_topology(T, R, r, comm)

reconstructs the global topology associated with the local topologies `T`, the amount of ranks 
in `T` direction (`R`) and the local rank index `r`. If all ranks hold a `FullyConnected` topology,
the global topology is `Periodic`, otherwise it is `Bounded`
"""
function reconstruct_global_topology(T, R, r, r1, r2, comm)
    if R == 1
        return T
    end
    
    topologies = zeros(Int, R)
    if T == FullyConnected && r1 == 1 && r2 == 1
        topologies[r] = 1
    end

    MPI.Allreduce!(topologies, +, comm)

    if sum(topologies) == R
        return Periodic
    else
        return Bounded
    end
end