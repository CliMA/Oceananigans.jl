using MPI
using Oceananigans.Grids: validate_halo, validate_rectilinear_domain, validate_size, validate_topology, topology, size, halo_size, architecture
using Oceananigans.Grids: generate_coordinate, cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, pop_flat_elements

import Oceananigans.Grids: AbstractGrid, RectilinearGrid, LatitudeLongitudeGrid, with_halo


@inline get_local_coords(c::Tuple         , nc, R, index) = (c[1] + (index-1) * (c[2] - c[1]) / R,    c[1] + index * (c[2] - c[1]) / R)
@inline get_local_coords(c::AbstractVector, nc, R, index) = c[1 + (index-1) * nc : 1 + nc * index]

@inline get_global_coords(c::Tuple        , nc, R,  index, arch) = (c[2] - index * (c[2] - c[1]), c[2] - (index - R) * (c[2] - c[1]))

function get_global_coords(c::AbstractVector, nc, R, index, arch) 
    cG = zeros(eltype(c), nc*R+1)
    cG[1 + (index-1) * nc : nc * index] .= c[1:end-1]
    
    if index == R
        cG[end] = c[end]
    end
    MPI.Allreduce!(cG, +, arch.communicator)

    return cG
end

function RectilinearGrid(arch::MultiArch, FT = Float64;
                         size,
                         x = nothing,
                         y = nothing,
                         z = nothing,
                         halo = nothing,
                         extent = nothing,
                         topology = (Periodic, Periodic, Bounded))

    TX, TY, TZ = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)

    # Validate the rectilinear domain
    x, y, z = validate_rectilinear_domain(TX, TY, TZ, FT, extent, x, y, z)

    Nx, Ny, Nz = size
    hx, hy, hz = halo

    i, j, k    = arch.local_index
    Rx, Ry, Rz = arch.ranks

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    nx, ny, nz = local_size = Nx÷Rx, Ny÷Ry, Nz÷Rz

    xl = get_local_coords(x, nx, Rx, i)
    yl = get_local_coords(y, ny, Ry, j)
    zl = get_local_coords(z, nz, Rz, k)

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology[1], nx, hx, xl, arch.child_arch)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology[2], ny, hy, yl, arch.child_arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology[3], nz, hz, zl, arch.child_arch)

    FX   = typeof(Δxᶠᵃᵃ)
    FY   = typeof(Δyᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(xᶠᵃᵃ)
    VY   = typeof(yᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)

    architecture = MultiArch(child_architecture(arch), topology = topology, ranks = arch.ranks, communicator = arch.communicator)

    Arch = typeof(arch) 

    return RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
        nx, ny, nz, hx, hy, hz, Lx, Ly, Lz, Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

function LatitudeLongitudeGrid(arch::MultiArch,
    FT=Float64; 
    precompute_metrics=false,
    size,
    latitude,
    longitude,
    z,                      
    radius=R_Earth,
    halo=(1, 1, 1))

    λ₁, λ₂ = get_domain_extent(longitude, size[1])
    @assert λ₁ < λ₂ && λ₂ - λ₁ ≤ 360

    φ₁, φ₂ = get_domain_extent(latitude, size[2])
    @assert -90 <= φ₁ < φ₂ <= 90

    (φ₁ == -90 || φ₂ == 90) &&
    @warn "Are you sure you want to use a latitude-longitude grid with a grid point at the pole?"

    Lλ = λ₂ - λ₁
    Lφ = φ₂ - φ₁

    TX = Lλ == 360 ? Periodic : Bounded
    TY = Bounded
    TZ = Bounded
    topo = (TX, TY, TZ)

    Nλ, Nφ, Nz = N = validate_size(TX, TY, TZ, size)
    hλ, hφ, hz = H = validate_halo(TX, TY, TZ, halo)

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)

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

    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topo[1], nλ, hλ, λl, arch.child_architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topo[2], nφ, hφ, φl, arch.child_architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topo[3], nz, hz, zl, arch.child_architecture)

    FX   = typeof(Δλᶠᵃᵃ)
    FY   = typeof(Δφᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(λᶠᵃᵃ)
    VY   = typeof(φᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)

    architecture = MultiArch(child_architecture(arch), grid = grid, ranks = arch.ranks, communicator = arch.communicator)

    Arch = typeof(architecture) 


    if precompute_metrics == true
        grid = LatitudeLongitudeGrid{FT, TX, TY, TZ, Nothing, Nothing, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
                nλ, nφ, nz, hλ, hφ, hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
                nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, radius)

        Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ = allocate_metrics(FT, grid)
        wait(device_event(architecture))

        precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ )
        wait(device_event(architecture))

        Δyᶠᶜ, Δyᶜᶠ = precompute_Δy_metrics(grid, Δyᶠᶜ, Δyᶜᶠ)

        M  = typeof(Δxᶠᶜ)
        MY = typeof(Δyᶠᶜ)
    else
        metrics = (:Δxᶠᶜ, :Δxᶜᶠ, :Δxᶠᶠ, :Δxᶜᶜ, :Δyᶠᶜ, :Δyᶜᶠ, :Azᶠᶜ, :Azᶜᶠ, :Azᶠᶠ, :Azᶜᶜ)
        for metric in metrics
            @eval $metric = nothing
        end
        M    = Nothing
        MY   = Nothing
    end

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
    nλ, nφ, nz, hλ, hφ, hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
    Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ, radius)
end

function reconstruct_global_grid(grid)

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

    FX   = typeof(Δxᶠᵃᵃ)
    FY   = typeof(Δyᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(xᶠᵃᵃ)
    VY   = typeof(yᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)
    Arch = typeof(architecture) 

    return RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
    Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)

end

function with_halo(new_halo, grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:MultiArch}) 
    new_grid  = with_halo(new_halo, reconstruct_global_grid(grid))
    return scatter_local_grids(architecture(grid), new_grid)
end

function scatter_local_grids(arch::MultiArch, grid::RectilinearGrid)

    # Pull out face grid constructors
    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    topo = topology(grid)
    N    = pop_flat_elements(size(grid), topo)
    halo = pop_flat_elements(halo_size(grid), topo)

    local_grid = RectilinearGrid(arch, eltype(grid); size = N, x = x, y = y, z = z, halo = halo, topology = topo)

    return local_grid
end

function scatter_local_grids(arch::MultiArch, grid::LatitudeLongitudeGrid)
    
    # Pull out face grid constructors
    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    topo = topology(grid)
    N    = pop_flat_element(size(grid), topo)
    halo = pop_flat_elements(halo_size(grid), topo)

    local_grid = LatitudeLongitudeGrid(arch, eltype(grid); size = N, longitude = x, latitude = y, z = z, halo = halo)
    
    return local_grid
end
