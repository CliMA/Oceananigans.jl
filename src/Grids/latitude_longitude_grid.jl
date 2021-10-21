import Oceananigans.Architectures: architecture
using KernelAbstractions: @kernel, @index

const R_Earth = 6371.0e3    # Mean radius of the Earth [m] https://en.wikipedia.org/wiki/Earth

struct LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}
        architecture::Arch
        Nx :: Int
        Ny :: Int
        Nz :: Int
        Hx :: Int
        Hy :: Int
        Hz :: Int
        Lx :: FT
        Ly :: FT
        Lz :: FT
      # All directions can be either regular (FX, FY, FZ) <: Number or stretched (FX, FY, FZ)<: AbstractVector
      Δλᶠᵃᵃ :: FX
      Δλᶜᵃᵃ :: FX
      λᶠᵃᵃ  :: VX
      λᶜᵃᵃ  :: VX
      Δφᵃᶠᵃ :: FY
      Δφᵃᶜᵃ :: FY
      φᵃᶠᵃ  :: VY
      φᵃᶜᵃ  :: VY
      Δzᵃᵃᶠ :: FZ 
      Δzᵃᵃᶜ :: FZ
      zᵃᵃᶠ  :: VZ
      zᵃᵃᶜ  :: VZ
      # Precomputed metrics M <: Nothing means the metrics will be computed on the fly
      Δxᶠᶜᵃ :: M
      Δxᶜᶠᵃ :: M
      Δyᶜᶠᵃ :: MY
      Azᶠᶠᵃ :: M
      Azᶜᶜᵃ :: M
    radius  :: FT
end

const LLGF  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing}
const LLGFX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number}
const LLGFY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Any, <:Number}
const LLGFB = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number, <:Number}

# latitude, longitude and z can be a 2-tuple that specifies the end of the domain (see RegularRectilinearDomain) or an array or function that specifies the faces (see VerticallyStretchedRectilinearGrid)

function LatitudeLongitudeGrid(FT=Float64; 
                               arch=CPU(),
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
    Hλ, Hφ, Hz = H = validate_halo(TX, TY, TZ, halo)
    
    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topo[1], Nλ, Hλ, longitude, arch)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topo[2], Nφ, Hφ, latitude,  arch)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topo[3], Nz, Hz, z,         arch)

    FX   = typeof(Δλᶠᵃᵃ)
    FY   = typeof(Δφᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(λᶠᵃᵃ)
    VY   = typeof(φᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)
    Arch = typeof(arch) 

    Δxᶠᶜ = nothing
    Δxᶜᶠ = nothing
    Δyᶜᶠ = nothing
    Azᶠᶠ = nothing
    Azᶜᶜ = nothing
    M = MY = Nothing

    grid = LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(arch,
            Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶜᵃ, Δφᵃᶠᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
            Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ, radius)

    if precompute_metrics
        Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ = preallocate_metrics(FT, grid)
        precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ)
        M  = typeof(Δxᶠᶜ)
        MY = typeof(Δyᶜᶠ)
    end

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(arch,
            Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶜᵃ, Δφᵃᶠᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
            Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ, radius)
end

function domain_string(grid::LatitudeLongitudeGrid)
    λ₁, λ₂ = domain(topology(grid, 1), grid.Nx, grid.λᶠᵃᵃ)
    φ₁, φ₂ = domain(topology(grid, 2), grid.Ny, grid.φᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "longitude λ ∈ [$λ₁, $λ₂], latitude ∈ [$φ₁, $φ₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::LatitudeLongitudeGrid{FT, TX, TY, TZ, M}) where {FT, TX, TY, TZ, M<:Nothing}
    print(io, "LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ} \n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid in λ: ", show_coordinate(g.Δλᶜᵃᵃ, TX), '\n',
              "grid in φ: ", show_coordinate(g.Δφᵃᶜᵃ, TY), '\n',
              "grid in z: ", show_coordinate(g.Δzᵃᵃᶜ, TZ), '\n',
              "metrics are computed on the fly")
end

function show(io::IO, g::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ} \n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid in λ: ", show_coordinate(g.Δλᶜᵃᵃ, TX), '\n',
              "grid in φ: ", show_coordinate(g.Δφᵃᶜᵃ, TY), '\n',
              "grid in z: ", show_coordinate(g.Δzᵃᵃᶜ, TZ), '\n',
              "metrics are pre-computed")
end

# Node by node
@inline xnode(::Center, i, grid::LatitudeLongitudeGrid) = @inbounds grid.λᶜᵃᵃ[i]
@inline xnode(::Face,   i, grid::LatitudeLongitudeGrid) = @inbounds grid.λᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶜᵃ[j]
@inline ynode(::Face,   j, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶠᵃ[j]

@inline znode(::Center, k, grid::LatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(::Face,   k, grid::LatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶜ[k]

all_x_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.λᶜᵃᵃ
all_x_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.λᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.φᵃᶜᵃ
all_y_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.φᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.zᵃᵃᶠ

architecture(::LatitudeLongitudeGrid) = nothing

@inline x_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.λᶠᵃᵃ)
@inline y_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.φᵃᶠᵃ)
@inline z_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

Adapt.adapt_structure(to, grid::LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ}) where {FT, TX, TY, TZ, M, MY, FX, FY, FZ} =
    LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ,
                            typeof(grid.λᶠᵃᵃ),
                            typeof(grid.φᵃᶠᵃ),
                            typeof(grid.zᵃᵃᶠ),
                            Nothing}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        Adapt.adapt(to, grid.Δλᶠᵃᵃ),
        Adapt.adapt(to, grid.Δλᶜᵃᵃ),
        Adapt.adapt(to, grid.λᶠᵃᵃ),
        Adapt.adapt(to, grid.λᶜᵃᵃ),
        Adapt.adapt(to, grid.Δφᵃᶠᵃ),
        Adapt.adapt(to, grid.Δφᵃᶜᵃ),
        Adapt.adapt(to, grid.φᵃᶠᵃ),
        Adapt.adapt(to, grid.φᵃᶜᵃ),
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.zᵃᵃᶠ),
        Adapt.adapt(to, grid.zᵃᵃᶜ),
        Adapt.adapt(to, grid.Δxᶠᶜᵃ),
        Adapt.adapt(to, grid.Δxᶜᶠᵃ),
        Adapt.adapt(to, grid.Δyᶜᶠᵃ),
        Adapt.adapt(to, grid.Azᶠᶠᵃ),
        Adapt.adapt(to, grid.Azᶜᶜᵃ),
        grid.radius)

#####
##### Pre compute LatitudeLongitudeGrid metrics
#####


@inline metric_worksize(grid::LLGF)  = (grid.Nx, grid.Ny) 
@inline metric_worksize(grid::LLGFY) = (grid.Nx, grid.Ny) 
@inline metric_worksize(grid::LLGFX) = (grid.Ny, ) 
@inline metric_worksize(grid::LLGFB) = (grid.Ny, ) 
@inline metric_workgroup(grid)        = (16, 16) 
@inline metric_workgroup(grid::LLGFX) = (16, 16) 
@inline metric_workgroup(grid::LLGFB) = (16, 16) 

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)                   

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGFY) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ)

@inline Azᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))


function  precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    arch = grid.architecture
    precompute_curvilinear_metrics! = precompute_metrics_kernel!(Oceananigans.Architectures.device(arch), metric_workgroup(grid), metric_worksize(grid))
    event = precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ; dependencies=device_event(arch))
    wait(Architectures.device(arch), event)
    return nothing
end

# kernel to pre_compute metrics
@kernel function precompute_metrics_kernel!(grid::LLGF, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Δxᶠᶜ[i, j] = Δxᶠᶜᵃ(i, j, 1, grid)
        Δxᶜᶠ[i, j] = Δxᶜᶠᵃ(i, j, 1, grid)
        Azᶠᶠ[i, j] = Azᶠᶠᵃ(i, j, 1, grid)
        Azᶜᶜ[i, j] = Azᶜᶜᵃ(i, j, 1, grid)
    end
    if i == 1
        Δyᶜᶠ[j]    = Δyᶜᶠᵃ(i, j, 1, grid)
    end
end

@kernel function precompute_metrics_kernel!(grid::LLGFX, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    j = @index(Global, NTuple)
    @inbounds begin
        Δxᶠᶜ[j] = Δxᶠᶜᵃ(1, j[1], 1, grid)
        Δxᶜᶠ[j] = Δxᶜᶠᵃ(1, j[1], 1, grid)
        Δyᶜᶠ[j] = Δyᶜᶠᵃ(1, j[1], 1, grid)
        Azᶠᶠ[j] = Azᶠᶠᵃ(1, j[1], 1, grid)
        Azᶜᶜ[j] = Azᶜᶜᵃ(1, j[1], 1, grid)
    end
end

@kernel function precompute_metrics_kernel!(grid::LLGFY, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Δxᶠᶜ[i, j] = Δxᶠᶜᵃ(i, j, 1, grid)
        Δxᶜᶠ[i, j] = Δxᶜᶠᵃ(i, j, 1, grid)
        Azᶠᶠ[i, j] = Azᶠᶠᵃ(i, j, 1, grid)
        Azᶜᶜ[i, j] = Azᶜᶜᵃ(i, j, 1, grid)
    end
    if j == 1 && i == 1
        Δyᶜᶠ       = Δyᶜᶠᵃ(1, 1, 1, grid)
    end
end

@kernel function precompute_metrics_kernel!(grid::LLGFB, Δxᶠᶜ, Δxᶜᶠ, Δyᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    j = @index(Global, NTuple)
    @inbounds begin
        Δxᶠᶜ[j] = Δxᶠᶜᵃ(1, j[1], 1, grid)
        Δxᶜᶠ[j] = Δxᶜᶠᵃ(1, j[1], 1, grid)
        Azᶠᶠ[j] = Azᶠᶠᵃ(1, j[1], 1, grid)
        Azᶜᶜ[j] = Azᶜᶜᵃ(1, j[1], 1, grid)
    end
    if j[1] == 1
        Δyᶜᶠ    = Δyᶜᶠᵃ(1, j[1], 1, grid)
    end
end


function preallocate_metrics(FT, grid::LLGF)
    
    # preallocate quantities to ensure correct type and size

    Δyᶜᶠᵃ_underlying = zeros(FT, length(grid.Δφᵃᶠᵃ))
    Δxᶜᶠᵃ_underlying = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶠᵃ))
    Δxᶠᶜᵃ_underlying = zeros(FT, length(grid.Δλᶠᵃᵃ), length(grid.φᵃᶜᵃ))
    Azᶠᶠᵃ_underlying = zeros(FT, length(grid.Δλᶠᵃᵃ), length(grid.φᵃᶜᵃ))
    Azᶜᶜᵃ_underlying = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))

    Δyᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, Δyᶜᶠᵃ_underlying), grid.Δφᵃᶠᵃ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, Δxᶜᶠᵃ_underlying), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶠᵃ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, Δxᶠᶜᵃ_underlying), grid.Δλᶠᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, Azᶠᶠᵃ_underlying), grid.Δλᶠᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1]+1)
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, Azᶜᶜᵃ_underlying), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function preallocate_metrics(FT, grid::LLGFX)
    
    # preallocate quantities to ensure correct type and size

    Δyᶜᶠᵃ_underlying = zeros(FT, length(grid.Δφᵃᶠᵃ))
    Δxᶜᶠᵃ_underlying = zeros(FT, length(grid.φᵃᶠᵃ))
    Δxᶠᶜᵃ_underlying = zeros(FT, length(grid.φᵃᶜᵃ))
    Azᶠᶠᵃ_underlying = zeros(FT, length(grid.φᵃᶜᵃ))
    Azᶜᶜᵃ_underlying = zeros(FT, length(grid.φᵃᶜᵃ))

    Δyᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, Δyᶜᶠᵃ_underlying), grid.Δφᵃᶠᵃ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, Δxᶜᶠᵃ_underlying), grid.φᵃᶠᵃ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, Δxᶠᶜᵃ_underlying), grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, Azᶠᶠᵃ_underlying), grid.φᵃᶜᵃ.offsets[1]+1)
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, Azᶜᶜᵃ_underlying), grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function preallocate_metrics(FT, grid::LLGFY)
    
    # preallocate quantities to ensure correct type and size

    Δxᶜᶠᵃ_underlying = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶠᵃ))
    Δxᶠᶜᵃ_underlying = zeros(FT, length(grid.Δλᶠᵃᵃ), length(grid.φᵃᶜᵃ))
    Azᶠᶠᵃ_underlying = zeros(FT, length(grid.Δλᶠᵃᵃ), length(grid.φᵃᶜᵃ))
    Azᶜᶜᵃ_underlying = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))

    Δyᶜᶠᵃ = FT(0.0)
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, Δxᶜᶠᵃ_underlying), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶠᵃ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, Δxᶠᶜᵃ_underlying), grid.Δλᶠᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, Azᶠᶠᵃ_underlying), grid.Δλᶠᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1]+1)
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, Azᶜᶜᵃ_underlying), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function preallocate_metrics(FT, grid::LLGFB)
    
    # preallocate quantities to ensure correct type and size

    Δxᶜᶠᵃ_underlying = zeros(FT, length(grid.φᵃᶠᵃ))
    Δxᶠᶜᵃ_underlying = zeros(FT, length(grid.φᵃᶜᵃ))
    Azᶠᶠᵃ_underlying = zeros(FT, length(grid.φᵃᶜᵃ))
    Azᶜᶜᵃ_underlying = zeros(FT, length(grid.φᵃᶜᵃ))

    Δyᶜᶠᵃ = FT(0.0)
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, Δxᶜᶠᵃ_underlying), grid.φᵃᶠᵃ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, Δxᶠᶜᵃ_underlying), grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, Azᶠᶠᵃ_underlying), grid.φᵃᶜᵃ.offsets[1]+1)
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, Azᶜᶜᵃ_underlying), grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end