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
      Δxᶠᶠᵃ :: M
      Δxᶜᶜᵃ :: M
      Δyᶠᶜᵃ :: MY
      Δyᶜᶠᵃ :: MY
      Azᶠᶜᵃ :: M
      Azᶜᶠᵃ :: M
      Azᶠᶠᵃ :: M
      Azᶜᶜᵃ :: M
    radius  :: FT
end

const LLGF  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing}
const LLGFX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number}
const LLGFY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Any, <:Number}
const LLGFB = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number, <:Number}
const LLGP  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any}
const LLGPX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const LLGPY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const LLGPB = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number, <:Number}

# latitude, longitude and z can be a 2-tuple that specifies the end of the domain (see RegularRectilinearDomain) or an array or function that specifies the faces (see VerticallyStretchedRectilinearGrid)

function LatitudeLongitudeGrid(FT=Float64; 
                               architecture=CPU(),
                               precompute_metrics=false,
                               size,
                               latitude,
                               longitude,
                               z,                      
                               radius=R_Earth,
                               halo=(1, 1, 1))

    arch = architecture
    
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
    Δxᶠᶠ = nothing
    Δxᶜᶜ = nothing
    Δyᶠᶜ = nothing
    Δyᶜᶠ = nothing
    Azᶠᶜ = nothing
    Azᶜᶠ = nothing
    Azᶠᶠ = nothing
    Azᶜᶜ = nothing
    M = MY = Nothing

    grid = LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(arch,
            Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶜᵃ, Δφᵃᶠᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
            Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ, radius)

    if precompute_metrics
        Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ = preallocate_metrics(FT, grid)
        precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ )
        Δyᶠᶜ, Δyᶜᶠ = precompute_Δy_metrics(grid, Δyᶠᶜ, Δyᶜᶠ)
        M  = typeof(Δxᶠᶜ)
        MY = typeof(Δyᶜᶠ)
    end

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(arch,
            Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶜᵃ, Δφᵃᶠᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
            Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Δyᶠᶜ, Δyᶜᶠ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ, radius)
end

function domain_string(grid::LatitudeLongitudeGrid)
    λ₁, λ₂ = domain(topology(grid, 1), grid.Nx, grid.λᶠᵃᵃ)
    φ₁, φ₂ = domain(topology(grid, 2), grid.Ny, grid.φᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "longitude λ ∈ [$λ₁, $λ₂], latitude ∈ [$φ₁, $φ₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::LatitudeLongitudeGrid{FT, TX, TY, TZ, M}) where {FT, TX, TY, TZ, M<:Nothing}
    print(io, "LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ} on the $(g.architecture) \n",
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
    print(io, "LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ}  on the $(g.architecture) \n",
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

Adapt.adapt_structure(to, grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    LatitudeLongitudeGrid{FT, TX, TY, TZ,
                            typeof(Adapt.adapt(to, grid.Δxᶠᶜᵃ)),
                            typeof(Adapt.adapt(to, grid.Δyᶜᶠᵃ)),
                            typeof(Adapt.adapt(to, grid.Δλᶠᵃᵃ)),
                            typeof(Adapt.adapt(to, grid.Δφᵃᶠᵃ)),
                            typeof(Adapt.adapt(to, grid.Δzᵃᵃᶠ)),
                            typeof(Adapt.adapt(to, grid.λᶠᵃᵃ)),
                            typeof(Adapt.adapt(to, grid.φᵃᶠᵃ)),
                            typeof(Adapt.adapt(to, grid.zᵃᵃᶠ)),
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
        Adapt.adapt(to, grid.Δxᶠᶠᵃ),
        Adapt.adapt(to, grid.Δxᶜᶜᵃ),
        Adapt.adapt(to, grid.Δyᶠᶜᵃ),
        Adapt.adapt(to, grid.Δyᶜᶠᵃ),
        Adapt.adapt(to, grid.Azᶠᶜᵃ),
        Adapt.adapt(to, grid.Azᶜᶠᵃ),
        Adapt.adapt(to, grid.Azᶠᶠᵃ),
        Adapt.adapt(to, grid.Azᶜᶜᵃ),
        grid.radius)

#####
##### Pre compute LatitudeLongitudeGrid metrics
#####

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

@inline Δxᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)   
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ[i])
@inline Δxᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶠᵃ[j]) * deg2rad(grid.Δλᶠᵃᵃ)
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ[i])
@inline Δxᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius * hack_cosd(grid.φᵃᶜᵃ[j]) * deg2rad(grid.Δλᶜᵃᵃ)   

@inline Δyᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ[j])
@inline Δyᶜᶠᵃ(i, j, k, grid::LLGFY) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶠᵃ)
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ[j])
@inline Δyᶠᶜᵃ(i, j, k, grid::LLGFY) = @inbounds grid.radius * deg2rad(grid.Δφᵃᶜᵃ)

@inline Azᶠᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶠᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ[i]) * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶠᶠᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶠᵃᵃ)    * (hack_sind(grid.φᵃᶜᵃ[j])   - hack_sind(grid.φᵃᶜᵃ[j-1]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGF)  = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ[i]) * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))
@inline Azᶜᶜᵃ(i, j, k, grid::LLGFX) = @inbounds grid.radius^2 * deg2rad(grid.Δλᶜᵃᵃ)    * (hack_sind(grid.φᵃᶠᵃ[j+1]) - hack_sind(grid.φᵃᶠᵃ[j]))

#######
####### Kernels to precompute the x- and z-metric
#######

@inline metric_worksize(grid::LLGF)   = (length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ) - 1) 
@inline metric_worksize(grid::LLGFX)  =  length(grid.φᵃᶜᵃ) - 1 
@inline metric_workgroup(grid::LLGF)  = (16, 16) 
@inline metric_workgroup(grid::LLGFX) =  16

function  precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    arch = grid.architecture
    precompute_curvilinear_metrics! = precompute_metrics_kernel!(Architectures.device(arch), metric_workgroup(grid), metric_worksize(grid))
    event = precompute_curvilinear_metrics!(grid, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ; dependencies=device_event(arch))
    wait(Architectures.device(arch), event)
    return nothing
end

@kernel function precompute_metrics_kernel!(grid::LLGF, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    i, j = @index(Global, NTuple)
    i += grid.Δλᶜᵃᵃ.offsets[1] 
    j += grid.φᵃᶜᵃ.offsets[1] + 1
    @inbounds begin
        Δxᶠᶜ[i, j] = Δxᶠᶜᵃ(i, j, 1, grid)
        Δxᶜᶠ[i, j] = Δxᶜᶠᵃ(i, j, 1, grid)
        Δxᶠᶠ[i, j] = Δxᶠᶠᵃ(i, j, 1, grid)
        Δxᶜᶜ[i, j] = Δxᶜᶜᵃ(i, j, 1, grid)
        Azᶠᶜ[i, j] = Azᶠᶜᵃ(i, j, 1, grid)
        Azᶜᶠ[i, j] = Azᶜᶠᵃ(i, j, 1, grid)
        Azᶠᶠ[i, j] = Azᶠᶠᵃ(i, j, 1, grid)
        Azᶜᶜ[i, j] = Azᶜᶜᵃ(i, j, 1, grid)
    end
end

@kernel function precompute_metrics_kernel!(grid::LLGFX, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δxᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ, Azᶜᶜ)
    j = @index(Global, Linear)
    j += grid.φᵃᶜᵃ.offsets[1] + 1
    @inbounds begin
        Δxᶠᶜ[j] = Δxᶠᶜᵃ(1, j, 1, grid)
        Δxᶜᶠ[j] = Δxᶜᶠᵃ(1, j, 1, grid)
        Δxᶠᶠ[j] = Δxᶠᶠᵃ(1, j, 1, grid)
        Δxᶜᶜ[j] = Δxᶜᶜᵃ(1, j, 1, grid)
        Azᶠᶜ[j] = Azᶠᶜᵃ(1, j, 1, grid)
        Azᶜᶠ[j] = Azᶜᶠᵃ(1, j, 1, grid)
        Azᶠᶠ[j] = Azᶠᶠᵃ(1, j, 1, grid)
        Azᶜᶜ[j] = Azᶜᶜᵃ(1, j, 1, grid)
    end
end

#######
####### Kernels that precompute the y-metric
#######

function  precompute_Δy_metrics(grid::LLGF, Δyᶠᶜ, Δyᶜᶠ)
    arch = grid.architecture
    precompute_Δy! = precompute_Δy_kernel!(Architectures.device(arch), 16, length(grid.Δφᵃᶜᵃ))
    event = precompute_Δy!(grid, Δyᶠᶜ, Δyᶜᶠ; dependencies=device_event(arch))
    wait(Architectures.device(arch), event)
    return Δyᶠᶜ, Δyᶜᶠ
end

function  precompute_Δy_metrics(grid::LLGFY, Δyᶠᶜ, Δyᶜᶠ)
    Δyᶜᶠ =  Δyᶜᶠᵃ(1, 1, 1, grid)
    Δyᶠᶜ =  Δyᶠᶜᵃ(1, 1, 1, grid)
    return Δyᶠᶜ, Δyᶜᶠ
end

@kernel function precompute_Δy_kernel!(grid, Δyᶠᶜ, Δyᶜᶠ)
    j = @index(Global, Linear)
    j += grid.Δφᵃᶜᵃ.offsets[1]
    @inbounds begin
        Δyᶜᶠ[j] = Δyᶜᶠᵃ(1, j, 1, grid)
        Δyᶠᶜ[j] = Δyᶜᶠᵃ(1, j, 1, grid)
    end
end

#######
####### Kernels that preallocate memory on the device for all metrics
#######


function preallocate_metrics(FT, grid::LLGF)
    
    # preallocate quantities to ensure correct type and size

    underlying_Δyᶠᶜᵃ = zeros(FT, length(grid.Δφᵃᶜᵃ))
    underlying_Δyᶜᶠᵃ = zeros(FT, length(grid.Δφᵃᶜᵃ))
    underlying_Δxᶠᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Δxᶠᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    
    Δyᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δyᶠᶜᵃ), grid.Δφᵃᶜᵃ.offsets[1])
    Δyᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δyᶜᶠᵃ), grid.Δφᵃᶜᵃ.offsets[1])
    
    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Δxᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    
    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function preallocate_metrics(FT, grid::LLGFX)
    
    # preallocate quantities to ensure correct type and size

    underlying_Δyᶠᶜᵃ = zeros(FT, length(grid.Δφᵃᶜᵃ))
    underlying_Δyᶜᶠᵃ = zeros(FT, length(grid.Δφᵃᶜᵃ))
    underlying_Δxᶠᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Δxᶠᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))

    Δyᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δyᶠᶜᵃ), grid.Δφᵃᶜᵃ.offsets[1])
    Δyᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δyᶜᶠᵃ), grid.Δφᵃᶜᵃ.offsets[1])

    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶜᵃ), grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Δxᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶜᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶜᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶜᵃ), grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function preallocate_metrics(FT, grid::LLGFY)
    
    # preallocate quantities to ensure correct type and size

    underlying_Δxᶠᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Δxᶠᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶠᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶜᵃ = zeros(FT, length(grid.Δλᶜᵃᵃ), length(grid.φᵃᶜᵃ))

    Δyᶠᶜᵃ = FT(0.0)
    Δyᶜᶠᵃ = FT(0.0)

    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Δxᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶠᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶜᵃ), grid.Δλᶜᵃᵃ.offsets[1], grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function preallocate_metrics(FT, grid::LLGFB)
    
    # preallocate quantities to ensure correct type and size
  
    Δyᶠᶜᵃ = FT(0.0)
    Δyᶜᶠᵃ = FT(0.0)
    underlying_Δxᶠᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Δxᶠᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Δxᶜᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶠᶠᵃ = zeros(FT, length(grid.φᵃᶜᵃ))
    underlying_Azᶜᶜᵃ = zeros(FT, length(grid.φᵃᶜᵃ))


    Δxᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶜᵃ), grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Δxᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶠᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Δxᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Δxᶜᶜᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶜᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶠᶠᵃ), grid.φᵃᶜᵃ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(arch_array(grid.architecture, underlying_Azᶜᶜᵃ), grid.φᵃᶜᵃ.offsets[1])

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end