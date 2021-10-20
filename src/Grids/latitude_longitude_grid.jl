import Oceananigans.Architectures: architecture

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
    
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, topo[1], Nλ, Hλ, longitude, architecture)
    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, topo[2], Nφ, Hφ, latitude,  architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topo[3], Nz, Hz, z,         architecture)

    if precompute_metrics
        Δxᶠᶠᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ = generate_curvilinear_operators(FT, Δλᶠᵃᵃ, Δλᶜᵃᵃ, Δφᵃᶠᵃ, φᵃᶠᵃ, φᵃᶜᵃ, radius) 
        M  = typeof(Δxᶠᶠᵃ)
        MY = typeof(Δyᶜᶠᵃ)
    else
        Δxᶠᶠᵃ = nothing
        Δxᶜᶠᵃ = nothing
        Δyᶜᶠᵃ = nothing
        Azᶠᶠᵃ = nothing
        Azᶜᶜᵃ = nothing
        M = MY = Nothing
    end

    FX   = typeof(Δλᶠᵃᵃ)
    FY   = typeof(Δφᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(λᶠᵃᵃ)
    VY   = typeof(φᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)
    Arch = typeof(architecture) 

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, M, MY, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
            Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, Δφᵃᶜᵃ, Δφᵃᶠᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ,
            Δxᶠᶠᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ, radius)
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
        adapt_if_vector(to, grid.Δλᶠᵃᵃ),
        adapt_if_vector(to, grid.Δλᶜᵃᵃ),
        adapt_if_vector(to, grid.λᶠᵃᵃ),
        adapt_if_vector(to, grid.λᶜᵃᵃ),
        adapt_if_vector(to, grid.Δφᵃᶠᵃ),
        adapt_if_vector(to, grid.Δφᵃᶜᵃ),
        adapt_if_vector(to, grid.φᵃᶠᵃ),
        adapt_if_vector(to, grid.φᵃᶜᵃ),
        adapt_if_vector(to, grid.Δzᵃᵃᶠ),
        adapt_if_vector(to, grid.Δzᵃᵃᶜ),
        adapt_if_vector(to, grid.zᵃᵃᶠ),
        adapt_if_vector(to, grid.zᵃᵃᶜ),
        adapt_if_vector(to, grid.Δxᶠᶠᵃ),
        adapt_if_vector(to, grid.Δxᶜᶠᵃ),
        adapt_if_vector(to, grid.Δyᶜᶠᵃ),
        adapt_if_vector(to, grid.Azᶠᶠᵃ),
        adapt_if_vector(to, grid.Azᶜᶜᵃ))
