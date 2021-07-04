import Oceananigans.Architectures: architecture

const R_Earth = 6371.0e3    # Mean radius of the Earth [m] https://en.wikipedia.org/wiki/Earth

struct RegularLatitudeLongitudeGrid{FT, TX, TY, TZ, A} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}
        Nx :: Int
        Ny :: Int
        Nz :: Int
        Hx :: Int
        Hy :: Int
        Hz :: Int
        Lx :: FT
        Ly :: FT
        Lz :: FT
        Δλ :: FT
        Δφ :: FT
        Δz :: FT
      λᶠᵃᵃ :: A
      λᶜᵃᵃ :: A
      φᵃᶠᵃ :: A
      φᵃᶜᵃ :: A
      zᵃᵃᶠ :: A
      zᵃᵃᶜ :: A
    radius :: FT
end

function RegularLatitudeLongitudeGrid(FT=Float64; size, latitude, longitude, z, radius=R_Earth, halo=(1, 1, 1))
    @assert length(latitude) == 2
    @assert length(longitude) == 2
    @assert length(z) == 2

    λ₁, λ₂ = longitude
    @assert λ₁ < λ₂ && λ₂ - λ₁ ≤ 360

    φ₁, φ₂ = latitude
    @assert -90 <= φ₁ < φ₂ <= 90

    (φ₁ == -90 || φ₂ == 90) &&
        @warn "Are you sure you want to use a latitude-longitude grid with a grid point at the pole?"

    z₁, z₂ = z
    @assert z₁ < z₂

    Lλ = λ₂ - λ₁
    Lφ = φ₂ - φ₁
    Lz = z₂ - z₁

    TX = Lλ == 360 ? Periodic : Bounded
    TY = Bounded
    TZ = Bounded
    topo = (TX, TY, TZ)

    Nλ, Nφ, Nz = N = validate_size(TX, TY, TZ, size)
    Hλ, Hφ, Hz = H = validate_halo(TX, TY, TZ, halo)

            Λ₁ = (λ₁, φ₁, z₁)
            L  = (Lλ, Lφ, Lz)
    Δλ, Δφ, Δz = Δ = @. L / N

    # Calculate end points for cell faces and centers
    λF₋, φF₋, zF₋ = ΛF₋ = @. Λ₁ - H * Δ
    λF₊, φF₊, zF₊ = ΛF₊ = @. ΛF₋ + total_extent(topo, H, Δ, L)

    λC₋, φC₋, zC₋ = ΛC₋ = @. ΛF₋ + Δ / 2
    λC₊, φC₊, zC₊ = ΛC₊ = @. ΛC₋ + L + Δ * (2H - 1)

    TFλ, TFφ, TFz = total_length.(Face, topo, N, H)
    TCλ, TCφ, TCz = total_length.(Center, topo, N, H)

    λᶠᵃᵃ = range(λF₋, λF₊, length = TFλ)
    φᵃᶠᵃ = range(φF₋, φF₊, length = TFφ)
    zᵃᵃᶠ = range(zF₋, zF₊, length = TFz)

    λᶜᵃᵃ = range(λC₋, λC₊, length = TCλ)
    φᵃᶜᵃ = range(φC₋, φC₊, length = TCφ)
    zᵃᵃᶜ = range(zC₋, zC₊, length = TCz)

    λᶠᵃᵃ = OffsetArray(λᶠᵃᵃ, -Hλ)
    φᵃᶠᵃ = OffsetArray(φᵃᶠᵃ, -Hφ)
    zᵃᵃᶠ = OffsetArray(zᵃᵃᶠ, -Hz)

    λᶜᵃᵃ = OffsetArray(λᶜᵃᵃ, -Hλ)
    φᵃᶜᵃ = OffsetArray(φᵃᶜᵃ, -Hφ)
    zᵃᵃᶜ = OffsetArray(zᵃᵃᶜ, -Hz)

    return RegularLatitudeLongitudeGrid{FT, TX, TY, TZ, typeof(λᶠᵃᵃ)}(Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλ, Δφ, Δz, λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ, zᵃᵃᶠ, zᵃᵃᶜ, radius)
end

function domain_string(grid::RegularLatitudeLongitudeGrid)
    λ₁, λ₂ = domain(topology(grid, 1), grid.Nx, grid.λᶠᵃᵃ)
    φ₁, φ₂ = domain(topology(grid, 2), grid.Ny, grid.φᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "longitude λ ∈ [$λ₁, $λ₂], latitude ∈ [$φ₁, $φ₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::RegularLatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RegularLatitudeLongitudeGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δλ, Δφ, Δz): ", (g.Δλ, g.Δφ, g.Δz))
end

#####
##### << Spherical nodes >>
#####

# Node by node
@inline xnode(::Center, i, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.λᶜᵃᵃ[i]
@inline xnode(::Face,   i, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.λᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.φᵃᶜᵃ[j]
@inline ynode(::Face,   j, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.φᵃᶠᵃ[j]

@inline znode(::Center, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(::Face,   k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶜ[k]

all_x_nodes(::Type{Center}, grid::RegularLatitudeLongitudeGrid) = grid.λᶜᵃᵃ
all_x_nodes(::Type{Face},   grid::RegularLatitudeLongitudeGrid) = grid.λᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::RegularLatitudeLongitudeGrid) = grid.φᵃᶜᵃ
all_y_nodes(::Type{Face},   grid::RegularLatitudeLongitudeGrid) = grid.φᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::RegularLatitudeLongitudeGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face},   grid::RegularLatitudeLongitudeGrid) = grid.zᵃᵃᶠ

architecture(::RegularLatitudeLongitudeGrid) = nothing
