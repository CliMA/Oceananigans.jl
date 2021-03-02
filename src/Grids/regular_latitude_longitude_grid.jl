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
        Δϕ :: FT
        Δz :: FT
      λᶠᵃᵃ :: A
      λᶜᵃᵃ :: A
      ϕᵃᶠᵃ :: A
      ϕᵃᶜᵃ :: A
      zᵃᵃᶠ :: A
      zᵃᵃᶜ :: A
    radius :: FT
end

function RegularLatitudeLongitudeGrid(FT=Float64; size, latitude, longitude, z, radius=R_Earth, halo=(1, 1, 1))
    @assert length(latitude) == 2
    @assert length(longitude) == 2
    @assert length(z) == 2

    λ₁, λ₂ = longitude
    @assert -180 <= λ₁ < λ₂ <= 180

    ϕ₁, ϕ₂ = latitude
    @assert -90 <= ϕ₁ < ϕ₂ <= 90

    (ϕ₁ == -90 || ϕ₂ == 90) &&
        @warn "Are you sure you want to use a latitude-longitude grid with a grid point at the pole?"

    z₁, z₂ = z
    @assert z₁ < z₂

    TX = λ₁ == -180 && λ₂ == 180 ? Periodic : Bounded
    TY = Bounded
    TZ = Bounded
    topo = (TX, TY, TZ)

    Nλ, Nϕ, Nz = N = validate_size(TX, TY, TZ, size)
    Hλ, Hϕ, Hz = H = validate_halo(TX, TY, TZ, halo)

    Lλ = λ₂ - λ₁
    Lϕ = ϕ₂ - ϕ₁
    Lz = z₂ - z₁

            Λ₁ = (λ₁, ϕ₁, z₁)
            L  = (Lλ, Lϕ, Lz)
    Δλ, Δϕ, Δz = Δ = @. L / N

    # Calculate end points for cell faces and centers
    λF₋, ϕF₋, zF₋ = ΛF₋ = @. Λ₁ - H * Δ
    λF₊, ϕF₊, zF₊ = ΛF₊ = @. ΛF₋ + total_extent(topo, H, Δ, L)

    λC₋, ϕC₋, zC₋ = ΛC₋ = @. ΛF₋ + Δ / 2
    λC₊, ϕC₊, zC₊ = ΛC₊ = @. ΛC₋ + L + Δ * (2H - 1)

    TFλ, TFϕ, TFz = total_length.(Face, topo, N, H)
    TCλ, TCϕ, TCz = total_length.(Center, topo, N, H)

    λᶠᵃᵃ = range(λF₋, λF₊, length = TFλ)
    ϕᵃᶠᵃ = range(ϕF₋, ϕF₊, length = TFϕ)
    zᵃᵃᶠ = range(zF₋, zF₊, length = TFz)

    λᶜᵃᵃ = range(λC₋, λC₊, length = TCλ)
    ϕᵃᶜᵃ = range(ϕC₋, ϕC₊, length = TCϕ)
    zᵃᵃᶜ = range(zC₋, zC₊, length = TCz)

    λᶠᵃᵃ = OffsetArray(λᶠᵃᵃ, -Hλ)
    ϕᵃᶠᵃ = OffsetArray(ϕᵃᶠᵃ, -Hϕ)
    zᵃᵃᶠ = OffsetArray(zᵃᵃᶠ, -Hz)

    λᶜᵃᵃ = OffsetArray(λᶜᵃᵃ, -Hλ)
    ϕᵃᶜᵃ = OffsetArray(ϕᵃᶜᵃ, -Hϕ)
    zᵃᵃᶜ = OffsetArray(zᵃᵃᶜ, -Hz)

    return RegularLatitudeLongitudeGrid{FT, TX, TY, TZ, typeof(λᶠᵃᵃ)}(Nλ, Nϕ, Nz, Hλ, Hϕ, Hz, Lλ, Lϕ, Lz, Δλ, Δϕ, Δz, λᶠᵃᵃ, λᶜᵃᵃ, ϕᵃᶠᵃ, ϕᵃᶜᵃ, zᵃᵃᶠ, zᵃᵃᶜ, radius)
end

function domain_string(grid::RegularLatitudeLongitudeGrid)
    λ₁, λ₂ = domain(topology(grid, 1), grid.Nx, grid.λᶠᵃᵃ)
    ϕ₁, ϕ₂ = domain(topology(grid, 2), grid.Ny, grid.ϕᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "longitude λ ∈ [$λ₁, $λ₂], latitude ∈ [$ϕ₁, $ϕ₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::RegularLatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RegularLatitudeLongitudeGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δλ, Δϕ, Δz): ", (g.Δλ, g.Δϕ, g.Δz))
end

#####
##### << Spherical nodes >>
#####

# Node by node
@inline xnode(::Type{Center}, i, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.λᶜᵃᵃ[i]
@inline xnode(::Type{Face},   i, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.λᶠᵃᵃ[i]

@inline ynode(::Type{Center}, j, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.ϕᵃᶜᵃ[j]
@inline ynode(::Type{Face},   j, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.ϕᵃᶠᵃ[j]

@inline znode(::Type{Center}, k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(::Type{Face},   k, grid::RegularLatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶜ[k]

all_x_nodes(::Type{Center}, grid::RegularLatitudeLongitudeGrid) = grid.λᶜᵃᵃ
all_x_nodes(::Type{Face},   grid::RegularLatitudeLongitudeGrid) = grid.λᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::RegularLatitudeLongitudeGrid) = grid.ϕᵃᶜᵃ
all_y_nodes(::Type{Face},   grid::RegularLatitudeLongitudeGrid) = grid.ϕᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::RegularLatitudeLongitudeGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face},   grid::RegularLatitudeLongitudeGrid) = grid.zᵃᵃᶠ
