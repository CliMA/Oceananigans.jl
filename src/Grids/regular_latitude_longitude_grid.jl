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

    z₁, z₂ = z
    @assert z₁ < z₂

    TX = ϕ₁ == -180 && ϕ₂ == 180 ? Periodic : Bounded
    TY = Bounded
    TZ = Bounded
    topo = (TX, TY, TZ)

    Nx, Ny, Nz = N = validate_size(TX, TY, TZ, size)
    Hx, Hy, Hz = H = validate_halo(TX, TY, TZ, halo)

    Lλ = λ₂ - λ₁
    Lϕ = ϕ₂ - ϕ₁
    Lz = z₂ - z₁

    Δϕ = Lλ / Nx
    Δλ = Lϕ / Ny
    Δz = Lz / Nz

    # Now including halos
    λF₋ = λ₁ - Hx * Δλ
    ϕF₋ = ϕ₁ - Hy * Δϕ
    zF₋ = z₁ - Hz * Δz

    λF₊ = λ₂ + Hx * Δλ
    ϕF₊ = ϕ₂ + Hy * Δϕ
    zF₊ = z₂ + Hz * Δz

    λC₋ = λF₋ + Δλ / 2
    ϕC₋ = ϕF₋ + Δϕ / 2
    zC₋ = zF₋ + Δz / 2

    λC₊ = λF₋ + total_extent(TX, Hx, Δλ, Lλ)
    ϕC₊ = ϕF₋ + total_extent(TY, Hy, Δϕ, Lϕ)
    zC₊ = zF₋ + total_extent(TZ, Hz, Δz, Lz)

    TFλ, TFϕ, TFz = total_length.(Face, topo, N, H)
    TCλ, TCϕ, TCz = total_length.(Center, topo, N, H)

    # FIXME? MITgcm generates (xG, yG) vorticity grid points then interpolates them to generate (xC, yC):
    # https://github.com/MITgcm/MITgcm/blob/fc300b65987b52171b1110c7930f580ca71dead0/model/src/ini_spherical_polar_grid.F#L86-L89

    λᶠᵃᵃ = range(λF₋, λF₊, length = TFλ)
    ϕᵃᶠᵃ = range(ϕF₋, ϕF₊, length = TFϕ)
    zᵃᵃᶠ = range(zF₋, zF₊, length = TFz)

    λᶜᵃᵃ = range(λC₋, λC₊, length = TCλ)
    ϕᵃᶜᵃ = range(ϕC₋, ϕC₊, length = TCϕ)
    zᵃᵃᶜ = range(zC₋, zC₊, length = TCz)

    λᶠᵃᵃ = OffsetArray(λᶠᵃᵃ, -Hx)
    ϕᵃᶠᵃ = OffsetArray(ϕᵃᶠᵃ, -Hy)
    zᵃᵃᶠ = OffsetArray(zᵃᵃᶠ, -Hz)

    λᶜᵃᵃ = OffsetArray(λᶜᵃᵃ, -Hx)
    ϕᵃᶜᵃ = OffsetArray(ϕᵃᶜᵃ, -Hy)
    zᵃᵃᶜ = OffsetArray(zᵃᵃᶜ, -Hz)

    return RegularLatitudeLongitudeGrid{FT, TX, TY, TZ, typeof(λᶠᵃᵃ)}(Nx, Ny, Nz, Hx, Hy, Hz, Lλ, Lϕ, Lz, Δλ, Δϕ, Δz, λᶠᵃᵃ, λᶜᵃᵃ, ϕᵃᶠᵃ, ϕᵃᶜᵃ, zᵃᵃᶠ, zᵃᵃᶜ, radius)
end

domain_string(grid::RegularLatitudeLongitudeGrid) =
    "longitude λ ∈ [$(grid.λᶠᵃᵃ[1]), $(grid.λᶠᵃᵃ[end])], latitude ∈ [$(grid.ϕᵃᶠᵃ[1]), $(grid.ϕᵃᶠᵃ[end])], z ∈ [$(grid.zᵃᵃᶠ[1]), $(grid.zᵃᵃᶠ[end])]"

function show(io::IO, g::RegularLatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RegularLatitudeLongitudeGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δλ, Δϕ, Δz): ", (g.Δλ, g.Δϕ, g.Δz))
end

# TODO: Move to Oceananigans.Operators (or define operators first?)

# zonal length between cell centers (MITgcm dxC)
Δxᶠᵃᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.radius * cosd(grid.ϕᵃᶠᵃ[j]) * deg2rad(grid.Δλ)

# meridional length between cell centers (MITgcm dyC)
Δyᵃᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.radius * deg2rad(grid.Δϕ)

# lengths between cell faces through the center (MITgcm dxF, dyF)
Δxᶜᵃᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.radius * cosd(grid.ϕᵃᶜᵃ[j]) * deg2rad(grid.Δλ)
Δyᶜᵃᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.radius * deg2rad(grid.Δϕ)

# lengths along cell boundaries (MITgcm dxG, dyG)
Δxᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.radius * cosd(grid.ϕᵃᶠᵃ[j]) * deg2rad(grid.Δλ)
Δyᶠᶠᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.radius * deg2rad(grid.Δϕ)

Δzᵃᵃᵃ(i, j, k, grid::RegularLatitudeLongitudeGrid) = grid.Δz
