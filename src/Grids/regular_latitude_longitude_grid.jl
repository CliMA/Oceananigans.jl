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

    ϕ₁, ϕ₂ = latitude
    @assert -180 <= ϕ₁ < ϕ₂ <= 180

    λ₁, λ₂ = longitude
    @assert -90 <= λ₁ < λ₂ <= 90

    z₁, z₂ = z
    @assert z₁ < z₂

    TX = ϕ₁ == -180 && ϕ₂ == 180 ? Periodic : Bounded
    TY = Bounded
    TZ = Bounded
    topo = (TX, TY, TZ)

    Nx, Ny, Nz = size = validate_size(TX, TY, TZ, size)
    Hx, Hy, Hz = halo = validate_halo(TX, TY, TZ, halo)

    Lx = λ₂ - λ₁
    Ly = ϕ₂ - ϕ₁
    Lz = z₂ - z₁

    Δϕ = Lx / Nx
    Δλ = Ly / Ny
    Δz = Lz / Nz

    # FIXME? MITgcm generates (xG, yG) vorticity grid points then interpolates them to generate (xC, yC):
    # https://github.com/MITgcm/MITgcm/blob/fc300b65987b52171b1110c7930f580ca71dead0/model/src/ini_spherical_polar_grid.F#L86-L89

    λᶠᵃᵃ = range(λ₁, λ₂, length=Ny+1)
    λᶜᵃᵃ = range(λ₁ + Δλ/2, λ₂ - Δλ/2, length=Ny+1)

    ϕᵃᶠᵃ = range(ϕ₁, ϕ₂, length=Nx+1)
    ϕᵃᶜᵃ = range(ϕ₁ + Δϕ/2, ϕ₂ - Δϕ/2, length=Nx)

    zᵃᵃᶠ = range(z₁, z₂, length=Nz+1)
    zᵃᵃᶜ = range(z₁ + Δz/2, z₂ - Δz/2, length=Nz)

    return RegularLatitudeLongitudeGrid{FT, TX, TY, TZ, typeof(λᶠᵃᵃ)}(Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δλ, Δϕ, Δz, λᶠᵃᵃ, λᶜᵃᵃ, ϕᵃᶠᵃ, ϕᵃᶜᵃ, zᵃᵃᶠ, zᵃᵃᶜ, radius)
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
