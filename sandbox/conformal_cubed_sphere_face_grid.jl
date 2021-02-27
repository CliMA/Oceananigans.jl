using OffsetArrays
using CubedSphere

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Grids: R_Earth

struct ConformalCubedSphereFaceGrid{FT, A, R}
        Nx :: Int
        Ny :: Int
        Nz :: Int
        Hx :: Int
        Hy :: Int
        Hz :: Int
      λᶜᶜᶜ :: A
      λᶠᶜᶜ :: A
      λᶜᶠᶜ :: A
      λᶠᶠᶜ :: A
      ϕᶜᶜᶜ :: A
      ϕᶠᶜᶜ :: A
      ϕᶜᶠᶜ :: A
      ϕᶠᶠᶜ :: A
      zᵃᵃᶠ :: R
      zᵃᵃᶜ :: R
     Δz    :: FT
    radius :: FT
end

function ConformalCubedSphereFaceGrid(FT=Float64; size, ξ=(-1, 1), η=(-1, 1), z, radius=R_Earth, halo=(1, 1, 1), rotation=nothing)
    Nξ, Nη, Nz = size
    Hx, Hy, Hz = halo

    # Use a regular grid for the face of the cube

    ξη_grid = RegularRectilinearGrid(FT, topology=(Bounded, Bounded, Bounded), size=(Nξ, Nη, Nz), x=ξ, y=η, z=z, halo=halo)

    ξᶠᵃᵃ = xnodes(Face, ξη_grid)
    ξᶜᵃᵃ = xnodes(Center, ξη_grid)
    ηᵃᶠᵃ = ynodes(Face, ξη_grid)
    ηᵃᶜᵃ = ynodes(Center, ξη_grid)

    # The vertical coordinates can come out of the regular rectilinear grid!

    Δz = ξη_grid.Δz
    zᵃᵃᶠ = ξη_grid.zF
    zᵃᵃᶜ = ξη_grid.zC

    # Compute staggered grid Cartesian coordinates (X, Y, Z) on the unit sphere.

    Xᶜᶜᶜ = zeros(Nξ,   Nη)
    Xᶠᶜᶜ = zeros(Nξ+1, Nη)
    Xᶜᶠᶜ = zeros(Nξ,   Nη+1)
    Xᶠᶠᶜ = zeros(Nξ+1, Nη+1)

    Yᶜᶜᶜ = zeros(Nξ,   Nη)
    Yᶠᶜᶜ = zeros(Nξ+1, Nη)
    Yᶜᶠᶜ = zeros(Nξ,   Nη+1)
    Yᶠᶠᶜ = zeros(Nξ+1, Nη+1)

    Zᶜᶜᶜ = zeros(Nξ,   Nη)
    Zᶠᶜᶜ = zeros(Nξ+1, Nη)
    Zᶜᶠᶜ = zeros(Nξ,   Nη+1)
    Zᶠᶠᶜ = zeros(Nξ+1, Nη+1)

    ξS = (ξᶜᵃᵃ, ξᶠᵃᵃ, ξᶜᵃᵃ, ξᶠᵃᵃ)
    ηS = (ηᵃᶜᵃ, ηᵃᶜᵃ, ηᵃᶠᵃ, ηᵃᶠᵃ)
    XS = (Xᶜᶜᶜ, Xᶠᶜᶜ, Xᶜᶠᶜ, Xᶠᶠᶜ)
    YS = (Yᶜᶜᶜ, Yᶠᶜᶜ, Yᶜᶠᶜ, Yᶠᶠᶜ)
    ZS = (Zᶜᶜᶜ, Zᶠᶜᶜ, Zᶜᶠᶜ, Zᶠᶠᶜ)

    for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
        for i in 1:length(ξ), j in 1:length(η)
            @inbounds X[i, j], Y[i, j], Z[i, j] = conformal_cubed_sphere_mapping(ξ[i], η[j])
        end
    end

    # Rotate the face if it's not the +z face (the one containing the North Pole).

    if !isnothing(rotation)
        for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
            for i in 1:length(ξ), j in 1:length(η)
                @inbounds X[i, j], Y[i, j], Z[i, j] = rotation * [X[i, j], Y[i, j], Z[i, j]]
            end
        end
    end

    # Compute staggered grid latitude-longitude (ϕ, λ) coordinates.

    λᶜᶜᶜ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    λᶠᶜᶜ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    λᶜᶠᶜ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    λᶠᶠᶜ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    ϕᶜᶜᶜ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    ϕᶠᶜᶜ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    ϕᶜᶠᶜ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    ϕᶠᶠᶜ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    λS = (λᶜᶜᶜ, λᶠᶜᶜ, λᶜᶠᶜ, λᶠᶠᶜ)
    ϕS = (ϕᶜᶜᶜ, ϕᶠᶜᶜ, ϕᶜᶠᶜ, ϕᶠᶠᶜ)

    for (ξ, η, X, Y, Z, λ, ϕ) in zip(ξS, ηS, XS, YS, ZS, λS, ϕS)
        for i in 1:length(ξ), j in 1:length(η)
            @inbounds ϕ[i, j], λ[i, j] = cartesian_to_lat_lon(X[i, j], Y[i, j], Z[i, j])
        end
    end

    any(any.(isnan, λS)) &&
        @warn "Your cubed sphere face contains a grid point at a pole so its longitude λ is undefined (NaN)."

    return ConformalCubedSphereFaceGrid(Nξ, Nη, Nz, Hx, Hy, Hz, λᶜᶜᶜ, λᶠᶜᶜ, λᶜᶠᶜ, λᶠᶠᶜ, ϕᶜᶜᶜ, ϕᶠᶜᶜ, ϕᶜᶠᶜ, ϕᶠᶠᶜ, zᵃᵃᶠ, zᵃᵃᶜ, Δz , radius)
end

# @inline Δxᶜᶜᶜ(i, j, k, grid::ConformalCubedSphereFaceField) = @inbounds grid.radius * cosd(grid.ϕᶜᶜᶜ[i, j]) * deg2rad(grid.λᶠᶜᶜ[i+1, j] - grid.λᶠᶜᶜ[i, j])
