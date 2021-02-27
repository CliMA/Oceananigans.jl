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

    Xᶜᶜᶜ = zeros(Nξ, Nη)
    Xᶠᶜᶜ = zeros(Nξ, Nη)
    Xᶜᶠᶜ = zeros(Nξ, Nη)
    Xᶠᶠᶜ = zeros(Nξ, Nη)

    Yᶜᶜᶜ = zeros(Nξ, Nη)
    Yᶠᶜᶜ = zeros(Nξ, Nη)
    Yᶜᶠᶜ = zeros(Nξ, Nη)
    Yᶠᶠᶜ = zeros(Nξ, Nη)

    Zᶜᶜᶜ = zeros(Nξ, Nη)
    Zᶠᶜᶜ = zeros(Nξ, Nη)
    Zᶜᶠᶜ = zeros(Nξ, Nη)
    Zᶠᶠᶜ = zeros(Nξ, Nη)

    for i in 1:Nξ, j in 1:Nη
        @inbounds Xᶜᶜᶜ[i, j], Yᶜᶜᶜ[i, j], Zᶜᶜᶜ[i, j] = conformal_cubed_sphere_mapping(ξᶜᵃᵃ[i], ηᵃᶜᵃ[j])
        @inbounds Xᶠᶜᶜ[i, j], Yᶠᶜᶜ[i, j], Zᶠᶜᶜ[i, j] = conformal_cubed_sphere_mapping(ξᶠᵃᵃ[i], ηᵃᶜᵃ[j])
        @inbounds Xᶜᶠᶜ[i, j], Yᶜᶠᶜ[i, j], Zᶜᶠᶜ[i, j] = conformal_cubed_sphere_mapping(ξᶜᵃᵃ[i], ηᵃᶠᵃ[j])
        @inbounds Xᶠᶠᶜ[i, j], Yᶠᶠᶜ[i, j], Zᶠᶠᶜ[i, j] = conformal_cubed_sphere_mapping(ξᶠᵃᵃ[i], ηᵃᶠᵃ[j])
    end

    # Rotate the face if it's not the +z face (the one containing the North Pole).

    XS = (Xᶜᶜᶜ, Xᶠᶜᶜ, Xᶜᶠᶜ, Xᶠᶠᶜ)
    YS = (Yᶜᶜᶜ, Yᶠᶜᶜ, Yᶜᶠᶜ, Yᶠᶠᶜ)
    ZS = (Zᶜᶜᶜ, Zᶠᶜᶜ, Zᶜᶠᶜ, Zᶠᶠᶜ)

    if !isnothing(rotation)
        for (X, Y, Z) in zip(XS, YS, ZS), i in 1:Nξ, j in 1:Nη
            X[I], Y[I], Z[I] = rotation * [X[I], Y[I], Z[I]]
        end
    end

    # Compute staggered grid latitude-longitude (ϕ, λ) coordinates.

    λᶜᶜᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)
    λᶠᶜᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)
    λᶜᶠᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)
    λᶠᶠᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)

    ϕᶜᶜᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)
    ϕᶠᶜᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)
    ϕᶜᶠᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)
    ϕᶠᶠᶜ = OffsetArray(zeros(Nξ + 2Hx, Nη + 2Hy), -Hx, -Hy)

    for i in 1:Nξ, j in 1:Nη
        @inbounds ϕᶜᶜᶜ[i, j], λᶜᶜᶜ[i, j] = cartesian_to_lat_lon(Xᶜᶜᶜ[i, j], Yᶜᶜᶜ[i, j], Zᶜᶜᶜ[i, j])
        @inbounds ϕᶠᶜᶜ[i, j], λᶠᶜᶜ[i, j] = cartesian_to_lat_lon(Xᶠᶜᶜ[i, j], Yᶠᶜᶜ[i, j], Zᶠᶜᶜ[i, j])
        @inbounds ϕᶜᶠᶜ[i, j], λᶜᶠᶜ[i, j] = cartesian_to_lat_lon(Xᶜᶠᶜ[i, j], Yᶜᶠᶜ[i, j], Zᶜᶠᶜ[i, j])
        @inbounds ϕᶠᶠᶜ[i, j], λᶠᶠᶜ[i, j] = cartesian_to_lat_lon(Xᶠᶠᶜ[i, j], Yᶠᶠᶜ[i, j], Zᶠᶠᶜ[i, j])
    end

    λS = (λᶜᶜᶜ, λᶠᶜᶜ, λᶜᶠᶜ, λᶠᶠᶜ)
    ϕS = (ϕᶜᶜᶜ, ϕᶠᶜᶜ, ϕᶜᶠᶜ, ϕᶠᶠᶜ)

    any(any.(isnan, λS)) &&
        @warn "Your cubed sphere face contains a grid point at a pole so its longitude λ is undefined (NaN)."

    return ConformalCubedSphereFaceGrid(Nξ, Nη, Nz, Hx, Hy, Hz, λᶜᶜᶜ, λᶠᶜᶜ, λᶜᶠᶜ, λᶠᶠᶜ, ϕᶜᶜᶜ, ϕᶠᶜᶜ, ϕᶜᶠᶜ, ϕᶠᶠᶜ, zᵃᵃᶠ, zᵃᵃᶜ, Δz , radius)
end

# @inline Δxᶜᶜᶜ(i, j, k, grid::ConformalCubedSphereFaceField) = @inbounds grid.radius * cosd(grid.ϕᶜᶜᶜ[i, j]) * deg2rad(grid.λᶠᶜᶜ[i+1, j] - grid.λᶠᶜᶜ[i, j])
