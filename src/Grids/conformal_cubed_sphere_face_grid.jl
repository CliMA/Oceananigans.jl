using CubedSphere
using JLD2
using OffsetArrays

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid, R_Earth

import Base: show

struct ConformalCubedSphereFaceGrid{FT, TX, TY, TZ, A, R} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}
        Nx :: Int
        Ny :: Int
        Nz :: Int
        Hx :: Int
        Hy :: Int
        Hz :: Int
      λᶜᶜᵃ :: A
      λᶠᶜᵃ :: A
      λᶜᶠᵃ :: A
      λᶠᶠᵃ :: A
      φᶜᶜᵃ :: A
      φᶠᶜᵃ :: A
      φᶜᶠᵃ :: A
      φᶠᶠᵃ :: A
      zᵃᵃᶜ :: R
      zᵃᵃᶠ :: R
     Δxᶜᶜᵃ :: A
     Δxᶠᶜᵃ :: A
     Δxᶜᶠᵃ :: A
     Δxᶠᶠᵃ :: A
     Δyᶜᶜᵃ :: A
     Δyᶜᶠᵃ :: A
     Δyᶠᶜᵃ :: A
     Δyᶠᶠᵃ :: A
     Δz    :: FT
     Azᶜᶜᵃ :: A
     Azᶠᶜᵃ :: A
     Azᶜᶠᵃ :: A
     Azᶠᶠᵃ :: A
    radius :: FT
end

function ConformalCubedSphereFaceGrid(FT = Float64; size, z,
                                      topology = (Bounded, Bounded, Bounded),
                                             ξ = (-1, 1),
                                             η = (-1, 1),
                                        radius = R_Earth,
                                          halo = (1, 1, 1),
                                      rotation = nothing)
    TX, TY, TZ = topology
    Nξ, Nη, Nz = size
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the face of the cube

    ξη_grid = RegularRectilinearGrid(FT, topology=topology, size=(Nξ, Nη, Nz), x=ξ, y=η, z=z, halo=halo)

    ξᶠᵃᵃ = xnodes(Face, ξη_grid)
    ξᶜᵃᵃ = xnodes(Center, ξη_grid)
    ηᵃᶠᵃ = ynodes(Face, ξη_grid)
    ηᵃᶜᵃ = ynodes(Center, ξη_grid)

    ## The vertical coordinates can come out of the regular rectilinear grid!

    Δz = ξη_grid.Δz
    zᵃᵃᶠ = ξη_grid.zF
    zᵃᵃᶜ = ξη_grid.zC

    ## Compute staggered grid Cartesian coordinates (X, Y, Z) on the unit sphere.

    Xᶜᶜᵃ = zeros(Nξ,   Nη  )
    Xᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Xᶜᶠᵃ = zeros(Nξ,   Nη+1)
    Xᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    Yᶜᶜᵃ = zeros(Nξ,   Nη  )
    Yᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Yᶜᶠᵃ = zeros(Nξ,   Nη+1)
    Yᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    Zᶜᶜᵃ = zeros(Nξ,   Nη  )
    Zᶠᶜᵃ = zeros(Nξ+1, Nη  )
    Zᶜᶠᵃ = zeros(Nξ,   Nη+1)
    Zᶠᶠᵃ = zeros(Nξ+1, Nη+1)

    ξS = (ξᶜᵃᵃ, ξᶠᵃᵃ, ξᶜᵃᵃ, ξᶠᵃᵃ)
    ηS = (ηᵃᶜᵃ, ηᵃᶜᵃ, ηᵃᶠᵃ, ηᵃᶠᵃ)
    XS = (Xᶜᶜᵃ, Xᶠᶜᵃ, Xᶜᶠᵃ, Xᶠᶠᵃ)
    YS = (Yᶜᶜᵃ, Yᶠᶜᵃ, Yᶜᶠᵃ, Yᶠᶠᵃ)
    ZS = (Zᶜᶜᵃ, Zᶠᶜᵃ, Zᶜᶠᵃ, Zᶠᶠᵃ)

    for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
        for i in 1:length(ξ), j in 1:length(η)
            @inbounds X[i, j], Y[i, j], Z[i, j] = conformal_cubed_sphere_mapping(ξ[i], η[j])
        end
    end

    ## Rotate the face if it's not the +z face (the one containing the North Pole).

    if !isnothing(rotation)
        for (ξ, η, X, Y, Z) in zip(ξS, ηS, XS, YS, ZS)
            for i in 1:length(ξ), j in 1:length(η)
                @inbounds X[i, j], Y[i, j], Z[i, j] = rotation * [X[i, j], Y[i, j], Z[i, j]]
            end
        end
    end

    ## Compute staggered grid latitude-longitude (φ, λ) coordinates.

    λᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    λᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    λᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    λᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    φᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    φᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    φᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    φᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    λS = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ)
    φS = (φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ)

    for (ξ, η, X, Y, Z, λ, φ) in zip(ξS, ηS, XS, YS, ZS, λS, φS)
        for i in 1:length(ξ), j in 1:length(η)
            @inbounds φ[i, j], λ[i, j] = cartesian_to_lat_lon(X[i, j], Y[i, j], Z[i, j])
        end
    end

    any(any.(isnan, λS)) &&
        @warn "Your cubed sphere face contains a grid point at a pole so its longitude λ is undefined (NaN)."

    ## Not sure how to compute these right now so how about zeros?

    Δxᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    Δxᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    Δxᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    Δxᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    Δyᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    Δyᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    Δyᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    Δyᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    Azᶜᶜᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy    ), -Hx, -Hy)
    Azᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    Azᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)
    Azᶠᶠᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy + 1), -Hx, -Hy)

    return ConformalCubedSphereFaceGrid{FT, TX, TY, TZ, typeof(λᶜᶜᵃ), typeof(zᵃᵃᶠ)}(
        Nξ, Nη, Nz, Hx, Hy, Hz,
         λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,  φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
        Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δz,
        Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

function offset_data_2d(data, loc, topo, N, H)
    LX, LY = loc
    TX, TY = topo
    Nx, Ny = N
    Hx, Hy = H

    Tx = total_length(LX, TX, Nx, Hx)
    Ty = total_length(LY, TY, Ny, Hy)

    new_data = zeros(Tx, Ty)

    ii = interior_parent_indices(LX, TX, Nx, Hx)
    jj = interior_parent_indices(LY, TY, Ny, Hy)

    new_data[ii, jj] .= data

    return OffsetArray(new_data, -Hx, -Hy)
end

function ConformalCubedSphereFaceGrid(filepath::AbstractString, FT = Float64; face, Nz, z,
                                      topology = (Bounded, Bounded, Bounded),
                                        radius = R_Earth,
                                          halo = (1, 1, 1),
                                      rotation = nothing)

    TX, TY, TZ = topo = topology
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the vertical grid
    ## The vertical coordinates can come out of the regular rectilinear grid!

    ξη_grid = RegularRectilinearGrid(FT, topology=topology, size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=z, halo=halo)

    Δz = ξη_grid.Δz
    zᵃᵃᶠ = ξη_grid.zF
    zᵃᵃᶜ = ξη_grid.zC

    ## Read everything else from the file

    file = jldopen(filepath, "r")
    cubed_sphere_data = file["face$face"]

    Nξ, Nη = size(cubed_sphere_data["λᶠᶠᵃ"]) .- 1

    N = (Nξ, Nη, Nz)
    H = halo

    loc_cc = (Center, Center)
    loc_cf = (Center, Face)
    loc_fc = (Face, Center)
    loc_ff = (Face, Face)

    c_inds = 1:Nξ
    f_inds = 1:Nξ+1

    topo_2d = (Bounded, Bounded)

    λᶜᶜᵃ = offset_data_2d(cubed_sphere_data["λᶜᶜᵃ"][c_inds, c_inds], loc_cc, topo_2d, N, H)
    λᶠᶠᵃ = offset_data_2d(cubed_sphere_data["λᶠᶠᵃ"][f_inds, f_inds], loc_ff, topo_2d, N, H)

    φᶜᶜᵃ = offset_data_2d(cubed_sphere_data["φᶜᶜᵃ"][c_inds, c_inds], loc_cc, topo_2d, N, H)
    φᶠᶠᵃ = offset_data_2d(cubed_sphere_data["φᶠᶠᵃ"][f_inds, f_inds], loc_ff, topo_2d, N, H)

    Δxᶜᶜᵃ = offset_data_2d(cubed_sphere_data["Δxᶜᶜᵃ"][c_inds, c_inds], loc_cc, topo_2d, N, H)
    Δxᶠᶜᵃ = offset_data_2d(cubed_sphere_data["Δxᶠᶜᵃ"][f_inds, c_inds], loc_fc, topo_2d, N, H)
    Δxᶜᶠᵃ = offset_data_2d(cubed_sphere_data["Δxᶜᶠᵃ"][c_inds, f_inds], loc_cf, topo_2d, N, H)
    Δxᶠᶠᵃ = offset_data_2d(cubed_sphere_data["Δxᶠᶠᵃ"][f_inds, f_inds], loc_ff, topo_2d, N, H)

    Δyᶜᶜᵃ = offset_data_2d(cubed_sphere_data["Δyᶜᶜᵃ"][c_inds, c_inds], loc_cc, topo_2d, N, H)
    Δyᶠᶜᵃ = offset_data_2d(cubed_sphere_data["Δyᶠᶜᵃ"][f_inds, c_inds], loc_fc, topo_2d, N, H)
    Δyᶜᶠᵃ = offset_data_2d(cubed_sphere_data["Δyᶜᶠᵃ"][c_inds, f_inds], loc_cf, topo_2d, N, H)
    Δyᶠᶠᵃ = offset_data_2d(cubed_sphere_data["Δyᶠᶠᵃ"][f_inds, f_inds], loc_ff, topo_2d, N, H)

    Azᶜᶜᵃ = offset_data_2d(cubed_sphere_data["Azᶜᶜᵃ"][c_inds, c_inds], loc_cc, topo_2d, N, H)
    Azᶠᶜᵃ = offset_data_2d(cubed_sphere_data["Azᶠᶜᵃ"][f_inds, c_inds], loc_fc, topo_2d, N, H)
    Azᶜᶠᵃ = offset_data_2d(cubed_sphere_data["Azᶜᶠᵃ"][c_inds, f_inds], loc_cf, topo_2d, N, H)
    Azᶠᶠᵃ = offset_data_2d(cubed_sphere_data["Azᶠᶠᵃ"][f_inds, f_inds], loc_ff, topo_2d, N, H)

    ## Maybe we won't need these?

    λᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    λᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)

    φᶠᶜᵃ = OffsetArray(zeros(Nξ + 2Hx + 1, Nη + 2Hy    ), -Hx, -Hy)
    φᶜᶠᵃ = OffsetArray(zeros(Nξ + 2Hx,     Nη + 2Hy + 1), -Hx, -Hy)

    return ConformalCubedSphereFaceGrid{FT, TX, TY, TZ, typeof(λᶜᶜᵃ), typeof(zᵃᵃᶠ)}(
        Nξ, Nη, Nz, Hx, Hy, Hz,
         λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,  φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
        Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δz,
        Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

function show(io::IO, g::ConformalCubedSphereFaceGrid{FT}) where FT
    print(io, "ConformalCubedSphereFaceGrid{$FT}\n",
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz))
end
