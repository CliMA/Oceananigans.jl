using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Adapt: adapt_structure

using Oceananigans

struct ConformalCubedSphereFaceGrid{FT, TX, TY, TZ, A, R, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
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

    function ConformalCubedSphereFaceGrid{TX, TY, TZ}(architecture::Arch,
                                                      Nx, Ny, Nz,
                                                      Hx, Hy, Hz,
                                                       λᶜᶜᵃ :: A,  λᶠᶜᵃ :: A,  λᶜᶠᵃ :: A,  λᶠᶠᵃ :: A,
                                                       φᶜᶜᵃ :: A,  φᶠᶜᵃ :: A,  φᶜᶠᵃ :: A,  φᶠᶠᵃ :: A, zᵃᵃᶜ :: R, zᵃᵃᶠ :: R,
                                                      Δxᶜᶜᵃ :: A, Δxᶠᶜᵃ :: A, Δxᶜᶠᵃ :: A, Δxᶠᶠᵃ :: A,
                                                      Δyᶜᶜᵃ :: A, Δyᶜᶠᵃ :: A, Δyᶠᶜᵃ :: A, Δyᶠᶠᵃ :: A, Δz :: FT,
                                                      Azᶜᶜᵃ :: A, Azᶠᶜᵃ :: A, Azᶜᶠᵃ :: A, Azᶠᶠᵃ :: A,
                                                      radius :: FT) where {TX, TY, TZ, FT, A, R, Arch}

        return new{FT, TX, TY, TZ, A, R, Arch}(architecture,
                                               Nx, Ny, Nz,
                                               Hx, Hy, Hz,
                                               λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                               φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                               Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                               Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δz,
                                               Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
    end
end

function ConformalCubedSphereFaceGrid(arch::AbstractArchitecture = CPU(),
                                      FT::DataType = Float64;
                                      size,
                                      z,
                                      topology = (Bounded, Bounded, Bounded),
                                      ξ = (-1, 1),
                                      η = (-1, 1),
                                      radius = R_Earth,
                                      halo = (1, 1, 1),
                                      rotation = nothing)

    TX, TY, TZ = topology
    Nξ, Nη, Nz = size
    Hx, Hy, Hz = halo
    Arch = typeof(arch)
    
    ## Use a regular rectilinear grid for the face of the cube

    ξη_grid = RectilinearGrid(arch, FT, topology=topology, size=(Nξ, Nη, Nz), x=ξ, y=η, z=z, halo=halo)

    ξᶠᵃᵃ = xnodes(Face, ξη_grid)
    ξᶜᵃᵃ = xnodes(Center, ξη_grid)
    ηᵃᶠᵃ = ynodes(Face, ξη_grid)
    ηᵃᶜᵃ = ynodes(Center, ξη_grid)

    ## The vertical coordinates can come out of the regular rectilinear grid!

    Δz = ξη_grid.Δzᵃᵃᶜ
    zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
    zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ

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

    return ConformalCubedSphereFaceGrid{TX, TY, TZ}(arch, Nξ, Nη, Nz, Hx, Hy, Hz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δz,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
ConformalCubedSphereFaceGrid(FT::DataType; kwargs...) = ConformalCubedSphereFaceGrid(CPU(), FT; kwargs...)

function load_and_offset_cubed_sphere_data(file, FT, arch, field_name, loc, topo, N, H)

    ii = interior_indices(loc[1], topo[1], N[1])
    jj = interior_indices(loc[2], topo[2], N[2])

    interior_data = arch_array(arch, file[field_name][ii, jj])

    underlying_data = zeros(FT, arch,
                            total_length(loc[1], topo[1], N[1], H[1]),
                            total_length(loc[2], topo[2], N[2], H[2]))

    ip = interior_parent_indices(loc[1], topo[1], N[1], H[1])
    jp = interior_parent_indices(loc[2], topo[2], N[2], H[2])

    view(underlying_data, ip, jp) .= interior_data

    return offset_data(underlying_data, loc, topo, N, H)
end

function ConformalCubedSphereFaceGrid(filepath::AbstractString, architecture = CPU(), FT = Float64;
                                      face, Nz, z,
                                      topology = (Bounded, Bounded, Bounded),
                                        radius = R_Earth,
                                          halo = (1, 1, 1),
                                      rotation = nothing)

    TX, TY, TZ = topo = topology
    Hx, Hy, Hz = halo

    ## Use a regular rectilinear grid for the vertical grid
    ## The vertical coordinates can come out of the regular rectilinear grid!

    ξη_grid = RectilinearGrid(architecture, FT, topology=topology, size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=z, halo=halo)

    Δz = ξη_grid.Δzᵃᵃᶜ
    zᵃᵃᶠ = ξη_grid.zᵃᵃᶠ
    zᵃᵃᶜ = ξη_grid.zᵃᵃᶜ

    ## Read everything else from the file

    file = jldopen(filepath, "r")["face$face"]
    Nξ, Nη = size(file["λᶠᶠᵃ"]) .- 1

    N = (Nξ, Nη, Nz)
    H = halo

    loc_cc = (Center, Center)
    loc_cf = (Center, Face)
    loc_fc = (Face, Center)
    loc_ff = (Face, Face)

    topo_bbb = (Bounded, Bounded, Bounded)

    λᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    λᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    φᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    φᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    Δxᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    Δxᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶜᵃ", loc_fc, topo_bbb, N, H)
    Δxᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶠᵃ", loc_cf, topo_bbb, N, H)
    Δxᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    Δyᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    Δyᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶜᵃ", loc_fc, topo_bbb, N, H)
    Δyᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶠᵃ", loc_cf, topo_bbb, N, H)
    Δyᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    Azᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶜᵃ", loc_cc, topo_bbb, N, H)
    Azᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶜᵃ", loc_fc, topo_bbb, N, H)
    Azᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶠᵃ", loc_cf, topo_bbb, N, H)
    Azᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶠᵃ", loc_ff, topo_bbb, N, H)

    ## Maybe we won't need these?
    Txᶠᶜ = total_length(loc_fc[1], topology[1], N[1], H[1])
    Txᶜᶠ = total_length(loc_cf[1], topology[1], N[1], H[1])
    Tyᶠᶜ = total_length(loc_fc[2], topology[2], N[2], H[2])
    Tyᶜᶠ = total_length(loc_cf[2], topology[2], N[2], H[2])

    λᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology, N, H)
    λᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology, N, H)
    φᶠᶜᵃ = offset_data(zeros(FT, architecture, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology, N, H)
    φᶜᶠᵃ = offset_data(zeros(FT, architecture, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology, N, H)

    return ConformalCubedSphereFaceGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz,
         λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,  φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
        Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δz,
        Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

function on_architecture(arch::AbstractArchitecture, grid::ConformalCubedSphereFaceGrid)
    horizontal_coordinates = (:λᶜᶜᵃ,
                              :λᶠᶜᵃ,
                              :λᶜᶠᵃ,
                              :λᶠᶠᵃ,
                              :φᶜᶜᵃ,
                              :φᶠᶜᵃ,
                              :φᶜᶠᵃ,
                              :φᶠᶠᵃ)

    horizontal_grid_spacings = (:Δxᶜᶜᵃ,
                                :Δxᶠᶜᵃ,
                                :Δxᶜᶠᵃ,
                                :Δxᶠᶠᵃ,
                                :Δyᶜᶜᵃ,
                                :Δyᶜᶠᵃ,
                                :Δyᶠᶜᵃ,
                                :Δyᶠᶠᵃ)

    horizontal_areas = (:Azᶜᶜᵃ,
                        :Azᶠᶜᵃ,
                        :Azᶜᶠᵃ,
                        :Azᶠᶠᵃ)

    horizontal_grid_spacing_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_grid_spacings)
    horizontal_coordinate_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_coordinates)
    horizontal_area_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_areas)

    zᵃᵃᶜ = arch_array(arch, grid.zᵃᵃᶜ)
    zᵃᵃᶠ = arch_array(arch, grid.zᵃᵃᶠ)

    TX, TY, TZ = topology(grid)

    new_grid = ConformalCubedSphereFaceGrid{TX, TY, TZ}(arch,
                                                        grid.Nx, grid.Ny, grid.Nz,
                                                        grid.Hx, grid.Hy, grid.Hz,
                                                        horizontal_coordinate_data..., zᵃᵃᶜ, zᵃᵃᶠ,
                                                        horizontal_grid_spacing_data..., grid.Δz,
                                                        horizontal_area_data..., grid.radius)

    return new_grid
end

function Adapt.adapt_structure(to, grid::ConformalCubedSphereFaceGrid)
    TX, TY, TZ = topology(grid)
    return ConformalCubedSphereFaceGrid{TX, TY, TZ}(nothing,
                                                    grid.Nx, grid.Ny, grid.Nz,
                                                    grid.Hx, grid.Hy, grid.Hz,
                                                    adapt(to, grid.λᶜᶜᵃ),  
                                                    adapt(to, grid.λᶠᶜᵃ),
                                                    adapt(to, grid.λᶜᶠᵃ),
                                                    adapt(to, grid.λᶠᶠᵃ),
                                                    adapt(to, grid.φᶜᶜᵃ),
                                                    adapt(to, grid.φᶠᶜᵃ),
                                                    adapt(to, grid.φᶜᶠᵃ),
                                                    adapt(to, grid.φᶠᶠᵃ),
                                                    adapt(to, grid.zᵃᵃᶜ),
                                                    adapt(to, grid.zᵃᵃᶠ),
                                                    adapt(to, grid.Δxᶜᶜᵃ), 
                                                    adapt(to, grid.Δxᶠᶜᵃ),
                                                    adapt(to, grid.Δxᶜᶠᵃ),
                                                    adapt(to, grid.Δxᶠᶠᵃ),
                                                    adapt(to, grid.Δyᶜᶜᵃ),
                                                    adapt(to, grid.Δyᶜᶠᵃ),
                                                    adapt(to, grid.Δyᶠᶜᵃ),
                                                    adapt(to, grid.Δyᶠᶠᵃ),
                                                    grid.Δz,
                                                    adapt(to, grid.Azᶜᶜᵃ), 
                                                    adapt(to, grid.Azᶠᶜᵃ),
                                                    adapt(to, grid.Azᶜᶠᵃ),
                                                    adapt(to, grid.Azᶠᶠᵃ),
                                                    grid.radius)
end

function Base.show(io::IO, g::ConformalCubedSphereFaceGrid{FT}) where FT
    print(io, "ConformalCubedSphereFaceGrid{$FT}\n",
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz))
end

@inline xnode(::Face,   ::Face,   LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.λᶠᶠᵃ[i, j]
@inline xnode(::Face,   ::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline xnode(::Center, ::Face,   LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline xnode(::Center, ::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.λᶜᶜᵃ[i, j]

@inline ynode(::Face,   ::Face,   LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.φᶠᶠᵃ[i, j]
@inline ynode(::Face,   ::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline ynode(::Center, ::Face,   LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline ynode(::Center, ::Center, LZ, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.φᶜᶜᵃ[i, j]

@inline znode(LX, LY, ::Face,   i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(LX, LY, ::Center, i, j, k, grid::ConformalCubedSphereFaceGrid) = @inbounds grid.zᵃᵃᶜ[k]
