using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Distances

using Adapt: adapt_structure

using Oceananigans
using Oceananigans.Grids: prettysummary, coordinate_summary, BoundedTopology, length

const AHCG = AbstractHorizontallyCurvilinearGrid

struct OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch, FT2} <: AHCG{FT, TX, TY, TZ, Z, Arch}
    architecture :: Arch
       Nx :: Int
       Ny :: Int
       Nz :: Int
       Hx :: Int
       Hy :: Int
       Hz :: Int
       Lz :: FT2
     λᶜᶜᵃ :: CC
     λᶠᶜᵃ :: FC
     λᶜᶠᵃ :: CF
     λᶠᶠᵃ :: FF
     φᶜᶜᵃ :: CC
     φᶠᶜᵃ :: FC
     φᶜᶠᵃ :: CF
     φᶠᶠᵃ :: FF
        z :: Z
    Δxᶜᶜᵃ :: CC
    Δxᶠᶜᵃ :: FC
    Δxᶜᶠᵃ :: CF
    Δxᶠᶠᵃ :: FF
    Δyᶜᶜᵃ :: CC
    Δyᶠᶜᵃ :: FC
    Δyᶜᶠᵃ :: CF
    Δyᶠᶠᵃ :: FF
    Azᶜᶜᵃ :: CC
    Azᶠᶜᵃ :: FC
    Azᶜᶠᵃ :: CF
    Azᶠᶠᵃ :: FF
    radius :: FT2
    conformal_mapping :: Map
end


function OrthogonalSphericalShellGrid{FT, TX, TY, TZ}(architecture::Arch,
                                                  Nx, Ny, Nz,
                                                  Hx, Hy, Hz,
                                                  Lz :: FT2,
                                                   λᶜᶜᵃ :: CC,  λᶠᶜᵃ :: FC,  λᶜᶠᵃ :: CF,  λᶠᶠᵃ :: FF,
                                                   φᶜᶜᵃ :: CC,  φᶠᶜᵃ :: FC,  φᶜᶠᵃ :: CF,  φᶠᶠᵃ :: FF, z :: Z,
                                                  Δxᶜᶜᵃ :: CC, Δxᶠᶜᵃ :: FC, Δxᶜᶠᵃ :: CF, Δxᶠᶠᵃ :: FF,
                                                  Δyᶜᶜᵃ :: CC, Δyᶠᶜᵃ :: FC, Δyᶜᶠᵃ :: CF, Δyᶠᶠᵃ :: FF,
                                                  Azᶜᶜᵃ :: CC, Azᶠᶜᵃ :: FC, Azᶜᶠᵃ :: CF, Azᶠᶠᵃ :: FF,
                                                  radius :: FT2,
                                                  conformal_mapping :: Map) where {TX, TY, TZ, FT, Z, Map,
                                                                                   CC, FC, CF, FF, Arch, FT2}
    return OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch, FT2}(architecture,
                                                  Nx, Ny, Nz,
                                                  Hx, Hy, Hz,
                                                  Lz,
                                                   λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                   φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z,
                                                  Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                  Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                                  Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                  radius,
                                                  conformal_mapping)
end

function OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture::Arch,
                                                  Nx, Ny, Nz,
                                                  Hx, Hy, Hz,
                                                  Lz :: FT,
                                                   λᶜᶜᵃ :: CC,  λᶠᶜᵃ :: FC,  λᶜᶠᵃ :: CF,  λᶠᶠᵃ :: FF,
                                                   φᶜᶜᵃ :: CC,  φᶠᶜᵃ :: FC,  φᶜᶠᵃ :: CF,  φᶠᶠᵃ :: FF, z :: Z,
                                                  Δxᶜᶜᵃ :: CC, Δxᶠᶜᵃ :: FC, Δxᶜᶠᵃ :: CF, Δxᶠᶠᵃ :: FF,
                                                  Δyᶜᶜᵃ :: CC, Δyᶠᶜᵃ :: FC, Δyᶜᶠᵃ :: CF, Δyᶠᶠᵃ :: FF,
                                                  Azᶜᶜᵃ :: CC, Azᶠᶜᵃ :: FC, Azᶜᶠᵃ :: CF, Azᶠᶠᵃ :: FF,
                                                  radius :: FT,
                                                  conformal_mapping :: Map) where {TX, TY, TZ, FT, Z, Map,
                                                                                   CC, FC, CF, FF, Arch}

    return OrthogonalSphericalShellGrid{FT, TX, TY, TZ}(architecture,
                                                              Nx, Ny, Nz,
                                                              Hx, Hy, Hz,
                                                              Lz,
                                                               λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                               φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z,
                                                              Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                              Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                                              Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                              radius, conformal_mapping)
end

const OSSG = OrthogonalSphericalShellGrid
const ZRegOSSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate}
const ZRegOrthogonalSphericalShellGrid = ZRegOSSG

# convenience constructor for OSSG without any conformal_mapping properties
OrthogonalSphericalShellGrid(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lz,
                              λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                              φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z,
                             Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                             Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                             Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius) =
    OrthogonalSphericalShellGrid(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lz,
                                  λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                  φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z,
                                 Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                 Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                 Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius, nothing)

"""
    fill_metric_halo_regions_x!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)

Fill the `x`-halo regions of the `metric` that lives on locations `ℓx`, `ℓy`, with halo size `Hx`, `Hy`,
and topology `tx`, `ty`.
"""
function fill_metric_halo_regions_x!(metric, ℓx, ℓy, tx::BoundedTopology, ty, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for j in 1:Ny⁺
            # fill west halos
            for i in 0:-1:-Hx+1
                metric[i, j] = metric[i+1, j]
            end

            # fill east halos
            for i in Nx⁺+1:Nx⁺+Hx
                metric[i, j] = metric[i-1, j]
            end
        end
    end

    return nothing
end

function fill_metric_halo_regions_x!(metric, ℓx, ℓy, tx::AbstractTopology, ty, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for j in 1:Ny⁺
            # fill west halos
            for i in 0:-1:-Hx+1
                metric[i, j] = metric[Nx+i, j]
            end
            # fill east halos
            for i in Nx⁺+1:Nx⁺+Hx
                metric[i, j] = metric[i-Nx, j]
            end
        end
    end

    return nothing
end

"""
    fill_metric_halo_regions_y!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)

Fill the `y`-halo regions of the `metric` that lives on locations `ℓx`, `ℓy`, with halo size `Hx`, `Hy`,
and topology `tx`, `ty`.
"""
function fill_metric_halo_regions_y!(metric, ℓx, ℓy, tx, ty::BoundedTopology, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for i in 1:Nx⁺
            # fill south halos
            for j in 0:-1:-Hy+1
                metric[i, j] = metric[i, j+1]
            end
            # fill north halos
            for j in Ny⁺+1:Ny⁺+Hy
                metric[i, j] = metric[i, j-1]
            end
        end
    end

    return nothing
end

function fill_metric_halo_regions_y!(metric, ℓx, ℓy, tx, ty::AbstractTopology, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for i in 1:Nx⁺
            # fill south halos
            for j in 0:-1:-Hy+1
                metric[i, j] = metric[i, Ny+j]
            end
            # fill north halos
            for j in Ny⁺+1:Ny⁺+Hy
                metric[i, j] = metric[i, j-Ny]
            end
        end
    end

    return nothing
end

"""
    fill_metric_halo_corner_regions!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)

Fill the corner halo regions of the `metric`  that lives on locations `ℓx`, `ℓy`,
and with halo size `Hx`, `Hy`. We choose to fill with the average of the neighboring
metric in the halo regions. Thus this requires that the metric in the `x`- and `y`-halo
regions have already been filled.
"""
function fill_metric_halo_corner_regions!(metric, ℓx, ℓy, tx, ty, Nx, Ny, Hx, Hy)
    # = N+1 for ::BoundedTopology or N otherwise
    Nx⁺ = length(ℓx, tx, Nx)
    Ny⁺ = length(ℓy, ty, Ny)

    @inbounds begin
        for j in 0:-1:-Hy+1, i in 0:-1:-Hx+1
            metric[i, j] = (metric[i+1, j] + metric[i, j+1]) / 2
        end
        for j in Ny⁺+1:Ny⁺+Hy, i in 0:-1:-Hx+1
            metric[i, j] = (metric[i+1, j] + metric[i, j-1]) / 2
        end
        for j in 0:-1:-Hy+1, i in Nx⁺+1:Nx⁺+Hx
            metric[i, j] = (metric[i-1, j] + metric[i, j+1]) / 2
        end
        for j in Ny⁺+1:Ny⁺+Hy, i in Nx⁺+1:Nx⁺+Hx
            metric[i, j] = (metric[i-1, j] + metric[i, j-1]) / 2
        end
    end

    return nothing
end

function fill_metric_halo_regions!(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    TX, TY, _ = topology(grid)

    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ = grid.Δxᶜᶜᵃ, grid.Δxᶠᶜᵃ, grid.Δxᶜᶠᵃ, grid.Δxᶠᶠᵃ
    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ = grid.Δyᶜᶜᵃ, grid.Δyᶜᶠᵃ, grid.Δyᶠᶜᵃ, grid.Δyᶠᶠᵃ
    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = grid.Azᶜᶜᵃ, grid.Azᶠᶜᵃ, grid.Azᶜᶠᵃ, grid.Azᶠᶠᵃ

    metric_arrays = (Δxᶜᶜᵃ,  Δxᶠᶜᵃ,  Δxᶜᶠᵃ,  Δxᶠᶠᵃ, Δyᶜᶜᵃ,  Δyᶜᶠᵃ,  Δyᶠᶜᵃ,  Δyᶠᶠᵃ, Azᶜᶜᵃ,  Azᶠᶜᵃ,  Azᶜᶠᵃ,  Azᶠᶠᵃ)
    LXs           = (Center, Face,   Center, Face,  Center, Center, Face,   Face,  Center, Face,   Center, Face)
    LYs           = (Center, Center, Face,   Face,  Center, Face,   Center, Face,  Center, Center, Face,   Face)

    for (metric, LX, LY) in zip(metric_arrays, LXs, LYs)
        fill_metric_halo_regions_x!(metric, LX(), LY(), TX(), TY(), Nx, Ny, Hx, Hy)
        fill_metric_halo_regions_y!(metric, LX(), LY(), TX(), TY(), Nx, Ny, Hx, Hy)
        fill_metric_halo_corner_regions!(metric, LX(), LY(), TX(), TY(), Nx, Ny, Hx, Hy)
    end

    return nothing
end

function lat_lon_to_cartesian(lat, lon, radius)
    abs(lat) > 90 && error("lat must be within -90 ≤ lat ≤ 90")
    return [lat_lon_to_x(lat, lon, radius), lat_lon_to_y(lat, lon, radius), lat_lon_to_z(lat, lon, radius)]
end

lat_lon_to_x(lat, lon, radius) = radius * cosd(lon) * cosd(lat)
lat_lon_to_y(lat, lon, radius) = radius * sind(lon) * cosd(lat)
lat_lon_to_z(lat, lon, radius) = radius * sind(lat)

function on_architecture(arch::AbstractSerialArchitecture, grid::OrthogonalSphericalShellGrid)

    coordinates = (:λᶜᶜᵃ,
                   :λᶠᶜᵃ,
                   :λᶜᶠᵃ,
                   :λᶠᶠᵃ,
                   :φᶜᶜᵃ,
                   :φᶠᶜᵃ,
                   :φᶜᶠᵃ,
                   :φᶠᶠᵃ,
                   :z)

    grid_spacings = (:Δxᶜᶜᵃ,
                     :Δxᶠᶜᵃ,
                     :Δxᶜᶠᵃ,
                     :Δxᶠᶠᵃ,
                     :Δyᶜᶜᵃ,
                     :Δyᶠᶜᵃ,
                     :Δyᶜᶠᵃ,
                     :Δyᶠᶠᵃ)

    horizontal_areas = (:Azᶜᶜᵃ,
                        :Azᶠᶜᵃ,
                        :Azᶜᶠᵃ,
                        :Azᶠᶠᵃ)

    coordinate_data = Tuple(on_architecture(arch, getproperty(grid, name)) for name in coordinates)
    grid_spacing_data = Tuple(on_architecture(arch, getproperty(grid, name)) for name in grid_spacings)
    horizontal_area_data = Tuple(on_architecture(arch, getproperty(grid, name)) for name in horizontal_areas)

    TX, TY, TZ = topology(grid)

    new_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                        grid.Nx, grid.Ny, grid.Nz,
                                                        grid.Hx, grid.Hy, grid.Hz,
                                                        grid.Lz,
                                                        coordinate_data...,
                                                        grid_spacing_data...,
                                                        horizontal_area_data...,
                                                        grid.radius,
                                                        grid.conformal_mapping)

    return new_grid
end

function Adapt.adapt_structure(to, grid::OrthogonalSphericalShellGrid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        adapt(to, grid.Lz),
        adapt(to,  grid.λᶜᶜᵃ), adapt(to,  grid.λᶠᶜᵃ), adapt(to,  grid.λᶜᶠᵃ), adapt(to,  grid.λᶠᶠᵃ),
        adapt(to,  grid.φᶜᶜᵃ), adapt(to,  grid.φᶠᶜᵃ), adapt(to,  grid.φᶜᶠᵃ), adapt(to,  grid.φᶠᶠᵃ), adapt(to, grid.z),
        adapt(to, grid.Δxᶜᶜᵃ), adapt(to, grid.Δxᶠᶜᵃ), adapt(to, grid.Δxᶜᶠᵃ), adapt(to, grid.Δxᶠᶠᵃ),
        adapt(to, grid.Δyᶜᶜᵃ), adapt(to, grid.Δyᶠᶜᵃ), adapt(to, grid.Δyᶜᶠᵃ), adapt(to, grid.Δyᶠᶠᵃ),
        adapt(to, grid.Azᶜᶜᵃ), adapt(to, grid.Azᶠᶜᵃ), adapt(to, grid.Azᶜᶠᵃ), adapt(to, grid.Azᶠᶠᵃ),
        adapt(to, grid.radius), adapt(to, grid.conformal_mapping))
end

function Base.summary(grid::OrthogonalSphericalShellGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    metric_computation = isnothing(grid.Δxᶠᶜᵃ) ? "without precomputed metrics" : "with precomputed metrics"

    return string(size_summary(size(grid)),
                  " OrthogonalSphericalShellGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo",
                  " and ", metric_computation)
end

function new_metric(FT, arch, (LX, LY), topo, (Nx, Ny), (Hx, Hy))
    # boost horizontal metrics by 2?
    Nx′ = Nx + 2
    Ny′ = Ny + 2
    Hx′ = Hx + 2
    Hy′ = Hy + 2
    metric = new_data(FT, arch, (LX, LY), topo, (Nx′, Ny′), (Hx′, Hy′))
    return metric
end

"""
    OrthogonalSphericalShellGrid(arch=CPU(), FT=Float64; size, z,
                                 radius = R_Earth,
                                 conformal_mapping = nothing,
                                 halo = (3, 3, 3),
                                 topology = (Bounded, Bounded, Bounded))

Create an OrthogonalSphericalShellGrid with empty horizontal metrics.
"""
function OrthogonalSphericalShellGrid(arch::AbstractArchitecture = CPU(),
                                      FT::DataType = Oceananigans.defaults.FloatType;
                                      size,
                                      z,
                                      radius = R_Earth,
                                      conformal_mapping = nothing,
                                      halo = (3, 3, 3), # TODO: support Flat directions
                                      topology = (Bounded, Bounded, Bounded))

    h_size = size[1:2]
    h_topo = topology[1:2]
    h_halo = halo[1:2]

     λᶜᶜᵃ = new_metric(FT, arch, (Center, Center), h_topo, h_size, h_halo)
     λᶠᶜᵃ = new_metric(FT, arch, (Face,   Center), h_topo, h_size, h_halo)
     λᶜᶠᵃ = new_metric(FT, arch, (Center, Face),   h_topo, h_size, h_halo)
     λᶠᶠᵃ = new_metric(FT, arch, (Face,   Face),   h_topo, h_size, h_halo)

     φᶜᶜᵃ = new_metric(FT, arch, (Center, Center), h_topo, h_size, h_halo)
     φᶠᶜᵃ = new_metric(FT, arch, (Face,   Center), h_topo, h_size, h_halo)
     φᶜᶠᵃ = new_metric(FT, arch, (Center, Face),   h_topo, h_size, h_halo)
     φᶠᶠᵃ = new_metric(FT, arch, (Face,   Face),   h_topo, h_size, h_halo)

    Δxᶜᶜᵃ = new_metric(FT, arch, (Center, Center), h_topo, h_size, h_halo)
    Δxᶠᶜᵃ = new_metric(FT, arch, (Face,   Center), h_topo, h_size, h_halo)
    Δxᶜᶠᵃ = new_metric(FT, arch, (Center, Face),   h_topo, h_size, h_halo)
    Δxᶠᶠᵃ = new_metric(FT, arch, (Face,   Face),   h_topo, h_size, h_halo)

    Δyᶜᶜᵃ = new_metric(FT, arch, (Center, Center), h_topo, h_size, h_halo)
    Δyᶠᶜᵃ = new_metric(FT, arch, (Face,   Center), h_topo, h_size, h_halo)
    Δyᶜᶠᵃ = new_metric(FT, arch, (Center, Face),   h_topo, h_size, h_halo)
    Δyᶠᶠᵃ = new_metric(FT, arch, (Face,   Face),   h_topo, h_size, h_halo)

    Azᶜᶜᵃ = new_metric(FT, arch, (Center, Center), h_topo, h_size, h_halo)
    Azᶠᶜᵃ = new_metric(FT, arch, (Face,   Center), h_topo, h_size, h_halo)
    Azᶜᶠᵃ = new_metric(FT, arch, (Center, Face),   h_topo, h_size, h_halo)
    Azᶠᶠᵃ = new_metric(FT, arch, (Face,   Face),   h_topo, h_size, h_halo)

    Lz, z = generate_coordinate(FT, topology, size, halo, z, :z, 3, arch)

    coordinate_arrays = (λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                         φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
                         z)

    metric_arrays = (Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                     Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                     Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo
    TX, TY, TZ = topology

    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch, Nx, Ny, Nz, Hx, Hy, Hz,
                                                    convert(FT, Lz),
                                                    coordinate_arrays...,
                                                    metric_arrays...,
                                                    convert(FT, radius),
                                                    conformal_mapping)

    return grid
end

"""
    get_center_and_extents_of_shell(grid::OSSG)

Return the latitude-longitude coordinates of the center of the shell `(λ_center, φ_center)`
and also the longitudinal and latitudinal extend of the shell `(extent_λ, extent_φ)`.
"""
function get_center_and_extents_of_shell(grid::OSSG)
    Nx, Ny, _ = size(grid)

    # find the indices that correspond to the center of the shell
    # ÷ ensures that expressions below work for both odd and even
    i_center = Nx÷2 + 1
    j_center = Ny÷2 + 1

    if mod(Nx, 2) == 0
        ℓx = Face()
    elseif mod(Nx, 2) == 1
        ℓx = Center()
    end

    if mod(Ny, 2) == 0
        ℓy = Face()
    elseif mod(Ny, 2) == 1
        ℓy = Center()
    end

    # latitude and longitudes of the shell's center
    λ_center = CUDA.@allowscalar λnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())
    φ_center = CUDA.@allowscalar φnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    if mod(Ny, 2) == 0
        extent_λ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δxᶜᶠᵃ[1:Nx, :], dims=1))) / grid.radius
    elseif mod(Ny, 2) == 1
        extent_λ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δxᶜᶜᵃ[1:Nx, :], dims=1))) / grid.radius
    end

    if mod(Nx, 2) == 0
        extent_φ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
    elseif mod(Nx, 2) == 1
        extent_φ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
    end

    return (λ_center, φ_center), (extent_λ, extent_φ)
end

function Base.show(io::IO, grid::OrthogonalSphericalShellGrid, withsummary=true)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)

    Nx_face, Ny_face = total_length(Face(), TX(), Nx, 0), total_length(Face(), TY(), Ny, 0)

    λ₁, λ₂ = minimum(grid.λᶠᶠᵃ[1:Nx_face, 1:Ny_face]), maximum(grid.λᶠᶠᵃ[1:Nx_face, 1:Ny_face])
    φ₁, φ₂ = minimum(grid.φᶠᶠᵃ[1:Nx_face, 1:Ny_face]), maximum(grid.φᶠᶠᵃ[1:Nx_face, 1:Ny_face])
    Ωz = domain(topology(grid, 3)(), Nz, grid.z.cᵃᵃᶠ)

    (λ_center, φ_center), (extent_λ, extent_φ) = get_center_and_extents_of_shell(grid)

    λ_center = round(λ_center, digits=4)
    φ_center = round(φ_center, digits=4)

    λ_center = ifelse(λ_center ≈ 0, 0.0, λ_center)
    φ_center = ifelse(φ_center ≈ 0, 0.0, φ_center)

    center_str = "centered at (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"

    if φ_center ≈ 90
        center_str = "centered at: North Pole, (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"
    end

    if φ_center ≈ -90
        center_str = "centered at: South Pole, (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"
    end

    λ_summary = "$(TX)  extent $(prettysummary(extent_λ)) degrees"
    φ_summary = "$(TY)  extent $(prettysummary(extent_φ)) degrees"
    z_summary = domain_summary(TZ(), "z", Ωz)

    longest = max(length(λ_summary), length(φ_summary), length(z_summary))

    padding_λ = length(λ_summary) < longest ? " "^(longest - length(λ_summary)) : ""
    padding_φ = length(φ_summary) < longest ? " "^(longest - length(φ_summary)) : ""

    λ_summary = "longitude: $(TX)  extent $(prettysummary(extent_λ)) degrees" * padding_λ * " " *
                coordinate_summary(TX, rad2deg.(grid.Δxᶠᶠᵃ[1:Nx_face, 1:Ny_face] ./ grid.radius), "λ")

    φ_summary = "latitude:  $(TY)  extent $(prettysummary(extent_φ)) degrees" * padding_φ * " " *
                coordinate_summary(TY, rad2deg.(grid.Δyᶠᶠᵃ[1:Nx_face, 1:Ny_face] ./ grid.radius), "φ")

    z_summary = "z:         " * dimension_summary(TZ(), "z", Ωz, grid.z, longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", center_str, "\n",
                     "├── ", λ_summary, "\n",
                     "├── ", φ_summary, "\n",
                     "└── ", z_summary)
end

function nodes(grid::OSSG, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos)
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        # λ and φ are 2D arrays
        N = (size(λ)..., size(z)...)
        λ = Base.reshape(λ, N[1], N[2], 1)
        φ = Base.reshape(φ, N[1], N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (λ, φ, z)
end

@inline λnodes(grid::OSSG, ℓx::Face,   ℓy::Face, ; with_halos=false) = with_halos ? grid.λᶠᶠᵃ :
    view(grid.λᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Face,   ℓy::Center; with_halos=false) = with_halos ? grid.λᶠᶜᵃ :
    view(grid.λᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Center, ℓy::Face, ; with_halos=false) = with_halos ? grid.λᶜᶠᵃ :
    view(grid.λᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) = with_halos ? grid.λᶜᶜᵃ :
    view(grid.λᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline φnodes(grid::OSSG, ℓx::Face,   ℓy::Face, ; with_halos=false) = with_halos ? grid.φᶠᶠᵃ :
    view(grid.φᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Face,   ℓy::Center; with_halos=false) = with_halos ? grid.φᶠᶜᵃ :
    view(grid.φᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Center, ℓy::Face, ; with_halos=false) = with_halos ? grid.φᶜᶠᵃ :
    view(grid.φᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) = with_halos ? grid.φᶜᶜᵃ :
    view(grid.φᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline xnodes(grid::OSSG, ℓx, ℓy; with_halos=false) = grid.radius * deg2rad.(λnodes(grid, ℓx, ℓy; with_halos=with_halos)) .* hack_cosd.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))
@inline ynodes(grid::OSSG, ℓx, ℓy; with_halos=false) = grid.radius * deg2rad.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))

# convenience
@inline λnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = λnodes(grid, ℓx, ℓy; with_halos)
@inline φnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = φnodes(grid, ℓx, ℓy; with_halos)
@inline xnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓx, ℓy; with_halos)

@inline λnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.λᶜᶜᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.λᶠᶠᵃ[i, j]

@inline φnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.φᶜᶜᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.φᶠᶠᵃ[i, j]

@inline xnode(i, j, grid::OSSG, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, j, grid, ℓx, ℓy)) * hack_cosd((φnode(i, j, grid, ℓx, ℓy)))
@inline ynode(i, j, grid::OSSG, ℓx, ℓy) = grid.radius * deg2rad(φnode(i, j, grid, ℓx, ℓy))

# convenience
@inline λnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline φnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline xnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = xnode(i, j, grid, ℓx, ℓy)
@inline ynode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = ynode(i, j, grid, ℓx, ℓy)

# Definitions for node
@inline ξnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline ηnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)

ξname(::OSSG) = :λ
ηname(::OSSG) = :φ
rname(::OSSG) = :z

#####
##### Grid spacings in x, y, z (in meters)
#####

@inline xspacings(grid::OSSG, ℓx, ℓy) = xspacings(grid, ℓx, ℓy, nothing)
@inline yspacings(grid::OSSG, ℓx, ℓy) = yspacings(grid, ℓx, ℓy, nothing)

@inline λspacings(grid::OSSG, ℓx, ℓy) = λspacings(grid, ℓx, ℓy, nothing)
@inline φspacings(grid::OSSG, ℓx, ℓy) = φspacings(grid, ℓx, ℓy, nothing)
