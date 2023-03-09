using Oceananigans.Architectures: architecture
using Oceananigans.Grids: R_Earth, halo_size, size_summary

using Rotations

const ConformalCubedSphereGrid{FT, TX, TY, TZ} = MultiRegionGrid{FT, TX, TY, TZ, <:CubedSpherePartition}

rotation_from_panel_index(idx) = idx == 1 ? RotX(π/2)*RotY(π/2) :
                                 idx == 2 ? RotY(π)*RotX(-π/2) :
                                 idx == 3 ? RotZ(π) :
                                 idx == 4 ? RotX(π)*RotY(-π/2) :
                                 idx == 5 ? RotY(π/2)*RotX(π/2) :
                                 RotZ(π/2)*RotX(π)

"""
    ConformalCubedSphereGrid(arch::AbstractArchitecture, FT=Float64;
                             panel_size,
                             z,
                             panel_halo = (1, 1, 1),
                             panel_topology = (FullyConnected, FullyConnected, Bounded),
                             radius = R_Earth,
                             partition = CubedSpherePartition(), 
                             devices = nothing)

Return a `ConformalCubedSphereGrid` that comprises of six [`OrthogonalSphericalShellGrid`](@ref);
we refer to each of these grids as a "panel". Each panel corresponds
to a face of the cube.

The keywords prescribe the properties of each of the panels.

The connectivity between the `ConformalCubedSphereGrid` faces is depicted below.

```
                          +==========+==========+
                          ∥    ↑     ∥    ↑     ∥
                          ∥    1W    ∥    1S    ∥
                          ∥←3N P5 6W→∥←5E P6 2S→∥
                          ∥    4N    ∥    4E    ∥
                          ∥    ↓     ∥    ↓     ∥
               +==========+==========+==========+
               ∥    ↑     ∥    ↑     ∥
               ∥    5W    ∥    5S    ∥
               ∥←1N P3 4W→∥←3E P4 6S→∥
               ∥    2N    ∥    2E    ∥
               ∥    ↓     ∥    ↓     ∥
    +==========+==========+==========+
    ∥    ↑     ∥    ↑     ∥
    ∥    3W    ∥    3S    ∥
    ∥←5N P1 2W→∥←1E P2 4S→∥
    ∥    6N    ∥    6E    ∥
    ∥    ↓     ∥    ↓     ∥
    +==========+==========+
```

The North Pole of the sphere is in panel 3 (P3) and the South Pole in panel 6 (P6).

A `CubedSpherePartition(; Rx=2)` implies partition in 2 in each
dimension of each panel resulting in 24 regions. In each partition
the intra-panel `x, y` indices are in written in the center and the
overall region index on the bottom right.

```
                                                +==========+==========+==========+==========+
                                                ∥    ↑     |    ↑     ∥    ↑     |    ↑     ∥
                                                ∥          |          ∥          |          ∥
                                                ∥← (1, 2) →|← (2, 2) →∥← (1, 2) →|← (2, 2) →∥
                                                ∥          |          ∥          |          ∥
                                                ∥    ↓  19 |    ↓  20 ∥    ↓  23 |    ↓  24 ∥
                                                +-------- P 5 --------+-------- P 6 --------+
                                                ∥    ↑     |    ↑     ∥    ↑     |    ↑     ∥
                                                ∥          |          ∥          |          ∥
                                                ∥← (1, 1) →|← (2, 1) →∥← (1, 1) →|← (2, 1) →∥
                                                ∥          |          ∥          |          ∥
                                                ∥    ↓  17 |    ↓  18 ∥    ↓  21 |    ↓  22 ∥
                          +==========+==========+==========+==========+==========+==========+
                          ∥    ↑     |    ↑     ∥    ↑     |    ↑     ∥
                          ∥          |          ∥          |          ∥
                          ∥← (1, 2) →|← (2, 2) →∥← (1, 2) →|← (2, 2) →∥
                          ∥          |          ∥          |          ∥
                          ∥    ↓ 11  |    ↓  12 ∥    ↓  15 |    ↓  16 ∥
                          +-------- P 3 --------+-------- P 4 --------+
                          ∥    ↑     |    ↑     ∥    ↑     |    ↑     ∥
                          ∥          |          ∥          |          ∥
                          ∥← (1, 1) →|← (2, 1) →∥← (1, 1) →|← (2, 1) →∥
                          ∥          |          ∥          |          ∥
                          ∥    ↓  9  |    ↓  10 ∥    ↓  13 |    ↓  14 ∥
    +==========+==========+==========+==========+==========+==========+
    ∥    ↑     |    ↑     ∥    ↑     |    ↑     ∥
    ∥          |          ∥          |          ∥
    ∥← (1, 2) →|← (2, 2) →∥← (1, 2) →|← (2, 2) →∥
    ∥          |          ∥          |          ∥
    ∥    ↓   3 |    ↓   4 ∥    ↓   7 |    ↓   8 ∥
    +-------- P 1 --------+-------- P 2 --------+
    ∥    ↑     |    ↑     ∥    ↑     |    ↑     ∥
    ∥          |          ∥          |          ∥
    ∥← (1, 1) →|← (2, 1) →∥← (1, 1) →|← (2, 1) →∥ 
    ∥          |          ∥          |          ∥
    ∥    ↓   1 |    ↓   2 ∥    ↓   5 |    ↓   6 ∥
    +==========+==========+==========+==========+
```

Below, we show in detail panels 1 and 2 and the connectivity
of each panel.

```
+===============+==============+==============+===============+
∥       ↑       |      ↑       ∥      ↑       |      ↑        ∥
∥      11W      |      9W      ∥      9S      |     10S       ∥
∥←19N (2, 1) 4W→|←3E (2, 2) 7W→∥←4E (2, 1) 8W→|←7E (2, 2) 13S→∥
∥       1N      |      2N      ∥      5N      |      6N       ∥
∥       ↓     3 |      ↓     4 ∥      ↓     7 |      ↓      8 ∥
+------------- P 1 ------------+------------ P 2 -------------+
∥       ↑       |      ↑       ∥      ↑       |      ↑        ∥
∥       3S      |      4S      ∥      7S      |      8S       ∥
∥←20N (1, 1) 2W→|←1E (2, 1) 5W→∥←2E (1, 1) 6W→|←5E (2, 1) 14S→∥
∥      23N      |     24N      ∥     24N      |     22N       ∥
∥       ↓     1 |      ↓     2 ∥      ↓     5 |      ↓      6 ∥
+===============+==============+==============+===============+
```

Example
=======

```@example
julia> using Oceananigans

julia> grid = ConformalCubedSphereGrid(panel_size=(10, 10, 1), z=(-1, 0), radius=1.0)
ConformalCubedSphereGrid{Float64, FullyConnected, FullyConnected, Bounded} partitioned on CPU():
├── grids: 10×10×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics 
├── partitioning: CubedSpherePartition with (1 region in each panel)
└── devices: (CPU(), CPU(), CPU(), CPU(), CPU(), CPU())
```

To determine all the connectivities of a grid we can call, e.g.,

```@example
julia> using Oceananigans.MultiRegion: inject_west_boundary, inject_east_boundary, inject_north_boundary, inject_south_boundary

julia> using Oceananigans.MultiRegion: CubedSphereConnectivity

julia> for j in 1:length(grid.partition); println("panel ", j, " :", inject_south_boundary(j, grid.partition, 1).condition); end
panel 1: CubedSphereConnectivity(1, 23, :south, :north)
panel 2: CubedSphereConnectivity(2, 24, :south, :north)
panel 3: CubedSphereConnectivity(3, 1, :south, :north)
panel 4: CubedSphereConnectivity(4, 2, :south, :north)
panel 5: CubedSphereConnectivity(5, 24, :south, :east)
panel 6: CubedSphereConnectivity(6, 22, :south, :east)
panel 7: CubedSphereConnectivity(7, 5, :south, :north)
panel 8: CubedSphereConnectivity(8, 6, :south, :north)
panel 9: CubedSphereConnectivity(9, 7, :south, :north)
panel 10: CubedSphereConnectivity(10, 8, :south, :north)
panel 11: CubedSphereConnectivity(11, 9, :south, :north)
panel 12: CubedSphereConnectivity(12, 10, :south, :north)
panel 13: CubedSphereConnectivity(13, 8, :south, :east)
panel 14: CubedSphereConnectivity(14, 6, :south, :east)
panel 15: CubedSphereConnectivity(15, 13, :south, :north)
panel 16: CubedSphereConnectivity(16, 14, :south, :north)
panel 17: CubedSphereConnectivity(17, 15, :south, :north)
panel 18: CubedSphereConnectivity(18, 16, :south, :north)
panel 19: CubedSphereConnectivity(19, 17, :south, :north)
panel 20: CubedSphereConnectivity(20, 18, :south, :north)
panel 21: CubedSphereConnectivity(21, 16, :south, :east)
panel 22: CubedSphereConnectivity(22, 14, :south, :east)
panel 23: CubedSphereConnectivity(23, 21, :south, :north)
panel 24: CubedSphereConnectivity(24, 22, :south, :north)
```
"""
function ConformalCubedSphereGrid(arch::AbstractArchitecture=CPU(), FT=Float64;
                                  panel_size,
                                  z,
                                  panel_halo = (1, 1, 1),
                                  panel_topology = (FullyConnected, FullyConnected, Bounded),
                                  radius = R_Earth,
                                  partition = CubedSpherePartition(; R=1),
                                  devices = nothing)

    devices = validate_devices(partition, arch, devices)
    devices = assign_devices(partition, devices)

    region_size = []
    region_η = []
    region_ξ = []
    region_rotation = []

    for r in 1:length(partition)
        # for a whole cube's face (ξ, η) ∈ [-1, 1]x[-1, 1]
        Lξᵢⱼ = 2 / Rx(r, partition)
        Lηᵢⱼ = 2 / Ry(r, partition)

        pᵢ = intra_panel_index_x(r, partition)
        pⱼ = intra_panel_index_y(r, partition)

        push!(region_size, (panel_size[1] ÷ Rx(r, partition), panel_size[2] ÷ Ry(r, partition), panel_size[3]))
        push!(region_ξ, (-1 + Lξᵢⱼ * (pᵢ - 1), -1 + Lξᵢⱼ * pᵢ))
        push!(region_η, (-1 + Lηᵢⱼ * (pⱼ - 1), -1 + Lηᵢⱼ * pⱼ))
        push!(region_rotation, rotation_from_panel_index(panel_index(r, partition)))
    end

    region_size = MultiRegionObject(tuple(region_size...), devices)
    region_ξ = Iterate(region_ξ)
    region_η = Iterate(region_η)
    region_rotation = Iterate(region_rotation)

    region_grids = construct_regionally(OrthogonalSphericalShellGrid, arch, FT;
                                        size = region_size,
                                        z,
                                        halo = panel_halo,
                                        radius,
                                        ξ = region_ξ,
                                        η = region_η,
                                        rotation = region_rotation)

    return MultiRegionGrid{FT, panel_topology[1], panel_topology[2], panel_topology[3]}(arch, partition, region_grids, devices)
end

"""
    ConformalCubedSphereGrid(filepath::AbstractString, arch::AbstractArchitecture=CPU(), FT=Float64;
                             Nz,
                             z,
                             panel_halo = (1, 1, 1),
                             panel_topology = (FullyConnected, FullyConnected, Bounded),
                             radius = R_Earth,
                             devices = nothing)

Load a `ConformalCubedSphereGrid` from `filepath`.
"""
function ConformalCubedSphereGrid(filepath::AbstractString, arch::AbstractArchitecture=CPU(), FT=Float64;
                                  Nz,
                                  z,
                                  panel_halo = (1, 1, 1),
                                  panel_topology = (FullyConnected, FullyConnected, Bounded),
                                  radius = R_Earth,
                                  devices = nothing)

    # to load a ConformalCubedSphereGrid from file we can only have a 6-panel partition
    partition = CubedSpherePartition(; R=1)

    devices = validate_devices(partition, arch, devices)
    devices = assign_devices(partition, devices)

    region_Nz = MultiRegionObject(Tuple(repeat([Nz], length(partition))), devices)
    region_panels = Iterate(Array(1:length(partition)))

    region_grids = construct_regionally(OrthogonalSphericalShellGrid, filepath, arch, FT;
                                        Nz = region_Nz,
                                        z,
                                        panel = region_panels,
                                        topology = panel_topology,
                                        halo = panel_halo,
                                        radius)

    return MultiRegionGrid{FT, panel_topology[1], panel_topology[2], panel_topology[3]}(arch, partition, region_grids, devices)
end

with_halo(new_halo, csg::ConformalCubedSphereGrid) = apply_regionally!(with_halo, new_halo, csg)

function Base.summary(grid::ConformalCubedSphereGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    return string(size_summary(size(grid)),
                  " ConformalCubedSphereGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

Base.show(io::IO, mrg::ConformalCubedSphereGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    print(io, "ConformalCubedSphereGrid{$FT, $TX, $TY, $TZ} partitioned on $(architecture(mrg)): \n",
              "├── grids: $(summary(mrg.region_grids[1])) \n",
              "├── partitioning: $(summary(mrg.partition)) \n",
              "└── devices: $(devices(mrg))")

"""
```
julia> grid = ConformalCubedSphereGrid(panel_size=(10, 10, 1), z=(-1, 0), radius=1.0)
┌ Warning: OrthogonalSphericalShellGrid is still under development. Use with caution!
└ @ Oceananigans.Grids ~/Research/OC.jl/src/Grids/orthogonal_spherical_shell_grid.jl:136
┌ Warning: OrthogonalSphericalShellGrid is still under development. Use with caution!
└ @ Oceananigans.Grids ~/Research/OC.jl/src/Grids/orthogonal_spherical_shell_grid.jl:136
┌ Warning: OrthogonalSphericalShellGrid is still under development. Use with caution!
└ @ Oceananigans.Grids ~/Research/OC.jl/src/Grids/orthogonal_spherical_shell_grid.jl:136
┌ Warning: OrthogonalSphericalShellGrid is still under development. Use with caution!
└ @ Oceananigans.Grids ~/Research/OC.jl/src/Grids/orthogonal_spherical_shell_grid.jl:136
┌ Warning: OrthogonalSphericalShellGrid is still under development. Use with caution!
└ @ Oceananigans.Grids ~/Research/OC.jl/src/Grids/orthogonal_spherical_shell_grid.jl:136
┌ Warning: OrthogonalSphericalShellGrid is still under development. Use with caution!
└ @ Oceananigans.Grids ~/Research/OC.jl/src/Grids/orthogonal_spherical_shell_grid.jl:136
ConformalCubedSphereGrid{Float64, FullyConnected, FullyConnected, Bounded} partitioned on CPU(): 
├── grids: 10×10×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics 
├── partitioning: CubedSpherePartition with (1 region in each panel) 
└── devices: (CPU(), CPU(), CPU(), CPU(), CPU(), CPU())

julia> field = CenterField(grid)

julia> @apply_regionally set!(field, (x, y, z) -> y)

julia> field = CenterField(grid)
10×10×1 Field{Center, Center, Center} on MultiRegionGrid on CPU
├── grid: MultiRegionGrid{Float64, FullyConnected, FullyConnected, Bounded} with CubedSpherePartition{Int64, Int64} on OrthogonalSphericalShellGrid
├── boundary conditions: MultiRegionObject{NTuple{6, FieldBoundaryConditions{BoundaryCondition{Oceananigans.BoundaryConditions.Communication, Oceananigans.MultiRegion.CubedSphereConnectivity}, BoundaryCondition{Oceananigans.BoundaryConditions.Communication, Oceananigans.MultiRegion.CubedSphereConnectivity}, BoundaryCondition{Oceananigans.BoundaryConditions.Communication, Oceananigans.MultiRegion.CubedSphereConnectivity}, BoundaryCondition{Oceananigans.BoundaryConditions.Communication, Oceananigans.MultiRegion.CubedSphereConnectivity}, BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}}}, NTuple{6, CPU}}
└── data: MultiRegionObject{NTuple{6, OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}}, NTuple{6, CPU}}
    └── max=0.0, min=0.0, mean=0.0

julia> using Oceananigans.MultiRegion: getregion

julia> field_panel_1 = getregion(field, 1)
10×10×1 Field{Center, Center, Center} on OrthogonalSphericalShellGrid on CPU
├── grid: 10×10×1 OrthogonalSphericalShellGrid{Float64, Bounded, Bounded, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── boundary conditions: FieldBoundaryConditions
│   └── west: MultiRegionCommunication, east: MultiRegionCommunication, south: MultiRegionCommunication, north: MultiRegionCommunication, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 12×12×3 OffsetArray(::Array{Float64, 3}, 0:11, 0:11, 0:2) with eltype Float64 with indices 0:11×0:11×0:2
    └── max=40.5277, min=-40.5277, mean=-2.13163e-16

julia> using Oceananigans.Utils: Iterate

julia> regions = Iterate(Tuple(j for j in 1:length(grid.partition)))
Iterate{NTuple{6, Int64}}((13, 13, 13, 13, 13, 13))

julia> set!(field, regions)

julia> using Oceananigans.BoundaryConditions: fill_halo_regions!

julia> fill_halo_regions!(field)
"""
