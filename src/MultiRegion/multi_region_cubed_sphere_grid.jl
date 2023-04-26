using Oceananigans.Architectures: architecture
using Oceananigans.Grids: R_Earth, halo_size, size_summary, total_length, topology

using Rotations

const ConformalCubedSphereGrid{FT, TX, TY, TZ} = MultiRegionGrid{FT, TX, TY, TZ, <:CubedSpherePartition}

rotation_from_panel_index(idx) = idx == 1 ? RotX(π/2)*RotY(π/2) :
                                 idx == 2 ? RotY(π)*RotX(-π/2) :
                                 idx == 3 ? RotZ(π) :
                                 idx == 4 ? RotX(π)*RotY(-π/2) :
                                 idx == 5 ? RotY(π/2)*RotX(π/2) :
                                 idx == 6 ? RotZ(π/2)*RotX(π) :
                                 error("invalid panel index")

"""
    ConformalCubedSphereGrid(arch::AbstractArchitecture, FT=Float64;
                             panel_size,
                             z,
                             horizontal_direction_halo = 1
                             z_halo = horizontal_direction_halo,
                             z_topology = Bounded,
                             radius = R_Earth,
                             partition = CubedSpherePartition(; R=1),
                             devices = nothing)

Return a `ConformalCubedSphereGrid` that comprises of six [`OrthogonalSphericalShellGrid`](@ref);
we refer to each of these grids as a "panel". Each panel corresponds
to a face of the cube.

The keywords prescribe the properties of each of the panels. Only the topology in the vertical
direction can be prescribed and that's done via the `z_topology` keyword argumet (default: `Bounded`).
Topologies in both horizontal directions for a `ConformalCubedSphereGrid` are _always_ [`FullyConnected`](@ref).

Halo size in both horizontal dimensions must be equal and they are prescribed via the
`horizontal_direction_halo` keyword argument. The number of halo points in the ``z``-directions
is given by `z_halo`.

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

A `CubedSpherePartition(; R=2)` implies partition in 2 in each
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

```jldoctest cubedspheregrid
julia> using Oceananigans

julia> grid = ConformalCubedSphereGrid(panel_size=(12, 12, 1), z=(-1, 0), radius=1)
ConformalCubedSphereGrid{Float64, FullyConnected, FullyConnected, Bounded} partitioned on CPU():
├── grids: 12×12×1 OrthogonalSphericalShellGrid{Float64, FullyConnected, FullyConnected, Bounded} on CPU with 1×1×1 halo and with precomputed metrics
├── partitioning: CubedSpherePartition with (1 region in each panel)
└── devices: (CPU(), CPU(), CPU(), CPU(), CPU(), CPU())
```

To determine all the connectivities of a grid we can call, e.g.,

```jldoctest cubedspheregrid
julia> using Oceananigans.MultiRegion: inject_west_boundary, inject_east_boundary, inject_north_boundary, inject_south_boundary, East, West, South, North

julia> using Oceananigans.MultiRegion: CubedSphereConnectivity

julia> for region in 1:length(grid.partition); println("panel ", region, " :", inject_south_boundary(region, grid.partition, 1).condition); end
panel 1 :CubedSphereConnectivity{South, North}(1, 6, South(), North())
panel 2 :CubedSphereConnectivity{South, East}(2, 6, South(), East())
panel 3 :CubedSphereConnectivity{South, North}(3, 2, South(), North())
panel 4 :CubedSphereConnectivity{South, East}(4, 2, South(), East())
panel 5 :CubedSphereConnectivity{South, North}(5, 4, South(), North())
panel 6 :CubedSphereConnectivity{South, East}(6, 4, South(), East())
```
"""
function ConformalCubedSphereGrid(arch::AbstractArchitecture=CPU(), FT=Float64;
                                  panel_size,
                                  z,
                                  horizontal_direction_halo = 1,
                                  z_halo = horizontal_direction_halo,
                                  z_topology = Bounded,
                                  radius = R_Earth,
                                  partition = CubedSpherePartition(; R=1),
                                  devices = nothing)

    Nx, Ny, _ = panel_size
    region_topology = (FullyConnected, FullyConnected, z_topology)
    region_halo = (horizontal_direction_halo, horizontal_direction_halo, z_halo)

    Nx !== Ny && error("Horizontal sizes for ConformalCubedSphereGrid must be equal; Nx=Ny.")

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
                                        halo = region_halo,
                                        topology = region_topology,
                                        radius,
                                        ξ = region_ξ,
                                        η = region_η,
                                        rotation = region_rotation)

    grid = MultiRegionGrid{FT, region_topology[1], region_topology[2], region_topology[3]}(arch, partition, region_grids, devices)

    CUDA.@allowscalar begin
        λcca = Field{Center, Center, Nothing}(grid)
        φcca = Field{Center, Center, Nothing}(grid)

        for region in 1:length(grid), j in 1:Ny, i in 1:Nx
            getregion(λcca, region).data[i, j] = getregion(grid, region).λᶜᶜᵃ[i, j]
            getregion(φcca, region).data[i, j] = getregion(grid, region).φᶜᶜᵃ[i, j]
        end

        fill_halo_regions!(λcca)
        fill_halo_regions!(φcca)

        for region in 1:length(grid)
            getregion(grid, region).λᶜᶜᵃ .= getregion(λcca, region).data
            getregion(grid, region).φᶜᶜᵃ .= getregion(φcca, region).data
        end

        λfca = Field{Face, Center, Nothing}(grid)
        φfca = Field{Face, Center, Nothing}(grid)

        for region in 1:length(grid), j in 1:Ny, i in 1:Nx
            getregion(λfca, region).data[i, j] = getregion(grid, region).λᶠᶜᵃ[i, j]
            getregion(φfca, region).data[i, j] = getregion(grid, region).φᶠᶜᵃ[i, j]
        end

        fill_halo_regions!(λfca)
        fill_halo_regions!(φfca)

        for region in 1:length(grid)
            getregion(grid, region).λᶠᶜᵃ .= getregion(λfca, region).data
            getregion(grid, region).φᶠᶜᵃ .= getregion(φfca, region).data
        end

        λcfa = Field{Center, Face, Nothing}(grid)
        φcfa = Field{Center, Face, Nothing}(grid)

        for region in 1:length(grid), j in 1:Ny, i in 1:Nx
            getregion(λcfa, region).data[i, j] = getregion(grid, region).λᶜᶠᵃ[i, j]
            getregion(φcfa, region).data[i, j] = getregion(grid, region).φᶜᶠᵃ[i, j]
        end

        fill_halo_regions!(λcfa)
        fill_halo_regions!(φcfa)

        for region in 1:length(grid)
            getregion(grid, region).λᶜᶠᵃ .= getregion(λcfa, region).data
            getregion(grid, region).φᶜᶠᵃ .= getregion(φcfa, region).data
        end

        λffa = Field{Face, Face, Nothing}(grid)
        φffa = Field{Face, Face, Nothing}(grid)

        for region in 1:length(grid), j in 1:Ny, i in 1:Nx
            getregion(λffa, region).data[i, j] = getregion(grid, region).λᶠᶠᵃ[i, j]
            getregion(φffa, region).data[i, j] = getregion(grid, region).φᶠᶠᵃ[i, j]
        end

        fill_halo_regions!(λffa)
        fill_halo_regions!(φffa)

        for region in 1:length(grid)
            getregion(grid, region).λᶠᶠᵃ .= getregion(λffa, region).data
            getregion(grid, region).φᶠᶠᵃ .= getregion(φffa, region).data
        end
    end

    return grid
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

    # to load a ConformalCubedSphereGrid from file we can only have a 6-panel partition, i.e. R=1
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

function with_halo(new_halo, csg::ConformalCubedSphereGrid) 
    
    region_rotation = []

    for r in 1:length(csg.partition)
        push!(region_rotation, rotation_from_panel_index(panel_index(r, csg.partition)))
    end

    apply_regionally!(with_halo, new_halo, csg; rotation = Iterate(region_rotation))

    return csg
end

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
