using Oceananigans.Architectures: architecture
using Oceananigans.Grids: conformal_cubed_sphere_panel,
                          R_Earth,
                          halo_size,
                          size_summary,
                          total_length,
                          topology

using CubedSphere
using Distances

import Oceananigans.Grids: grid_name

const ConformalCubedSphereGrid{FT, TX, TY, TZ} = MultiRegionGrid{FT, TX, TY, TZ, <:CubedSpherePartition}

"""
    ConformalCubedSphereGrid(arch=CPU(), FT=Float64;
                             panel_size,
                             z,
                             horizontal_direction_halo = 1,
                             z_halo = horizontal_direction_halo,
                             horizontal_topology = FullyConnected,
                             z_topology = Bounded,
                             radius = R_Earth,
                             partition = CubedSpherePartition(; R = 1),
                             devices = nothing)

Return a `ConformalCubedSphereGrid` that comprises of six [`conformal_cubed_sphere_panel`](@ref)
grids; we refer to each of these grids as a "panel". Each panel corresponds to a face of the cube.

The keyword arguments prescribe the properties of each of the panels. Only the topology in
the vertical direction can be prescribed and that's done via the `z_topology` keyword
argumet (default: `Bounded`). Topologies in both horizontal directions for a `ConformalCubedSphereGrid`
are _always_ [`FullyConnected`](@ref).

Halo size in both horizontal dimensions _must_ be equal; this is prescribed via the
`horizontal_halo :: Integer` keyword argument. The number of halo points in the ``z``-direction
is prescribed by the `z_halo :: Integer` keyword argument.

The connectivity between the `ConformalCubedSphereGrid` panels is depicted below.

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

The North Pole of the sphere lies in the center of panel 3 (P3) and the South Pole
in the center of panel 6 (P6).

The `partition` keyword argument prescribes the partitioning in regions within each 
panel; see [`CubedSpherePartition`](@ref). For example, a `CubedSpherePartition(; R=2)`
implies that each of the panels are partitioned into 2 regions in each dimension;
this adds up, e.g., to 24 regions for the  whole sphere. In the depiction below,
the intra-panel `x, y` indices are depicted in the center of each region and the overall
region index is shown at the bottom right of each region.

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
├── grids: 12×12×1 OrthogonalSphericalShellGrid{Float64, FullyConnected, FullyConnected, Bounded} on CPU with 3×3×3 halo and with precomputed metrics 
├── partitioning: CubedSpherePartition with (1 region in each panel) 
├── connectivity: CubedSphereConnectivity 
└── devices: (CPU(), CPU(), CPU(), CPU(), CPU(), CPU())
```

The connectivities of the regions of our grid are stored in `grid.connectivity`.
For example, to find out all connectivites on the South boundary of each region we call

```jldoctest cubedspheregrid
julia> using Oceananigans.MultiRegion: East, North, West, South

julia> for region in 1:length(grid); println(grid.connectivity.connections[region].south); end
CubedSphereRegionalConnectivity
├── from: Oceananigans.MultiRegion.North side, region 6
├── to:   Oceananigans.MultiRegion.South side, region 1
└── no rotation
CubedSphereRegionalConnectivity
├── from: Oceananigans.MultiRegion.East side, region 6
├── to:   Oceananigans.MultiRegion.South side, region 2
└── counter-clockwise rotation ↺
CubedSphereRegionalConnectivity
├── from: Oceananigans.MultiRegion.North side, region 2
├── to:   Oceananigans.MultiRegion.South side, region 3
└── no rotation
CubedSphereRegionalConnectivity
├── from: Oceananigans.MultiRegion.East side, region 2
├── to:   Oceananigans.MultiRegion.South side, region 4
└── counter-clockwise rotation ↺
CubedSphereRegionalConnectivity
├── from: Oceananigans.MultiRegion.North side, region 4
├── to:   Oceananigans.MultiRegion.South side, region 5
└── no rotation
CubedSphereRegionalConnectivity
├── from: Oceananigans.MultiRegion.East side, region 4
├── to:   Oceananigans.MultiRegion.South side, region 6
└── counter-clockwise rotation ↺
```
"""
function ConformalCubedSphereGrid(arch::AbstractArchitecture=CPU(), FT=Float64;
                                  panel_size,
                                  z,
                                  horizontal_direction_halo = 3,
                                  z_halo = horizontal_direction_halo,
                                  horizontal_topology = FullyConnected,
                                  z_topology = Bounded,
                                  radius = R_Earth,
                                  partition = CubedSpherePartition(; R = 1),
                                  devices = nothing)

    Nx, Ny, _ = panel_size
    region_topology = (horizontal_topology, horizontal_topology, z_topology)
    region_halo = (horizontal_direction_halo, horizontal_direction_halo, z_halo)

    Nx !== Ny && error("Horizontal sizes for ConformalCubedSphereGrid must be equal; Nx=Ny.")

    # first we construct the grid on CPU and convert to user-prescribed architecture later...
    devices = validate_devices(partition, CPU(), devices)
    devices = assign_devices(partition, devices)

    connectivity = CubedSphereConnectivity(devices, partition)

    region_size = []
    region_η = []
    region_ξ = []
    region_rotation = []

    for r in 1:length(partition)
        Lξ_total, Lη_total = 2, 2 # a cube's face has (ξ, η) ∈ [-1, 1] x [-1, 1]
        Lξᵢⱼ = Lξ_total / Rx(r, partition)
        Lηᵢⱼ = Lη_total / Ry(r, partition)

        pᵢ = intra_panel_index_x(r, partition)
        pⱼ = intra_panel_index_y(r, partition)

        push!(region_size, (panel_size[1] ÷ Rx(r, partition), panel_size[2] ÷ Ry(r, partition), panel_size[3]))
        push!(region_ξ, (-1 + Lξᵢⱼ * (pᵢ - 1), -1 + Lξᵢⱼ * pᵢ))
        push!(region_η, (-1 + Lηᵢⱼ * (pⱼ - 1), -1 + Lηᵢⱼ * pⱼ))
        push!(region_rotation, connectivity.rotations[panel_index(r, partition)])
    end

    region_size = MultiRegionObject(tuple(region_size...), devices)
    region_ξ = Iterate(region_ξ)
    region_η = Iterate(region_η)
    region_rotation = Iterate(region_rotation)

    # as mentioned above, construct the grid on CPU and convert to user-prescribed architecture later...
    region_grids = construct_regionally(conformal_cubed_sphere_panel, CPU(), FT;
                                        size = region_size,
                                        z,
                                        halo = region_halo,
                                        topology = region_topology,
                                        radius,
                                        ξ = region_ξ,
                                        η = region_η,
                                        rotation = region_rotation)

    grid = MultiRegionGrid{FT, region_topology...}(CPU(),
                                                   partition,
                                                   connectivity,
                                                   region_grids,
                                                   devices)

    λᶜᶜᵃ  = Field((Center, Center, Nothing), grid)
    φᶜᶜᵃ  = Field((Center, Center, Nothing), grid)
    Azᶜᶜᵃ = Field((Center, Center, Nothing), grid)
    λᶠᶠᵃ  = Field((Face,   Face,   Nothing), grid)
    φᶠᶠᵃ  = Field((Face,   Face,   Nothing), grid)
    Azᶠᶠᵃ = Field((Face,   Face,   Nothing), grid)

    for (field, name) in zip(( λᶜᶜᵃ, φᶜᶜᵃ,   Azᶜᶜᵃ,  λᶠᶠᵃ,  φᶠᶠᵃ,  Azᶠᶠᵃ),
                             (:λᶜᶜᵃ, :φᶜᶜᵃ, :Azᶜᶜᵃ, :λᶠᶠᵃ, :φᶠᶠᵃ, :Azᶠᶠᵃ))
        
        for region in 1:number_of_regions(grid)
            getregion(field, region).data .= getproperty(getregion(grid, region), name)
        end

        if horizontal_topology == FullyConnected
            fill_halo_regions!(field)
        end

        for region in 1:number_of_regions(grid)
            getproperty(getregion(grid, region), name) .= getregion(field, region).data
        end
    end

    Δxᶜᶜᵃ = Field((Center, Center, Nothing), grid)
    Δxᶠᶜᵃ = Field((Center, Center, Nothing), grid)
    Δyᶠᶜᵃ = Field((Center, Center, Nothing), grid)
    λᶠᶜᵃ  = Field((Face,   Face,   Nothing), grid)
    φᶠᶜᵃ  = Field((Face,   Face,   Nothing), grid)
    Azᶠᶜᵃ = Field((Face,   Face,   Nothing), grid)
    Δxᶠᶠᵃ = Field((Face,   Face,   Nothing), grid)
    
    fields₁ = ( Δxᶜᶜᵃ,   Δxᶠᶜᵃ,   Δyᶠᶜᵃ,   λᶠᶜᵃ,    φᶠᶜᵃ,    Azᶠᶜᵃ ,  Δxᶠᶠᵃ)
    names₁  = (:Δxᶜᶜᵃ,  :Δxᶠᶜᵃ,  :Δyᶠᶜᵃ,  :λᶠᶜᵃ,   :φᶠᶜᵃ,   :Azᶠᶜᵃ , :Δxᶠᶠᵃ)

    Δyᶜᶜᵃ = Field((Center, Center, Nothing), grid)
    Δyᶜᶠᵃ = Field((Center, Center, Nothing), grid)
    Δxᶜᶠᵃ = Field((Center, Center, Nothing), grid)
    λᶜᶠᵃ  = Field((Face,   Face,   Nothing), grid)
    φᶜᶠᵃ  = Field((Face,   Face,   Nothing), grid)
    Azᶜᶠᵃ = Field((Face,   Face,   Nothing), grid)
    Δyᶠᶠᵃ = Field((Face,   Face,   Nothing), grid)

    fields₂ = ( Δyᶜᶜᵃ,   Δyᶜᶠᵃ,   Δxᶜᶠᵃ,   λᶜᶠᵃ,    φᶜᶠᵃ,    Azᶜᶠᵃ ,  Δyᶠᶠᵃ)
    names₂  = (:Δyᶜᶜᵃ,  :Δyᶜᶠᵃ,  :Δxᶜᶠᵃ,  :λᶜᶠᵃ,   :φᶜᶠᵃ,   :Azᶜᶠᵃ , :Δyᶠᶠᵃ)

    for (field₁, field₂, name₁, name₂) in zip(fields₁, fields₂, names₁, names₂)
        for region in 1:number_of_regions(grid)
            getregion(field₁, region).data .= geproperty(getregion(grid, region), name₁)
            getregion(field₂, region).data .= geproperty(getregion(grid, region), name₂)
        end

        if horizontal_topology == FullyConnected
            fill_halo_regions!(field₁, field₂; signed = false)
        end

        for region in 1:number_of_regions(grid)
            geproperty(getregion(grid, region), name₁) .= getregion(field₁, region).data
            geproperty(getregion(grid, region), name₂) .= getregion(field₂, region).data
        end
    end

    ###################################################
    ## Code specific to one-region-per panel partitions

    ## hardcoding NW/SE corner values only works for a one-region-per panel partition

    number_of_regions(grid) !== 6 && error("requires cubed sphere grids with 1 region per panel")

    for region in 1:number_of_regions(grid)
        if isodd(region)
            # Coordinates of "missing" NW corner points on odd panels can't be read from the interior
            # so we compute them via conformal_cubed_sphere_mapping
            φc, λc = cartesian_to_lat_lon(conformal_cubed_sphere_mapping(1, -1)...)
            getregion(grid, region).φᶠᶠᵃ[1, Ny+1] = φc
            getregion(grid, region).λᶠᶠᵃ[1, Ny+1] = λc
        elseif iseven(region)
            # Coordinates of "missing" SE corner points on even panels can't be read from the interior
            # so we compute them via conformal_cubed_sphere_mapping
            φc, λc = -1 .* cartesian_to_lat_lon(conformal_cubed_sphere_mapping(-1, -1)...)
            getregion(grid, region).φᶠᶠᵃ[Nx+1, 1] = φc
            getregion(grid, region).λᶠᶠᵃ[Nx+1, 1] = λc
        end

        getregion(grid, region).λᶜᶜᵃ[getregion(grid, region).λᶜᶜᵃ .== -180] .= 180
        getregion(grid, region).λᶠᶜᵃ[getregion(grid, region).λᶠᶜᵃ .== -180] .= 180
        getregion(grid, region).λᶜᶠᵃ[getregion(grid, region).λᶜᶠᵃ .== -180] .= 180
        getregion(grid, region).λᶠᶠᵃ[getregion(grid, region).λᶠᶠᵃ .== -180] .= 180
    end

    ## End code specific to one-region-per panel partitions
    #######################################################

    # now convert to user-prescribed architecture
    region_grids = grid.region_grids
    @apply_regionally new_region_grids = on_architecture(arch, region_grids)

    new_devices = arch == CPU() ? Tuple(CPU() for _ in 1:length(partition)) : Tuple(CUDA.device() for _ in 1:length(partition))

    new_region_grids = MultiRegionObject(new_region_grids.regional_objects, new_devices)

    new_grid = MultiRegionGrid{FT, region_topology...}(arch,
                                                       partition,
                                                       connectivity,
                                                       new_region_grids,
                                                       new_devices)

    return new_grid
end

"""
    ConformalCubedSphereGrid(filepath::AbstractString, arch::AbstractArchitecture=CPU(), FT=Float64;
                             Nz,
                             z,
                             panel_halo = (4, 4, 4),
                             panel_topology = (FullyConnected, FullyConnected, Bounded),
                             radius = R_Earth,
                             devices = nothing)

Load a `ConformalCubedSphereGrid` from `filepath`.
"""
function ConformalCubedSphereGrid(filepath::AbstractString, arch::AbstractArchitecture=CPU(), FT=Float64;
                                  Nz,
                                  z,
                                  panel_halo = (4, 4, 4),
                                  panel_topology = (FullyConnected, FullyConnected, Bounded),
                                  radius = R_Earth,
                                  devices = nothing)

    # only 6-panel partition, i.e. R = 1, are allowed when loading a ConformalCubedSphereGrid from file
    partition = CubedSpherePartition(R = 1)

    devices = validate_devices(partition, arch, devices)
    devices = assign_devices(partition, devices)

    region_Nz = MultiRegionObject(Tuple(repeat([Nz], length(partition))), devices)
    region_panels = Iterate(Array(1:length(partition)))

    region_grids = construct_regionally(conformal_cubed_sphere_panel, filepath, arch, FT;
                                        Nz = region_Nz,
                                        z,
                                        panel = region_panels,
                                        topology = panel_topology,
                                        halo = panel_halo,
                                        radius)

    connectivity = CubedSphereConnectivity(devices, partition)

    return MultiRegionGrid{FT, panel_topology...}(arch, partition, connectivity, region_grids, devices)
end

function with_halo(new_halo, csg::ConformalCubedSphereGrid)
    region_rotation = []

    for region in 1:length(csg.partition)
        push!(region_rotation, csg[region].conformal_mapping.rotation)
    end

    apply_regionally!(with_halo, new_halo, csg; rotation = Iterate(region_rotation))

    return csg
end

function Base.summary(grid::ConformalCubedSphereGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    return string(size_summary(size(grid)),
                  " ConformalCubedSphereGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

radius(mrg::ConformalCubedSphereGrid) = first(mrg).radius

grid_name(mrg::ConformalCubedSphereGrid) = "ConformalCubedSphereGrid"
