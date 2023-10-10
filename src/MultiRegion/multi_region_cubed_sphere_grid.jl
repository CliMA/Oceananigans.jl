using Oceananigans.Architectures: architecture
using Oceananigans.Grids: conformal_cubed_sphere_panel,
                          R_Earth,
                          halo_size,
                          size_summary,
                          total_length,
                          topology

import Oceananigans.Grids: grid_name

using CubedSphere
using Distances

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

```@example cubedspheregrid; setup = :(using Oceananigans; using Oceananigans.MultiRegion: inject_west_boundary, inject_south_boundary, inject_east_boundary, inject_north_boundary, East, West, South, North, CubedSphereRegionalConnectivity)
using Oceananigans

grid = ConformalCubedSphereGrid(panel_size=(12, 12, 1), z=(-1, 0), radius=1)
```

We can find out all connectivities of the regions of our grid. For example, to determine the
connectivites on the South boundary of each region we can call

```@example cubedspheregrid; setup = :(using Oceananigans; using Oceananigans.MultiRegion: East, West, South, North, CubedSphereRegionalConnectivity)
using Oceananigans.MultiRegion: CubedSphereRegionalConnectivity, East, West, South, North, getregion

for region in 1:length(grid); println("panel ", region, ": ", getregion(grid.connectivity.connections, 3).south); end
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

    devices = validate_devices(partition, arch, devices)
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

    region_grids = construct_regionally(conformal_cubed_sphere_panel, arch, FT;
                                        size = region_size,
                                        z,
                                        halo = region_halo,
                                        topology = region_topology,
                                        radius,
                                        ξ = region_ξ,
                                        η = region_η,
                                        rotation = region_rotation)

    grid = MultiRegionGrid{FT, region_topology[1], region_topology[2], region_topology[3]}(arch,
                                                                                           partition,
                                                                                           connectivity,
                                                                                           region_grids,
                                                                                           devices)

    # fields₁ = (:Δxᶜᶜᵃ,  :Δxᶠᶜᵃ,  :Δxᶜᶠᵃ,  :Δxᶠᶠᵃ, :λᶜᶜᵃ,   :λᶠᶜᵃ,   :λᶜᶠᵃ,   :λᶠᶠᵃ, :Azᶜᶜᵃ,  :Azᶠᶜᵃ,  :Azᶠᶠᵃ)
    # LXs₁    = (:Center, :Face,   :Center, :Face,  :Center, :Face,   :Center, :Face, :Center, :Face,   :Face )
    # LYs₁    = (:Center, :Center, :Face,   :Face,  :Center, :Center, :Face,   :Face, :Center, :Center, :Face )
    fields₁ = (:Δxᶜᶜᵃ,  :Δxᶠᶜᵃ,  :Δxᶜᶠᵃ,  :λᶜᶜᵃ,   :λᶠᶜᵃ,   :λᶜᶠᵃ,   :Azᶜᶜᵃ,  :Azᶠᶜᵃ)
    LXs₁    = (:Center, :Face,   :Center, :Center, :Face,   :Center, :Center, :Face)
    LYs₁    = (:Center, :Center, :Face,   :Center, :Center, :Face,   :Center, :Center)

    # fields₂ = (:Δyᶜᶜᵃ,  :Δyᶜᶠᵃ,  :Δyᶠᶜᵃ,  :Δyᶠᶠᵃ, :φᶜᶜᵃ,   :φᶜᶠᵃ,   :φᶠᶜᵃ,   :φᶠᶠᵃ, :Azᶜᶜᵃ,  :Azᶜᶠᵃ,  :Azᶠᶠᵃ)
    # LXs₂    = (:Center, :Center, :Face,   :Face,  :Center, :Center, :Face,   :Face, :Center, :Center, :Face )
    # LYs₂    = (:Center, :Face,   :Center, :Face,  :Center, :Face,   :Center, :Face, :Center, :Face,   :Face )
    fields₂ = (:Δyᶜᶜᵃ,  :Δyᶜᶠᵃ,  :Δyᶠᶜᵃ,  :φᶜᶜᵃ,   :φᶜᶠᵃ,   :φᶠᶜᵃ,   :Azᶜᶜᵃ,  :Azᶜᶠᵃ)
    LXs₂    = (:Center, :Center, :Face,   :Center, :Center, :Face,   :Center, :Center)
    LYs₂    = (:Center, :Face,   :Center, :Center, :Face,   :Center, :Center, :Face)

    for (field₁, LX₁, LY₁, field₂, LX₂, LY₂) in zip(fields₁, LXs₁, LYs₁, fields₂, LXs₂, LYs₂)
        expr = quote
            $(Symbol(field₁)) = Field{$(Symbol(LX₁)), $(Symbol(LY₁)), Nothing}($(grid))
            $(Symbol(field₂)) = Field{$(Symbol(LX₂)), $(Symbol(LY₂)), Nothing}($(grid))

            CUDA.@allowscalar begin
                for region in 1:6
                    getregion($(Symbol(field₁)), region).data .= getregion($(grid), region).$(Symbol(field₁))
                    getregion($(Symbol(field₂)), region).data .= getregion($(grid), region).$(Symbol(field₂))
                end
            end

            if $(horizontal_topology) == FullyConnected
                for _ in 1:2
                    fill_halo_regions!($(Symbol(field₁)))
                    fill_halo_regions!($(Symbol(field₂)))

                    @apply_regionally replace_horizontal_vector_halos!((; u = $(Symbol(field₁)),
                                                                          v = $(Symbol(field₂)),
                                                                          w = nothing), $(grid), signed=false)
                end
            end

            CUDA.@allowscalar begin
                for region in 1:6
                    getregion($(grid), region).$(Symbol(field₁)) .= getregion($(Symbol(field₁)), region).data
                    getregion($(grid), region).$(Symbol(field₂)) .= getregion($(Symbol(field₂)), region).data
                end
            end
        end # quote

        eval(expr)
    end

    " Halo filling for Face-Face coordinates, hardcoded for the default cubed-sphere connectivity. "
    function fill_faceface_coordinates!(grid)
        length(grid.partition) != 6 && error("only works for CubedSpherePartition(R = 1) at the moment")

        CUDA.allowscalar() do
            getregion(grid, 1).φᶠᶠᵃ[2:Nx+1, Ny+1] = reshape(reverse(getregion(grid, 3).φᶠᶠᵃ[1:1, 1:Ny]), (Ny, 1))
            getregion(grid, 1).λᶠᶠᵃ[2:Nx+1, Ny+1] = reshape(reverse(getregion(grid, 3).λᶠᶠᵃ[1:1, 1:Ny]), (Ny, 1))
            getregion(grid, 1).φᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 2).φᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 1).λᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 2).λᶠᶠᵃ[1, 1:Ny]

            getregion(grid, 3).φᶠᶠᵃ[2:Nx+1, Ny+1] = reshape(reverse(getregion(grid, 5).φᶠᶠᵃ[1:1, 1:Ny]), (Ny, 1))
            getregion(grid, 3).λᶠᶠᵃ[2:Nx+1, Ny+1] = reshape(reverse(getregion(grid, 5).λᶠᶠᵃ[1:1, 1:Ny]), (Ny, 1))
            getregion(grid, 3).φᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 4).φᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 3).λᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 4).λᶠᶠᵃ[1, 1:Ny]

            getregion(grid, 5).φᶠᶠᵃ[2:Nx+1, Ny+1] = reshape(reverse(getregion(grid, 1).φᶠᶠᵃ[1:1, 1:Ny]), (Ny, 1))
            getregion(grid, 5).λᶠᶠᵃ[2:Nx+1, Ny+1] = reshape(reverse(getregion(grid, 1).λᶠᶠᵃ[1:1, 1:Ny]), (Ny, 1))
            getregion(grid, 5).φᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 6).φᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 5).λᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 6).λᶠᶠᵃ[1, 1:Ny]

            getregion(grid, 2).φᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 3).φᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 2).λᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 3).λᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 2).φᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 4).φᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 2).λᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 4).λᶠᶠᵃ[1:Nx, 1:1])

            getregion(grid, 4).φᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 5).φᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 4).λᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 5).λᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 4).φᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 6).φᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 4).λᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 6).λᶠᶠᵃ[1:Nx, 1:1])

            getregion(grid, 6).φᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 1).φᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 6).λᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 1).λᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 6).φᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 2).φᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 6).λᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 2).λᶠᶠᵃ[1:Nx, 1:1])

            for region in (1, 3, 5)
                φc, λc = cartesian_to_lat_lon(conformal_cubed_sphere_mapping(1, -1)...)
                getregion(grid, region).φᶠᶠᵃ[1, Ny+1] = φc
                getregion(grid, region).λᶠᶠᵃ[1, Ny+1] = λc
            end

            for region in (2, 4, 6)
                φc, λc = cartesian_to_lat_lon(conformal_cubed_sphere_mapping(-1, -1)...)
                getregion(grid, region).φᶠᶠᵃ[Nx+1, 1] = -φc
                getregion(grid, region).λᶠᶠᵃ[Nx+1, 1] = -λc
            end
        end

        return nothing
    end

    " Halo filling for Face-Face metrics, hardcoded for the default cubed-sphere connectivity. "
    function fill_faceface_metrics!(grid)
        length(grid.partition) != 6 && error("only works for CubedSpherePartition(R = 1) at the moment")

        CUDA.@allowscalar begin
            getregion(grid, 1).Δxᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 3).Δyᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 1).Δyᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 3).Δxᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 1).Azᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 3).Azᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 1).Δxᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 2).Δxᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 1).Δyᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 2).Δyᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 1).Azᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 2).Azᶠᶠᵃ[1, 1:Ny]

            getregion(grid, 3).Δxᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 5).Δyᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 3).Δyᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 5).Δxᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 3).Azᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 5).Azᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 3).Δxᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 4).Δxᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 3).Δyᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 4).Δyᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 3).Azᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 4).Azᶠᶠᵃ[1, 1:Ny]

            getregion(grid, 5).Δxᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 1).Δyᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 5).Δyᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 1).Δxᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 5).Azᶠᶠᵃ[2:Nx+1, Ny+1] = reverse(getregion(grid, 1).Azᶠᶠᵃ[1:1, 1:Ny])'
            getregion(grid, 5).Δxᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 6).Δxᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 5).Δyᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 6).Δyᶠᶠᵃ[1, 1:Ny]
            getregion(grid, 5).Azᶠᶠᵃ[Nx+1, 1:Ny]   = getregion(grid, 6).Azᶠᶠᵃ[1, 1:Ny]

            getregion(grid, 2).Δxᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 3).Δxᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 2).Δyᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 3).Δyᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 2).Azᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 3).Azᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 2).Δxᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 4).Δyᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 2).Δyᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 4).Δxᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 2).Azᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 4).Azᶠᶠᵃ[1:Nx, 1:1])

            getregion(grid, 4).Δxᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 5).Δxᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 4).Δyᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 5).Δyᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 4).Azᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 5).Azᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 4).Δxᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 6).Δyᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 4).Δyᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 6).Δxᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 4).Azᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 6).Azᶠᶠᵃ[1:Nx, 1:1])

            getregion(grid, 6).Δxᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 1).Δxᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 6).Δyᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 1).Δyᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 6).Azᶠᶠᵃ[1:Nx, Ny+1]   = getregion(grid, 1).Azᶠᶠᵃ[1:Nx, 1]
            getregion(grid, 6).Δxᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 2).Δyᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 6).Δyᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 2).Δxᶠᶠᵃ[1:Nx, 1:1])
            getregion(grid, 6).Azᶠᶠᵃ[Nx+1, 2:Ny+1] = reverse(getregion(grid, 2).Azᶠᶠᵃ[1:Nx, 1:1])

            for region in (1, 3, 5)
                getregion(grid, region).Δxᶠᶠᵃ[1, Ny+1] = getregion(grid, region).Δxᶠᶠᵃ[Nx+1, 1]
                getregion(grid, region).Δyᶠᶠᵃ[1, Ny+1] = getregion(grid, region).Δyᶠᶠᵃ[Nx+1, 1]
                getregion(grid, region).Azᶠᶠᵃ[1, Ny+1] = getregion(grid, region).Azᶠᶠᵃ[1, 1]
            end

            for region in (2, 4, 6)
                getregion(grid, region).Δxᶠᶠᵃ[Nx+1, 1] = getregion(grid, region).Δxᶠᶠᵃ[1, 1]
                getregion(grid, region).Δyᶠᶠᵃ[Nx+1, 1] = getregion(grid, region).Δyᶠᶠᵃ[1, 1]
                getregion(grid, region).Azᶠᶠᵃ[Nx+1, 1] = getregion(grid, region).Azᶠᶠᵃ[1, 1]
            end
        end

        return nothing
    end

    if horizontal_topology == FullyConnected
        fill_faceface_coordinates!(grid)
        fill_faceface_metrics!(grid)
    end

    CUDA.@allowscalar begin
        for region in 1:6
            getregion(grid, region).λᶜᶜᵃ[getregion(grid, region).λᶜᶜᵃ .== -180] .= 180
            getregion(grid, region).λᶠᶜᵃ[getregion(grid, region).λᶠᶜᵃ .== -180] .= 180
            getregion(grid, region).λᶜᶠᵃ[getregion(grid, region).λᶜᶠᵃ .== -180] .= 180
            getregion(grid, region).λᶠᶠᵃ[getregion(grid, region).λᶠᶠᵃ .== -180] .= 180
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

    # only 6-panel partition, i.e. R=1, are allowed when loading a ConformalCubedSphereGrid from file
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

    return MultiRegionGrid{FT, panel_topology[1], panel_topology[2], panel_topology[3]}(arch, partition, connectivity, region_grids, devices)
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
