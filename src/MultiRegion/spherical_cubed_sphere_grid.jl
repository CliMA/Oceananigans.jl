#####
##### Six-panel cubed spheres of `SphericalGrid` panels
#####

using Oceananigans.Grids: SphericalGrid, EquiangularGnomonicPanelGrid, isorthogonal
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, BottomAndTop, select_bc, fill_halo_kernel

import Oceananigans.BoundaryConditions: fill_halo_kernels

const ConnectedSides = Sides{Connected, Connected}
const SphericalCubedSpherePanel = SphericalGrid{<:Any, ConnectedSides, ConnectedSides}

Grids.isorthogonal(mrg::MultiRegionGrid) = all(isorthogonal, mrg.region_grids.regional_objects)

"""
    SphericalCubedSphereGrid(panels; partition = CubedSpherePartition(R = 1))

Compose six [`SphericalGrid`](@ref) panels into a `MultiRegionGrid` with `CubedSphereConnectivity`. Panel `n` is
region `n`; the panels must have equal horizontal sizes and `Sides{Connected, Connected}` horizontal topology, and
every panel's local axes must follow the orientation `CubedSphereConnectivity` assumes (that of
`ConformalCubedSphereGrid`). Panel connectivity is used only by halo filling.
"""
function SphericalCubedSphereGrid(panels::NTuple{6, SphericalGrid}; partition = CubedSpherePartition(R = 1))
    length(partition) == 6 || throw(ArgumentError("SphericalCubedSphereGrid requires one region per panel (R = 1)"))
    first_panel = first(panels)
    Nx, Ny, Nz = size(first_panel)
    Nx == Ny || throw(ArgumentError("cubed-sphere panels must be square; got $(Nx)×$(Ny)"))

    for (n, panel) in enumerate(panels)
        size(panel) == (Nx, Ny, Nz) || throw(ArgumentError("panel $n has size $(size(panel)) ≠ $((Nx, Ny, Nz))"))
        halo_size(panel) == halo_size(first_panel) || throw(ArgumentError("panel $n has a different halo size"))
        panel isa SphericalCubedSpherePanel ||
            throw(ArgumentError("panel $n must have Sides{Connected, Connected} horizontal topology"))
        architecture(panel) == architecture(first_panel) || throw(ArgumentError("panel $n is on another architecture"))
    end

    FT = eltype(first_panel)
    TX, TY, TZ = topology(first_panel)
    CZ = typeof(first_panel.z)
    connectivity = CubedSphereConnectivity(partition)

    return MultiRegionGrid{FT, TX, TY, TZ, CZ}(architecture(first_panel), partition, connectivity,
                                               MultiRegionObject(panels))
end

"""
    EquiangularCubedSphereGrid(arch = CPU(), FT = Oceananigans.defaults.FloatType;
                               panel_size, z,
                               horizontal_direction_halo = 3,
                               z_halo = horizontal_direction_halo,
                               z_topology = Bounded,
                               radius = Oceananigans.defaults.planet_radius)

Return the six-panel equiangular gnomonic cubed sphere: [`EquiangularGnomonicPanelGrid`](@ref) panels rotated like
the panels of `ConformalCubedSphereGrid` (`CubedSphereConnectivity` rotations) and composed by
[`SphericalCubedSphereGrid`](@ref). Each panel's halo geometry continues its own gnomonic map; halo values of
fields come from the neighbouring panels through `fill_halo_regions!`.
"""
function EquiangularCubedSphereGrid(arch::AbstractArchitecture = CPU(),
                                    FT::DataType = Oceananigans.defaults.FloatType;
                                    panel_size,
                                    z,
                                    horizontal_direction_halo = 3,
                                    z_halo = horizontal_direction_halo,
                                    z_topology = Bounded,
                                    radius = Oceananigans.defaults.planet_radius)

    partition = CubedSpherePartition(R = 1)
    rotations = CubedSphereConnectivity(partition).rotations
    topology = (ConnectedSides, ConnectedSides, z_topology)
    halo = (horizontal_direction_halo, horizontal_direction_halo, z_halo)

    panels = ntuple(n -> EquiangularGnomonicPanelGrid(arch, FT; size = panel_size, z, topology, halo, radius,
                                                      panel = n, rotation = getregion(rotations, n)), 6)

    return SphericalCubedSphereGrid(panels; partition)
end

# Vertical halos are filled across the full horizontal span of `data`, including the horizontal halo
# columns that panel corner exchanges read (as for `ConformalCubedSpherePanelGrid`).
@inline function fill_halo_kernels(bcs::FieldBoundaryConditions, data::OffsetArray, grid::SphericalCubedSpherePanel,
                                   loc, indices)
    reduced_dimensions = tuple(findall(x -> x isa Nothing, loc)...)
    size    = (Base.size(data, 1), Base.size(data, 2))
    offset  = (first(axes(data, 1)) - 1, first(axes(data, 2)) - 1)
    side    = BottomAndTop()
    bcs     = (bcs.bottom, bcs.top)
    bc      = select_bc(bcs)
    kernel! = fill_halo_kernel(side, bc, grid, size, offset, data, reduced_dimensions)

    return (; bottom_and_top = kernel!), (; bottom_and_top = bcs)
end
