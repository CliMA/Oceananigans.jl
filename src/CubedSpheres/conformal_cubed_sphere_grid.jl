using Rotations
using Suppressor
using Oceananigans.Grids
using Oceananigans.Grids: R_Earth, interior_indices

import Base: show, size, eltype
import Oceananigans.Grids: topology, architecture, halo_size, on_architecture, size_summary

struct CubedSpherePanelConnectivityDetails{P, S}
    panel :: P
     side :: S
end

short_string(deets::CubedSpherePanelConnectivityDetails) = "panel $(deets.panel) $(deets.side) side"

Base.show(io::IO, deets::CubedSpherePanelConnectivityDetails) =
    print(io, "CubedSpherePanelConnectivityDetails: $(short_string(deets))")

struct CubedSpherePanelConnectivity{W, E, S, N}
     west :: W
     east :: E
    south :: S
    north :: N
end

CubedSpherePanelConnectivity(; west, east, south, north) =
    CubedSpherePanelConnectivity(west, east, south, north)

function Base.show(io::IO, connectivity::CubedSpherePanelConnectivity)
    print(io, "CubedSpherePanelConnectivity:\n",
              "├── west: $(short_string(connectivity.west))\n",
              "├── east: $(short_string(connectivity.east))\n",
              "├── south: $(short_string(connectivity.south))\n",
              "└── north: $(short_string(connectivity.north))")
end

function default_panel_connectivity()
    # Adopted from figure 8.4 of https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-6tile
    #
    #                         panel P5   panel P6
    #                       +----------+----------+
    #                       |    ↑↑    |    ↑↑    |
    #                       |    1W    |    1S    |
    #                       |←3N P5 6W→|←5E P6 2S→|
    #                       |    4N    |    4E    |
    #              panel P3 |    ↓↓    |    ↓↓    |
    #            +----------+----------+----------+
    #            |    ↑↑    |    ↑↑    |
    #            |    5W    |    5S    |
    #            |←1N P3 4W→|←3E P4 6S→|
    #            |    2N    |    2E    |
    #            |    ↓↓    |    ↓↓    |
    # +----------+----------+----------+
    # |    ↑↑    |    ↑↑    | panel P4
    # |    3W    |    3S    |
    # |←5N P1 2W→|←1E P2 4S→|
    # |    6N    |    6E    |
    # |    ↓↓    |    ↓↓    |
    # +----------+----------+
    #   panel P1   panel P2

    panel1_connectivity = CubedSpherePanelConnectivity(
        west  = CubedSpherePanelConnectivityDetails(5, :north),
        east  = CubedSpherePanelConnectivityDetails(2, :west),
        south = CubedSpherePanelConnectivityDetails(6, :north),
        north = CubedSpherePanelConnectivityDetails(3, :west),
    )

    panel2_connectivity = CubedSpherePanelConnectivity(
        west  = CubedSpherePanelConnectivityDetails(1, :east),
        east  = CubedSpherePanelConnectivityDetails(4, :south),
        south = CubedSpherePanelConnectivityDetails(6, :east),
        north = CubedSpherePanelConnectivityDetails(3, :south),
    )

    panel3_connectivity = CubedSpherePanelConnectivity(
        west  = CubedSpherePanelConnectivityDetails(1, :north),
        east  = CubedSpherePanelConnectivityDetails(4, :west),
        south = CubedSpherePanelConnectivityDetails(2, :north),
        north = CubedSpherePanelConnectivityDetails(5, :west),
    )

    panel4_connectivity = CubedSpherePanelConnectivity(
        west  = CubedSpherePanelConnectivityDetails(3, :east),
        east  = CubedSpherePanelConnectivityDetails(6, :south),
        south = CubedSpherePanelConnectivityDetails(2, :east),
        north = CubedSpherePanelConnectivityDetails(5, :south),
    )

    panel5_connectivity = CubedSpherePanelConnectivity(
        west  = CubedSpherePanelConnectivityDetails(3, :north),
        east  = CubedSpherePanelConnectivityDetails(6, :west),
        south = CubedSpherePanelConnectivityDetails(4, :north),
        north = CubedSpherePanelConnectivityDetails(1, :west),
    )


    panel6_connectivity = CubedSpherePanelConnectivity(
        west  = CubedSpherePanelConnectivityDetails(5, :east),
        east  = CubedSpherePanelConnectivityDetails(2, :south),
        south = CubedSpherePanelConnectivityDetails(4, :east),
        north = CubedSpherePanelConnectivityDetails(1, :south),
    )

    panel_connectivity = (
        panel1_connectivity,
        panel2_connectivity,
        panel3_connectivity,
        panel4_connectivity,
        panel5_connectivity,
        panel6_connectivity
    )

    return panel_connectivity
end

# Note: I think we want to keep panels and panel_connectivity tuples
# so it's easy to support an arbitrary number of panels.

struct ConformalCubedSphereGrid{FT, F, C, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, FullyConnected, FullyConnected, Bounded, Arch}
          architecture :: Arch
                panels :: F
    panel_connectivity :: C
end

function ConformalCubedSphereGrid(arch = CPU(), FT=Float64;
                                  panel_size, z,
                                  panel_halo = (1, 1, 1),
                                  panel_topology = (FullyConnected, FullyConnected, Bounded),
                                  radius = R_Earth)

    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    size, halo, topology = panel_size, panel_halo, panel_topology

    panel_kwargs = (; z, size, topology, halo, radius)

    # +z panel (panel 1)
    z⁺_panel_grid = @suppress OrthogonalSphericalShellGrid(arch, FT; rotation=nothing, panel_kwargs...)

    # +x panel (panel 2)
    x⁺_panel_grid = @suppress OrthogonalSphericalShellGrid(arch, FT; rotation=RotX(π/2), panel_kwargs...)

    # +y panel (panel 3)
    y⁺_panel_grid = @suppress OrthogonalSphericalShellGrid(arch, FT; rotation=RotY(π/2), panel_kwargs...)

    # -x panel (panel 4)
    x⁻_panel_grid = @suppress OrthogonalSphericalShellGrid(arch, FT; rotation=RotX(-π/2), panel_kwargs...)

    # -y panel (panel 5)
    y⁻_panel_grid = @suppress OrthogonalSphericalShellGrid(arch, FT; rotation=RotY(-π/2), panel_kwargs...)

    # -z panel (panel 6)
    z⁻_panel_grid = @suppress OrthogonalSphericalShellGrid(arch, FT; rotation=RotX(π), panel_kwargs...)

    panels = (
        z⁺_panel_grid,
        x⁺_panel_grid,
        y⁺_panel_grid,
        x⁻_panel_grid,
        y⁻_panel_grid,
        z⁻_panel_grid
    )

    panel_connectivity = default_panel_connectivity()

    return ConformalCubedSphereGrid{FT, typeof(panels), typeof(panel_connectivity), typeof(arch)}(arch, panels, panel_connectivity)
end

function ConformalCubedSphereGrid(filepath::AbstractString, arch = CPU(), FT=Float64; Nz, z, radius = R_Earth, halo = (1, 1, 1))
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    panel_topo = (FullyConnected, FullyConnected, Bounded)
    panel_kwargs = (; Nz, z, topology=panel_topo, radius, halo)

    panels = Tuple(OrthogonalSphericalShellGrid(filepath, arch, FT; panel=n, panel_kwargs...) for n in 1:6)

    panel_connectivity = default_panel_connectivity()

    grid = ConformalCubedSphereGrid{FT, typeof(panels), typeof(panel_connectivity), typeof(arch)}(arch, panels, panel_connectivity)

    fill_grid_metric_halos!(grid)
    fill_grid_metric_halos!(grid)

    return grid
end

Base.summary(grid::OrthogonalSphericalShellGrid{FT, FullyConnected, FullyConnected, TZ}) where {FT, TZ} = 
    string(size_summary(size(grid)),
           " OrthogonalSphericalShellGrid with topology (FullyConnected, FullyConnected, $TZ)",
           " and with ", size_summary(halo_size(grid)), " halo")

function Base.summary(grid::ConformalCubedSphereGrid)
    Nx, Ny, Nz, Nf = size(grid)
    FT = eltype(grid)
    metric_computation = isnothing(grid.panels[1].Δxᶠᶜᵃ) ? "without precomputed metrics" : "with precomputed metrics"

    return string(size_summary(size(grid)), " × $Nf panels",
                  " ConformalCubedSphereGrid{$FT} on ", summary(architecture(grid)),
                  " ", metric_computation)
end

function Base.show(io::IO, grid::ConformalCubedSphereGrid, withsummary=true)
    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "|   Panels: \n",
                     "├── ", summary(grid.panels[1]), "\n",
                     "├── ", summary(grid.panels[2]), "\n",
                     "├── ", summary(grid.panels[3]), "\n",
                     "├── ", summary(grid.panels[4]), "\n",
                     "├── ", summary(grid.panels[5]), "\n",
                     "└── ", summary(grid.panels[6]))
end

#####
##### Nodes for OrthogonalSphericalShellGrid
#####

@inline λnode(LX::Face,   LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶠᶠᵃ[i, j]
@inline λnode(LX::Face,   LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline λnode(LX::Center, LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline λnode(LX::Center, LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.λᶜᶜᵃ[i, j]

@inline φnode(LX::Face,   LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶠᶠᵃ[i, j]
@inline φnode(LX::Face,   LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline φnode(LX::Center, LY::Face,   LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline φnode(LX::Center, LY::Center, LZ, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.φᶜᶜᵃ[i, j]

@inline znode(LX, LY, LZ::Face,   i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(LX, LY, LZ::Center, i, j, k, grid::OrthogonalSphericalShellGrid) = @inbounds grid.zᵃᵃᶜ[k]

λnodes(LX::Face, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶠᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Face, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶠᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Center, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶜᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

λnodes(LX::Center, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.λᶜᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Face, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶠᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Face, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶠᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Center, LY::Face, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶜᶠᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

φnodes(LX::Center, LY::Center, LZ, grid::OrthogonalSphericalShellGrid{TX, TY}) where {TX, TY} =
    view(grid.φᶜᶜᵃ, interior_indices(LX, TX, grid.Nx), interior_indices(LY, TY, grid.Ny))

#####
##### Grid utils
#####

Base.size(grid::ConformalCubedSphereGrid)      = (size(grid.panels[1])..., length(grid.panels))
Base.size(loc, grid::ConformalCubedSphereGrid) = size(loc, grid.panels[1])
Base.size(grid::ConformalCubedSphereGrid, i)   = size(grid)[i]
halo_size(ccsg::ConformalCubedSphereGrid)      = halo_size(first(ccsg.panels)) # hack

Base.eltype(grid::ConformalCubedSphereGrid{FT}) where FT = FT

topology(::ConformalCubedSphereGrid) = (Bounded, Bounded, Bounded)
topology(grid::ConformalCubedSphereGrid, i) = topology(grid)[i] 
architecture(grid::ConformalCubedSphereGrid) = grid.architecture

function on_architecture(arch, grid::ConformalCubedSphereGrid) 

    panels = Tuple(on_architecture(arch, grid.panels[n]) for n in 1:6)
    panel_connectivity = grid.panel_connectivity
    FT = eltype(grid)
    
    return ConformalCubedSphereGrid{FT, typeof(panels), typeof(panel_connectivity), typeof(arch)}(arch, panels, panel_connectivity)
end

#####
##### filling grid halos
#####

function grid_metric_halo(grid_metric, grid, location, topo, side)
    LX, LY = location
    TX, TY = topo
    side == :west  && return  underlying_west_halo(grid_metric, grid, LX, TX)
    side == :east  && return  underlying_east_halo(grid_metric, grid, LX, TX)
    side == :south && return underlying_south_halo(grid_metric, grid, LY, TY)
    side == :north && return underlying_north_halo(grid_metric, grid, LY, TY)
end

function grid_metric_boundary(grid_metric, grid, location, topo, side)
    LX, LY = location
    TX, TY = topo
    side == :west  && return  underlying_west_boundary(grid_metric, grid, LX, TX)
    side == :east  && return  underlying_east_boundary(grid_metric, grid, LX, TX)
    side == :south && return underlying_south_boundary(grid_metric, grid, LY, TY)
    side == :north && return underlying_north_boundary(grid_metric, grid, LY, TY)
end

function fill_grid_metric_halos!(grid)

    topo_bb = (Bounded, Bounded)

    loc_cc = (Center, Center)
    loc_cf = (Center, Face  )
    loc_fc = (Face,   Center)
    loc_ff = (Face,   Face  )

    for panel_number in 1:6, side in (:west, :east, :south, :north)

        connectivity_info = getproperty(grid.panel_connectivity[panel_number], side)
        src_panel_number = connectivity_info.panel
        src_side = connectivity_info.side

        grid_panel = grid.panels[panel_number]
        src_grid_panel = grid.panels[src_panel_number]

        if sides_in_the_same_dimension(side, src_side)
            grid_metric_halo(grid_panel.Δxᶜᶜᵃ, grid_panel, loc_cc, topo_bb, side) .= grid_metric_boundary(grid_panel.Δxᶜᶜᵃ, src_grid_panel, loc_cc, topo_bb, src_side)
            grid_metric_halo(grid_panel.Δyᶜᶜᵃ, grid_panel, loc_cc, topo_bb, side) .= grid_metric_boundary(grid_panel.Δyᶜᶜᵃ, src_grid_panel, loc_cc, topo_bb, src_side)
            grid_metric_halo(grid_panel.Azᶜᶜᵃ, grid_panel, loc_cc, topo_bb, side) .= grid_metric_boundary(grid_panel.Azᶜᶜᵃ, src_grid_panel, loc_cc, topo_bb, src_side)

            grid_metric_halo(grid_panel.Δxᶜᶠᵃ, grid_panel, loc_cf, topo_bb, side) .= grid_metric_boundary(grid_panel.Δxᶜᶠᵃ, src_grid_panel, loc_cf, topo_bb, src_side)
            grid_metric_halo(grid_panel.Δyᶜᶠᵃ, grid_panel, loc_cf, topo_bb, side) .= grid_metric_boundary(grid_panel.Δyᶜᶠᵃ, src_grid_panel, loc_cf, topo_bb, src_side)
            grid_metric_halo(grid_panel.Azᶜᶠᵃ, grid_panel, loc_cf, topo_bb, side) .= grid_metric_boundary(grid_panel.Azᶜᶠᵃ, src_grid_panel, loc_cf, topo_bb, src_side)

            grid_metric_halo(grid_panel.Δxᶠᶜᵃ, grid_panel, loc_fc, topo_bb, side) .= grid_metric_boundary(grid_panel.Δxᶠᶜᵃ, src_grid_panel, loc_fc, topo_bb, src_side)
            grid_metric_halo(grid_panel.Δyᶠᶜᵃ, grid_panel, loc_fc, topo_bb, side) .= grid_metric_boundary(grid_panel.Δyᶠᶜᵃ, src_grid_panel, loc_fc, topo_bb, src_side)
            grid_metric_halo(grid_panel.Azᶠᶜᵃ, grid_panel, loc_fc, topo_bb, side) .= grid_metric_boundary(grid_panel.Azᶠᶜᵃ, src_grid_panel, loc_fc, topo_bb, src_side)

            grid_metric_halo(grid_panel.Δxᶠᶠᵃ, grid_panel, loc_ff, topo_bb, side) .= grid_metric_boundary(grid_panel.Δxᶠᶠᵃ, src_grid_panel, loc_ff, topo_bb, src_side)
            grid_metric_halo(grid_panel.Δyᶠᶠᵃ, grid_panel, loc_ff, topo_bb, side) .= grid_metric_boundary(grid_panel.Δyᶠᶠᵃ, src_grid_panel, loc_ff, topo_bb, src_side)
            grid_metric_halo(grid_panel.Azᶠᶠᵃ, grid_panel, loc_ff, topo_bb, side) .= grid_metric_boundary(grid_panel.Azᶠᶠᵃ, src_grid_panel, loc_ff, topo_bb, src_side)
        else
            reverse_dim = src_side in (:west, :east) ? 1 : 2
            grid_metric_halo(grid_panel.Δxᶜᶜᵃ, grid_panel, loc_cc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δyᶜᶜᵃ, src_grid_panel, loc_cc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Δyᶜᶜᵃ, grid_panel, loc_cc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δxᶜᶜᵃ, src_grid_panel, loc_cc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Azᶜᶜᵃ, grid_panel, loc_cc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Azᶜᶜᵃ, src_grid_panel, loc_cc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_panel.Δxᶜᶠᵃ, grid_panel, loc_cf, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δyᶠᶜᵃ, src_grid_panel, loc_fc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Δyᶜᶠᵃ, grid_panel, loc_cf, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δxᶠᶜᵃ, src_grid_panel, loc_fc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Azᶜᶠᵃ, grid_panel, loc_cf, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Azᶠᶜᵃ, src_grid_panel, loc_fc, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_panel.Δxᶠᶜᵃ, grid_panel, loc_fc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δyᶜᶠᵃ, src_grid_panel, loc_cf, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Δyᶠᶜᵃ, grid_panel, loc_fc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δxᶜᶠᵃ, src_grid_panel, loc_cf, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Azᶠᶜᵃ, grid_panel, loc_fc, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Azᶜᶠᵃ, src_grid_panel, loc_cf, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_panel.Δxᶠᶠᵃ, grid_panel, loc_ff, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δyᶠᶠᵃ, src_grid_panel, loc_ff, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Δyᶠᶠᵃ, grid_panel, loc_ff, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Δxᶠᶠᵃ, src_grid_panel, loc_ff, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_panel.Azᶠᶠᵃ, grid_panel, loc_ff, topo_bb, side) .= reverse(permutedims(grid_metric_boundary(grid_panel.Azᶠᶠᵃ, src_grid_panel, loc_ff, topo_bb, src_side), (2, 1, 3)), dims=reverse_dim)
        end
    end

    return nothing
end
