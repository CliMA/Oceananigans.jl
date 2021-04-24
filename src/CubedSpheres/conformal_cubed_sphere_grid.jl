using Rotations
using Oceananigans.Grids
using Oceananigans.Grids: R_Earth, interior_indices

"""
    default_conformal_cubed_sphere_connectivity()

Default connectivity of the Oceananigans' MultiRegionGrid composed of six cube
faces conformally mapped to the surface of the sphere.
"""
function default_conformal_cubed_sphere_connectivity()
    # See figure 8.4 of https://mitgcm.readthedocs.io/en/latest/phys_pkgs/exch2.html?highlight=cube%20sphere#fig-6tile
    #
    #                         face  F5   face  F6
    #                       +----------+----------+
    #                       |    ↑↑    |    ↑↑    |
    #                       |    1W    |    1S    |
    #                       |←3N F5 6W→|←5E F6 2S→|
    #                       |    4N    |    4E    |
    #              face  F3 |    ↓↓    |    ↓↓    |
    #            +----------+----------+----------+
    #            |    ↑↑    |    ↑↑    |
    #            |    5W    |    5S    |
    #            |←1N F3 4W→|←3E F4 6S→|
    #            |    2N    |    2E    |
    #            |    ↓↓    |    ↓↓    |
    # +----------+----------+----------+
    # |    ↑↑    |    ↑↑    | face  F4
    # |    3W    |    3S    |
    # |←5N F1 2W→|←1E F2 4S→|
    # |    6N    |    6E    |
    # |    ↓↓    |    ↓↓    |
    # +----------+----------+
    #   face  F1   face  F2

    face1_connectivity = RegionConnectivity(
        west  = RegionConnectivityDetails(5, :north),
        east  = RegionConnectivityDetails(2, :west),
        south = RegionConnectivityDetails(6, :north),
        north = RegionConnectivityDetails(3, :west),
    )

    face2_connectivity = RegionConnectivity(
        west  = RegionConnectivityDetails(1, :east),
        east  = RegionConnectivityDetails(4, :south),
        south = RegionConnectivityDetails(6, :east),
        north = RegionConnectivityDetails(3, :south),
    )

    face3_connectivity = RegionConnectivity(
        west  = RegionConnectivityDetails(1, :north),
        east  = RegionConnectivityDetails(4, :west),
        south = RegionConnectivityDetails(2, :north),
        north = RegionConnectivityDetails(5, :west),
    )

    face4_connectivity = RegionConnectivity(
        west  = RegionConnectivityDetails(3, :east),
        east  = RegionConnectivityDetails(6, :south),
        south = RegionConnectivityDetails(2, :east),
        north = RegionConnectivityDetails(5, :south),
    )

    face5_connectivity = RegionConnectivity(
        west  = RegionConnectivityDetails(3, :north),
        east  = RegionConnectivityDetails(6, :west),
        south = RegionConnectivityDetails(4, :north),
        north = RegionConnectivityDetails(1, :west),
    )


    face6_connectivity = RegionConnectivity(
        west  = RegionConnectivityDetails(5, :east),
        east  = RegionConnectivityDetails(2, :south),
        south = RegionConnectivityDetails(4, :east),
        north = RegionConnectivityDetails(1, :south),
    )

    connectivity = (
        face1_connectivity,
        face2_connectivity,
        face3_connectivity,
        face4_connectivity,
        face5_connectivity,
        face6_connectivity
    )

    return connectivity
end

function ConformalCubedSphereGrid(FT=Float64; face_size, z, radius=R_Earth)
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    # +z face (face 1)
    z⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=nothing)

    # +x face (face 2)
    x⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(π/2))

    # +y face (face 3)
    y⁺_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotY(π/2))

    # -x face (face 4)
    x⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(-π/2))

    # -y face (face 5)
    y⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotY(-π/2))

    # -z face (face 6)
    z⁻_face_grid = ConformalCubedSphereFaceGrid(FT, size=face_size, z=z, radius=radius, rotation=RotX(π))

    faces = (
        z⁺_face_grid,
        x⁺_face_grid,
        y⁺_face_grid,
        x⁻_face_grid,
        y⁻_face_grid,
        z⁻_face_grid
    )

    connectivity = default_conformal_cubed_sphere_connectivity()

    return MultiRegionGrid(faces, connectivity)
end

function ConformalCubedSphereGrid(filepath::AbstractString, FT=Float64; Nz, z, architecture = CPU(), radius = R_Earth, halo = (1, 1, 1))
    @warn "ConformalCubedSphereGrid is experimental: use with caution!"

    face_topo = (Connected, Connected, Bounded)
    face_kwargs = (Nz=Nz, z=z, topology=face_topo, radius=radius, halo=halo, architecture=architecture)

    faces = Tuple(ConformalCubedSphereFaceGrid(filepath, FT; face=n, face_kwargs...) for n in 1:6)

    connectivity = default_conformal_cubed_sphere_connectivity()

    grid = MultiRegionGrid(faces, connectivity)

    fill_grid_metric_halos!(grid)

    return grid
end

#####
##### filling grid halos
#####

function grid_metric_halo(grid_metric, grid, location, side)
    LX, LY = location
    side == :west  && return  underlying_west_halo(grid_metric, grid, LX)
    side == :east  && return  underlying_east_halo(grid_metric, grid, LX)
    side == :south && return underlying_south_halo(grid_metric, grid, LY)
    side == :north && return underlying_north_halo(grid_metric, grid, LY)
end

function grid_metric_boundary(grid_metric, grid, location, side)
    LX, LY = location
    side == :west  && return  underlying_west_boundary(grid_metric, grid, LX)
    side == :east  && return  underlying_east_boundary(grid_metric, grid, LX)
    side == :south && return underlying_south_boundary(grid_metric, grid, LY)
    side == :north && return underlying_north_boundary(grid_metric, grid, LY)
end

function fill_grid_metric_halos!(grid)

    loc_cc = (Center, Center)
    loc_cf = (Center, Face  )
    loc_fc = (Face,   Center)
    loc_ff = (Face,   Face  )

    for face_number in 1:6, side in (:west, :east, :south, :north)

        connectivity_info = getproperty(grid.connectivity[face_number], side)
        src_face_number = connectivity_info.face
        src_side = connectivity_info.side

        grid_face = get_region(grid, face_number)
        src_grid_face = get_region(grid, src_face_number)

        if sides_in_the_same_dimension(side, src_side)
            grid_metric_halo(grid_face.Δxᶜᶜᵃ, grid_face, loc_cc, side) .= grid_metric_boundary(grid_face.Δxᶜᶜᵃ, src_grid_face, loc_cc, src_side)
            grid_metric_halo(grid_face.Δyᶜᶜᵃ, grid_face, loc_cc, side) .= grid_metric_boundary(grid_face.Δyᶜᶜᵃ, src_grid_face, loc_cc, src_side)
            grid_metric_halo(grid_face.Azᶜᶜᵃ, grid_face, loc_cc, side) .= grid_metric_boundary(grid_face.Azᶜᶜᵃ, src_grid_face, loc_cc, src_side)

            grid_metric_halo(grid_face.Δxᶜᶠᵃ, grid_face, loc_cf, side) .= grid_metric_boundary(grid_face.Δxᶜᶠᵃ, src_grid_face, loc_cf, src_side)
            grid_metric_halo(grid_face.Δyᶜᶠᵃ, grid_face, loc_cf, side) .= grid_metric_boundary(grid_face.Δyᶜᶠᵃ, src_grid_face, loc_cf, src_side)
            grid_metric_halo(grid_face.Azᶜᶠᵃ, grid_face, loc_cc, side) .= grid_metric_boundary(grid_face.Azᶜᶠᵃ, src_grid_face, loc_cc, src_side)

            grid_metric_halo(grid_face.Δxᶠᶜᵃ, grid_face, loc_fc, side) .= grid_metric_boundary(grid_face.Δxᶠᶜᵃ, src_grid_face, loc_fc, src_side)
            grid_metric_halo(grid_face.Δyᶠᶜᵃ, grid_face, loc_fc, side) .= grid_metric_boundary(grid_face.Δyᶠᶜᵃ, src_grid_face, loc_fc, src_side)
            grid_metric_halo(grid_face.Azᶠᶜᵃ, grid_face, loc_cc, side) .= grid_metric_boundary(grid_face.Azᶠᶜᵃ, src_grid_face, loc_cc, src_side)

            grid_metric_halo(grid_face.Δxᶠᶠᵃ, grid_face, loc_ff, side) .= grid_metric_boundary(grid_face.Δxᶠᶠᵃ, src_grid_face, loc_ff, src_side)
            grid_metric_halo(grid_face.Δyᶠᶠᵃ, grid_face, loc_ff, side) .= grid_metric_boundary(grid_face.Δyᶠᶠᵃ, src_grid_face, loc_ff, src_side)
            grid_metric_halo(grid_face.Azᶠᶠᵃ, grid_face, loc_cc, side) .= grid_metric_boundary(grid_face.Azᶠᶠᵃ, src_grid_face, loc_cc, src_side)
        else
            reverse_dim = src_side in (:west, :east) ? 1 : 2
            grid_metric_halo(grid_face.Δxᶜᶜᵃ, grid_face, loc_cc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶜᶜᵃ, src_grid_face, loc_cc, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶜᶜᵃ, grid_face, loc_cc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶜᶜᵃ, src_grid_face, loc_cc, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶜᶜᵃ, grid_face, loc_cc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶜᶜᵃ, src_grid_face, loc_cc, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_face.Δxᶜᶠᵃ, grid_face, loc_cf, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶠᶜᵃ, src_grid_face, loc_fc, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶜᶠᵃ, grid_face, loc_cf, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶠᶜᵃ, src_grid_face, loc_fc, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶜᶠᵃ, grid_face, loc_cc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶠᶜᵃ, src_grid_face, loc_cc, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_face.Δxᶠᶜᵃ, grid_face, loc_fc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶜᶠᵃ, src_grid_face, loc_cf, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶠᶜᵃ, grid_face, loc_fc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶜᶠᵃ, src_grid_face, loc_cf, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶠᶜᵃ, grid_face, loc_cc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶜᶠᵃ, src_grid_face, loc_cc, src_side), (2, 1, 3)), dims=reverse_dim)

            grid_metric_halo(grid_face.Δxᶠᶠᵃ, grid_face, loc_ff, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δyᶠᶠᵃ, src_grid_face, loc_ff, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Δyᶠᶠᵃ, grid_face, loc_ff, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Δxᶠᶠᵃ, src_grid_face, loc_ff, src_side), (2, 1, 3)), dims=reverse_dim)
            grid_metric_halo(grid_face.Azᶠᶠᵃ, grid_face, loc_cc, side) .= reverse(permutedims(grid_metric_boundary(grid_face.Azᶠᶠᵃ, src_grid_face, loc_cc, src_side), (2, 1, 3)), dims=reverse_dim)
        end
    end

    return nothing
end
