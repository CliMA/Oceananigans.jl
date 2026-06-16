using Oceananigans.Grids: topology, Flat, StaticVerticalDiscretization, MutableVerticalDiscretization, AbstractVerticalCoordinate
using Oceananigans.OrthogonalSphericalShellGrids: OrthogonalSphericalShellGrid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

# Short aliases for compact dispatch in name-generator method tables.
const OSSG = OrthogonalSphericalShellGrid
const SVD  = StaticVerticalDiscretization
const MVD  = MutableVerticalDiscretization

#####
##### Vertical coordinate naming
#####
#
# The 1D vertical coordinate variable name written to NetCDF depends on the kind of vertical
# discretization:
#   - `StaticVerticalDiscretization`: the reference and physical vertical coordinates coincide,
#     so we use "z".
#   - `MutableVerticalDiscretization` (z-star, σ-coordinates): the saved 1D coordinate is the
#     reference (Lagrangian) coordinate; the physical `z = z(r, η, …)` is reconstructible from
#     `r` and the time-varying free surface `η`. We use "r" for the 1D reference coordinate.
#

vertical_coordinate_name(::SVD) = "z"
vertical_coordinate_name(::MVD) = "r"
vertical_coordinate_name(grid::AbstractGrid) = vertical_coordinate_name(grid.z)
vertical_coordinate_name(grid::ImmersedBoundaryGrid) = vertical_coordinate_name(grid.underlying_grid)

#####
##### Dimension name generators
#####

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX}, LX, LY, LZ, dim::Val{:x}; connector="_", location_letters) where {FT, TX}
    if TX == Flat || isnothing(LX)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX, TY}, LX, LY, LZ, dim::Val{:y}; connector="_", location_letters) where {FT, TX, TY}
    if TY == Flat || isnothing(LY)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

function suffixed_dim_name_generator(var_name, grid::AbstractGrid{FT, TX, TY, TZ}, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) where {FT, TX, TY, TZ}
    if TZ == Flat || isnothing(LZ)
        return ""
    else
        return "$(var_name)" * connector * location_letters
    end
end

suffixed_dim_name_generator(var_name, ::AbstractVerticalCoordinate, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) = var_name * connector * location_letters

loc2letter(::Face, full=true) = "f"
loc2letter(::Center, full=true) = "c"
loc2letter(::Nothing, full=true) = full ? "a" : ""

minimal_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX, false)
minimal_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LY, false)

minimal_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim; connector="_", location_letters=minimal_location_string(grid, LX, LY, LZ, dim))
minimal_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = minimal_dim_name(var_name, grid.underlying_grid, args...)

trilocation_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * "aa"
trilocation_location_string(::RectilinearGrid, LX, LY, LZ, ::Val{:y}) = "a" * loc2letter(LY) * "a"

trilocation_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * loc2letter(LY) * "a"
trilocation_location_string(::LatitudeLongitudeGrid, LX, LY, LZ, ::Val{:y}) = loc2letter(LX) * loc2letter(LY) * "a"

trilocation_location_string(::OSSG, LX, LY, LZ, ::Val{:x}) = loc2letter(LX) * loc2letter(LY) * "a"
trilocation_location_string(::OSSG, LX, LY, LZ, ::Val{:y}) = loc2letter(LX) * loc2letter(LY) * "a"

trilocation_location_string(grid::AbstractGrid,         LX, LY, LZ, dim::Val{:z}) = trilocation_location_string(grid.z, LX, LY, LZ, dim)
trilocation_location_string(::AbstractVerticalCoordinate, LX, LY, LZ, dim::Val{:z}) = "aa" * loc2letter(LZ)
trilocation_location_string(grid,                       LX, LY, LZ, dim)          = loc2letter(LX) * loc2letter(LY) * loc2letter(LZ)

trilocation_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim, connector="_", location_letters=trilocation_location_string(grid, LX, LY, LZ, dim))

trilocation_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = trilocation_dim_name(var_name, grid.underlying_grid, args...)

dimension_name_generator_free_surface(dimension_name_generator, var_name, grid, LX, LY, LZ, dim) = dimension_name_generator(var_name, grid, LX, LY, LZ, dim)
dimension_name_generator_free_surface(dimension_name_generator, var_name, grid, LX, LY, LZ, dim::Val{:z}) = dimension_name_generator(var_name, grid, LX, LY, LZ, dim) * "_displacement"

add_grid_suffix(name, grid_index) = isempty(name) ? name : name * "_grid$(grid_index)"
add_grid_suffix(name, ::Nothing) = name