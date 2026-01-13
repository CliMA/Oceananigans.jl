#####
##### NetCDFWriter struct definition
#####
##### NetCDFWriter functionality is implemented in ext/OceananigansNCDatasetsExt
#####

using Oceananigans.Grids: topology, Flat, StaticVerticalDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

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

suffixed_dim_name_generator(var_name, ::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters) = var_name * connector * location_letters

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

trilocation_location_string(grid::AbstractGrid,             LX, LY, LZ, dim::Val{:z}) = trilocation_location_string(grid.z, LX, LY, LZ, dim)
trilocation_location_string(::StaticVerticalDiscretization, LX, LY, LZ, dim::Val{:z}) = "aa" * loc2letter(LZ)
trilocation_location_string(grid,                           LX, LY, LZ, dim)          = loc2letter(LX) * loc2letter(LY) * loc2letter(LZ)

trilocation_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim, connector="_", location_letters=trilocation_location_string(grid, LX, LY, LZ, dim))

trilocation_dim_name(var_name, grid::ImmersedBoundaryGrid, args...) = trilocation_dim_name(var_name, grid.underlying_grid, args...)

dimension_name_generator_free_surface(dimension_name_generator, var_name, grid, LX, LY, LZ, dim) = dimension_name_generator(var_name, grid, LX, LY, LZ, dim)
dimension_name_generator_free_surface(dimension_name_generator, var_name, grid, LX, LY, LZ, dim::Val{:z}) = dimension_name_generator(var_name, grid, LX, LY, LZ, dim) * "_displacement"

mutable struct NetCDFWriter{G, D, O, T, A, FS, DN, DT} <: AbstractOutputWriter
    grid :: G
    filepath :: String
    dataset :: D
    outputs :: O
    schedule :: T
    array_type :: A
    indices :: Tuple
    global_attributes :: Dict
    output_attributes :: Dict
    dimensions :: Dict
    with_halos :: Bool
    include_grid_metrics :: Bool
    overwrite_existing :: Bool
    verbose :: Bool
    deflatelevel :: Int
    part :: Int
    file_splitting :: FS
    dimension_name_generator :: DN
    dimension_type :: DT
end

function NetCDFWriter(model, outputs; kw...)
    @warn "`using NCDatasets` is required (without erroring!) to use `NetCDFWriter`."
    throw(MethodError(NetCDFWriter, (model, outputs)))
end

function write_grid_reconstruction_data! end
function convert_for_netcdf end
function materialize_from_netcdf end
function reconstruct_grid end
