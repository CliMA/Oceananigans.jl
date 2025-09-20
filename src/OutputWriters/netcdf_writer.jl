#####
##### NetCDFWriter struct definition
#####
##### NetCDFWriter functionality is implemented in ext/OceananigansNCDatasetsExt
#####

using Oceananigans.Grids: topology, Flat

#####
##### Dimension name generators
#####

"""
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim; connector="_", location_letters)

Generate dimension names by suffixing the variable name with location letters.
"""
function suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim; connector="_", location_letters)
    # Check if topology is Flat for the given dimension
    topo = topology(grid)

    if dim == Val(:x)
        TX = topo[1]
        if TX == Flat || isnothing(LX)
            return ""
        else
            return "$(var_name)" * connector * location_letters
        end
    elseif dim == Val(:y)
        TY = topo[2]
        if TY == Flat || isnothing(LY)
            return ""
        else
            return "$(var_name)" * connector * location_letters
        end
    elseif dim == Val(:z)
        TZ = topo[3]
        if TZ == Flat || isnothing(LZ)
            return ""
        else
            return "$(var_name)" * connector * location_letters
        end
    end

    return "$(var_name)" * connector * location_letters
end

# Special case for StaticVerticalDiscretization (will be overridden in extension)
function suffixed_dim_name_generator(var_name, coordinate, LX, LY, LZ, dim::Val{:z}; connector="_", location_letters)
    return var_name * connector * location_letters
end

"""
    loc2letter(location, full=true)

Convert location types to letter representations.
"""
loc2letter(::Any, full=true) = full ? "a" : ""

"""
    trilocation_location_string(grid, LX, LY, LZ, dim)

Generate location strings for trilocation naming convention.
"""
function trilocation_location_string(grid, LX, LY, LZ, dim)
    # Default implementation - will be overridden in extension for specific grid types
    return loc2letter(LX, true) * loc2letter(LY, true) * loc2letter(LZ, true)
end

"""
    trilocation_dim_name(var_name, grid, LX, LY, LZ, dim)

Generate dimension names using trilocation naming convention.
This is the default dimension name generator for NetCDF output.
"""
trilocation_dim_name(var_name, grid, LX, LY, LZ, dim) =
    suffixed_dim_name_generator(var_name, grid, LX, LY, LZ, dim, connector="_", location_letters=trilocation_location_string(grid, LX, LY, LZ, dim))

trilocation_dim_name(var_name, grid, args...) = trilocation_dim_name(var_name, grid, args...)

mutable struct NetCDFWriter{G, D, O, T, A, FS, DN} <: AbstractOutputWriter
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
end

function NetCDFWriter(model, outputs; kw...)
    @warn "`using NCDatasets` is required (without erroring!) to use `NetCDFWriter`."
    throw(MethodError(NetCDFWriter, (model, outputs)))
end

function write_grid_reconstruction_data! end
function convert_for_netcdf end
function materialize_from_netcdf end
function reconstruct_grid end