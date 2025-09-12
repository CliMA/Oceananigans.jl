module OceananigansPythonCallExt

using PythonCall
using CondaPkg
using SparseArrays

import Oceananigans.Fields: regridding_weights
import Oceananigans.Grids: λnodes, φnodes, Center, Face
import Oceananigans.Architectures: on_architecture, CPU

"""
    install_xesmf()

Install the xesmf package using CondaPkg.
Returns a NamedTuple containing package information if successful.
"""
function add_package(name, channel="conda-forge"; verbose=true)
    verbose && @info "Installing $(name)..."
    CondaPkg.add(name; channel)
    pkg = CondaPkg.which(name)
    verbose && @info "... $name has been installed at $(pkg)."
    return pkg
end

function add_import_pkg(name, channel="conda-forge")
    pkg = try
        pyimport(name)
    catch
        add_package(name, channel)
        pyimport(name)
    end

    return pkg
end

flip(::Center) = Face()
flip(::Face) = Center()
flip(::Nothing) = nothing

#=
function coordinate_data_arrays(λ, φ)
    xarray = add_import_pkg("xarray")

    λ_da = xarray.DataArray(
        λ',
        dims=["y", "x"],
        coords= Dict(
            "lat" => (["y", "x"], φ'),
            "lon" => (["y", "x"], λ')
        ),
        name="longitude"
    )

    φ_da = xarray.DataArray(
        φ',
        dims=["y", "x"],
        coords= Dict(
            "lat" => (["y", "x"], φ'),
            "lon" => (["y", "x"], λ')
        ),
        name="latitude"
    )

    return λ_da, φ_da
end
=#

function regridding_weights(dst_field, src_field; method="conservative")

    # Extract center coordinates from both fields
    λᵈ = λnodes(dst_field)
    φᵈ = φnodes(dst_field)
    λˢ = λnodes(src_field)
    φˢ = φnodes(src_field)

    # Extract boundary coordinates
    dst_loc = Oceananigans.Fields.instantiated_location(dst_field)
    src_loc = Oceananigans.Fields.instantiated_location(src_field)
    flipped_dst_loc = (flip(dst_loc[1]), flip(dst_loc[2]), dst_loc[3])
    flipped_src_loc = (flip(src_loc[1]), flip(src_loc[2]), src_loc[3])

    λᵈᵇ = λnodes(dst_field.grid, flipped_dst_loc...)
    φᵈᵇ = φnodes(dst_field.grid, flipped_dst_loc...)
    λˢᵇ = λnodes(src_field.grid, flipped_src_loc...)
    φˢᵇ = φnodes(src_field.grid, flipped_src_loc...)

    # Ensure coordinates are on CPU
    λᵈ = on_architecture(CPU(), λᵈ)
    φᵈ = on_architecture(CPU(), φᵈ)
    λˢ = on_architecture(CPU(), λˢ)
    φˢ = on_architecture(CPU(), φˢ)

    λᵈᵇ = on_architecture(CPU(), λᵈᵇ)
    φᵈᵇ = on_architecture(CPU(), φᵈᵇ)
    λˢᵇ = on_architecture(CPU(), λˢᵇ)
    φˢᵇ = on_architecture(CPU(), φˢᵇ)

    #=
    # Convert 1D coordinates to 2D if needed
    if ndims(λᵈ) == 1
        λᵈ = repeat(λᵈ, 1, size(φᵈ, 1))
        φᵈ = repeat(φᵈ, size(λᵈ, 1), 1)
        λᵈᵇ = repeat(λᵈᵇ, 1, size(φᵈᵇ, 1))
        φᵈᵇ = repeat(φᵈᵇ, size(λᵈᵇ, 1), 1)
    end

    if ndims(λˢ) == 1
        λˢ = repeat(λˢ, 1, size(φˢ, 1))
        φˢ = repeat(φˢ, size(λˢ, 1), 1)
        λˢᵇ = repeat(λˢᵇ, 1, size(φˢᵇ, 1))
        φˢᵇ = repeat(φˢᵇ, size(λˢᵇ, 1), 1)
    end
    =#

    # λᵈx,  φᵈx  = coordinate_data_arrays(λᵈ, φᵈ)
    # λˢx,  φˢx  = coordinate_data_arrays(λˢ, φˢ)
    # λᵈᵇx, φᵈᵇx = coordinate_data_arrays(λᵈᵇ, φᵈᵇ)
    # λˢᵇx, φˢᵇx = coordinate_data_arrays(λˢᵇ, φˢᵇ)

    dst_coordinates = Dict("lat"   => λᵈ, 
                           "lon"   => φᵈ,
                           "lat_b" => λᵈᵇ,
                           "lon_b" => φᵈᵇ)
        

    src_coordinates = Dict("lat"   => λˢ, 
                           "lon"   => φˢ,
                           "lat_b" => λˢᵇ,
                           "lon_b" => φˢᵇ)
        
    xesmf = add_import_pkg("xesmf")
    regridder = xesmf.Regridder(src_coordinates, dst_coordinates, method)
    
    return nothing
end

end # module