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

function regridding_weights(dst_field, src_field; method="conservative")
    xesmf = add_import_pkg("xesmf")
    numpy = add_import_pkg("numpy")
    xarray = add_import_pkg("xarray")

    # Extract center coordinates from both fields
    λd = λnodes(dst_field)
    φd = φnodes(dst_field)
    λs = λnodes(src_field)
    φs = φnodes(src_field)

    # Extract boundary coordinates
    

    λd_b = λnodes(dst_field.grid, Face(), Face(), Center())
    φd_b = φnodes(dst_field.grid, Face(), Face(), Center())
    λs_b = λnodes(src_field.grid, Face(), Face(), Center())
    φs_b = φnodes(src_field.grid, Face(), Face(), Center())

    # Ensure coordinates are on CPU
    λd = on_architecture(CPU(), λd)
    φd = on_architecture(CPU(), φd)
    λs = on_architecture(CPU(), λs)
    φs = on_architecture(CPU(), φs)
    λd_b = on_architecture(CPU(), λd_b)
    φd_b = on_architecture(CPU(), φd_b)
    λs_b = on_architecture(CPU(), λs_b)
    φs_b = on_architecture(CPU(), φs_b)

    # Convert 1D coordinates to 2D if needed
    if ndims(λd) == 1
        λd = repeat(λd', size(φd, 1))
        φd = repeat(φd, 1, size(λd, 2))
    end

    if ndims(λs) == 1
        λs = repeat(λs', size(φs, 1))
        φs = repeat(φs, 1, size(λs, 2))
    end

    if ndims(λd_b) == 1
        λd_b = repeat(λd_b', size(φd_b, 1))
        φd_b = repeat(φd_b, 1, size(λd_b, 2))
    end

    if ndims(λs_b) == 1
        λs_b = repeat(λs_b', size(φs_b, 1))
        φs_b = repeat(φs_b, 1, size(λs_b, 2))
    end

    # Create coordinate datasets
    src_ds = structured_coordinate_dataset(φs, λs, φs_b, λs_b, numpy, xarray)
    dst_ds = structured_coordinate_dataset(φd, λd, φd_b, λd_b, numpy, xarray)

    # Create regridder and extract weights
    regridder = xesmf.Regridder(src_ds, dst_ds, method, periodic=pytrue)
    
    return nothing
end

end # module