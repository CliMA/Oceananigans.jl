module OceananigansPythonCallExt

using PythonCall
using CondaPkg
using SparseArrays

using Oceananigans
using Oceananigans.Grids: λnodes, φnodes, Center, Face
using Oceananigans.Architectures: on_architecture, CPU

import Oceananigans.Fields: regridding_weights

"""
    add_package(package_name, channel="conda-forge"; verbose=true)

Install the Conda package `name` from `channel` using `CondaPkg.add`.

If `verbose` is true, log progress messages.

Returns a `NamedTuple` with package information (`name`, `version`, `channel`)
if successful, or `nothing` if installation could not be verified.
"""
function add_package(name; channel="conda-forge", verbose=true)
    verbose && @info "Installing $name from $channel..."
    CondaPkg.add(name; channel)

    status = CondaPkg.status()
    if haskey(status, name)
        version = status[name].version
        verbose && @info "... $name $version installed."
        return (name=name, version=version, channel=channel)
    else
        verbose && @warn "$name was added but not found in CondaPkg.status()"
        return nothing
    end
end

"""
    add_import_pkg(name; channel="conda-forge", verbose=true)

Ensure that the Python package `name` is available through PythonCall.

Attempts to `pyimport(name)`. If import fails, installs the package via
[`add_package`](@ref) from `channel`, then retries the import.

Returns the imported Python module object on success.
"""
function add_import_pkg(name; channel="conda-forge", verbose=true)
    try
        return pyimport(name)
    catch e
        verbose && @warn "Python package $name not found, installing..."
        add_package(name; channel, verbose)
        return pyimport(name)  # may still throw if package is broken
    end
end

x_node_array(x::AbstractVector, Nx, Ny) = view(x, 1:Nx) |> Array
y_node_array(x::AbstractVector, Nx, Ny) = view(x, 1:Ny) |> Array
x_node_array(x::AbstractMatrix, Nx, Ny) = view(x, 1:Nx, 1:Ny) |> Array

x_vertex_array(x::AbstractVector, Nx, Ny) = view(x, 1:Nx+1) |> Array
y_vertex_array(x::AbstractVector, Nx, Ny) = view(x, 1:Ny+1) |> Array
x_vertex_array(x::AbstractMatrix, Nx, Ny) = view(x, 1:Nx+1, 1:Ny+1) |> Array

y_node_array(x::AbstractMatrix, Nx, Ny) = x_node_array(x, Nx, Ny)
y_vertex_array(x::AbstractMatrix, Nx, Ny) = x_vertex_array(x, Nx, Ny)

"""
    regridding_weights(dst_field, src_field; method="conservative")

Return the regridding weights from `src_field` to `dst_field` using the specified `method`.
The regridding weights are obtained via xESMF Python package. xESMF exposes five different
regridding algorithms from the ESMF library, specified with the `method` keyword argument:

* `"bilinear"`: ESMF.RegridMethod.BILINEAR
* `"conservative"`: ESMF.RegridMethod.CONSERVE
* `"conservative_normed"`: ESMF.RegridMethod.CONSERVE
* `"patch"`: ESMF.RegridMethod.PATCH
* `"nearest_s2d"`: ESMF.RegridMethod.NEAREST_STOD
* `"nearest_d2s"`: ESMF.RegridMethod.NEAREST_DTOS

where `conservative_normed` is just the conservative method with the normalization set to
`ESMF.NormType.FRACAREA` instead of the default `norm_type = ESMF.NormType.DSTAREA`.

For more information, see the xESMF documentation at:

> https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html

"""
function regridding_weights(dst_field, src_field; method="conservative")

    ℓx, ℓy, ℓz = Oceananigans.Fields.instantiated_location(src_field)

    # We only support regridding between centered fields.
    @assert ℓx isa Center
    @assert ℓy isa Center
    @assert (ℓx, ℓy, ℓz) == Oceananigans.Fields.instantiated_location(dst_field)

    dst_grid = dst_field.grid
    src_grid = src_field.grid

    # Extract center coordinates from both fields
    λᵈ = λnodes(dst_grid, Center(), Center(), ℓz, with_halos=true)
    φᵈ = φnodes(dst_grid, Center(), Center(), ℓz, with_halos=true)
    λˢ = λnodes(src_grid, Center(), Center(), ℓz, with_halos=true)
    φˢ = φnodes(src_grid, Center(), Center(), ℓz, with_halos=true)

    # Extract cell vertices
    λvᵈ = λnodes(dst_grid, Face(), Face(), ℓz, with_halos=true)
    φvᵈ = φnodes(dst_grid, Face(), Face(), ℓz, with_halos=true)
    λvˢ = λnodes(src_grid, Face(), Face(), ℓz, with_halos=true)
    φvˢ = φnodes(src_grid, Face(), Face(), ℓz, with_halos=true)

    # Build data structures expected by xESMF.
    Nˢx, Nˢy, Nˢz = size(src_field)
    Nᵈx, Nᵈy, Nᵈz = size(dst_field)

    λᵈ = x_node_array(λᵈ, Nᵈx, Nᵈy)
    φᵈ = y_node_array(φᵈ, Nᵈx, Nᵈy)
    λˢ = x_node_array(λˢ, Nˢx, Nˢy)
    φˢ = y_node_array(φˢ, Nˢx, Nˢy)

    λvᵈ = x_vertex_array(λvᵈ, Nᵈx, Nᵈy)
    φvᵈ = y_vertex_array(φvᵈ, Nᵈx, Nᵈy)
    λvˢ = x_vertex_array(λvˢ, Nˢx, Nˢy)
    φvˢ = y_vertex_array(φvˢ, Nˢx, Nˢy)

    dst_coordinates = Dict("lat"   => λᵈ,
                           "lon"   => φᵈ,
                           "lat_b" => λvᵈ,
                           "lon_b" => φvᵈ)

    src_coordinates = Dict("lat"   => λˢ,
                           "lon"   => φˢ,
                           "lat_b" => λvˢ,
                           "lon_b" => φvˢ)

    periodic = Oceananigans.Grids.topology(dst_field.grid, 1) === Periodic

    xesmf = add_import_pkg("xesmf")
    regridder = xesmf.Regridder(src_coordinates, dst_coordinates, method; periodic)

    # Move back to Julia
    # Convert the regridder weights to a Julia sparse matrix
    FT = eltype(dst_grid)
    coords = regridder.weights.data
    shape  = pyconvert(Tuple{Int, Int}, coords.shape)
    vals   = pyconvert(Array{FT}, coords.data)
    coords = pyconvert(Array{FT}, coords.coords)
    rows = coords[1, :] .+ 1
    cols = coords[2, :] .+ 1

    weights = sparse(rows, cols, vals, shape[1], shape[2])

    return weights
end

end # module
