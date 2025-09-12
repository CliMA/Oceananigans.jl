module OceananigansPythonCallExt

using Oceananigans
using PythonCall
using CondaPkg
using SparseArrays

import Oceananigans.Fields: regridding_weights
import Oceananigans.Grids: λnodes, φnodes, Center, Face
import Oceananigans.Architectures: on_architecture, CPU

"""
    add_package(package_name, channel="conda-forge"; verbose=true)

Install `package_name` with `CondaPkg.add` from `channel`, printing
a few messages if `verbose == true`.
Return a NamedTuple containing package information if successful.
"""
function add_package(name, channel="conda-forge"; verbose=true)
    verbose && @info "Installing $(name)..."
    CondaPkg.add(name; channel)
    pkg = CondaPkg.which(name)
    verbose && @info "... $name has been installed at $(pkg)."
    return pkg
end

"""
    add_import_pkg(package_name, channel="conda-forge")

Import and return `package_name` with `PythonCall.pyimport`,
installing it with `add_package` if it is not found.
"""
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

x_node_array(x::AbstractVector, Nx, Ny) = view(x, 1:Nx) |> Array
y_node_array(x::AbstractVector, Nx, Ny) = view(x, 1:Ny) |> Array
x_node_array(x::AbstractMatrix, Nx, Ny) = view(x, 1:Nx, 1:Ny) |> Array

x_vertex_array(x::AbstractVector, Nx, Ny) = view(x, 1:Nx+1) |> Array
y_vertex_array(x::AbstractVector, Nx, Ny) = view(x, 1:Ny+1) |> Array
x_vertex_array(x::AbstractMatrix, Nx, Ny) = view(x, 1:Nx+1, 1:Ny+1) |> Array

y_node_array(x::AbstractMatrix, Nx, Ny) = x_node_array(x, Nx, Ny)
y_vertex_array(x::AbstractMatrix, Nx, Ny) = x_vertex_array(x, Nx, Ny)

function regridding_weights(dst_field, src_field; method="conservative")

    ℓx, ℓy, ℓz = Oceananigans.Fields.instantiated_location(src_field)
    @assert ℓx isa Center
    @assert ℓy isa Center

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

    # Ensure coordinates are on CPU
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