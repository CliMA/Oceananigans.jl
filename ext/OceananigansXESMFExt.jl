module OceananigansXESMFExt

using XESMF
using CUDA
using SparseArrays
using Oceananigans
using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Grids: λnodes, φnodes, Center, Face

import Oceananigans.Architectures: on_architecture
import Oceananigans.Fields: Regridder


function x_node_array(x::AbstractVector, Nx, Ny)
    return Array(repeat(view(x, 1:Nx), 1, Ny))'
end
function  y_node_array(x::AbstractVector, Nx, Ny)
    return Array(repeat(view(x, 1:Ny)', Nx, 1))'
end
x_node_array(x::AbstractMatrix, Nx, Ny) = Array(view(x, 1:Nx, 1:Ny))'

function x_vertex_array(x::AbstractVector, Nx, Ny)
    return Array(repeat(view(x, 1:Nx+1), 1, Ny+1))'
end
function y_vertex_array(x::AbstractVector, Nx, Ny)
    return Array(repeat(view(x, 1:Ny+1)', Nx+1, 1))'
end
x_vertex_array(x::AbstractMatrix, Nx, Ny) = Array(view(x, 1:Nx+1, 1:Ny+1))'

y_node_array(x::AbstractMatrix, Nx, Ny) = x_node_array(x, Nx, Ny)
y_vertex_array(x::AbstractMatrix, Nx, Ny) = x_vertex_array(x, Nx, Ny)

function extract_xesmf_coordinates_structure(dst_field, src_field)

    ℓx, ℓy, ℓz = Oceananigans.Fields.instantiated_location(src_field)

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

    # Build data structures expected by xESMF
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

    dst_coordinates = Dict("lat"   => φᵈ,  # φ is latitude
                           "lon"   => λᵈ,  # λ is longitude
                           "lat_b" => φvᵈ,
                           "lon_b" => λvᵈ)

    src_coordinates = Dict("lat"   => φˢ,  # φ is latitude
                           "lon"   => λˢ,  # λ is longitude
                           "lat_b" => φvˢ,
                           "lon_b" => λvˢ)

    return dst_coordinates, src_coordinates
end

#TODO: fix this method so it converts to CuSparseMatrixCSC
on_architecture(arch, m::SparseMatrixCSC) = m

"""
    Regridder(dst_field, src_field; method="conservative")

Return a regridder from `src_field` to `dst_field` using the specified `method`.
The regridder contains a sparse matrix with the regridding weights.
The regridding weights are obtained via xESMF Python package.
xESMF exposes five different regridding algorithms from the ESMF library,
specified with the `method` keyword argument:

* `"bilinear"`: `ESMF.RegridMethod.BILINEAR`
* `"conservative"`: `ESMF.RegridMethod.CONSERVE`
* `"conservative_normed"`: `ESMF.RegridMethod.CONSERVE`
* `"patch"`: `ESMF.RegridMethod.PATCH`
* `"nearest_s2d"`: `ESMF.RegridMethod.NEAREST_STOD`
* `"nearest_d2s"`: `ESMF.RegridMethod.NEAREST_DTOS`

where `conservative_normed` is just the conservative method with the normalization set to
`ESMF.NormType.FRACAREA` instead of the default `norm_type = ESMF.NormType.DSTAREA`.

For more information, see the Python xESMF documentation at:

> https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html

Example
=======

```@example
using Oceananigans
using XESMF

z = (-1, 0)
tg = TripolarGrid(; size=(360, 170, 1), z, southernmost_latitude = -80)
llg = LatitudeLongitudeGrid(; size=(360, 180, 1), z,
                            longitude=(0, 360), latitude=(-82, 90))

src_field = CenterField(tg)
dst_field = CenterField(llg)

regridder = Oceananigans.Fields.Regridder(dst_field, src_field, method="conservative")
```
"""
function Regridder(dst_field, src_field; method="conservative")

    ℓx, ℓy, ℓz = Oceananigans.Fields.instantiated_location(src_field)

    # We only support regridding between centered fields
    @assert ℓx isa Center
    @assert ℓy isa Center
    @assert (ℓx, ℓy, ℓz) == Oceananigans.Fields.instantiated_location(dst_field)

    src_Nz = size(src_field)[3]
    dst_Nz = size(dst_field)[3]
    @assert src_field.grid.z.cᵃᵃᶠ[1:src_Nz+1] == dst_field.grid.z.cᵃᵃᶠ[1:dst_Nz+1]

    dst_coordinates, src_coordinates = extract_xesmf_coordinates_structure(dst_field, src_field)

    periodic = Oceananigans.Grids.topology(src_field.grid, 1) === Periodic ? true : false

    xesmf = XESMF.xesmf
    regridder = xesmf.Regridder(src_coordinates, dst_coordinates, method; periodic)

    weights = XESMF.sparse_regridder_weights(regridder)

    arch = architecture(src_field)
    FT = eltype(src_field)

    weights = on_architecture(arch, weights)

    Nx, Ny, Nz = size(src_field)
    temp_src = on_architecture(arch, zeros(FT, Nx * Ny))

    Nx, Ny, Nz = size(dst_field)
    temp_dst = on_architecture(arch, zeros(FT, Nx * Ny))

    return Regridder(arch, weights, src_field.grid, dst_field.grid, temp_src, temp_dst)
end

end # module
