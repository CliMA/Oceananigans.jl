module OceananigansXESMFExt

using XESMF
using Oceananigans
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Fields: AbstractField, topology, location
using Oceananigans.Grids: λnodes, φnodes, Center, Face, total_length

import Oceananigans.Fields: regrid!
import XESMF: Regridder, extract_xesmf_coordinates_structure

node_array(ξ::AbstractMatrix, Nx, Ny) = view(ξ, 1:Nx, 1:Ny)

x_node_array(x::AbstractVector, Nx, Ny) = repeat(view(x, 1:Nx), 1, Ny)
x_node_array(x::AbstractMatrix, Nx, Ny) = node_array(x, Nx, Ny)

y_node_array(y::AbstractVector, Nx, Ny) = repeat(transpose(view(y, 1:Ny)), Nx, 1)
y_node_array(y::AbstractMatrix, Nx, Ny) = node_array(y, Nx, Ny)

vertex_array(ξ::AbstractMatrix, Nx, Ny) = view(ξ, 1:Nx+1, 1:Ny+1)

x_vertex_array(x::AbstractVector, Nx, Ny) = repeat(view(x, 1:Nx+1), 1, Ny+1)
x_vertex_array(x::AbstractMatrix, Nx, Ny) = vertex_array(x, Nx, Ny)

y_vertex_array(y::AbstractVector, Nx, Ny) = repeat(transpose(view(y, 1:Ny+1)), Nx+1, 1)
y_vertex_array(y::AbstractMatrix, Nx, Ny) = vertex_array(y, Nx, Ny)

"""
    extract_xesmf_coordinates_structure(dst_field::AbstractField, src_field::AbstractField)

Extract the coordinates (latitude/longitude) and the coordinates' bounds from the
`src_field` and `dst_field`.
"""
function extract_xesmf_coordinates_structure(dst_field::AbstractField, src_field::AbstractField)

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

    # Python's xESMF expects 2D arrays with (x, y) coordinates
    # in which y varies in dim=1 and x varies in dim=2
    # therefore we transpose the coordinate matrices

    dst_coordinates = Dict("lat"   => permutedims(φᵈ, (2, 1)),  # φ is latitude
                           "lon"   => permutedims(λᵈ, (2, 1)),  # λ is longitude
                           "lat_b" => permutedims(φvᵈ, (2, 1)),
                           "lon_b" => permutedims(λvᵈ, (2, 1)))

    src_coordinates = Dict("lat"   => permutedims(φˢ, (2, 1)),  # φ is latitude
                           "lon"   => permutedims(λˢ, (2, 1)),  # λ is longitude
                           "lat_b" => permutedims(φvˢ, (2, 1)),
                           "lon_b" => permutedims(λvˢ, (2, 1)))

    return dst_coordinates, src_coordinates
end

"""
    Regridder(dst_field::AbstractField, src_field::AbstractField; method="conservative")

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

To create a regridder for two fields that live on different grids.

```@example regridding
using Oceananigans
using XESMF

z = (-1, 0)
tg = TripolarGrid(; size=(180, 85, 1), z, southernmost_latitude = -80)
llg = LatitudeLongitudeGrid(; size=(170, 80, 1), z,
                            longitude=(0, 360), latitude=(-82, 90))

src_field = CenterField(tg)
dst_field = CenterField(llg)

regridder = XESMF.Regridder(dst_field, src_field, method="conservative")
```

Then we can use this regridder to regrid:

```@example regridding
set!(src_field, 1)

regrid!(dst_field, regridder, src_field)
```
"""
function Regridder(dst_field::AbstractField, src_field::AbstractField; method="conservative")

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

    regridder = XESMF.Regridder(src_coordinates, dst_coordinates; method, periodic)
    weights = regridder.weights

    arch = architecture(src_field)

    weights = on_architecture(arch, weights)

    temp_src = on_architecture(architecture(src_field), regridder.src_temp)
    temp_dst = on_architecture(architecture(dst_field), regridder.dst_temp)

    return XESMF.Regridder(method, weights, temp_src, temp_dst)
end

"""
    regrid!(dst_field, regrider::XESMF.Regridder, src_field)

Regrid `src_field` onto the grid of field `dst_field` using the regrider `r`.

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

λ₀, φ₀ = 150, 30.  # degrees
width = 12         # degrees
set!(src_field, (λ, φ, z) -> exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2width^2))

regridder = XESMF.Regridder(dst_field, src_field, method="conservative")

regrid!(dst_field, regridder, src_field)

first(Field(Integral(dst_field, dims=(1, 2))))
```
"""
function regrid!(dst_field,  regridder::XESMF.Regridder, src_field)
    Nz = size(src_field.grid)[3]
    topo_z = topology(src_field)[3]()
    ℓz = location(src_field)[3]()

    for k in 1:total_length(ℓz, topo_z, Nz)
        src = vec(interior(src_field, :, :, k))
        dst = vec(interior(dst_field, :, :, k))
        regridder(dst, src)
    end

    return dst_field
end

end # module
