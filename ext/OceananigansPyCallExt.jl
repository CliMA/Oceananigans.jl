module OceananigansPyCallExt

using Oceananigans
using Oceananigans.Fields: location, ReducedField
using Oceananigans.Grids: AbstractGrid

using PyCall
using SparseArrays
using LinearAlgebra

export regrid_tracers!, regridder_weights

"""
    compute_regridding_weights(to_field, from_field; method = "conservative")

Return the regridding weights obtained via xESMF Python package to regrid a field
from the grid that corresponds to the `from_field` onto the grid that corresponds
to the `to_field` using the specified regridding method.

xESMF exposes five different regridding algorithms from the ESMF library, specified
with the `method` keyword argument:

- `"bilinear"`: ESMF.RegridMethod.BILINEAR
- `"conservative"`: ESMF.RegridMethod.CONSERVE
- `"conservative_normed"`: ESMF.RegridMethod.CONSERVE
- `"patch"`: ESMF.RegridMethod.PATCH
- `"nearest_s2d"`: ESMF.RegridMethod.NEAREST_STOD
- `"nearest_d2s"`: ESMF.RegridMethod.NEAREST_DTOS

where `conservative_normed` is just the conservative method with the normalization set
to `ESMF.NormType.FRACAREA` instead of the default `norm_type=ESMF.NormType.DSTAREA`.

For more information, refer to the xESMF documentation at:
> https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html
"""
function compute_regridding_weights(to_field, from_field, method::String = "conservative")

    # Create source and destination fields

    src_ds = coordinate_dataset(from_field.grid)
    dst_ds = coordinate_dataset(to_field.grid)

    xesmf = pyimport("xesmf")
    regridder = xesmf.Regridder(src_ds, dst_ds, method, periodic=PyObject(true))

    # Move back to Julia
    # Convert the regridder weights to a Julia sparse matrix
    coo = regridder.weights.data
    coords = coo[:coords]
    rows = coords[1,:].+1
    cols = coords[2,:].+1
    vals = Float64.(coo[:data])

    shape = Tuple(Int.(coo[:shape]))
    weights = sparse(rows, cols, vals, shape[1], shape[2])

    return weights
end

const SomeTripolarGrid = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
const SomeLatitudeLongitudeGrid = Union{LatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}}
const TripolarOrLatLonGrid = Union{SomeTripolarGrid, SomeLatitudeLongitudeGrid}

function two_dimensionalize(lat::AbstractVector, lon::AbstractVector)
    Nx = length(lon)
    Ny = length(lat)
    lat = repeat(lat', Nx)
    lon = repeat(lon, 1, Ny)

    return lat, lon
end

function coordinate_dataset(grid::SomeLatitudeLongitudeGrid)
    lat = Array(φnodes(grid, Center(), Center(), Center()))
    lon = Array(λnodes(grid, Center(), Center(), Center()))

    lat_b = Array(φnodes(grid, Face(), Face(), Center()))
    lon_b = Array(grid.λᶠᵃᵃ[1:grid.Nx+1])

    lat,   lon   = two_dimensionalize(lat,   lon)
    lat_b, lon_b = two_dimensionalize(lat_b, lon_b)

    return structured_coordinate_dataset(lat, lon, lat_b, lon_b)
end

function coordinate_dataset(grid::SomeTripolarGrid)
    lat = Array(grid.φᶜᶜᵃ[1:grid.Nx, 1:grid.Ny])
    lon = Array(grid.λᶜᶜᵃ[1:grid.Nx, 1:grid.Ny])

    lat_b = Array(grid.φᶠᶠᵃ[1:grid.Nx+1, 1:grid.Ny+1])
    lon_b = Array(grid.λᶠᶠᵃ[1:grid.Nx+1, 1:grid.Ny+1])

    return structured_coordinate_dataset(lat, lon, lat_b, lon_b)
end

function structured_coordinate_dataset(lat, lon, lat_b, lon_b)
    numpy = pyimport("numpy")
    xarray = xarray("xarray")

    lat = numpy.array(lat)
    lon = numpy.array(lon)

    lat_b = numpy.array(lat_b)
    lon_b = numpy.array(lon_b)

    ds_lat = xarray.DataArray(
        lat',
        dims=["y", "x"],
        coords= PyObject(Dict(
            "lat" => (["y", "x"], lat'),
            "lon" => (["y", "x"], lon')
        )),
        name="latitude"
    )

    ds_lon = xarray.DataArray(
        lon',
        dims=["y", "x"],
        coords= PyObject(Dict(
            "lat" => (["y", "x"], lat'),
            "lon" => (["y", "x"], lon')
        )),
        name="longitude"
    )

    ds_lat_b = xarray.DataArray(
        lat_b',
        dims=["y_b", "x_b"],
        coords= PyObject(Dict(
            "lat_b" => (["y_b", "x_b"], lat_b'),
            "lon_b" => (["y_b", "x_b"], lon_b')
        )),
    )

    ds_lon_b = xarray.DataArray(
        lon_b',
        dims=["y_b", "x_b"],
        coords= PyObject(Dict(
            "lat_b" => (["y_b", "x_b"], lat_b'),
            "lon_b" => (["y_b", "x_b"], lon_b')
        )),
    )

    return  xarray.Dataset(
        PyObject(
            Dict("lat"   => ds_lat,
                "lon"   => ds_lon,
                "lat_b" => ds_lat_b,
                "lon_b" => ds_lon_b))
    )
end

end # module
