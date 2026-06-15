module OceananigansMakieExt

export geo_surface!, geo_surface, spherical_coordinates

using Oceananigans
using Oceananigans.Grids: OrthogonalSphericalShellGrid, LatitudeLongitudeGrid, topology,
                          xnode, ynode, znode
using Oceananigans.Fields: AbstractField, location, interior
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans: quadmesh, quadmesh!   # extend the main-package stubs

const SphericalGrid = Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}

using Makie: Observable, AbstractPlot, Axis, Axis3, Figure, NoShading, @lift,
             Point2f, Point3f, GLTriangleFace, mesh!

import Makie: convert_arguments, _create_plot, args_preferred_axis, surface!, surface

# Extending args_preferred_axis here ensures that Field
# do not overstate a preference for being plotted in a 3D LScene.
# Because often we are trying to plot 1D and 2D Field, even though
# (perhaps incorrectly) all Field are AbstractArray{3}.
args_preferred_axis(::AbstractField) = nothing

function drop_singleton_indices(N)
    if N == 1
        return 1
    else
        return Colon()
    end
end

"""
    deduce_dimensionality(f)

Deduce the dimensionality of the Field or FieldTimeSeries `f` and return a 3-tuple `d1, d2, D`, where
`d1` is the first dimension along which `f` varies, `d2` is the second dimension (if any),
and `D` is the total dimensionality of `f`.
"""
function deduce_dimensionality(f)
    # Find indices of the dimensions along which `f` varies
    d1 = findfirst(n -> n > 1, size(f))
    d2 =  findlast(n -> n > 1, size(f))

    # Deduce total dimensionality
    D = sum((d > 1) for d in size(f))

    return d1, d2, D
end

axis_str(::RectilinearGrid, dim) = ("x", "y", "z", "Time")[dim]
axis_str(::LatitudeLongitudeGrid, dim) = ("Longitude (deg)", "Latitude (deg)", "z", "Time")[dim]
axis_str(::OrthogonalSphericalShellGrid, dim) = ""
axis_str(grid::ImmersedBoundaryGrid, dim) = axis_str(grid.underlying_grid, dim)

const LLGOrIBLLG = Union{LatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}}

const FieldOrFTS = Union{Field, FieldTimeSeries}

function _create_plot(F::Function, attributes::Dict, f::FieldOrFTS)
    converted_args = convert_field_argument(f)

    if !(:axis ∈ keys(attributes)) # Let's try to automatically add labels and ticks
        d1, d2, D = deduce_dimensionality(f)
        grid = f.grid

        if D === 1 # 1D plot

            # See `convert_field_argument` for this horizontal/vertical plotting convention.
            if d1 === 1 || d1 === 4 # This is a horizontal or time series plot, so we add xlabel
                axis = (; xlabel=axis_str(grid, d1))
            else # vertical plot with a ylabel
                axis = (; ylabel=axis_str(grid, d1))
            end

        elseif D === 2 # it's a two-dimensional plot
            if d2 === 4
                # Always plot time on horizontal axis
                axis = (xlabel=axis_str(grid, d2), ylabel=axis_str(grid, d1))
            else
                axis = (xlabel=axis_str(grid, d1), ylabel=axis_str(grid, d2))
            end
        else
            throw(ArgumentError("Cannot create axis labels for a 3D field!"))
        end

        # if longitude wraps around the globe then adjust the longitude ticks
        if grid isa LLGOrIBLLG && grid.Lx == 360 && topology(grid, 1) == Periodic
            axis = merge(axis, (; xticks = -360:60:360))
        end

        attributes[:axis] = axis
    end

    return _create_plot(F, attributes, converted_args...)
end

function _create_plot(F::Function, attributes::Dict, op::AbstractOperation)
    f = Field(op)
    return _create_plot(F, attributes, f)
end

_create_plot(F::Function, attributes::Dict, f::Observable{<:Field}) =
    _create_plot(F, attributes, f[])

convert_arguments(pl::Type{<:AbstractPlot}, f::FieldOrFTS) =
    convert_arguments(pl, convert_field_argument(f)...)

function convert_arguments(pl::Type{<:AbstractPlot}, op::AbstractOperation)
    f = Field(op)
    return convert_arguments(pl, f)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, op::AbstractOperation)
    f = Field(op)
    return convert_arguments(pl, ξ1, f)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, ξ2::AbstractArray, op::AbstractOperation)
    f = Field(op)
    return convert_arguments(pl, ξ1, ξ2, f)
end

"""
    make_plottable_array(f)

Convert a field `f` to an array that can be plotted with Makie by

- masking immersed cells (for fields on immersed boundary
grids) with NaNs;
- dropping singleton dimensions, and
- transferring data from GPU to CPU if necessary.
"""
function make_plottable_array(f)
    compute!(f)
    mask_immersed_field!(f, NaN)

    Nx, Ny, Nz = size(f)

    ii = drop_singleton_indices(Nx)
    jj = drop_singleton_indices(Ny)
    kk = drop_singleton_indices(Nz)

    fi = interior(f, ii, jj, kk)
    fi_cpu = on_architecture(CPU(), fi)

    if architecture(f) isa CPU
        fi_cpu = deepcopy(fi_cpu) # so we can re-zero peripheral nodes
    end

    mask_immersed_field!(f)

    return fi_cpu
end

"""
    make_plottable_array(fts::FieldTimeSeries)

Convert a field time series `fts` to an array tha can be plotted with Makie by
iterating fields corresponding to all time indices, converting each field to a
plottable array and stacking along first dimension as time is always assumed to
be plotted on horizontal axis.
"""
make_plottable_array(fts::FieldTimeSeries) = stack(make_plottable_array(fts[i]) for i in 1:length(fts); dims=1)

nodes_and_possibly_times(f::Field) = nodes(f)
nodes_and_possibly_times(f::FieldTimeSeries) = (nodes(f)..., f.times)

function convert_field_argument(f::FieldOrFTS)

    fi_cpu = make_plottable_array(f)
    d1, d2, D = deduce_dimensionality(f)
    fnodes = nodes_and_possibly_times(f)

    if D == 1

        ξ1 = fnodes[d1]
        ξ1_cpu = on_architecture(CPU(), ξ1)

        # Shenanigans
        if d1 === 1 || d1 === 4 # horizontal or time series plot, in x
            return ξ1_cpu, fi_cpu
        else # vertical plot instead
            return fi_cpu, ξ1_cpu
        end

    elseif D == 2

        # If time series plot swap time to be horizontal (x) axis
        d1, d2 = (d2 == 4) ? (d2, d1) : (d1, d2)
        ξ1 = fnodes[d1]
        ξ2 = fnodes[d2]

        ξ1_cpu = on_architecture(CPU(), ξ1)
        ξ2_cpu = on_architecture(CPU(), ξ2)

        return ξ1_cpu, ξ2_cpu, fi_cpu

    elseif D == 3
        throw(ArgumentError("Cannot convert_arguments for a 3D field!"))
    end
end

# For Fields on OrthogonalSphericalShellGrid, just return the interior without coordinates
# TODO: support plotting in geographic coordinates using mesh
# See for example
# https://github.com/navidcy/Imaginocean.jl/blob/f5cc5f27dd2e99e0af490e8dca5a53daf6837ead/src/Imaginocean.jl#L259
const OSSGField = Field{<:Any, <:Any, <:Any, <:Any, <:OrthogonalSphericalShellGrid}
convert_field_argument(f::OSSGField) = make_plottable_array(f)

#####
##### When nodes are provided
#####

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, f::FieldOrFTS)
    fi_cpu = make_plottable_array(f)
    return convert_arguments(pl, ξ1, fi_cpu)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, ξ2::AbstractArray, f::FieldOrFTS)
    fi_cpu = make_plottable_array(f)
    return convert_arguments(pl, ξ1, ξ2, fi_cpu)
end

# For vertical plots
function convert_arguments(pl::Type{<:AbstractPlot}, f::FieldOrFTS, ξ1::AbstractArray)
    fi_cpu = make_plottable_array(f)
    return convert_arguments(pl, fi_cpu, ξ1)
end

#####
##### Spherical plotting utilities
#####

"""
    spherical_coordinates(λ, φ, r=1)

Convert longitude `λ` (degrees), latitude `φ` (degrees), and radius `r` to
Cartesian coordinates `(x, y, z)` for 3D plotting on a sphere.

Arguments
=========
- `λ`: Longitude in degrees (can be a scalar, vector, or array)
- `φ`: Latitude in degrees (can be a scalar, vector, or array)
- `r`: Radius (default: 1). Can be a scalar for uniform radius, or an array
       for plotting data as radial displacement on the sphere.

Returns
=======
A tuple `(x, y, z)` of Cartesian coordinates.

Examples
========

```jldoctest spherical
using Oceananigans
using CairoMakie
ext = Base.get_extension(Oceananigans, :OceananigansMakieExt)
spherical_coordinates = ext.spherical_coordinates

# Point on the equator at 0° longitude
x, y, z = spherical_coordinates(0.0, 0.0)
(x, y, z)

# output
(1.0, 0.0, 0.0)
```

```jldoctest spherical
# North pole
x, y, z = spherical_coordinates(0.0, 90.0)
(round(x, digits=10), round(y, digits=10), z)

# output
(0.0, 0.0, 1.0)
```

```jldoctest spherical
# Array of longitudes along equator
λ = [0.0, 90.0, 180.0]
φ = [0.0, 0.0, 0.0]
x, y, z = spherical_coordinates(λ, φ)
round.(x, digits=10)

# output
3-element Vector{Float64}:
  1.0
  0.0
 -1.0
```
"""
function spherical_coordinates(λ, φ, r=1)
    # Convert degrees to radians
    λ_rad = deg2rad.(λ)
    φ_rad = deg2rad.(φ)

    x = @. r * cos(φ_rad) * cos(λ_rad)
    y = @. r * cos(φ_rad) * sin(λ_rad)
    z = @. r * sin(φ_rad)

    return x, y, z
end

"""
    spherical_coordinates(grid, ℓx, ℓy)

Extract longitude and latitude coordinates from a spherical `grid` at locations `(ℓx, ℓy)`,
and convert them to Cartesian coordinates for 3D plotting.

Arguments
=========
- `grid`: A `LatitudeLongitudeGrid` or `OrthogonalSphericalShellGrid`
- `ℓx`: Location in x-direction (`Center()` or `Face()`)
- `ℓy`: Location in y-direction (`Center()` or `Face()`)

Returns
=======
A tuple `(x, y, z)` of Cartesian coordinates suitable for `surface!` plotting.
"""
function spherical_coordinates(grid::LatitudeLongitudeGrid, ℓx, ℓy)
    λ = on_architecture(CPU(), λnodes(grid, ℓx))
    φ = on_architecture(CPU(), φnodes(grid, ℓy))

    # Create 2D meshgrid
    Λ = [λi for λi in λ, φi in φ]
    Φ = [φi for λi in λ, φi in φ]

    return spherical_coordinates(Λ, Φ)
end

function spherical_coordinates(grid::OrthogonalSphericalShellGrid, ℓx, ℓy)
    λ = on_architecture(CPU(), λnodes(grid, ℓx, ℓy))
    φ = on_architecture(CPU(), φnodes(grid, ℓx, ℓy))

    return spherical_coordinates(λ, φ)
end

function spherical_coordinates(grid::ImmersedBoundaryGrid, ℓx, ℓy)
    return spherical_coordinates(grid.underlying_grid, ℓx, ℓy)
end

"""
    spherical_coordinates(f::Field)

Extract spherical coordinates from the grid of field `f` at the field's native location,
and convert them to Cartesian coordinates for 3D plotting.
"""
function spherical_coordinates(f::Field)
    ℓx, ℓy, ℓz = location(f)
    return spherical_coordinates(f.grid, ℓx(), ℓy())
end

"""
    geo_surface!(ax, f::Field; kwargs...)

Plot a 2D horizontal slice of field `f` on a 3D sphere in axis `ax`.

This function extracts the grid coordinates, converts them to Cartesian
coordinates on a unit sphere, and plots the field data as a surface.
This is useful for visualizing global ocean model output on an Axis3.

Arguments
=========
- `ax`: A Makie `Axis3` to plot into
- `f`: A 2D `Field` on a spherical grid (LatitudeLongitudeGrid or OrthogonalSphericalShellGrid)

Keyword arguments
=================
All keyword arguments are passed to `surface!`.

Example
=======

```jldoctest
using CairoMakie
using Oceananigans

ext = Base.get_extension(Oceananigans, :OceananigansMakieExt)
geo_surface! = ext.geo_surface!

grid = LatitudeLongitudeGrid(size=(36, 18, 1),
                             longitude=(0, 360),
                             latitude=(-90, 90),
                             z=(0, 1))

T = CenterField(grid)
set!(T, (λ, φ, z) -> cosd(φ) * sind(λ))

fig = Figure()
ax = Axis3(fig[1, 1]; aspect=:data)
plt = geo_surface!(ax, T)

# output
:Plot
```
"""
function geo_surface!(ax, f::Field; kwargs...)
    compute!(f)
    mask_immersed_field!(f, NaN)

    fi = make_plottable_array(f)
    x, y, z = spherical_coordinates(f)

    plt = surface!(ax, x, y, z; color=fi, shading=NoShading, kwargs...)

    mask_immersed_field!(f)

    return plt
end

"""
    geo_surface!(ax, grid, data; ℓx=Center(), ℓy=Center(), kwargs...)

Plot 2D `data` on a 3D sphere using coordinates from `grid`.

Arguments
=========
- `ax`: A Makie `Axis3` to plot into
- `grid`: A spherical grid (LatitudeLongitudeGrid or OrthogonalSphericalShellGrid)
- `data`: 2D array of data to plot

Keyword arguments
=================
- `ℓx`: Location in x-direction for extracting coordinates (default: `Center()`)
- `ℓy`: Location in y-direction for extracting coordinates (default: `Center()`)
- All other keyword arguments are passed to `surface!`.
"""
function geo_surface!(ax, grid, data; ℓx=Center(), ℓy=Center(), kwargs...)
    data_cpu = on_architecture(CPU(), data)
    x, y, z = spherical_coordinates(grid, ℓx, ℓy)

    return surface!(ax, x, y, z; color=data_cpu, shading=NoShading, kwargs...)
end

"""
    geo_surface(f::Field; kwargs...)

Create a new figure with an Axis3 and plot field `f` on a 3D sphere.

Returns `(fig, ax, plt)` tuple.

See also: [`geo_surface!`](@ref)
"""
function geo_surface(f::Field; figure_kwargs=(;), axis_kwargs=(;), kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = Axis3(fig[1, 1]; aspect=:data, axis_kwargs...)
    plt = geo_surface!(ax, f; kwargs...)
    return fig, ax, plt
end

#####
##### Extend surface! for Fields on spherical grids
#####

const SphericalGrid = Union{LatitudeLongitudeGrid,
                            OrthogonalSphericalShellGrid,
                            ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid},
                            ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:OrthogonalSphericalShellGrid}}

const SphericalField = Field{<:Any, <:Any, <:Any, <:Any, <:SphericalGrid}

"""
    surface!(ax::Axis3, f::SphericalField; kwargs...)

Plot a spherical field `f` on a 3D sphere in `Axis3`.

When a field on a `LatitudeLongitudeGrid` or `OrthogonalSphericalShellGrid` is plotted
using `surface!` on an `Axis3`, this method automatically converts the geographic
coordinates to Cartesian coordinates and renders the data on a unit sphere.

Example
=======

```jldoctest
using CairoMakie
using Oceananigans

grid = LatitudeLongitudeGrid(size=(36, 18, 1),
                             longitude=(0, 360),
                             latitude=(-90, 90),
                             z=(0, 1))

T = CenterField(grid)
set!(T, (λ, φ, z) -> cosd(φ) * sind(λ))

fig = Figure()
ax = Axis3(fig[1, 1]; aspect=:data)
plt = surface!(ax, T; colormap=:viridis)
typeof(plt).name.name

# output
:Plot
```
"""
function surface!(ax::Axis3, f::SphericalField; kwargs...)
    return geo_surface!(ax, f; kwargs...)
end

"""
    surface!(ax::Axis3, f_obs::Observable{<:SphericalField}; kwargs...)

Plot an observable spherical field on a 3D sphere in `Axis3`.

This method handles animations where the field is wrapped in an Observable.
The coordinates are computed from the initial field, and the color data
is updated reactively when the observable changes.
"""
function surface!(ax::Axis3, f_obs::Observable{<:SphericalField}; kwargs...)
    # Get initial field to compute coordinates (grid doesn't change)
    f0 = f_obs[]
    x, y, z = spherical_coordinates(f0)

    # Create an observable for the color data that updates when f_obs changes
    color_obs = @lift begin
        f = $f_obs
        compute!(f)
        mask_immersed_field!(f, NaN)
        fi = make_plottable_array(f)
        mask_immersed_field!(f)
        fi
    end

    return surface!(ax, x, y, z; color=color_obs, shading=NoShading, kwargs...)
end

#####
##### quadmesh!: flat-shaded curvilinear quadrilateral mesh
#####
##### A `pcolormesh`-style plot: each grid cell is drawn as a quadrilateral from
##### its four explicit corners and filled with a single flat color. Unlike
##### `heatmap!` (rectangular cells from 1D axes) this renders fields on
##### curvilinear meshes — terrain-following vertical slices, LatitudeLongitude /
##### OrthogonalSphericalShell panels — in their true geometry; unlike `surface!`
##### the color is flat per cell rather than Gouraud-interpolated.
#####
##### Implementation: 4 duplicated vertices per quad + 2 triangles, with the cell
##### value repeated across its 4 vertices and passed as the `color` keyword. The
##### equal corner colors degenerate the mesh's Gouraud shading to a flat fill —
##### the one path that renders identically across Makie backends (CairoMakie has
##### no per-face color path).

function _quad_faces(ncell)
    faces = Vector{GLTriangleFace}(undef, 2ncell)
    @inbounds for q in 1:ncell
        v = 4(q - 1)
        faces[2q - 1] = GLTriangleFace(v + 1, v + 2, v + 3)
        faces[2q]     = GLTriangleFace(v + 1, v + 3, v + 4)
    end
    return faces
end

# Linear indices of the cells to draw (all, or only the non-NaN cells).
_cell_keep(vals, drop_nan_cells) = drop_nan_cells ? findall(!isnan, vec(vals)) : collect(1:length(vals))

# Cell values repeated 4× (once per duplicated quad vertex), as Float32.
function _quad_colors(vals::AbstractMatrix, keep)
    v = vec(vals)
    c = Vector{Float32}(undef, 4length(keep))
    @inbounds for (q, lin) in enumerate(keep)
        c[4q-3] = c[4q-2] = c[4q-1] = c[4q] = Float32(v[lin])
    end
    return c
end

function _quad_vertices(xc::AbstractMatrix, yc::AbstractMatrix, Np, Nq, keep)
    verts = Vector{Point2f}(undef, 4length(keep))
    CI = CartesianIndices((Np, Nq))
    @inbounds for (q, lin) in enumerate(keep)
        i, j = Tuple(CI[lin]); v = 4(q - 1)
        verts[v+1] = Point2f(xc[i,   j  ], yc[i,   j  ])
        verts[v+2] = Point2f(xc[i+1, j  ], yc[i+1, j  ])
        verts[v+3] = Point2f(xc[i+1, j+1], yc[i+1, j+1])
        verts[v+4] = Point2f(xc[i,   j+1], yc[i,   j+1])
    end
    return verts
end

function _quad_vertices(xc::AbstractMatrix, yc::AbstractMatrix, zc::AbstractMatrix, Np, Nq, keep)
    verts = Vector{Point3f}(undef, 4length(keep))
    CI = CartesianIndices((Np, Nq))
    @inbounds for (q, lin) in enumerate(keep)
        i, j = Tuple(CI[lin]); v = 4(q - 1)
        verts[v+1] = Point3f(xc[i,   j  ], yc[i,   j  ], zc[i,   j  ])
        verts[v+2] = Point3f(xc[i+1, j  ], yc[i+1, j  ], zc[i+1, j  ])
        verts[v+3] = Point3f(xc[i+1, j+1], yc[i+1, j+1], zc[i+1, j+1])
        verts[v+4] = Point3f(xc[i,   j+1], yc[i,   j+1], zc[i,   j+1])
    end
    return verts
end

_valsmat(vals::Observable) = vals[]
_valsmat(vals) = vals
_color_arg(vals::Observable, keep) = map(v -> _quad_colors(v, keep), vals)
_color_arg(vals, keep) = _quad_colors(vals, keep)

"""
    quadmesh!(ax, xc, yc, vals; drop_nan_cells=false, kwargs...)
    quadmesh!(ax, xc, yc, zc, vals; drop_nan_cells=false, kwargs...)

Plot cell values `vals` (size `(P, Q)`) as flat-colored quadrilaterals whose
corners are the coordinate matrices `xc, yc` (and `zc` for a panel embedded in 3D,
e.g. on an `Axis3`), each of size `(P+1, Q+1)`. This renders fields on curvilinear
meshes — terrain-following vertical slices, spherical-grid panels — in their true
geometry, where `heatmap!` would draw a rectangle. Each cell gets one flat color
(cf. matplotlib `pcolormesh`), unlike the Gouraud interpolation of `surface!`.

`vals` may be an `Observable{<:AbstractMatrix}` for animations — the geometry is
built once and only the color updates. With `drop_nan_cells=true`, cells whose
value is `NaN` are omitted from the mesh (the mask is taken from `vals` once).
Other keywords (`colormap`, `colorrange`, `nan_color`, `alpha`, …) pass through to
`mesh!`. Returns the `Makie.Mesh` plot, so `Colorbar(fig[…], plt)` works.
"""
function quadmesh!(ax, xc::AbstractMatrix, yc::AbstractMatrix, vals; drop_nan_cells=false, kwargs...)
    vm = _valsmat(vals); Np, Nq = size(vm)
    size(xc) == size(yc) == (Np + 1, Nq + 1) ||
        throw(ArgumentError("corner matrices must be size (P+1, Q+1) = $((Np+1, Nq+1)); got $(size(xc)), $(size(yc))"))
    keep = _cell_keep(vm, drop_nan_cells)
    verts = _quad_vertices(xc, yc, Np, Nq, keep)
    return mesh!(ax, verts, _quad_faces(length(keep)); color=_color_arg(vals, keep), shading=NoShading, kwargs...)
end

function quadmesh!(ax, xc::AbstractMatrix, yc::AbstractMatrix, zc::AbstractMatrix, vals; drop_nan_cells=false, kwargs...)
    vm = _valsmat(vals); Np, Nq = size(vm)
    size(xc) == size(yc) == size(zc) == (Np + 1, Nq + 1) ||
        throw(ArgumentError("corner matrices must be size (P+1, Q+1) = $((Np+1, Nq+1))"))
    keep = _cell_keep(vm, drop_nan_cells)
    verts = _quad_vertices(xc, yc, zc, Np, Nq, keep)
    return mesh!(ax, verts, _quad_faces(length(keep)); color=_color_arg(vals, keep), shading=NoShading, kwargs...)
end

"""
    quadmesh(xc, yc, vals; figure_kwargs=(;), axis_kwargs=(;), kwargs...)

Non-mutating [`quadmesh!`](@ref): build a `Figure` and `Axis`, draw, and return
`(figure, axis, plot)`.
"""
function quadmesh(xc::AbstractMatrix, yc::AbstractMatrix, vals; figure_kwargs=(;), axis_kwargs=(;), kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = Axis(fig[1, 1]; axis_kwargs...)
    plt = quadmesh!(ax, xc, yc, vals; kwargs...)
    return fig, ax, plt
end

#####
##### Field method: derive the corner coordinates from the grid automatically
#####

# Pad a Face-node corner array (P,Q)-ish up to (P+1, Q+1): wrap the first row
# back (Periodic) and copy the last column (Bounded) as needed.
function _pad_corners(A, P, Q)
    size(A) == (P + 1, Q + 1) && return A
    a = A
    size(a, 1) == P && (a = vcat(a, reshape(a[1, :], 1, :)))
    size(a, 2) == Q && (a = hcat(a, a[:, end:end]))
    size(a) == (P + 1, Q + 1) ||
        throw(ArgumentError("could not build (P+1,Q+1)=$((P+1,Q+1)) corner grid from node array of size $(size(A))"))
    return a
end

_nodefun(d) = d == 1 ? xnode : d == 2 ? ynode : znode

# Promote a field interior to a 3D array indexed by coordinate dims (x, y, z),
# inserting a singleton for a Flat dimension (whose interior arrives 2D).
function _interior3d(fcpu)
    v = Array(interior(fcpu))
    ndims(v) == 3 && return v
    flat = findfirst(T -> T === Flat, topology(fcpu.grid))
    flat === nothing && throw(ArgumentError("expected a 2D field on a 3D grid, or a field on a grid with a Flat dimension"))
    return reshape(v, ntuple(d -> d == flat ? 1 : size(v, d < flat ? d : d - 1), 3))
end

# Build (P+1, Q+1) physical corner matrices for the two active dims of a 2D field
# by evaluating the scalar node functions over the corner indices (Face in the
# active dims, the slice's Center in the reduced dim). Using the scalar `znode`
# means terrain-following vertical coordinates render with their true curvature.
function _rectilinear_corners(grid, active, reduced, P, Q)
    a, b = active
    ℓ = ntuple(d -> d == reduced ? Center() : Face(), 3)
    fa, fb = _nodefun(a), _nodefun(b)
    Ca = Matrix{Float64}(undef, P + 1, Q + 1)
    Cb = Matrix{Float64}(undef, P + 1, Q + 1)
    for q in 1:Q + 1, p in 1:P + 1
        ijk = ntuple(d -> d == a ? p : d == b ? q : 1, 3)
        Ca[p, q] = fa(ijk..., grid, ℓ...)
        Cb[p, q] = fb(ijk..., grid, ℓ...)
    end
    return Ca, Cb
end

"""
    quadmesh!(ax, f::AbstractField; kwargs...)

Draw a two-dimensional `Field` as a flat-shaded curvilinear mesh, deriving the
cell-corner coordinates from `f`'s grid — so terrain-following vertical slices and
spherical-grid panels render in their true geometry with no manual coordinate
bookkeeping. `f` must be two-dimensional (one dimension reduced or `Flat`). On a
`LatitudeLongitudeGrid` / `OrthogonalSphericalShellGrid` (horizontal field) it is
drawn as a 3-D Cartesian shell — use an `Axis3`; otherwise it is a 2-D slice in
the two active coordinates (vertical slices follow the terrain via `znode`).
"""
function quadmesh!(ax, f::AbstractField; kwargs...)
    fcpu = on_architecture(CPU(), f)
    vals3 = _interior3d(fcpu)
    sz = size(vals3)
    reduced_dims = findall(==(1), sz)
    length(reduced_dims) == 1 ||
        throw(ArgumentError("quadmesh!(ax, f) needs a 2D field (exactly one reduced dimension); got interior size $sz"))
    reduced = reduced_dims[1]
    active = Tuple(d for d in 1:3 if d != reduced)
    vals = dropdims(vals3; dims = reduced)
    P, Q = size(vals)
    grid = fcpu.grid

    if grid isa SphericalGrid && active == (1, 2)
        x, y, z = spherical_coordinates(grid, Face(), Face())
        xc = _pad_corners(Array(x), P, Q)
        yc = _pad_corners(Array(y), P, Q)
        zc = _pad_corners(Array(z), P, Q)
        return quadmesh!(ax, xc, yc, zc, vals; kwargs...)
    else
        Ca, Cb = _rectilinear_corners(grid, active, reduced, P, Q)
        return quadmesh!(ax, Ca, Cb, vals; kwargs...)
    end
end

"""
    quadmesh(f::AbstractField; figure_kwargs=(;), axis_kwargs=(;), kwargs...)

Non-mutating [`quadmesh!`](@ref) for a `Field`: build a `Figure` and an `Axis`
(or `Axis3` for a spherical grid), draw, and return `(figure, axis, plot)`.
"""
function quadmesh(f::AbstractField; figure_kwargs=(;), axis_kwargs=(;), kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = (f.grid isa SphericalGrid) ? Axis3(fig[1, 1]; axis_kwargs...) : Axis(fig[1, 1]; axis_kwargs...)
    plt = quadmesh!(ax, f; kwargs...)
    return fig, ax, plt
end

end # module
