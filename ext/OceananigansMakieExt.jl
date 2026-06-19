module OceananigansMakieExt

export geo_surface!, geo_surface, spherical_coordinates

using Oceananigans
using Oceananigans.Grids: AbstractGrid, OrthogonalSphericalShellGrid, LatitudeLongitudeGrid,
                          topology, xnode, ynode, znode, λnodes, φnodes
using Oceananigans.Fields: AbstractField, location, interior
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans: quadmesh, quadmesh!   # extend the main-package stubs

using Makie: Observable, AbstractPlot, Axis, Axis3, Figure, NoShading, @lift,
             Point, GLTriangleFace, mesh!, lines!

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

# For Fields on OrthogonalSphericalShellGrid (or an ImmersedBoundaryGrid wrapping
# one), just return the interior without coordinates. `nodes(f)` returns 2D
# (λ, φ) matrices which Makie's CellGrid heatmap can't accept.
# TODO: support plotting in geographic coordinates using mesh
# See for example
# https://github.com/navidcy/Imaginocean.jl/blob/f5cc5f27dd2e99e0af490e8dca5a53daf6837ead/src/Imaginocean.jl#L259
const OSSGOrIBGOSSG = Union{OrthogonalSphericalShellGrid,
                            ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any,
                                                 <:OrthogonalSphericalShellGrid}}
const OSSGField = Field{<:Any, <:Any, <:Any, <:Any, <:OSSGOrIBGOSSG}
# Wrap in a 1-tuple so the splat in `convert_arguments(pl, convert_field_argument(f)...)`
# passes the matrix as a single argument rather than iterating its elements.
convert_field_argument(f::OSSGField) = (make_plottable_array(f),)

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
##### Each cell is drawn as a quadrilateral from its four corner coordinates and
##### filled with one flat color (cf. matplotlib `pcolormesh`). Unlike `heatmap!`
##### (rectangular cells, 1D axes) this renders curvilinear grids — terrain-
##### following slices, spherical panels — in their true geometry; unlike
##### `surface!` the color is flat per cell, not Gouraud-interpolated. Built as 4
##### duplicated vertices + 2 triangles per quad with the cell value repeated
##### across the 4 vertices and passed as `color`: equal corner colors degenerate
##### Gouraud to a flat fill, the one path identical across backends.

function quad_faces(ncell)
    faces = Vector{GLTriangleFace}(undef, 2ncell)
    for q in 1:ncell
        v = 4(q - 1)
        faces[2q-1] = GLTriangleFace(v+1, v+2, v+3)
        faces[2q]   = GLTriangleFace(v+1, v+3, v+4)
    end
    return faces
end

kept_cells(vals, drop_nan_cells) = drop_nan_cells ? findall(!isnan, vec(vals)) : eachindex(vals)

# Cell values repeated 4× (once per duplicated quad vertex), as Float32.
quad_colors(vals, keep) = Float32[vals[lin] for lin in keep for _ in 1:4]

function quad_vertices(keep, Np, Nq, coords::Vararg{AbstractMatrix, D}) where D
    verts = Vector{Point{D, Float32}}(undef, 4length(keep))
    CI = CartesianIndices((Np, Nq))
    @inbounds for (q, lin) in enumerate(keep)
        i, j = Tuple(CI[lin]); v = 4(q - 1)
        verts[v+1] = Point(ntuple(d -> coords[d][i,   j  ], D))
        verts[v+2] = Point(ntuple(d -> coords[d][i+1, j  ], D))
        verts[v+3] = Point(ntuple(d -> coords[d][i+1, j+1], D))
        verts[v+4] = Point(ntuple(d -> coords[d][i,   j+1], D))
    end
    return verts
end

function build_quadmesh!(ax, coords, vals; drop_nan_cells=false, kwargs...)
    vm = vals isa Observable ? vals[] : vals
    Np, Nq = size(vm)
    all(c -> size(c) == (Np + 1, Nq + 1), coords) ||
        throw(ArgumentError("corner matrices must be size (P+1, Q+1) = $((Np+1, Nq+1))"))
    keep = kept_cells(vm, drop_nan_cells)
    verts = quad_vertices(keep, Np, Nq, coords...)
    color = vals isa Observable ? map(v -> quad_colors(v, keep), vals) : quad_colors(vm, keep)
    return mesh!(ax, verts, quad_faces(length(keep)); color, shading=NoShading, kwargs...)
end

"""
    quadmesh!(ax, xc, yc, vals; drop_nan_cells=false, kwargs...)
    quadmesh!(ax, xc, yc, zc, vals; drop_nan_cells=false, kwargs...)

Plot cell values `vals` (size `(P, Q)`) as flat-colored quadrilaterals whose
corners are the coordinate matrices `xc, yc` (plus `zc` for a panel in 3D, e.g.
on an `Axis3`), each of size `(P+1, Q+1)`. Renders curvilinear grids in their true
geometry where `heatmap!` would draw a rectangle. `vals` may be an `Observable`
for animations — geometry is built once and only the color updates;
`drop_nan_cells=true` omits NaN cells. `colormap`/`colorrange`/`nan_color`/`alpha`
pass through; returns the `Mesh` plot so `Colorbar(fig[…], plt)` works.
"""
quadmesh!(ax, xc::AbstractMatrix, yc::AbstractMatrix, vals; kw...) = build_quadmesh!(ax, (xc, yc), vals; kw...)
quadmesh!(ax, xc::AbstractMatrix, yc::AbstractMatrix, zc::AbstractMatrix, vals; kw...) = build_quadmesh!(ax, (xc, yc, zc), vals; kw...)

"""
    quadmesh(xc, yc, vals; figure_kwargs=(;), axis_kwargs=(;), kwargs...)

Non-mutating [`quadmesh!`](@ref): build a `Figure` and `Axis`, draw, return
`(figure, axis, plot)`.
"""
function quadmesh(xc::AbstractMatrix, yc::AbstractMatrix, vals; figure_kwargs=(;), axis_kwargs=(;), kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = Axis(fig[1, 1]; axis_kwargs...)
    return fig, ax, quadmesh!(ax, xc, yc, vals; kwargs...)
end

#####
##### Field / grid methods: derive the corner coordinates automatically
#####

# (P+1, Q+1) longitude/latitude corners for a spherical grid. The interior
# `λnodes(grid, Face(), Face())` is only (Nx, Ny) — the closing boundary corner
# lives in the halo — so we index the halo'd Face-Face nodes over 1:P+1, 1:Q+1
# (the (P+1)-th wraps to the first, closing the periodic seam with no gap).
function spherical_corners(grid::OrthogonalSphericalShellGrid, P, Q)
    λ = λnodes(grid, Face(), Face(); with_halos=true)[1:P+1, 1:Q+1]
    φ = φnodes(grid, Face(), Face(); with_halos=true)[1:P+1, 1:Q+1]
    return (@. cosd(φ) * cosd(λ)), (@. cosd(φ) * sind(λ)), (@. sind(φ))
end

function spherical_corners(grid::LatitudeLongitudeGrid, P, Q)
    λ1 = λnodes(grid, Face(); with_halos=true)[1:P+1]
    φ1 = φnodes(grid, Face(); with_halos=true)[1:Q+1]
    λ = [λ1[i] for i in 1:P+1, _ in 1:Q+1]; φ = [φ1[j] for _ in 1:P+1, j in 1:Q+1]
    return (@. cosd(φ) * cosd(λ)), (@. cosd(φ) * sind(λ)), (@. sind(φ))
end

spherical_corners(grid::ImmersedBoundaryGrid, P, Q) = spherical_corners(grid.underlying_grid, P, Q)

node_function(d) = d == 1 ? xnode : d == 2 ? ynode : znode

# Promote a field interior to 3D indexed by (x, y, z), inserting a singleton for
# a Flat dimension (whose interior arrives 2D).
function interior_3d(fcpu)
    v = Array(interior(fcpu))
    ndims(v) == 3 && return v
    flat = findfirst(T -> T === Flat, topology(fcpu.grid))
    flat === nothing && throw(ArgumentError("expected a 2D field, or a field on a grid with a Flat dimension"))
    return reshape(v, ntuple(d -> d == flat ? 1 : size(v, d < flat ? d : d - 1), 3))
end

# (P+1, Q+1) physical corners for the two `active` dims, evaluating the scalar
# node functions over the corner indices (Face in the active dims, Center in the
# `reduced` dim). The scalar `znode` carries terrain-following curvature.
function physical_corners(grid, active, reduced, P, Q)
    a, b = active
    ℓ = ntuple(d -> d == reduced ? Center() : Face(), 3)
    fa, fb = node_function(a), node_function(b)
    Ca = Matrix{Float64}(undef, P + 1, Q + 1); Cb = similar(Ca)
    for q in 1:Q+1, p in 1:P+1
        ijk = ntuple(d -> d == a ? p : d == b ? q : 1, 3)
        Ca[p, q] = fa(ijk..., grid, ℓ...); Cb[p, q] = fb(ijk..., grid, ℓ...)
    end
    return Ca, Cb
end

"""
    quadmesh!(ax, f::AbstractField; kwargs...)

Draw a two-dimensional `Field` as a flat-shaded curvilinear mesh, deriving the
cell corners from `f`'s grid — no coordinate bookkeeping. `f` must be 2D (one
reduced or `Flat` dimension). A horizontal field on a `LatitudeLongitudeGrid` /
`OrthogonalSphericalShellGrid` is drawn as a 3-D Cartesian shell (use an `Axis3`);
otherwise it is a 2-D slice in the two active coordinates (vertical slices follow
the terrain via `znode`).
"""
function quadmesh!(ax, f::AbstractField; kwargs...)
    fcpu = on_architecture(CPU(), f)
    vals3 = interior_3d(fcpu)
    reduced_dims = findall(==(1), size(vals3))
    length(reduced_dims) == 1 ||
        throw(ArgumentError("quadmesh!(ax, f) needs a 2D field (exactly one reduced dimension); got interior size $(size(vals3))"))
    reduced = reduced_dims[1]
    active = Tuple(d for d in 1:3 if d != reduced)
    vals = dropdims(vals3; dims=reduced)
    P, Q = size(vals)

    if fcpu.grid isa SphericalGrid && active == (1, 2)
        return quadmesh!(ax, spherical_corners(fcpu.grid, P, Q)..., vals; kwargs...)
    else
        return quadmesh!(ax, physical_corners(fcpu.grid, active, reduced, P, Q)..., vals; kwargs...)
    end
end

"""
    quadmesh(f::AbstractField; figure_kwargs=(;), axis_kwargs=(;), kwargs...)

Non-mutating [`quadmesh!`](@ref) for a `Field`: build a `Figure` and an `Axis`
(or `Axis3` for a spherical grid), draw, return `(figure, axis, plot)`.
"""
function quadmesh(f::AbstractField; figure_kwargs=(;), axis_kwargs=(;), kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = (f.grid isa SphericalGrid) ? Axis3(fig[1, 1]; axis_kwargs...) : Axis(fig[1, 1]; axis_kwargs...)
    return fig, ax, quadmesh!(ax, f; kwargs...)
end

# Draw the cell-edge polylines (rows + columns) of a corner mesh — clean quad
# edges, unlike `wireframe!` of the triangulated mesh which shows the diagonals.
function wireframe_lines!(ax, coords...; kwargs...)
    plt = nothing
    for i in axes(coords[1], 1); plt = lines!(ax, (c[i, :] for c in coords)...; kwargs...); end
    for j in axes(coords[1], 2); plt = lines!(ax, (c[:, j] for c in coords)...; kwargs...); end
    return plt
end

"""
    quadmesh!(ax, grid; color=(:black, 0.6), linewidth=0.75, kwargs...)

Draw `grid` itself as a wireframe — cell edges, no fill — from the same corners
`quadmesh!` would fill. A spherical grid is drawn as a 3-D shell graticule (use an
`Axis3`); a grid with one `Flat` dimension as a 2-D wireframe. (For other 3-D
grids, slice first or pass corner arrays.)
"""
function quadmesh!(ax, grid::SphericalGrid; color=(:black, 0.6), linewidth=0.75, kwargs...)
    Nx, Ny, _ = size(grid)
    return wireframe_lines!(ax, spherical_corners(grid, Nx, Ny)...; color, linewidth, kwargs...)
end

function quadmesh!(ax, grid::AbstractGrid; color=(:black, 0.6), linewidth=0.75, kwargs...)
    flat = findfirst(T -> T === Flat, topology(grid))
    flat === nothing &&
        throw(ArgumentError("quadmesh!(ax, grid) needs a spherical grid or a grid with one Flat dimension; otherwise pass corner arrays"))
    active = Tuple(d for d in 1:3 if d != flat)
    N = size(grid)
    return wireframe_lines!(ax, physical_corners(grid, active, flat, N[active[1]], N[active[2]])...; color, linewidth, kwargs...)
end

end # module
