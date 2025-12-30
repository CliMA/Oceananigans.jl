module OceananigansMakieExt

export geo_surface!, geo_surface, spherical_coordinates

using Oceananigans
using Oceananigans.Grids: OrthogonalSphericalShellGrid, topology
using Oceananigans.Fields: AbstractField, location
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using Makie: Observable, AbstractPlot, Axis3, Figure, NoShading, @lift

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

Deduce the dimensionality of the field `f` and return a 3-tuple `d1, d2, D`, where
`d1` is the first dimension along which `f` varies, `d2` is the second dimension (if any),
and `D` is the total dimensionality of `f`.
"""
function deduce_dimensionality(f)
    # Find indices of the dimensions along which `f` varies
    d1 = findfirst(n -> n > 1, size(f))
    d2 =  findlast(n -> n > 1, size(f))

    # Deduce total dimensionality
    Nx, Ny, Nz = size(f)
    D = (Nx > 1) + (Ny > 1) + (Nz > 1)

    return d1, d2, D
end

axis_str(::RectilinearGrid, dim) = ("x", "y", "z")[dim]
axis_str(::LatitudeLongitudeGrid, dim) = ("Longitude (deg)", "Latitude (deg)", "z")[dim]
axis_str(::OrthogonalSphericalShellGrid, dim) = ""
axis_str(grid::ImmersedBoundaryGrid, dim) = axis_str(grid.underlying_grid, dim)

const LLGOrIBLLG = Union{LatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}}

function _create_plot(F::Function, attributes::Dict, f::Field)
    converted_args = convert_field_argument(f)

    if !(:axis ∈ keys(attributes)) # Let's try to automatically add labels and ticks
        d1, d2, D = deduce_dimensionality(f)
        grid = f.grid

        if D === 1 # 1D plot

            # See `convert_field_argument` for this horizontal/vertical plotting convention.
            if d1 === 1 # This is a horizontal plot, so we add xlabel
                axis = (; xlabel=axis_str(grid, 1))
            else # vertical plot with a ylabel
                axis = (; ylabel=axis_str(grid, d1))
            end

        elseif D === 2 # it's a two-dimensional plot
            axis = (xlabel=axis_str(grid, d1), ylabel=axis_str(grid, d2))
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

convert_arguments(pl::Type{<:AbstractPlot}, f::Field) =
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

function convert_field_argument(f::Field)

    fi_cpu = make_plottable_array(f)
    d1, d2, D = deduce_dimensionality(f)
    fnodes = nodes(f)

    if D == 1

        ξ1 = fnodes[d1]
        ξ1_cpu = on_architecture(CPU(), ξ1)

        # Shenanigans
        if d1 === 1 # horizontal plot, in x
            return ξ1_cpu, fi_cpu
        else # vertical plot instead
            return fi_cpu, ξ1_cpu
        end

    elseif D == 2

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

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, f::Field)
    fi_cpu = make_plottable_array(f)
    return convert_arguments(pl, ξ1, fi_cpu)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, ξ2::AbstractArray, f::Field)
    fi_cpu = make_plottable_array(f)
    return convert_arguments(pl, ξ1, ξ2, fi_cpu)
end

# For vertical plots
function convert_arguments(pl::Type{<:AbstractPlot}, f::Field, ξ1::AbstractArray)
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

end # module
