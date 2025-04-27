module OceananigansMakieExt

using Oceananigans
using Oceananigans.Grids: OrthogonalSphericalShellGrid, topology
using Oceananigans.Fields: AbstractField
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using Makie: Observable
using MakieCore: AbstractPlot
import MakieCore: convert_arguments, _create_plot
import Makie: args_preferred_axis

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
            axis = merge(axis, (xticks = -360:60:360,))
        end

        attributes[:axis] = axis
    end

    return _create_plot(F, attributes, converted_args...)
end

function _create_plot(F::Function, attributes::Dict, op::AbstractOperation)
    f = Field(op)
    compute!(f)
    return _create_plot(F::Function, attributes::Dict, f)
end

_create_plot(F::Function, attributes::Dict, f::Observable{<:Field}) =
    _create_plot(F, attributes, f[])

convert_arguments(pl::Type{<:AbstractPlot}, f::Field) =
    convert_arguments(pl, convert_field_argument(f)...)

function convert_arguments(pl::Type{<:AbstractPlot}, op::AbstractOperation)
    f = Field(op)
    compute!(f)
    return convert_arguments(pl, f)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, op::AbstractOperation)
    f = Field(op)
    compute!(f)
    return convert_arguments(pl, ξ1, f)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, ξ2::AbstractArray, op::AbstractOperation)
    f = Field(op)
    compute!(f)
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

end # module
