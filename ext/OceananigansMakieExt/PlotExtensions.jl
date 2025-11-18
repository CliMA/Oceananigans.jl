module PlotExtensions

using Oceananigans
using Oceananigans.Grids: topology
using Oceananigans.Fields: AbstractField
using Oceananigans.AbstractOperations: AbstractOperation

using Makie: Observable

import MakieCore: _create_plot
import Makie: args_preferred_axis

using ..DimensionalityUtils
using ..MakieConversions

# Extending args_preferred_axis here ensures that Field do not overstate a preference for being plotted in a 3D LScene.
# Because often we are trying to plot 1D and 2D Field, even though (perhaps incorrectly) all Field are AbstractArray{3}.
args_preferred_axis(::AbstractField) = nothing

function _create_plot(F::Function, attributes::Dict, f::Field)
    converted_args = convert_field_argument(f)

    if !(:axis ∈ keys(attributes)) # Let's try to automatically add labels and ticks.
        d1, d2, D = deduce_dimensionality(f)
        grid = f.grid

        if D === 1 # 1D plot
            # See `convert_field_argument` for this horizontal/vertical plotting convention.
            if d1 === 1 # This is a horizontal plot, so we add xlabel.
                axis = (; xlabel=axis_str(grid, 1))
            else # vertical plot with a ylabel
                axis = (; ylabel=axis_str(grid, d1))
            end
        elseif D === 2 # it's a two-dimensional plot
            axis = (xlabel=axis_str(grid, d1), ylabel=axis_str(grid, d2))
        else
            throw(ArgumentError("Cannot create axis labels for a 3D field!"))
        end

        # If longitude wraps around the globe, then adjust the longitude ticks.
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

end # module PlotExtensions
