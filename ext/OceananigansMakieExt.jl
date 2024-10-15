module OceananigansMakieExt

using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using MakieCore: AbstractPlot
import MakieCore: convert_arguments, _create_plot
import Makie: args_preferred_axis

# Extending args_preferred_axis here ensures that Field
# do not overstate a preference for being plotted in a 3D LScene.
# Because often we are trying to plot 1D and 2D Field, even though
# (perhaps incorrectly) all Field are AbstractArray{3}.
args_preferred_axis(::Field) = nothing

function drop_singleton_indices(N)
    if N == 1
        return 1
    else
        return Colon()
    end
end

function _create_plot(F::Function, attributes::Dict, f::Field)
    converted_args = convert_field_argument(f)
    return _create_plot(F, attributes, converted_args...)
end

function _create_plot(F::Function, attributes::Dict, op::AbstractOperation)
    f = Field(op)
    compute!(f)
    return _create_plot(F::Function, attributes::Dict, f)
end

convert_arguments(pl::Type{<:AbstractPlot}, f::Field) =
    convert_arguments(pl, convert_field_argument(f)...)

function convert_arguments(pl::Type{<:AbstractPlot}, fop::AbstractOperation)
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

    # Indices of the non-zero dimensions
    d1 = findfirst(n -> n > 1, size(f))
    d2 =  findlast(n -> n > 1, size(f))
    
    # Nodes shenanigans
    fnodes = nodes(f)

    # Deduce dimensionality
    Nx, Ny, Nz = size(f)
    D = (Nx > 1) + (Ny > 1) + (Nz > 1)

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
