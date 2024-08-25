module OceananigansMakieExt

using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using MakieCore: AbstractPlot
import MakieCore: convert_arguments, _create_plot

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

convert_arguments(pl::Type{<:AbstractPlot}, f::Field) =
    convert_arguments(pl, convert_field_argument(f)...)

"""
    make_plottable_field(f)

Convert a field `f` to an array that can be plotted with Makie.
Specifically:

- masking immersed cells (for fields on immersed boundary
grids) with NaNs;
- dropping singleton dimensions, and
- transferring data from GPU to CPU if necessary.
"""
function make_plottable_field(f)

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

    fi_cpu = make_plottable_field(f)

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
        if d1 === 3 # vertical plot...
            return fi_cpu, ξ1_cpu
        else
            return ξ1_cpu, fi_cpu
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
    fi_cpu = make_plottable_field(f)
    return convert_arguments(pl, ξ1, fi_cpu)
end

function convert_arguments(pl::Type{<:AbstractPlot}, ξ1::AbstractArray, ξ2::AbstractArray, f::Field)
    fi_cpu = make_plottable_field(f)
    return convert_arguments(pl, ξ1, ξ2, fi_cpu)
end

end # module
