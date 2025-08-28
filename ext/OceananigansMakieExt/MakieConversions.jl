module MakieConversions

using Oceananigans
using Oceananigans.Grids: OrthogonalSphericalShellGrid
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using MakieCore: AbstractPlot

import MakieCore: convert_arguments
import ..DimensionalityUtils: deduce_dimensionality, drop_singleton_indices

export make_plottable_array, convert_field_argument

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
- masking immersed cells (for fields on immersed boundary grids) with NaNs;
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

# For Fields on OrthogonalSphericalShellGrid, just return the interior without coordinates.
# TODO: Support plotting in geographic coordinates using mesh.
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

end # module MakieConversions
