module OceananigansMakieExt

using Oceananigans
using Oceananigans.Architectures: on_architecture

using MakieCore: AbstractPlot
import MakieCore: convert_arguments

function drop_singleton_indices(N)
    if N == 1
        return 1
    else
        return Colon()
    end
end

function convert_arguments(pl::Type{<:AbstractPlot}, f::Field)

    # Deduce dimensionality
    Nx, Ny, Nz = size(f)
    D = (Nx > 1) + (Ny > 1) + (Nz > 1)

    # Indices of the non-zero dimensions
    d1 = findfirst(n -> n > 1, size(f))
    d2 =  findlast(n -> n > 1, size(f))

    # Drop singleton dimensions
    ii = drop_singleton_indices(Nx)
    jj = drop_singleton_indices(Ny)
    kk = drop_singleton_indices(Nz)
    fi = interior(f, ii, jj, kk)
    fi_cpu = on_architecture(CPU(), fi)

    # Nodes shenanigans
    fnodes = nodes(f)
    if D == 1

        ξ1 = fnodes[d1]
        ξ1_cpu = on_architecture(CPU(), ξ1)
        return convert_arguments(pl, ξ1_cpu, fi_cpu)

    elseif D == 2

        ξ1 = fnodes[d1]
        ξ2 = fnodes[d2]

        ξ1_cpu = on_architecture(CPU(), ξ1)
        ξ2_cpu = on_architecture(CPU(), ξ2)

        return convert_arguments(pl, ξ1_cpu, ξ2_cpu, fi_cpu)

    elseif D == 3
        throw(ArgumentError("Cannot convert_arguments for a 3D field!"))
    end
end

end # module
