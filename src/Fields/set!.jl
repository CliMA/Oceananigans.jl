using CUDA
using CUDAKernels
using KernelAbstractions: @kernel, @index
using Adapt: adapt_structure

using Oceananigans.Grids: on_architecture
using Oceananigans.Architectures: device, GPU, CPU, AbstractMultiArchitecture
using Oceananigans.Utils: work_layout

function set!(Φ::NamedTuple; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(Φ, fldname)
        set!(ϕ, value)
    end
    return nothing
end

set!(u::AbstractField, v) = u .= v # fallback

# Niceties
const CPUField = Field{LX, LY, LZ, O, <:CPU} where {LX, LY, LZ, O}

""" Set the CPU field `u` data to the function `f(x, y, z)`. """
function set!(u::Field, f::Union{Array, Function})
    f_field = field(location(u), f, u.grid)
    set!(u, f_field)
    return nothing
end

set!(u::Field, v::Field) = copyto!(parent(u), parent(v))
