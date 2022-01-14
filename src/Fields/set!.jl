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

set!(u::Field, v) = u .= v # fallback

function set!(u::Field, f::Function)
    if architecture(u) isa GPU
        cpu_grid = on_architecture(CPU(), u.grid)
        u_cpu = Field(location(u), cpu_grid)
        f_field = field(location(u), f, cpu_grid)
        set!(u_cpu, f_field)
        set!(u, u_cpu)
    elseif architecture(u) isa CPU
        f_field = field(location(u), f, u.grid)
        set!(u, f_field)
    end

    return nothing
end

function set!(u::Field, f::Union{Array, CuArray, OffsetArray})
    f = arch_array(architecture(u), f)
    u .= f
    return nothing
end

function set!(u::Field, v::Field)
    if architecture(u) === architecture(v)
        parent(u) .= parent(v)
    else
        copyto!(parent(u), parent(v))
    end
    return nothing
end

