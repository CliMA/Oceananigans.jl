using CUDA
using CUDAKernels
using KernelAbstractions: @kernel, @index
using Adapt: adapt_structure

using Oceananigans.Grids: on_architecture
using Oceananigans.Architectures: device, GPU, CPU
using Oceananigans.Utils: work_layout

function set!(Φ::NamedTuple; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(Φ, fldname)
        set!(ϕ, value)
    end
    return nothing
end

function set!(u::Field, v)
    u .= v # fallback
    return u
end

function set!(u::Field, f::Function)
    if architecture(u) isa GPU
        cpu_grid = on_architecture(CPU(), u.grid)
        u_cpu = Field(location(u), cpu_grid; indices = indices(u))
        f_field = field(location(u), f, cpu_grid)
        set!(u_cpu, f_field)
        set!(u, u_cpu)
    elseif architecture(u) isa CPU
        f_field = field(location(u), f, u.grid)
        set!(u, f_field)
    end

    return u
end

function set!(u::Field, f::Union{Array, CuArray, OffsetArray})
    f = arch_array(architecture(u), f)
    u .= f
    return u
end

function set!(u::Field, v::Field)
    # Note: we only copy interior points.
    # To copy halos use `parent(u) .= parent(v)`.
    
    if architecture(u) === architecture(v)
        interior(u) .= interior(v)
    else
        v_data = arch_array(architecture(u), v.data)
        interior(u) .= interior(v_data, location(v), v.grid, v.indices)
    end

    return u
end
