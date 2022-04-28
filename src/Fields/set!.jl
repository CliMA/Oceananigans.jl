using CUDA
using CUDAKernels
using AMDGPU
using KernelAbstractions: @kernel, @index
using Adapt: adapt_structure

using Oceananigans.Grids: on_architecture
using Oceananigans.Architectures: device, CUDAGPU, ROCMGPU, CPU, AbstractMultiArchitecture
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

    return u
end

function set!(u::Field, f::Union{Array, CuArray, ROCArray, OffsetArray})
    f = arch_array(architecture(u), f)
    u .= f
    return u
end

function set!(u::Field, v::Field)
    if architecture(u) === architecture(v)
        try # to transfer halos between architectures
            parent(u) .= parent(v)
        catch # just copy interior points
            interior(u) .= interior(v)
        end
    else
        try # to transfer halos between architectures
            u_parent = parent(u)
            v_parent = parent(v)
            # If u_parent is a view, we have to convert to an Array.
            # If u_parent or v_parent is on the GPU, we don't expect
            # SubArray.
            u_parent isa SubArray && (u_parent = Array(u_parent))
            v_parent isa SubArray && (v_parent = Array(v_parent))
            copyto!(u_parent, v_parent)
        catch # just copy interior points
            v_data = arch_array(architecture(u), v.data)
            interior(u) .= interior(v_data, location(v), v.grid)
        end
    end

    return u
end

