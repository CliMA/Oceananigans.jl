using CUDA
using CUDAKernels
using KernelAbstractions: @kernel, @index
using Adapt: adapt_structure

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
const AbstractCPUField = AbstractField{X, Y, Z, <:CPU} where {X, Y, Z}

""" Set the CPU field `u` data to the function `f(x, y, z)`. """
function set!(u::AbstractCPUField, f::Function)
    f_field = FunctionField(location(u), f, u.grid)
    u .= f_field
    return nothing
end

#####
##### set! for fields on the GPU
#####

const AbstractGPUField = AbstractField{X, Y, Z, <:GPU} where {X, Y, Z}

""" Set the GPU field `u` to the array or function `v`. """
function set!(u::AbstractGPUField, v::Union{Array, Function})
    cpu_grid = on_architecture(CPU(), u.grid)
    v_field = similar(u, cpu_grid)
    set!(v_field, v)
    set!(u, v_field)
    return nothing
end

#####
##### Setting fields to fields
#####

""" Set the CPU field `u` data to the GPU field data of `v`. """
set!(u::AbstractCPUField, v::AbstractGPUField) = copyto!(parent(u), parent(v))

""" Set the GPU field `u` data to the CPU field data of `v`. """
set!(u::AbstractGPUField, v::AbstractCPUField) = copyto!(parent(u), parent(v))

set!(u::AbstractCPUField, v::AbstractCPUField) = parent(u) .= parent(v)
set!(u::AbstractGPUField, v::AbstractGPUField) = parent(u) .= parent(v)
