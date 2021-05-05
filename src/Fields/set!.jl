using CUDA
using KernelAbstractions: @kernel, @index, CUDADevice
using Adapt: adapt_structure

using Oceananigans.Architectures: device, GPU, AbstractCPUArchitecture, AbstractGPUArchitecture
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
const AbstractCPUField = AbstractField{X, Y, Z, <:AbstractCPUArchitecture} where {X, Y, Z}
const AbstractReducedCPUField = AbstractReducedField{X, Y, Z, <:AbstractCPUArchitecture} where {X, Y, Z}

""" Returns an AbstractReducedField on the CPU. """
function similar_cpu_field(u::AbstractReducedField)
    FieldType = typeof(u).name.wrapper
    return FieldType(location(u), CPU(), u.grid; dims=u.dims)
end

""" Set the CPU field `u` data to the function `f(x, y, z)`. """
function set!(u::AbstractCPUField, f::Function)
    f_field = FunctionField(location(u), f, u.grid)
    u .= f_field
    return nothing
end

#####
##### set! for fields on the GPU
#####

const AbstractGPUField = AbstractField{X, Y, Z, <:AbstractGPUArchitecture} where {X, Y, Z}
const AbstractReducedGPUField = AbstractReducedField{X, Y, Z, <:AbstractGPUArchitecture} where {X, Y, Z}

""" Returns a field on the CPU with `nothing` boundary conditions. """
function similar_cpu_field(u)
    FieldType = typeof(u).name.wrapper
    return FieldType(location(u), CPU(), adapt_structure(CPU(), u.grid), nothing)
end

""" Set the GPU field `u` to the array or function `v`. """
function set!(u::AbstractGPUField, v::Union{Array, Function})
    v_field = similar_cpu_field(u)
    set!(v_field, v)
    set!(u, v_field)
    return nothing
end

""" Set the CPU field `u` data to the GPU field data of `v`. """
set!(u::AbstractCPUField, v::AbstractGPUField) = copyto!(parent(u), parent(v))

""" Set the GPU field `u` data to the CPU field data of `v`. """
set!(u::AbstractGPUField, v::AbstractCPUField) = copyto!(parent(u), parent(v))

set!(u::AbstractCPUField, v::AbstractCPUField) = parent(u) .= parent(v)
set!(u::AbstractGPUField, v::AbstractGPUField) = parent(u) .= parent(v)
