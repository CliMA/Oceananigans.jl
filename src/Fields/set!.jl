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
function set!(u::CPUField, f::Function)
    f_field = FunctionField(location(u), f, u.grid)
    u .= f_field
    return nothing
end

#####
##### set! for fields on the GPU
#####

const GPUField = Field{LX, LY, LZ, O, <:GPU} where {LX, LY, LZ, O}

""" Set the GPU field `u` to the array or function `v`. """
function set!(u::GPUField, v::Union{Array, Function})
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
set!(u::CPUField, v::GPUField) = copyto!(parent(u), parent(v))

""" Set the GPU field `u` data to the CPU field data of `v`. """
set!(u::GPUField, v::CPUField) = copyto!(parent(u), parent(v))

set!(u::CPUField, v::CPUField) = parent(u) .= parent(v)
set!(u::GPUField, v::GPUField) = parent(u) .= parent(v)
