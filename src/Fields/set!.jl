using CUDAnative
using GPUifyLoops: @launch, @loop
using Oceananigans.Architectures: device
using Oceananigans.Utils: @loop_xyz

"""
    set!(model; kwargs...)

Set velocity and tracer fields of `model`. The keyword arguments
`kwargs...` take the form `name=data`, where `name` refers to one of the
fields of `model.velocities` or `model.tracers`, and the `data` may be an array,
a function with arguments `(x, y, z)`, or any data type for which a
`set!(ϕ::AbstractField, data)` function exists.

Example
=======
```julia
model = IncompressibleModel(grid=RegularCartesianGrid(size=(32, 32, 32), length=(1, 1, 1))

# Set u to a parabolic function of z, v to random numbers damped
# at top and bottom, and T to some silly array of half zeros,
# half random numbers.

u₀(x, y, z) = z/model.grid.Lz * (1 + z/model.grid.Lz)
v₀(x, y, z) = 1e-3 * rand() * u₀(x, y, z)

T₀ = rand(size(model.grid)...)
T₀[T₀ .< 0.5] .= 0

set!(model, u=u₀, v=v₀, T=T₀)
```
"""
function set!(model; kwargs...)
    for (fldname, value) in kwargs
        if fldname ∈ propertynames(model.velocities)
            ϕ = getproperty(model.velocities, fldname)
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.velocities or model.tracers."))
        end
        set!(ϕ, value)
    end
    return nothing
end

function set!(Φ::NamedTuple; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(Φ, fldname)
        set!(ϕ, value)
    end
    return nothing
end

set!(u::Field, v::Number) = @. u.data = v

set!(u::Field{X, Y, Z, A}, v::Field{X, Y, Z, A}) where {X, Y, Z, A} =
    @. u.data.parent = v.data.parent

# Niceties
const AbstractCPUField =
    AbstractField{X, Y, Z, A, G} where {X, Y, Z, A<:OffsetArray{T, D, <:Array} where {T, D}, G}

@hascuda const AbstractGPUField =
    AbstractField{X, Y, Z, A, G} where {X, Y, Z, A<:OffsetArray{T, D, <:CuArray} where {T, D}, G}


"Set the CPU field `u` to the array `v`."
function set!(u::AbstractCPUField, v::Array)
    for k in 1:u.grid.Nz, j in 1:u.grid.Ny, i in 1:u.grid.Nx
        u[i, j, k] = v[i, j, k]
    end
    return nothing
end

# Set the GPU field `u` to the array `v`.
@hascuda function set!(u::AbstractGPUField, v::Array)
    # Just need a temporary field so we set bcs = nothing.
    v_field = Field(location(u), CPU(), u.grid, nothing)
    set!(v_field, v)
    set!(u, v_field)
    return nothing
end

# Set the GPU field `u` to the CuArray `v`.
@hascuda function set!(u::AbstractGPUField, v::CuArray)
    @launch device(GPU()) config=launch_config(u.grid, :xyz) _set_gpu!(u.data, v, u.grid)
    return nothing
end

function _set_gpu!(u, v, grid)
	@loop_xyz i j k grid begin
        @inbounds u[i, j, k] = v[i, j, k]
    end
    return nothing
end

# Set the GPU field `u` data to the CPU field data of `v`.
@hascuda set!(u::AbstractGPUField, v::AbstractCPUField) = copyto!(u.data.parent, v.data.parent)

# Set the CPU field `u` data to the GPU field data of `v`.
@hascuda set!(u::AbstractCPUField, v::AbstractGPUField) = u.data.parent .= Array(v.data.parent)

"Set the CPU field `u` data to the function `f(x, y, z)`."
set!(u::AbstractCPUField, f::Function) = interior(u) .= f.(nodes(u)...)

# Set the GPU field `u` data to the function `f(x, y, z)`.
@hascuda function set!(u::AbstractGPUField, f::Function)
    # Just need a temporary field so we set bcs = nothing.
    u_cpu = Field(location(u), CPU(), u.grid, nothing)
    set!(u_cpu, f)
    set!(u, u_cpu)
    return nothing
end
