"""
    Cell

A type describing the location at the center of a grid cell.
"""
struct Cell end

"""
	Face

A type describing the location at the face of a grid cell.
"""
struct Face end

"""
    Field{X, Y, Z, A, G} <: AbstractLocatedField{X, Y, Z, A, G}

A field defined at the location (`X`, `Y`, `Z`) which can be either `Cell` or `Face`.
"""
struct Field{X, Y, Z, A, G} <: AbstractLocatedField{X, Y, Z, A, G}
    data :: A
    grid :: G
    function Field{X, Y, Z}(data, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(data), typeof(grid)}(data, grid)
    end
end

"""
	Field(L::Tuple, data::AbstractArray, grid)

Construct a `Field` on `grid` using the array `data` with location defined by the tuple `L` 
of length 3 whose elements are `Cell` or `Face`.
"""
Field(L::Tuple, data::AbstractArray, grid) = Field{L[1], L[2], L[3]}(data, grid)

"""
    Field(L::Tuple, arch::AbstractArchitecture, grid)

Construct a `Field` on architecture `arch` and `grid` at location `L`,
where `L` is a tuple of `Cell` or `Face` types.
"""
Field(L::Tuple, arch::AbstractArchitecture, grid) = 
    Field{L[1], L[2], L[3]}(zeros(arch, grid), grid)

"""
    Field(X, Y, Z, arch::AbstractArchitecture, grid)

Construct a `Field` on architecture `arch` and `grid` at location `X`, `Y`, `Z`, 
where each of `X, Y, Z` is `Cell` or `Face`.
"""
Field(X, Y, Z, arch::AbstractArchitecture, grid) =  Field((X, Y, Z), arch, grid)
   
"""
    CellField([T=eltype(grid)], arch, grid)

Return a `Field{Cell, Cell, Cell}` on architecture `arch` and `grid`.
Used for tracers and pressure fields.
"""
CellField(T, arch, grid) = Field{Cell, Cell, Cell}(zeros(T, arch, grid), grid)

"""
    FaceFieldX([T=eltype(grid)], arch, grid)

Return a `Field{Face, Cell, Cell}` on architecture `arch` and `grid`.
Used for the x-velocity field.
"""
FaceFieldX(T, arch, grid) = Field{Face, Cell, Cell}(zeros(T, arch, grid), grid)

"""
    FaceFieldY([T=eltype(grid)], arch, grid)

Return a `Field{Cell, Face, Cell}` on architecture `arch` and `grid`.
Used for the y-velocity field.
"""
FaceFieldY(T, arch, grid) = Field{Cell, Face, Cell}(zeros(T, arch, grid), grid)

"""
    FaceFieldZ([T=eltype(grid)], arch, grid)

Return a `Field{Cell, Cell, Face}` on architecture `arch` and `grid`.
Used for the z-velocity field.
"""
FaceFieldZ(T, arch, grid) = Field{Cell, Cell, Face}(zeros(T, arch, grid), grid)

 CellField(arch, grid) = Field((Cell, Cell, Cell), arch, grid)
FaceFieldX(arch, grid) = Field((Face, Cell, Cell), arch, grid)
FaceFieldY(arch, grid) = Field((Cell, Face, Cell), arch, grid)
FaceFieldZ(arch, grid) = Field((Cell, Cell, Face), arch, grid)

location(a) = nothing
location(::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = (X, Y, Z)

architecture(f::Field) = architecture(f.data)
architecture(o::OffsetArray) = architecture(o.parent)

@inline size(f::AbstractField) = size(f.grid)
@inline length(f::Field) = length(f.data)

@propagate_inbounds getindex(f::Field, inds...) = @inbounds getindex(f.data, inds...)
@propagate_inbounds setindex!(f::Field, v, inds...) = @inbounds setindex!(f.data, v, inds...)
@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)

"Returns `f.data` for `f::Field` or `f` for `f::AbstractArray."
@inline data(a) = a
@inline data(f::Field) = f.data

# Endpoint for recursive `datatuple` function:
@inline datatuple(obj::AbstractField) = data(obj)

"Returns `f.data.parent` for `f::Field`."
@inline Base.parent(f::Field) = f.data.parent

"Returns a view over the interior points of the `field.data`."
@inline interior(f::Field) = view(f.data, 1:f.grid.Nx, 1:f.grid.Ny, 1:f.grid.Nz)

"Returns a reference to the interior points of `field.data.parent.`"
@inline interiorparent(f::Field) = @inbounds f.data.parent[1+f.grid.Hx:f.grid.Nx+f.grid.Hx,
                                                           1+f.grid.Hy:f.grid.Ny+f.grid.Hy,
                                                           1+f.grid.Hz:f.grid.Nz+f.grid.Hz]

iterate(f::Field, state=1) = iterate(f.data, state)

@inline xnode(::Type{Cell}, i, grid) = @inbounds grid.xC[i]
@inline xnode(::Type{Face}, i, grid) = @inbounds grid.xF[i]

@inline ynode(::Type{Cell}, j, grid) = @inbounds grid.yC[j]
@inline ynode(::Type{Face}, j, grid) = @inbounds grid.yF[j]

@inline znode(::Type{Cell}, k, grid) = @inbounds grid.zC[k]
@inline znode(::Type{Face}, k, grid) = @inbounds grid.zF[k]

@inline xnode(i, ϕ::Field{X, Y, Z}) where {X, Y, Z} = xnode(X, i, ϕ.grid)
@inline ynode(j, ϕ::Field{X, Y, Z}) where {X, Y, Z} = ynode(Y, j, ϕ.grid)
@inline znode(k, ϕ::Field{X, Y, Z}) where {X, Y, Z} = znode(Z, k, ϕ.grid)

xnodes(ϕ::AbstractField) = reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1)
ynodes(ϕ::AbstractField) = reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1)
znodes(ϕ::AbstractField) = reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz)

xnodes(ϕ::Field{Face})                    = reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1)
ynodes(ϕ::Field{X, Face}) where X         = reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1)
znodes(ϕ::Field{X, Y, Face}) where {X, Y} = reshape(ϕ.grid.zF[1:end-1], 1, 1, ϕ.grid.Nz)

nodes(ϕ) = (xnodes(ϕ), ynodes(ϕ), znodes(ϕ))

# Niceties
const AbstractCPUField = 
    AbstractField{A, G} where {A<:OffsetArray{T, D, <:Array} where {T, D}, G}

@hascuda const AbstractGPUField = 
    AbstractField{A, G} where {A<:OffsetArray{T, D, <:CuArray} where {T, D}, G}

set!(u::Field, v::Number) = @. u.data = v

set!(u::Field{X, Y, Z, A}, v::Field{X, Y, Z, A}) where {X, Y, Z, A} = 
    @. u.data.parent = v.data.parent

"Set the CPU field `u` to the array `v`."
function set!(u::AbstractCPUField, v::Array)
    for k in 1:u.grid.Nz, j in 1:u.grid.Ny, i in 1:u.grid.Nx
        u[i, j, k] = v[i, j, k]
    end
    return nothing
end

# Set the GPU field `u` to the array `v`.
@hascuda function set!(u::AbstractGPUField, v::Array)
    v_field = Field(location(u), CPU(), u.grid)
    set!(v_field, v)
    set!(u, v_field)
    return nothing
end

# Set the GPU field `u` to the CuArray `v`.
@hascuda function set!(u::AbstractGPUField, v::CuArray)
    @launch device(GPU()) config=launch_config(u.grid, 3) _set_gpu!(u.data, v, u.grid)
    return nothing
end

function _set_gpu!(u, v, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
	            u[i, j, k] = v[i, j, k]
	        end
	    end
    end
    return nothing
end

# Set the GPU field `u` data to the CPU field data of `v`.
@hascuda set!(u::AbstractGPUField, v::AbstractCPUField) = copyto!(u.data.parent, v.data.parent)

# Set the CPU field `u` data to the GPU field data of `v`.
@hascuda set!(u::AbstractCPUField, v::AbstractGPUField) = u.data.parent .= Array(v.data.parent)

"Set the CPU field `u` data to the function `f(x, y, z)`."
set!(u::Field, f::Function) = interior(u) .= f.(nodes(u)...)

# Set the GPU field `u` data to the function `f(x, y, z)`.
@hascuda function set!(u::AbstractGPUField, f::Function)
    u_cpu = Field(location(u), CPU(), u.grid)
    set!(u_cpu, f)
    set!(u, u_cpu)
    return nothing
end

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
model = Model(grid=RegularCartesianGrid(size=(32, 32, 32), length=(1, 1, 1))

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
function set!(model::AbstractModel; kwargs...)
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

set_ic!(model; kwargs...) = set!(model; kwargs...) # legacy wrapper

Adapt.adapt_structure(to, field::Field{X, Y, Z}) where {X, Y, Z} =
    Field{X, Y, Z}(adapt(to, data), field.grid)

show_location(X, Y, Z) = string("(", string(typeof(X())), ", ",
                                     string(typeof(Y())), ", ",
                                     string(typeof(Z())), ")")

show_location(field::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = show_location(X, Y, Z)

short_show(a) = string(typeof(a))
shortname(a::Array) = string(typeof(a).name.wrapper)
                                                                            
show(io::IO, field::Field) =
    print(io, 
          short_show(field), '\n',
          "├── data: ", typeof(field.data), '\n',
          "└── grid: ", typeof(field.grid), '\n',
          "    ├── size: ", size(field.grid), '\n',
          "    └── domain: ", show_domain(field.grid), '\n')

short_show(field::AbstractLocatedField) = string("Field at ", show_location(field))
