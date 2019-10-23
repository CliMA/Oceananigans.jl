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
    Field{LX, LY, LZ, A, G} <: AbstractField{A, G}

A field defined at the location (`LX`, `LY`, `LZ`) which can be either `Cell` or `Face`.
"""
struct Field{Lx, Ly, Lz, A, G} <: AbstractField{A, G}
    data :: A
    grid :: G
end

"""
	Field(L::Tuple, arch::AbstractArchitecture, grid)

Construct a `Field` on architecture `arch` and `grid` with location defined by the tuple `L` of length 3 whose elements
are `Cell` or `Face`.
"""
function Field(L::Tuple, arch::AbstractArchitecture, grid)
    data = zeros(arch, grid)
    return Field(L, data, grid)
end

"""
	Field(L::Tuple, data::AbstractArray, grid)

Construct a `Field` on `grid` using the array `data` with location defined by the tuple `L` of length 3 whose elements
are `Cell` or `Face`.
"""
Field(L::Tuple, data::AbstractArray, grid) = Field{L[1], L[2], L[3], typeof(data), typeof(grid)}(data, grid)

"""
	Field(Lx, Ly, Lz, data, grid)

Construct a `Field` on `grid` using the array `data` with location defined by `Lx`, `Ly`, and `Lz` which are `Cell`
or `Face`.
"""
Field(Lx, Ly, Lz, data, grid) = Field((Lx, Ly, Lz), data, grid)

"""
	CellField

A field defined at the cell centers. Used for pressure and tracers.
"""
const CellField  = Field{Cell, Cell, Cell}

"""
	FaceFieldX

A field defined at the faces along the x-direction. Used for horizontal velocity u.
"""
const FaceFieldX = Field{Face, Cell, Cell}

"""
	FaceFieldY

A field defined at the faces along the y-direction. Used for horizontal velocity v.
"""
const FaceFieldY = Field{Cell, Face, Cell}

"""
	FaceFieldZ

A field defined at the faces along the z-direction. Used for vertical velocity w.
"""
const FaceFieldZ = Field{Cell, Cell, Face}

"""
    CellField([T=eltype(grid)], arch, grid)

Return a `CellField` on architecture `arch` and `grid`.
"""
CellField(T, arch, grid) = Field(Cell, Cell, Cell, zeros(T, arch, grid), grid)

"""
    FaceFieldX([T=eltype(grid)], arch, grid)

Return a `FaceFieldX` on architecture `arch` and `grid`.
"""
FaceFieldX(T, arch, grid) = Field(Face, Cell, Cell, zeros(T, arch, grid), grid)

"""
    FaceFieldY([T=eltype(grid)], arch, grid)

Return a `FaceFieldY` on architecture `arch` and `grid`.
"""
FaceFieldY(T, arch, grid) = Field(Cell, Face, Cell, zeros(T, arch, grid), grid)

"""
    FaceFieldZ([T=eltype(grid)], arch, grid)

Return a `FaceFieldZ` on architecture `arch` and `grid`.
"""
FaceFieldZ(T, arch, grid) = Field(Cell, Cell, Face, zeros(T, arch, grid), grid)

 CellField(arch, grid) = Field(Cell, Cell, Cell, arch, grid)
FaceFieldX(arch, grid) = Field(Face, Cell, Cell, arch, grid)
FaceFieldY(arch, grid) = Field(Cell, Face, Cell, arch, grid)
FaceFieldZ(arch, grid) = Field(Cell, Cell, Face, arch, grid)

fieldtype(f::AbstractField) = typeof(f).name.wrapper
location(::Field{Lx, Ly, Lz}) where {Lx, Ly, Lz} = (Lx, Ly, Lz)

@inline size(f::AbstractField) = size(f.grid)
@inline length(f::AbstractField) = length(f.data)

@propagate_inbounds getindex(f::AbstractField, inds...) = getindex(f.data, inds...)
@propagate_inbounds setindex!(f::AbstractField, v, inds...) = setindex!(f.data, v, inds...)
@inline lastindex(f::AbstractField) = lastindex(f.data)
@inline lastindex(f::AbstractField, dim) = lastindex(f.data, dim)

"Returns a view over the interior points of the `field.data`."
@inline data(f::AbstractField) = view(f.data, 1:f.grid.Nx, 1:f.grid.Ny, 1:f.grid.Nz)

"Returns a reference to the interior points of `field.data.parent.`"
@inline parentdata(f::AbstractField) = @inbounds f.data.parent[1+f.grid.Hx:f.grid.Nx+f.grid.Hx,
                                                               1+f.grid.Hy:f.grid.Ny+f.grid.Hy,
                                                               1+f.grid.Hz:f.grid.Nz+f.grid.Hz]

show(io::IO, f::AbstractField) = show(io, f.data)
iterate(f::AbstractField, state=1) = iterate(f.data, state)

# Define +, -, and * on fields as element-wise calculations on their data. This
# is only true for fields of the same type, e.g. when adding a FaceFieldY to
# another FaceFieldY, otherwise some interpolation or averaging must be done so
# that the two fields are defined at the same point, so the operation which
# will not be commutative anymore.
for ft in (:CellField, :FaceFieldX, :FaceFieldY, :FaceFieldZ)
    for op in (:+, :-, :*)
        @eval begin
            # +, -, * a Field by a Number on the left.
            function $op(num::Number, f::$ft)
                ff = similar(f)
                @. ff.data = $op(num, f.data)
                ff
            end

            # +, -, * a Field by a Number on the right.
            $op(f::$ft, num::Number) = $op(num, f)

            # Multiplying two fields together
            function $op(f1::$ft, f2::$ft)
                f3 = similar(f1)
                @. f3.data = $op(f1.data, f2.data)
                f3
            end
        end
    end
end

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
const AbstractCPUField = AbstractField{A, G} where {A<:OffsetArray{T, D, <:Array} where {T, D}, G}
@hascuda const AbstractGPUField = AbstractField{A, G} where {A<:OffsetArray{T, D, <:CuArray} where {T, D}, G}

set!(u::AbstractField, v::Number) = @. u.data = v
set!(u::AbstractField{A}, v::AbstractField{A}) where A = @. u.data.parent = v.data.parent

"Set the CPU field `u` to the array `v`."
function set!(u::AbstractCPUField, v::Array)
    for k in 1:u.grid.Nz, j in 1:u.grid.Ny, i in 1:u.grid.Nx
        u[i, j, k] = v[i, j, k]
    end
    return nothing
end

# Set the GPU field `u` to the array `v`.
@hascuda function set!(u::AbstractGPUField, v::Array)
    FieldType = fieldtype(u)
    v_field = FieldType(location(u), CPU(), u.grid)
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
set!(u::AbstractField, f::Function) = data(u) .= f.(nodes(u)...)

# Set the GPU field `u` data to the function `f(x, y, z)`.
@hascuda function set!(u::AbstractGPUField, f::Function)
    FieldType = fieldtype(u)
    u_cpu = FieldType(location(u), CPU(), u.grid)
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
