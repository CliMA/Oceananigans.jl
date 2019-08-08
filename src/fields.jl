"""
    CellField{A<:AbstractArray, G<:Grid} <: Field

A cell-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct CellField{A<:AbstractArray, G<:Grid} <: Field{A, G}
    data::A
    grid::G
end

"""
    FaceFieldX{A<:AbstractArray, G<:Grid} <: FaceField

An x-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldX{A<:AbstractArray, G<:Grid} <: FaceField{A, G}
    data::A
    grid::G
end

"""
    FaceFieldY{A<:AbstractArray, G<:Grid} <: FaceField

A y-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldY{A<:AbstractArray, G<:Grid} <: FaceField{A, G}
    data::A
    grid::G
end

"""
    FaceFieldZ{A<:AbstractArray, G<:Grid} <: Field

A z-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldZ{A<:AbstractArray, G<:Grid} <: FaceField{A, G}
    data::A
    grid::G
end

# Constructors

"""
    CellField([T=eltype(grid)], arch, grid)

Return a `CellField` with element type `T` on `arch` and `grid`.
`T` defaults to the element type of `grid`.
"""
CellField(T, arch, grid) = CellField(zeros(T, arch, grid), grid)

"""
    FaceFieldX([T=eltype(grid)], arch, grid)

Return a `FaceFieldX` with element type `T` on `arch` and `grid`.
`T` defaults to the element type of `grid`.
"""
FaceFieldX(T, arch, grid) = FaceFieldX(zeros(T, arch, grid), grid)

"""
    FaceFieldY([T=eltype(grid)], arch, grid)

Return a `FaceFieldY` with element type `T` on `arch` and `grid`.
`T` defaults to the element type of `grid`.
"""
FaceFieldY(T, arch, grid) = FaceFieldY(zeros(T, arch, grid), grid)

"""
    FaceFieldZ([T=eltype(grid)], arch, grid)

Return a `FaceFieldZ` with element type `T` on `arch` and `grid`.
`T` defaults to the element type of `grid`.
"""
FaceFieldZ(T, arch, grid) = FaceFieldZ(zeros(T, arch, grid), grid)

 CellField(arch, grid) =  CellField(zeros(arch, grid), grid)
FaceFieldX(arch, grid) = FaceFieldX(zeros(arch, grid), grid)
FaceFieldY(arch, grid) = FaceFieldY(zeros(arch, grid), grid)
FaceFieldZ(arch, grid) = FaceFieldZ(zeros(arch, grid), grid)

fieldtype(::CellField) = CellField
fieldtype(::FaceFieldX) = FaceFieldX
fieldtype(::FaceFieldY) = FaceFieldY
fieldtype(::FaceFieldZ) = FaceFieldZ

@inline size(f::Field) = size(f.grid)
@inline length(f::Field) = length(f.data)

@inline getindex(f::Field, inds...) = getindex(f.data, inds...)
@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)
@inline setindex!(f::Field, v, inds...) = setindex!(f.data, v, inds...)

@inline data(f::Field) = view(f.data, 1:f.grid.Nx, 1:f.grid.Ny, 1:f.grid.Nz)

@inline ardata_view(f::Field) = view(f.data.parent, 1+f.grid.Hx:f.grid.Nx+f.grid.Hx, 
                                                    1+f.grid.Hy:f.grid.Ny+f.grid.Hy, 
                                                    1+f.grid.Hz:f.grid.Nz+f.grid.Hz)

@inline ardata(f::Field) = f.data.parent[1+f.grid.Hx:f.grid.Nx+f.grid.Hx, 
                                         1+f.grid.Hy:f.grid.Ny+f.grid.Hy, 
                                         1+f.grid.Hz:f.grid.Nz+f.grid.Hz]

@inline underlying_data(f::Field) = f.data.parent

show(io::IO, f::Field) = show(io, f.data)
iterate(f::Field, state=1) = iterate(f.data, state)

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

xnodes(ϕ::Field) = reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1)
ynodes(ϕ::Field) = reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1)
znodes(ϕ::Field) = reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz)

xnodes(ϕ::FaceFieldX) = reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1)
ynodes(ϕ::FaceFieldY) = reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1)
znodes(ϕ::FaceFieldZ) = reshape(ϕ.grid.zF[1:end-1], 1, 1, ϕ.grid.Nz)

nodes(ϕ) = (xnodes(ϕ), ynodes(ϕ), znodes(ϕ))

zerofunk(args...) = 0

set!(u::Field, v::Number) = @. u.data = v
set!(u::Field{A}, v::Field{A}) where A = @. u.data.parent = v.data.parent

"Set the CPU field `u` to the array `v`."
function set!(u::Field{A}, v::Array) where {
    A <: OffsetArray{T, D, <:Array} where {T, D}}
    for k in 1:u.grid.Nz, j in 1:u.grid.Ny, i in 1:u.grid.Nx
        u[i, j, k] = v[i, j, k]
    end
    return nothing
end

"Set the GPU field `u` to the array `v`."
@hascuda function set!(u::Field{A}, v::Array) where {
    A <: OffsetArray{T, D, <:CuArray} where {T, D}}

    FieldType = fieldtype(u)
    v_field = FieldType(CPU(), u.grid)
    set!(v_field, v)
    set!(u, v_field)
    return nothing
end

"Set the GPU field `u` to the CuArray `v`."
@hascuda function set!(u::Field{A}, v::CuArray) where {
    A <: OffsetArray{T, D, <:CuArray} where {T, D}}
    @launch device(GPU()) config=launch_config(u.grid, 3) _set_gpu!(u.data, v, u.grid)
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

"Set the GPU field `u` data to the CPU field data of `v`."
@hascuda function set!(u::Field{Au}, v::Field{Av}) where {
    Au<:OffsetArray{T1, D, <:CuArray}, Av<:OffsetArray{T2, D, <:Array}} where {T1, T2, D}

    copyto!(u.data.parent, v.data.parent)
    return nothing
end

"Set the CPU field `u` data to the GPU field data of `v`."
@hascuda function set!(u::Field{Au}, v::Field{Av}) where {
    Au<:OffsetArray{T1, D, <:Array}, Av<:OffsetArray{T2, D, <:CuArray}} where {T1, T2, D}

    u.data.parent .= Array(v.data.parent)
    return nothing
end

"Set the CPU field `u` data to the function `f(x, y, z)`."
set!(u::Field, f::Function) = data(u) .= f.(nodes(u)...)

"Set the GPU field `u` data to the function `f(x, y, z)`."
@hascuda function set!(u::Field{A1}, f::Function) where {
    A1 <: OffsetArray{T, D, <:CuArray} where {T, D}}

    # Get type of u, e.g. CellField or FaceFieldX.
    field_type = typeof(u).name.wrapper
    u_cpu = field_type(CPU(), u.grid)

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
`set!(ϕ::Field, data)` function exists.

Example
=======

model = Model(grid=RegularCartesianGrid(N=(32, 32, 32), L=(1, 1, 1))

# Set u to a parabolic function of z, v to random numbers damped
# at top and bottom, and T to some silly array of half zeros, 
# half random numbers.

u₀(x, y, z) = z/model.grid.Lz * (1 + z/model.grid.Lz)
v₀(x, y, z) = 1e-3 * rand() * u₀(x, y, z)

T₀ = rand(size(model.grid)...)
T₀[T₀ .< 0.5] .= 0

set!(model, u=u₀, v=v₀, T=T₀)
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

set_ic!(model; kwargs...) = set!(model; kwargs...)  # legacy wrapper
