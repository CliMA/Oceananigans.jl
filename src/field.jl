import Base:
    size, length,
    getindex, lastindex, setindex!,
    iterate, similar, *, +, -

"""
    CellField{T,G<:Grid{T}} <: Field{G}

A cell-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct CellField{T<:AbstractArray} <: Field
    data::T
    grid::Grid
end

"""
    FaceFieldX{T,G<:Grid{T}} <: FaceField{G}

An x-face-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct FaceFieldX{T<:AbstractArray} <: FaceField
    data::T
    grid::Grid
end

"""
    FaceFieldY{T,G<:Grid{T}} <: FaceField{G}

A y-face-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct FaceFieldY{T<:AbstractArray} <: FaceField
    data::T
    grid::Grid
end

"""
    FaceFieldZ{T,G<:Grid{T}} <: FaceField{G}

A z-face-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct FaceFieldZ{T<:AbstractArray} <: FaceField
    data::T
    grid::Grid
end

"""
    CellField(grid::Grid)

Construct a `CellField` whose values are defined at the center of a cell.
"""
function CellField(grid::Grid, T=Float64, dim=3)
    data = zeros(T, size(grid))
    CellField{Array{T,dim}}(data, grid)
end

"""
    FaceFieldX(grid::Grid)

A `Field` whose values are defined on the x-face of a cell.
"""
function FaceFieldX(grid::Grid, T=Float64, dim=3)
    data = zeros(T, size(grid))
    FaceFieldX{Array{T,dim}}(data, grid)
end

"""
    FaceFieldY(grid::Grid)

A `Field` whose values are defined on the y-face of a cell.
"""
function FaceFieldY(grid::Grid, T=Float64, dim=3)
    data = zeros(T, size(grid))
    FaceFieldY{Array{T,dim}}(data, grid)
end

"""
    FaceFieldZ(grid::Grid)

A `Field` whose values are defined on the z-face of a cell.
"""
function FaceFieldZ(grid::Grid, T=Float64, dim=3)
    data = zeros(T, size(grid))
    FaceFieldZ{Array{T,dim}}(data, grid)
end

size(f::Field) = size(f.grid)
length(f::Field) = length(f.data)

getindex(f::Field, inds...) = getindex(f.data, inds...)

lastindex(f::Field) = lastindex(f.data)
lastindex(f::Field, dim) = lastindex(f.data, dim)

setindex!(f::Field, v, inds...) = setindex!(f.data, v, inds...)

show(io::IO, f::Field) = show(io, f.data)

iterate(f::Field, state=1) = iterate(f.data, state)
# iterate(f::Field, state=1) = state > length(f) ? nothing : (f.data[state], state+1)

similar(f::CellField) = CellField(f.grid)
similar(f::FaceFieldX{T}) where {T} = FaceFieldX(f.grid)
similar(f::FaceFieldY{T}) where {T} = FaceFieldY(f.grid)
similar(f::FaceFieldZ{T}) where {T} = FaceFieldZ(f.grid)

# TODO: This will not work if T=Float32 and v::Irrational.
set!(u::Field, v) = @. u.data = v
set!(u::Field, v::Field) = @. u.data = v.data

# TODO: Revise this using just xC, yC, zC.
# set!(u::Field{G}, f::Function) where {G<:RegularCartesianGrid} = @. u.data = f(u.grid.xCA, u.grid.yCA, u.grid.zCA)

# Define +, -, and * on fields as element-wise calculations on their data. This
# is only true for fields of the same type, e.g. when adding a FaceFieldY to
# another FaceFieldY, otherwise some interpolation or averaging must be done so
# that the two fields are defined at the same point, so the operation which
# will not be commutative anymore.
for ft in (:CellField, :FaceFieldX, :FaceFieldY, :FaceFieldZ)
    for op in (:+, :-, :*)
        # TODO: @eval does things in global scope, is this the desired behavior?
        @eval begin
            function $op(f1::$ft, f2::$ft)
                f3 = similar(f1)
                @. f3.data = $op(f1.data, f2.data)
                f3
            end
        end
    end
end
