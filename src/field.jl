import Base: size, getindex, similar, *, +, -

"""
    CellField(grid::Grid)

Construct a `CellField` whose values are defined at the center of a cell.
"""
struct CellField{T,G<:Grid{T}} <: Field{G}
    data::AbstractArray{T}
    grid::G
end

"""
    FaceFieldX(grid::Grid)

A `Field` whose values are defined on the x-face of a cell.
"""
struct FaceFieldX{T,G<:Grid{T}} <: FaceField{G}
    data::AbstractArray{T}
    grid::G
end

"""
    FaceFieldY(grid::Grid)

A `Field` whose values are defined on the y-face of a cell.
"""
struct FaceFieldY{T,G<:Grid{T}} <: FaceField{G}
    data::AbstractArray{T}
    grid::G
end

"""
    FaceFieldZ(grid::Grid)

A `Field` whose values are defined on the z-face of a cell.
"""
struct FaceFieldZ{T,G<:Grid{T}} <: FaceField{G}
    data::AbstractArray{T}
    grid::G
end

function CellField(grid::Grid{T}) where T
    sz = size(grid)
    data = zeros(T, sz)
    CellField(data, grid)
end

function FaceFieldX(grid::Grid{T}) where T
    sz = size(grid)
    data = zeros(T, sz)
    FaceFieldX(data, grid)
end

function FaceFieldY(grid::Grid{T}) where T
    sz = size(grid)
    data = zeros(T, sz)
    FaceFieldY(data, grid)
end

function FaceFieldZ(grid::Grid{T}) where T
    sz = size(grid)
    data = zeros(T, sz)
    FaceFieldZ(data, grid)
end

similar(f::CellField) = CellField(f.grid)
size(f::CellField) = size(f.grid)

set!(u::Field, v) = @. u.data = v
set!(u::Field, v::Field) = @. u.data = v.data

# TODO: Revise this using just xC, yC, zC.
# set!(u::Field{G}, f::Function) where {G<:RegularCartesianGrid} = @. u.data = f(u.grid.xCA, u.grid.yCA, u.grid.zCA)

similar(f::FaceFieldX{T,G}) where {T,G} = FaceFieldX(f.grid)
similar(f::FaceFieldY{T,G}) where {T,G} = FaceFieldY(f.grid)
similar(f::FaceFieldZ{T,G}) where {T,G} = FaceFieldZ(f.grid)
size(f::FaceFieldX{T,G}) where {T,G} = size(f.grid)
size(f::FaceFieldY{T,G}) where {T,G} = size(f.grid)
size(f::FaceFieldZ{T,G}) where {T,G} = size(f.grid)

getindex(f::Field, inds...) = getindex(f.data, inds...)

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
