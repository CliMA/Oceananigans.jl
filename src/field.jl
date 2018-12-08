import Base: size, getindex, similar, *, +, -


struct CellField{T,G<:Grid{T}} <: Field{G}
    data::AbstractArray{T}
    grid::G
end

struct FaceFieldX{T,G<:Grid{T}} <: FaceField{G}
    data::AbstractArray{T}
    grid::G
end

struct FaceFieldY{T,G<:Grid{T}} <: FaceField{G}
    data::AbstractArray{T}
    grid::G
end

struct FaceFieldZ{T,G<:Grid{T}} <: FaceField{G}
    data::AbstractArray{T}
    grid::G
end

"""
    CellField(grid::Grid)

Construct a `CellField` whose values are defined at the center of a cell.
"""
function CellField(grid::Grid{T}) where T <: AbstractFloat
    sz = size(grid)
    data = zeros(T, sz)
    CellField(data, grid)
end

"""
    FaceFieldX(grid::Grid)

A `Field` whose values are defined on the x-face of a cell.
"""
function FaceFieldX(grid::Grid{T}) where T <: AbstractFloat
    sz = size(grid)
    data = zeros(T, sz)
    FaceFieldX(data, grid)
end

"""
    FaceFieldY(grid::Grid)

A `Field` whose values are defined on the y-face of a cell.
"""
function FaceFieldY(grid::Grid{T}) where T <: AbstractFloat
    sz = size(grid)
    data = zeros(T, sz)
    FaceFieldY(data, grid)
end

"""
    FaceFieldZ(grid::Grid)

A `Field` whose values are defined on the z-face of a cell.
"""
function FaceFieldZ(grid::Grid{T}) where T <: AbstractFloat
    sz = size(grid)
    data = zeros(T, sz)
    FaceFieldZ(data, grid)
end

size(f::Field) = size(f.grid)
show(io::IO, f::Field) = show(io, f.data)

# TODO: This will not work if T=Float32 and v::Irrational.
set!(u::Field, v) = @. u.data = v
set!(u::Field, v::Field) = @. u.data = v.data

similar(f::CellField) = CellField(f.grid)
similar(f::FaceFieldX{T,G}) where {T,G} = FaceFieldX(f.grid)
similar(f::FaceFieldY{T,G}) where {T,G} = FaceFieldY(f.grid)
similar(f::FaceFieldZ{T,G}) where {T,G} = FaceFieldZ(f.grid)

# TODO: Revise this using just xC, yC, zC.
# set!(u::Field{G}, f::Function) where {G<:RegularCartesianGrid} = @. u.data = f(u.grid.xCA, u.grid.yCA, u.grid.zCA)
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
