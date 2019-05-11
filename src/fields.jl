import Base:
    size, length,
    getindex, lastindex, setindex!,
    iterate, similar, *, +, -

"""
    CellField{A<:AbstractArray, G<:Grid} <: Field

A cell-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct CellField{A<:AbstractArray, G<:Grid} <: Field
    data::A
    grid::G
end

CellField(T, arch, grid) = CellField(zeros(T, arch, grid), grid)
CellField(arch, grid) = CellField(zeros(arch, grid), grid)

"""
    FaceFieldX{A<:AbstractArray, G<:Grid} <: FaceField

An x-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldX{A<:AbstractArray, G<:Grid} <: FaceField
    data::A
    grid::G
end

FaceFieldX(T, arch, grid) = FaceFieldX(zeros(T, arch, grid), grid)
FaceFieldX(arch, grid) = FaceFieldX(zeros(arch, grid), grid)

"""
    FaceFieldY{A<:AbstractArray, G<:Grid} <: FaceField

A y-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldY{A<:AbstractArray, G<:Grid} <: FaceField
    data::A
    grid::G
end

FaceFieldY(T, arch, g) = FaceFieldY(zeros(T, arch, g), g)
FaceFieldY(arch, g) = FaceFieldY(zeros(arch, g), g)

"""
    FaceFieldZ{T, G<:Grid{T}} <: FaceField{G}

A z-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldZ{A<:AbstractArray, G<:Grid} <: FaceField
    data::A
    grid::G
end

FaceFieldZ(T, arch, grid) = FaceFieldZ(zeros(T, arch, grid), grid)
FaceFieldZ(arch, grid) = FaceFieldZ(zeros(arch, grid), grid)

"""
    EdgeField{T<:AbstractArray} <: Field

An edge-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct EdgeField{A<:AbstractArray, G<:Grid} <: Field
    data::A
    grid::G
end

EdgeField(T, arch, grid) = EdgeField(zeros(T, arch, grid), grid)
EdgeField(arch, grid) = EdgeField(zeros(arch, grid), grid)

abstract type Location end
struct Cell <: Location end
struct Interface <: Location end

"""
    GeneralField{𝕃𝕩<:AbstractArray} <: Field

A general field at the location determined by Lx, Ly, Lz.
"""
struct GeneralField{Lˣ, Lʸ, Lᶻ, A, G} <: Field
    data :: A
    grid :: G
end

function GeneralField(Lˣ, Lʸ, Lᶻ, arch, grid)
    data = zeros(arch, grid)
    return GeneralField{Lˣ, Lʸ, Lᶻ, typeof(data), typeof(grid)}(data, grid)
end

@inline size(f::Field) = size(f.grid)
@inline length(f::Field) = length(f.data)

@inline getindex(f::Field, inds...) = getindex(f.data, inds...)
# @inline getindex(f::Field, inds...) = f.data[inds...]

@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)

@inline setindex!(f::Field, v, inds...) = setindex!(f.data, v, inds...)
# @inline function setindex!(f::Field, v, inds...)
#     f.data[inds...] = v
# end

show(io::IO, f::Field) = show(io, f.data)

iterate(f::Field, state=1) = iterate(f.data, state)
# iterate(f::Field, state=1) = state > length(f) ? nothing : (f.data[state], state+1)

similar(f::CellField{T})  where {T} = CellField(f.metadata, f.grid, f.metadata.float_type)
similar(f::FaceFieldX{T}) where {T} = FaceFieldX(f.metadata, f.grid, f.metadata.float_type)
similar(f::FaceFieldY{T}) where {T} = FaceFieldY(f.metadata, f.grid, f.metadata.float_type)
similar(f::FaceFieldZ{T}) where {T} = FaceFieldZ(f.metadata, f.grid, f.metadata.float_type)
similar(f::EdgeField{T})  where {T} = EdgeField(f.metadata, f.grid, f.metadata.float_type)

set!(u::Field, v) = u.data .= convert(eltype(u.grid), v)
set!(u::Field, v::Field) = @. u.data = v.data

# set!(u::Field{G}, f::Function) where {G<:RegularCartesianGrid} = @. u.data = f(u.grid.xCA, u.grid.yCA, u.grid.zCA)

# Define +, -, and * on fields as element-wise calculations on their data. This
# is only true for fields of the same type, e.g. when adding a FaceFieldY to
# another FaceFieldY, otherwise some interpolation or averaging must be done so
# that the two fields are defined at the same point, so the operation which
# will not be commutative anymore.
for ft in (:CellField, :FaceFieldX, :FaceFieldY, :FaceFieldZ, :EdgeField)
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
