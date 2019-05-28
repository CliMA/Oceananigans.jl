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

"""
    FaceFieldX{A<:AbstractArray, G<:Grid} <: FaceField

An x-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldX{A<:AbstractArray, G<:Grid} <: FaceField
    data::A
    grid::G
end

"""
    FaceFieldY{A<:AbstractArray, G<:Grid} <: FaceField

A y-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldY{A<:AbstractArray, G<:Grid} <: FaceField
    data::A
    grid::G
end

"""
    FaceFieldZ{A<:AbstractArray, G<:Grid} <: Field

A z-face-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct FaceFieldZ{A<:AbstractArray, G<:Grid} <: FaceField
    data::A
    grid::G
end

"""
    EdgeField{A<:AbstractArray, G<:Grid} <: Field

An edge-centered field defined on a grid `G` whose values are stored in an `A`.
"""
struct EdgeField{A<:AbstractArray, G<:Grid} <: Field
    data::A
    grid::G
end

# Constructors

"""
    CellField([T=eltype(g)], arch, g)

Return a `CellField` with element type `T` on `arch` and grid `g`.
`T` defaults to the element type of `g`.
"""
CellField(T, arch, g) = CellField(zeros(T, arch, g), g)

"""
    FaceFieldX(T, arch, g)

Return a `FaceFieldX` with element type `T` on `arch` and grid `g`.
`T` defaults to the element type of `g`.
"""
FaceFieldX(T, arch, g) = FaceFieldX(zeros(T, arch, g), g)

"""
    FaceFieldY(T, arch, g)

Return a `FaceFieldY` with element type `T` on `arch` and grid `g`.
`T` defaults to the element type of `g`.
"""
FaceFieldY(T, arch, g) = FaceFieldY(zeros(T, arch, g), g)

"""
    FaceFieldZ(T, arch, g)

Return a `FaceFieldZ` with element type `T` on `arch` and grid `g`.
`T` defaults to the element type of `g`.
"""
FaceFieldZ(T, arch, g) = FaceFieldZ(zeros(T, arch, g), g)

"""
    EdgeField(T, arch, g)

Return a `EdgeField` with element type `T` on `arch` and grid `g`.
`T` defaults to the element type of `g`.
"""
 EdgeField(T, arch, g) =  EdgeField(zeros(T, arch, g), g)

 CellField(arch, g) =  CellField(zeros(arch, g), g)
FaceFieldX(arch, g) = FaceFieldX(zeros(arch, g), g)
FaceFieldY(arch, g) = FaceFieldY(zeros(arch, g), g)
FaceFieldZ(arch, g) = FaceFieldZ(zeros(arch, g), g)
 EdgeField(arch, g) =  EdgeField(zeros(arch, g), g)

@inline size(f::Field) = size(f.grid)
@inline length(f::Field) = length(f.data)

@inline getindex(f::Field, inds...) = getindex(f.data, inds...)
@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)
@inline setindex!(f::Field, v, inds...) = setindex!(f.data, v, inds...)

show(io::IO, f::Field) = show(io, f.data)

iterate(f::Field, state=1) = iterate(f.data, state)

set!(u::Field, v) = u.data .= convert(eltype(u.grid), v)
set!(u::Field, v::Field) = @. u.data = v.data

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
