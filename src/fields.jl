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

"""
    EdgeField([T=eltype(grid)], arch, grid)

Return an `EdgeField` with element type `T` on `arch` and `grid`.
`T` defaults to the element type of `grid`.
"""
 EdgeField(T, arch, grid) =  EdgeField(zeros(T, arch, grid), grid)

 CellField(arch, grid) =  CellField(zeros(arch, grid), grid)
FaceFieldX(arch, grid) = FaceFieldX(zeros(arch, grid), grid)
FaceFieldY(arch, grid) = FaceFieldY(zeros(arch, grid), grid)
FaceFieldZ(arch, grid) = FaceFieldZ(zeros(arch, grid), grid)
 EdgeField(arch, grid) =  EdgeField(zeros(arch, grid), grid)

@inline size(f::Field) = size(f.grid)
@inline length(f::Field) = length(f.data)

@inline getindex(f::Field, inds...) = getindex(f.data, inds...)
@inline lastindex(f::Field) = lastindex(f.data)
@inline lastindex(f::Field, dim) = lastindex(f.data, dim)
@inline setindex!(f::Field, v, inds...) = setindex!(f.data, v, inds...)

@inline data(f::Field) = view(f.data, 1:f.grid.Nx, 1:f.grid.Ny, 1:f.grid.Nz)

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
