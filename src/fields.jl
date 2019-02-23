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
    metadata::ModelMetadata
    grid::Grid
    data::T
end

"""
    FaceFieldX{T,G<:Grid{T}} <: FaceField{G}

An x-face-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct FaceFieldX{T<:AbstractArray} <: FaceField
    metadata::ModelMetadata
    grid::Grid
    data::T
end

"""
    FaceFieldY{T,G<:Grid{T}} <: FaceField{G}

A y-face-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct FaceFieldY{T<:AbstractArray} <: FaceField
    metadata::ModelMetadata
    grid::Grid
    data::T
end

"""
    FaceFieldZ{T,G<:Grid{T}} <: FaceField{G}

A z-face-centered field defined on a grid `G` whose values are stored as
floating-point values of type T.
"""
struct FaceFieldZ{T<:AbstractArray} <: FaceField
    metadata::ModelMetadata
    grid::Grid
    data::T
end

"""
    EdgeField{T<:AbstractArray} <: Field

A field defined on a grid `G` whose values lie on the edges of the cells.
"""
struct EdgeField{T<:AbstractArray} <: Field
    metadata::ModelMetadata
    grid::Grid
    data::T
end

"""
    CellField(metadata::ModelMetadata, grid::Grid, T)

Construct a `CellField` whose values are defined at the center of a cell.
"""
function CellField(metadata::ModelMetadata, grid::Grid, T)
    if metadata.arch == :cpu
        data = zeros(T, size(grid))
        return CellField{Array{T,3}}(metadata, grid, data)
    elseif metadata.arch == :gpu
        # data = cu(zeros(T, size(grid)))
        data = CuArray{T}(undef, grid.Nx, grid.Ny, grid.Nz)
        data .= 0.0
        return CellField{CuArray{T,3}}(metadata, grid, data)
    end
end

CellField(mm::ModelMetadata, g::Grid) = CellField(mm, g, eltype(g))

"""
    FaceFieldX(metadata::ModelMetadata, grid::Grid, T)

A `Field` whose values are defined on the x-face of a cell.
"""
function FaceFieldX(metadata::ModelMetadata, grid::Grid, T)
    if metadata.arch == :cpu
        data = zeros(eltype(grid), size(grid))
        return FaceFieldX{Array{eltype(grid),3}}(metadata, grid, data)
    elseif metadata.arch == :gpu
        data = CuArray{T}(undef, grid.Nx, grid.Ny, grid.Nz)
        data .= 0.0
        return FaceFieldX{CuArray{T,3}}(metadata, grid, data)
    end
end

"""
    FaceFieldY(metadata::ModelMetadata, grid::Grid, T)

A `Field` whose values are defined on the y-face of a cell.
"""
function FaceFieldY(metadata::ModelMetadata, grid::Grid, T)
    if metadata.arch == :cpu
        data = zeros(eltype(grid), size(grid))
        return FaceFieldY{Array{eltype(grid),3}}(metadata, grid, data)
    elseif metadata.arch == :gpu
        data = CuArray{T}(undef, grid.Nx, grid.Ny, grid.Nz)
        data .= 0.0
        return FaceFieldY{CuArray{T,3}}(metadata, grid, data)
    end
end

"""
    FaceFieldZ(metadata::ModelMetadata, grid::Grid, T)

A `Field` whose values are defined on the z-face of a cell.
"""
function FaceFieldZ(metadata::ModelMetadata, grid::Grid, T)
    if metadata.arch == :cpu
        data = zeros(eltype(grid), size(grid))
        return FaceFieldZ{Array{eltype(grid),3}}(metadata, grid, data)
    elseif metadata.arch == :gpu
        data = CuArray{T}(undef, grid.Nx, grid.Ny, grid.Nz)
        data .= 0.0
        return FaceFieldZ{CuArray{T,3}}(metadata, grid, data)
    end
end

"""
    FEdgeField(metadata::ModelMetadata, grid::Grid, T)

A `Field` whose values are defined on the edges of a cell.
"""
function EdgeField(metadata::ModelMetadata, grid::Grid, T)
    if metadata.arch == :cpu
        data = zeros(eltype(grid), size(grid))
        return EdgeField{Array{eltype(grid),3}}(metadata, grid, data)
    elseif metadata.arch == :gpu
        data = CuArray{T}(undef, grid.Nx, grid.Ny, grid.Nz)
        data .= 0.0
        return EdgeField{CuArray{T,3}}(metadata, grid, data)
    end
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

set!(u::Field, v) = @. u.data = v
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
