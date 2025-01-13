import Krylov
import Krylov.FloatOrComplex

using KernelAbstractions: @index, @kernel

struct KrylovField{T, F <: AbstractField} <: AbstractVector{T}
    field::F
end

function KrylovField(field::F) where F <: AbstractField
    T = eltype(field)
    return KrylovField{T,F}(field)
end

Base.size(kf::KrylovField) = size(kf.field)
Base.length(kf::KrylovField) = length(kf.field)
Base.getindex(kf::KrylovField, i::Int) = getindex(kf.field, i)

function Krylov.kdot(n::Int, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    _y = y.field
    res = zero(T)
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                res += _x[i,j,k] * _y[i,j,k]
            end
        end
    end
    return res
end

function Krylov.knorm(n::Int, x::KrylovField{T}) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    res = zero(T)
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                res += _x[i,j,k]^2
            end
        end
    end
    return sqrt(res)
end

function Krylov.kscal!(n::Int, s::T, x::KrylovField{T}) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                _x[i,j,k] = s * _x[i,j,k]
            end
        end
    end
    return x
end

function Krylov.kaxpy!(n::Int, s::T, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    _y = y.field
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                _y[i,j,k] += s * _x[i,j,k]
            end
        end
    end
    return y
end

function Krylov.kaxpby!(n::Int, s::T, x::KrylovField{T}, t::T, y::KrylovField{T}) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    _y = y.field
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                _y[i,j,k] = s * _x[i,j,k] + t * _y[i,j,k]
            end
        end
    end
    return y
end

function Krylov.kcopy!(n::Int, y::KrylovField{T}, x::KrylovField{T}) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    _y = y.field
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                _y[i,j,k] = _x[i,j,k]
            end
        end
    end
    return y
end

function Krylov.kfill!(x::KrylovField{T}, val::T) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                _x[i,j,k] = val
            end
        end
    end
    return x
end

function Krylov.kref!(n::Integer, x::KrylovField{T}, y::KrylovField{T}, c::T, s::T) where T <: FloatOrComplex
    mx, nx, kx = size(x.field)
    _x = x.field
    _y = y.field
    for i = 1:mx
        for j = 1:nx
            for k = 1:kx
                x_ijk = _x[i,j,k]
                y_ijk = _y[i,j,k]
                _x[i,j,k] = c       * x_ijk + s * y_ijk
                _x[i,j,k] = conj(s) * x_ijk - c * y_ijk
            end
        end
    end
    return x, y
end

# function Krylov.knorm(kf::KrylovField)
#     conditional_kf = condition_operand(kf.field, condition, 0)
#     result = zeros(kf.field.grid, 1)
#     Base.mapreducedim!(x -> x * x, +, result, conditional_kf)
#     return CUDA.@allowscalar sqrt(first(result))
# end

# function Krylov.kdot(kf_a::KrylovField, kf_b::KrylovField)
#     conditional_kf_a = condition_operand(kf_a.field, condition, 0)
#     conditional_kf_b = condition_operand(kf_b.field, condition, 0)
#     result = zeros(kf_a.field.grid, 1)
#     Base.mapreducedim!((x, y) -> x * y, +, result, conditional_kf_a, conditional_kf_b)
#     return CUDA.@allowscalar first(result)
# end

# kdot(n::Integer, x::Field, dx::Integer, y::Field, dy::Integer) = dot(x, y)
# knrm2(n::Integer, x::Field, dx::Integer) = norm(x)
# kcopy!(n::Integer, x::Field, dx::Integer, y::Field, dy :: Integer) =

# function kaxpy!(n::Integer, s::F, a::Field, dx::Integer, y::Field, dy::Integer) where F<:AbstractFloat
#     grid = a.grid
#     arch = architecture(grid)
#     launch!(arch, grid, size(a), _axpy!, y, s, x)
#     return nothing
# end

# @kernel function _axpy!(y, s, x)
#     i, j, k = @index(Global, NTuple)
#     @inbounds y[i, j, k] += s * x[i, j, k]
# end

# function kaxpby!(n::Integer, s::F, a::Field, dx::Integer, t::F, y::Field, dy::Integer) where F<:AbstractFloat
#     grid = a.grid
#     arch = architecture(grid)
#     launch!(arch, grid, size(a), _axpy!, y, t, s, x)
#     return nothing
# end

# @kernel function _axpby!(y, t, s, x)
#     i, j, k = @index(Global, NTuple)
#     @inbounds y[i, j, k] = s * x[i, j, k] + t * y[i, j, k]
# end
