import Krylov: kaxpby!, kaxpby!, kdot, knrm2

using KernelAbstractions: @index, @kernel

kdot(n::Integer, x::Field, dx::Integer, y::Field, dy::Integer) = dot(x, y)
knrm2(n::Integer, x::Field, dx::Integer) = norm(x)
kcopy!(n::Integer, x::Field, dx::Integer, y::Field, dy :: Integer) = copyto!(y, x)

function kaxpy!(n::Integer, s::T, a::Field, dx::Integer, y::Field, dy::Integer) where T<:AbstractFloat
    grid = a.grid
    arch = architecture(grid)
    launch!(arch, grid, size(a), _axpy!, y, s, x)
    return nothing
end

@kernel function _axpy!(y, s, x)
    i, j, k = @index(Global, NTuple)
    @inbounds y[i, j, k] += s * x[i, j, k]
end

function kaxpby!(n::Integer, s::T, a::Field, dx::Integer, t::T, y::Field, dy::Integer) where T<:AbstractFloat
    grid = a.grid
    arch = architecture(grid)
    launch!(arch, grid, size(a), _axpy!, y, t, s, x)
    return nothing
end

@kernel function _axpby!(y, t, s, x)
    i, j, k = @index(Global, NTuple)
    @inbounds y[i, j, k] = s * x[i, j, k] + t * y[i, j, k]
end

