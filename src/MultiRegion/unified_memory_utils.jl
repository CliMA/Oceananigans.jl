using CUDA
using CUDA.CUSPARSE
using Oceananigans
using BenchmarkTools

function unified_array(arr) 
    buf = Mem.alloc(Mem.Unified, sizeof(arr))
    vec = unsafe_wrap(CuArray{eltype(arr),length(size(arr))}, convert(CuPtr{eltype(arr)}, buf), size(arr))
    finalizer(vec) do _
        Mem.free(buf)
    end
    copyto!(vec, arr)
    return vec
end

unified_array(int::Int) = int

using LinearAlgebra
using SparseArrays
import LinearAlgebra: mul!, ldiv!

# function mul!(r, A::NTuple{N}, b) where N
#     for i in 1:N
#         CUDA.device!(i-1)
#         mul!(r[i], A[i], b)
#     end
# end

function mul!(r, A::NTuple{N}, b) where {N}
    r = reshape(r, :, N)
    CUDA.@sync begin
        @sync begin
           Threads.@spawn begin
            CUDA.device!(0)
            # @show CUDA.stream(), CUDA.current_context()
            mul!(view(r, :, 1), A[1], b)
            # @show "finished mul, $(CUDA.current_context())"
            end
            Threads.@spawn begin
                CUDA.device!(1)
                # @show CUDA.stream(), CUDA.current_context()
                mul!(view(r, :, 2), A[2], b)
                # @show "finished mul, $(CUDA.current_context())"
            end
            Threads.@spawn begin
                CUDA.device!(2)
                # @show CUDA.stream(), CUDA.current_context()
                mul!(view(r, :, 3), A[3], b)
                # @show "finished mul, $(CUDA.current_context())"
            end
        end
    end
end

# function mul3!(r, A::NTuple{N}, b, ::Val{L}) where {N, L}
#     @inbounds for i in 1:N
#         CUDA.device!(i-1)
#         mul!(view(r, L*(i-1)+1:L*i), A[i], b)
#     end
# end

import Oceananigans.Solvers: constructors, arch_sparse_matrix
using Oceananigans.Utils

struct Unified end

@inline constructors(::Unified, A::SparseMatrixCSC)       = (unified_array(A.colptr), unified_array(A.rowval), unified_array(A.nzval),  (A.m, A.n))
@inline arch_sparse_matrix(::Unified, A::SparseMatrixCSC) = CuSparseMatrixCSC(constructors(Unified(), A)...)
@inline constructors(::GPU, A::SparseMatrixCSC)           = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))

using Oceananigans.Solvers: asymptotic_diagonal_inverse_preconditioner, SparseInversePreconditioner

### 3 Cases : CPU, GPU Device Buffers ()

## Sparse Matrix Test

n  = 180000

num_dev = 3
m = Int(n/num_dev)

# Solution
x  = zeros(n)
cx = CuArray(x)
ux  = unified_array(x)

# Matrix
A  = sprand(n, n, 0.000001)
A  = A + A' + 1I
cA = arch_sparse_matrix(GPU(), A)
mat = []
for i in 1:num_dev
    push!(mat, A[m*(i-1)+1:m*i, :])
end
uA = ()
for i in 1:num_dev
    CUDA.device!(i-1)
    uA = (uA..., arch_sparse_matrix(Unified(), mat[i]))
end
CUDA.device!(0)

# RHS
x_real  = ones(n)
b  = A * x_real
cb = CuArray(b)
ub = unified_array(b)

# Preconditioner
P  = asymptotic_diagonal_inverse_preconditioner(A; asymptotic_order = 1)
cP = asymptotic_diagonal_inverse_preconditioner(cA; asymptotic_order = 1)
pre = []
for i in 1:num_dev
    push!(pre, P.Minv[m*(i-1)+1:m*i, :])
end
uMinv = ()
for i in 1:num_dev
    CUDA.device!(i-1)
    uMinv = (uMinv..., arch_sparse_matrix(Unified(), pre[i]))
end
CUDA.device!(0)
uP = SparseInversePreconditioner(uMinv)



CUDA.device!(0)

using IterativeSolvers

using IterativeSolvers: done, start
import IterativeSolvers: iterate, cg!, cg_iterator!, start, converged, done

mutable struct UCGIterable{matT, solT, vecT, numT <: Real, L, N}
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    tol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int

    function UCGIterable{L, N}(A::matT, x::solT, r::vecT, c::vecT, u::vecT,
        tolerance::numT, residual::numT, prev_residual::numT,
        maxiter::Int, mv_products::Int) where {L, N, matT, solT, vecT, numT}
        return new{matT, solT, vecT, numT, L, N}(A, x, r, c, u,
                                                 tolerance, residual, prev_residual,
                                                 maxiter, mv_products)
    end
end

mutable struct UPCGIterable{precT, matT, solT, vecT, numT <: Real, paramT <: Number, L, N}
    Pl::precT
    A::matT
    x::solT
    r::vecT
    c::vecT
    u::vecT
    tol::numT
    residual::numT    
    ρ::paramT
    maxiter::Int
    mv_products::Int

    function UPCGIterable{L, N}(Pl::precT, A::matT, x::solT, r::vecT, c::vecT, u::vecT,
        tolerance::numT, residual::numT, ρ::paramT,
        maxiter::Int, mv_products::Int) where {L, N, precT, matT, solT, vecT, numT, paramT}

        return new{precT, matT, solT, vecT, numT, paramT, L, N}(Pl, A, x, r, c, u,
                                                 tolerance, residual, ρ,
                                                 maxiter, mv_products)
    end
end


@inline converged(it::Union{UPCGIterable, UCGIterable}) = it.residual ≤ it.tol
@inline start(it::Union{UPCGIterable, UCGIterable}) = 0
@inline done(it::Union{UPCGIterable, UCGIterable}, iteration::Int) = iteration ≥ it.maxiter || converged(it)

# function cg_iterator!(x, A::NTuple{N}, b, ::Val{L}, Pl = Identity();
#                       abstol::Real = zero(real(eltype(b))),
#                       reltol::Real = sqrt(eps(real(eltype(b)))),
#                       maxiter::Int = size(A, 2),
#                       statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
#                       initially_zero::Bool = false) where {N, L}
                        
#     u = statevars.u
#     r = statevars.r
#     c = statevars.c
#     fill!(u, zero(eltype(x)))
#     copyto!(r, b)

#     # Compute r with an MV-product or not.
#     if initially_zero
#         mv_products = 0
#     else
#         mv_products = 1
#         mul!(reshape(c, L, N), A, x)
#         r .-= c
#     end
#     residual = norm(r)
#     tolerance = max(reltol * residual, abstol)

#     # Return the iterable
#     if isa(Pl, Identity)
#         return UCGIterable{L, N}(A, x, r, c, u,
#                                  tolerance, residual, one(residual),
#                                  maxiter, mv_products)
#     else
#         return UPCGIterable{L, N}(Pl, A, x, r, c, u,
#                                   tolerance, residual, one(eltype(x)),
#                                   maxiter, mv_products)
#     end
# end

using Printf

function cg!(x, A::NTuple{N}, b;
             abstol::Real = zero(real(eltype(b))),
             reltol::Real = sqrt(eps(real(eltype(b)))),
             maxiter::Int = size(A[1], 2),
             log::Bool = false,
             statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
             verbose::Bool = false,
             Pl = Identity(),
             kwargs...) where N

    L = Int(length(x) / N)
    # Actually perform CG
    iterable = cg_iterator!(x, A, b, Pl; abstol = abstol, reltol = reltol, maxiter = maxiter,
                            statevars = statevars, kwargs...)
    
    for (iteration, item) = enumerate(iterable)
        verbose && @printf("%3d\t%1.2e\n", iteration, iterable.residual)
    end
    
    iterable.x
end

function iterate(it::UPCGIterable{P, A, B, C, D, R, L, N}, iteration::Int=start(it)) where {P, A, B, C, D, R, L, N}
    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # Apply left preconditioner
    CUDA.@sync ldiv!(reshape(it.c, L, N), it.Pl, it.r)
    
    ρ_prev = it.ρ
    it.ρ = multi_dot(it.c, it.r)

    # u := c + βu (almost an axpy)
    β = it.ρ / ρ_prev
    it.u .= it.c .+ β .* it.u

    # c = A * u
    CUDA.@sync mul!(reshape(it.c, L, N), it.A, it.u)

    α = it.ρ / multi_dot(it.u, it.c)

    # Improve solution and residual
    it.x .+= α .* it.u
    it.r .-= α .* it.c

    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

multi_dot(a, b) = dot(a, b)

function iterate(it::UCGIterable{A, B, C, D, L, N}, iteration::Int=start(it)) where {A, B, C, D, L, N}

    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # u := r + βu (almost an axpy)
    β = it.residual^2 / it.prev_residual^2
    it.u .= it.r .+ β .* it.u

    # c = A * u
    mul!(reshape(it.c, L, N), it.A, it.u)
    for i in 1:N
        CUDA.device!(i-1)
        synchronize()
    end

    α = it.residual^2 / dot(it.u, it.c)

    # Improve solution and residual
    it.x .+= α .* it.u
    it.r .-= α .* it.c

    it.prev_residual = it.residual
    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end



trial = @benchmark begin
    CUDA.@sync blocking = true cg!(x, A, b, Pl = P, maxiter=10)
end samples = 10

CUDA.device!(0)

trial = @benchmark begin
    CUDA.@sync blocking = true cg!(cx, cA, cb, Pl = cP, maxiter=10)
end samples = 10

trial = @benchmark begin
    CUDA.@sync blocking = true cg!(ux, uA, ub, Pl = uP, maxiter=10)
end samples = 10