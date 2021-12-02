using CUDA, CUDA.CUSPARSE
using LinearAlgebra, SparseArrays
using IncompleteLU: ILUFactorization

import SparseArrays: nnz
import LinearAlgebra: ldiv!
import Base.\

import IncompleteLU.forward_substitution!
import IncompleteLU.backward_substitution!
import IncompleteLU.ilu

using Adapt

# Utils for sparse matrix manipulation

@inline constructors(::CPU, A::SparseMatrixCSC) = (A.n, A.n, A.colptr, A.rowval, A.nzval)
@inline constructors(::GPU, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.n, A.n))
@inline constructors(::CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Array(A.colPtr), Array(A.rowVal), Array(A.nzVal))
@inline constructors(::GPU, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)
@inline constructors(::CPU, n::Number, constr::Tuple) = (n, n, constr...)
@inline constructors(::GPU, n::Number, constr::Tuple) = (constr..., (n, n))

@inline unpack_constructors(::CPU, constr::Tuple) = (constr[3], constr[4], constr[5])
@inline unpack_constructors(::GPU, constr::Tuple) = (constr[1], constr[2], constr[3])

@inline arch_sparse_matrix(::CPU, constr::Tuple) = SparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::GPU, constr::Tuple) = CuSparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::CPU, A::CuSparseMatrixCSC) = SparseMatrixCSC(constructors(CPU(), A)...)
@inline arch_sparse_matrix(::GPU, A::SparseMatrixCSC)   = CuSparseMatrixCSC(constructors(GPU(), A)...)

"""
Extending the ILUFactorization methods to a ILUFactorizationGPU type which lives on the GPU
This allow us to create a LU preconditioner on the GPU and use it whitin the IterativeSolvers
directly on the GPU

Next step should be to make an efficient parallel backward - forward substitution on the GPU
At the moment it is done with threads == blocks == 1. 

This is extremely unefficient!

"""

struct ILUFactorizationGPU{Tv}
    L::CuSparseMatrixCSC{Tv}
    U::CuSparseMatrixCSC{Tv}
end

@inline ILUFactorizationGPU(p::ILUFactorization) = ILUFactorizationGPU(arch_sparse_matrix(GPU(), p.L), arch_sparse_matrix(GPU(), p.U))

nnz(F::ILUFactorizationGPU) = nnz(F.L) + nnz(F.U)

function ldiv!(F::ILUFactorizationGPU, y::AbstractVector)
    forward_substitution!(F, y)
    backward_substitution!(F, y)
end

function ldiv!(y::AbstractVector, F::ILUFactorizationGPU, x::AbstractVector)
    y .= x
    ldiv!(F, y)
end

(\)(F::ILUFactorizationGPU, y::AbstractVector) = ldiv!(F, copy(y))

function backward_substitution!(v::AbstractVector, F::ILUFactorizationGPU, y::AbstractVector)
    v .= y
    backward_substitution!(F, v)
end

function backward_substitution!(F::ILUFactorizationGPU, y::AbstractVector)
    U = F.U
    # for the moment just one thread does the substitution, maybe we wanna parallelize this
    @cuda threads=1 _backward_substitution!(U.dims[1], U.colPtr, U.rowVal, U.nzVal, y)   
    y
end

## Parallel version of the algorithm

##   DO  PARALLEL I = 1, N
##   X(I) = Y(I)
##   END DO
##   DO J = N, 1, -1
##   X(J) = X(J) / U(J,J)
##   DO PARALLEL I = 1, J-1
##      X(I) = X(I) - U(I,J)*X(J)
##   END DO
##   END DO

## Which in Julia would be

## x .= y
## for j = n:-1:1
## x[j] = x[j] / nzval(diag)
## parallelly: i = 1:j-1
##   x[i] = x[i] - nzval(i,j) * x(j)
## end
## end

function _backward_substitution!(n, colptr, rowval, nzval, y)
    @inbounds begin
        for col = n : -1 : 1
            for idx = colptr[col + 1] - 1 : -1 : colptr[col] + 1
                y[col] -= nzval[idx] * y[rowval[idx]]
            end
            y[col] /= nzval[colptr[col]]
        end
    end
end

function forward_substitution!(F::ILUFactorizationGPU, y::AbstractVector)
    L = F.L
    # for the moment just one thread does the substitution, maybe we wanna parallelize this
    @cuda threads=1 _forward_substitution!(L.dims[1], L.colPtr, L.rowVal, L.nzVal, y)  
    y
end

function _forward_substitution!(n, colptr, rowval, nzval, y)
    @inbounds begin
    for col = 1 : n - 1
            for idx = colptr[col] : colptr[col + 1] - 1
                y[rowval[idx]] -= nzval[idx] * y[col]
            end
        end
    end
end

function ilu(dA::CuSparseMatrix{Tv}; τ = 1e-3) where {Tv}
    A = arch_sparse_matrix(CPU(), dA)
    p = ilu(A, τ = τ)

    return ILUFactorizationGPU(p)
end


