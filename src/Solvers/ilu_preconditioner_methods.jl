using CUDA, CUDA.CUSPARSE
using LinearAlgebra, SparseArrays
using IncompleteLU
using IncompleteLU: ILUFactorization
import LinearAlgebra: ldiv!

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
This allow us to create an incomplete LU preconditioner on the GPU and use it within the IterativeSolvers
directly on the GPU

Next step should be to make an efficient parallel backward - forward substitution on the GPU
At the moment it is done with threads == blocks == 1. 

This is extremely unefficient!
"""

@inline arch_preconditioner(::Val{false}, args...) = Identity()
@inline arch_preconditioner(::Val{true}, ::CPU, matrix) =  ilu(matrix, τ = 0.01)
@inline arch_preconditioner(::Val{true}, ::GPU, matrix) =  Identity()
# @inline arch_preconditioner(::Val{true}, ::GPU, matrix) =  CudaILUFactorization(ilu(arch_sparse_matrix(CPU(), matrix), τ = 1.0))

@inline def_τ(::CPU) = 0.01
@inline def_τ(::GPU) = 1.0

struct CudaILUFactorization{TL, TU}
    L::TL
    U::TU
end

function CudaILUFactorization(f::ILUFactorization)
    L = LowerTriangular(arch_sparse_matrix(GPU(), f.L+I))
    U = UpperTriangular(arch_sparse_matrix(GPU(), sparse(transpose(f.U))))

    return CudaILUFactorization(L, U)
end

LinearAlgebra.ldiv!(f::CudaILUFactorization, x) = ldiv!(f.U, ldiv!(f.L, x))

function LinearAlgebra.ldiv!(y, f::CudaILUFactorization, x) 
    copyto!(y, x)
    ldiv!(f.U, ldiv!(f.L, y))
    return y
end
