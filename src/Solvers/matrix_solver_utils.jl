using Oceananigans.Architectures
using Oceananigans.Architectures: device
import Oceananigans.Architectures: architecture
using CUDA, CUDA.CUSPARSE
using KernelAbstractions: @kernel, @index

using IterativeSolvers
using LinearAlgebra, SparseArrays, IncompleteLU
using IncompleteLU
using SparseArrays: fkeep!, nnz

import LinearAlgebra.ldiv!
import Base: size

# Utils for sparse matrix manipulation

@inline constructors(::CPU, A::SparseMatrixCSC) = (A.n, A.n, A.colptr, A.rowval, A.nzval)
@inline constructors(::GPU, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.n, A.n))
@inline constructors(::CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Array(A.colPtr), Array(A.rowVal), Array(A.nzVal))
@inline constructors(::GPU, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)
@inline constructors(::CPU, n::Number, constr::Tuple) = (n, n, constr...)
@inline constructors(::GPU, n::Number, constr::Tuple) = (constr..., (n, n))

@inline unpack_constructors(::CPU, constr::Tuple) = (constr[3], constr[4], constr[5])
@inline unpack_constructors(::GPU, constr::Tuple) = (constr[1], constr[2], constr[3])
@inline copy_unpack_constructors(::CPU, constr::Tuple) = deepcopy((constr[3], constr[4], constr[5]))
@inline copy_unpack_constructors(::GPU, constr::Tuple) = deepcopy((constr[1], constr[2], constr[3]))
@inline size(::CPU, constr::Tuple) = constr[1]
@inline size(::GPU, constr::Tuple) = constr[4][1]

@inline arch_sparse_matrix(::CPU, constr::Tuple) = SparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::GPU, constr::Tuple) = CuSparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::CPU, A::CuSparseMatrixCSC) = SparseMatrixCSC(constructors(CPU(), A)...)
@inline arch_sparse_matrix(::GPU, A::SparseMatrixCSC)   = CuSparseMatrixCSC(constructors(GPU(), A)...)
@inline arch_sparse_matrix(::CPU, A::SparseMatrixCSC)   = A
@inline arch_sparse_matrix(::GPU, A::CuSparseMatrixCSC) = A


# We need to update the diagonal element each time the time step changes!!
function update_diag!(constr, arch, problem_size, diag, Δt)   
    M = prod(problem_size)
    colptr, rowval, nzval = unpack_constructors(arch, constr)
   
    loop! = _update_diag!(device(arch), 256, M)
    event = loop!(nzval, colptr, rowval, diag, Δt; dependencies=Event(device(arch)))
    wait(event)

    constr = constructors(arch, M, (colptr, rowval, nzval))
end

@kernel function _update_diag!(nzval, colptr, rowval, diag, Δt)
    col = @index(Global, Linear)
    map = 1
    for idx in colptr[col]:colptr[col+1] - 1
       if rowval[idx] == col
           map = idx 
            break
        end
    end
    nzval[map] += diag[col] / Δt^2 
end

@kernel function _get_inv_diag!(invdiag, colptr, rowval, nzval)
    col = @index(Global, Linear)
    map = 1
    for idx in colptr[col]:colptr[col+1] - 1
        if rowval[idx] == col
            map = idx 
            break
        end
    end
    if nzval[map] == 0.0 
        invdiag[col] = 0 
    else
        invdiag[col] = 1.0 / nzval[map]
    end
end

#unfortunately this cannot run on a GPU so we have to resort to that ugly loop in _update_diag!
@inline map_row_to_diag_element(i, rowval, colptr) =  colptr[i] - 1 + findfirst(rowval[colptr[i]:colptr[i+1]-1] .== i)

@inline function validate_laplacian_direction(N, topo, reduced_dim)  
    dim = N > 1 && reduced_dim == false
    if N < 3 && topo == Bounded && dim == true
        throw(ArgumentError("cannot calculate laplacian in bounded domain with N < 3"))
    end

    return dim
end

@inline validate_laplacian_size(N, dim) = dim == true ? N : 1
  
@inline ensure_diagonal_elements_are_present!(A) = fkeep!(A, (i, j, x) -> (i == j || !iszero(x)))

"""
No matter what I try, to precondition the solver on the GPU() with a direct LU (or Choleski) type 
of preconditioner would require a 

This is extremely unefficient!

choices of preconditioners on the CPU
Identity() (no preconditioner)
ilu()
sparse_inverse_preconditioner()

on the GPU
Identity() (no preconditioner)
sparse_inverse_preconditioner()

"""

@inline arch_preconditioner(::Val{false}, args...)      = Identity()
@inline arch_preconditioner(::Val{true}, ::CPU, matrix) = sparse_inverse_preconditioner(matrix, ε=0.1) # mit_gcm_preconditioner(matrix) # #ilu(matrix, τ=0.001) 
@inline arch_preconditioner(::Val{true}, ::GPU, matrix) = sparse_inverse_preconditioner(matrix, ε=0.1) # Identity()     

@inline architecture(::CuSparseMatrixCSC) = GPU()
@inline architecture(::SparseMatrixCSC)   = CPU()

struct JacobiPreconditioner{V}
    invdiag::V
end

# Constructor:
function JacobiPreconditioner(A::AbstractMatrix)
    M    = size(A, 1)
    arch = architecture(A)
    dev  = device(arch)
    
    invdiag = arch_array(arch, zeros(eltype(A), M))
    col, row, val = unpack_constructors(arch, constructors(arch, A))

    loop! = _get_inv_diag!(dev, 256, M)
    event = loop!(invdiag, col, row, val; dependencies=Event(dev))
    wait(dev, event)

    return JacobiPreconditioner(invdiag)
end

@kernel function _multiply_in_place!(u, invdiag, v)
    i = @index(Global, Linear)
    @inbounds begin
        u[i] = invdiag[i] * v[i]
    end
end  

function  LinearAlgebra.ldiv!(u, precon::JacobiPreconditioner, v)
    invdiag = precon.invdiag
    arch = architecture(invdiag)
    dev  = device(arch)
    
    M = length(invdiag)
    loop! = _multiply_in_place!(dev, 256, M)
    event = loop!(u, invdiag, v; dependencies=Event(dev))
    wait(dev, event)
end

function  LinearAlgebra.ldiv!(precon::JacobiPreconditioner, v)
    ldiv!(v,precon,v)
end

abstract type AbstractInversePreconditioner{M} end

function  LinearAlgebra.ldiv!(u, precon::AbstractInversePreconditioner, v)
    mul!(u, matrix(precon), v)
end

function  LinearAlgebra.ldiv!(precon::AbstractInversePreconditioner, v)
    mul!(v, matrix(precon), v)
end

struct MITGCMPreconditioner{M} <: AbstractInversePreconditioner{M}
    Minv :: M
end

function mit_gcm_preconditioner(A::AbstractMatrix)
    
    arch                  = architecture(A)
    constr                = deepcopy(constructors(arch, A)) 
    colptr, rowval, nzval = copy_unpack_constructors(arch, constr)
    dev                   = device(arch)
    
    M = size(arch, constr)

    invdiag = arch_array(arch, zeros(eltype(nzval), M))

    loop! = _get_inv_diag!(dev, 256, M)
    event = loop!(invdiag, colptr, rowval, nzval; dependencies=Event(dev))
    wait(dev, event)

    loop! = _initialize_MIT_preconditioner!(dev, 256, M)
    event = loop!(nzval, colptr, rowval, invdiag; dependencies=Event(dev))
    wait(dev, event)
    
    constr_new = (colptr, rowval, nzval)

    Minv = arch_sparse_matrix(arch, constructors(arch, M, constr_new))

    return MITGCMPreconditioner(Minv)
end

@kernel function _initialize_MIT_preconditioner!(nzval, colptr, rowval, invdiag)
    col = @index(Global, Linear)

    for idx = colptr[col] : colptr[col+1] - 1
        if rowval[idx] == col
            nzval[idx] = - invdiag[col]
        else
            nzval[idx] = nzval[idx] * invdiag[rowval[idx]]
        end
    end
end

struct SparseInversePreconditioner{M} <: AbstractInversePreconditioner{M}
    Minv :: M
end

@inline matrix(p::MITGCMPreconditioner)         = p.Minv
@inline matrix(p::SparseInversePreconditioner)  = p.Minv

function sparse_inverse_preconditioner(A::AbstractMatrix; ε)

   # let's choose an initial sparsity => diagonal
   A_cpu    = arch_sparse_matrix(CPU(), A)
   Minv_cpu = spai_preconditioner(A_cpu, ε = ε)
   
   Minv = arch_sparse_matrix(architecture(A), Minv_cpu)
   return SparseInversePreconditioner(Minv)
end
