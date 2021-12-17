using Oceananigans.Architectures
using Oceananigans.Architectures: device
import Oceananigans.Architectures: architecture
using CUDA, CUDA.CUSPARSE
using KernelAbstractions: @kernel, @index

using LinearAlgebra, SparseArrays, IncompleteLU
using IncompleteLU
using SparseArrays: fkeep!, nnz

import LinearAlgebra.ldiv!

# Utils for sparse matrix manipulation

@inline constructors(::CPU, A::SparseMatrixCSC) = (A.n, A.n, A.colptr, A.rowval, A.nzval)
@inline constructors(::GPU, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.n, A.n))
@inline constructors(::CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline constructors(::GPU, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)
@inline constructors(::CPU, n::Number, constr::Tuple) = (n, n, constr...)
@inline constructors(::GPU, n::Number, constr::Tuple) = (constr..., (n, n))

@inline unpack_constructors(::CPU, constr::Tuple) = (constr[3], constr[4], constr[5])
@inline unpack_constructors(::GPU, constr::Tuple) = (constr[1], constr[2], constr[3])
@inline copy_unpack_constructors(::CPU, constr::Tuple) = deepcopy((constr[3], constr[4], constr[5]))
@inline copy_unpack_constructors(::GPU, constr::Tuple) = deepcopy((constr[1], constr[2], constr[3]))

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

@kernel function _get_diag!(diag, colptr, rowval, nzval)
    col = @index(Global, Linear)
    map = 1
    for idx in colptr[col]:colptr[col+1] - 1
        if rowval[idx] == col
            map = idx 
            break
        end
    end
    diag[col] = nzval[map]
end

#unfortunately this cannot run on a GPU so we have to resort to that ugly loop in _update_diag!
@inline map_row_to_diag_element(i, rowval, colptr) =  colptr[i] - 1 + findfirst(rowval[colptr[i]:colptr[i+1]-1] .== i)

@inline function validate_laplacian_direction(N, topo, reduced_dim)  
    dim = N > 1 && reduced_dim == false
    if N < 3 && topo == Bounded && dim == true
        throw(ArgumentError("Cannot calculate laplacian in bounded domain with N < 3!"))
    end

    return dim
end

@inline validate_laplacian_size(N, dim) = dim == true ? N : 1
  
@inline ensure_diagonal_elements_are_present!(A) = fkeep!(A, (i, j, x) -> (i == j || !iszero(x)))

"""
`JacobiPreconditioner`
    stores only the diagonal `d` of D⁻¹ where `D = diag(A)`
    is applied to `r` with a vector multiplication `d .* r`

`ILUFactorization`
    stores two sparse lower and upper trianguilar matrices `L` and `U` such that `LU ≈ A`
    is applied to `r` with `forward_substitution!(L, r)` followed by `backward_substitution(U, r)`
    constructed with `ilu(A, τ = drop_tolerance)`
    
`SparseInversePreconditioner`
    stores a sparse matrix `M` such that `M ≈ A⁻¹` 
    is applied to `r` with a matrix multiplication `M * r`
    constructed with
    `simplified_inverse_preconditioner(A)`
        -> assumes that the sparsity of `M` is the same as the sparsity of `A`
    `sparse_approximate_preconditioner(A, ε = tolerance, nzrel = relative_maximum_number_of_elements)`
        -> starts constructing the sparse inverse of A from identity matrix until, either a tolerance (ε) is met or nnz(M) = nzrel * nnz(A) 

The suggested preconditioners are

on the `CPU`
`ilu()` (superior to everything always and in every situation!)

on the `GPU`
`sparse_inverse_preconditioner()` (if `Δt` is constant)
`simplified_inverse_preconditioner()` or `JacobiPreconditioner` (if Δt is variable)

`ilu()` cannot be used on the GPU because preconditioning the solver with a direct LU (or Choleski) type 
of preconditioner would require too much computation for the `ldiv!(P, r)` step completely hindering the performances

"""

@inline arch_preconditioner(::Val{false}, args...)                       = Identity()
@inline arch_preconditioner(::Val{true}, ::CPU, A, τ = 0.001)            = ilu(A, τ = τ) 
@inline arch_preconditioner(::Val{true}, ::GPU, A, ε = 0.1, nzrel = 2.0) = sparse_inverse_preconditioner(A, ε = ε, nzrel = nzrel) 

@inline architecture(::CuSparseMatrixCSC) = GPU()
@inline architecture(::SparseMatrixCSC)   = CPU()

struct JacobiPreconditioner{V}
    invdiag::V
end

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
    @inbounds u[i] = invdiag[i] * v[i]
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

struct SparseInversePreconditioner{M} <: AbstractInversePreconditioner{M}
    Minv :: M
end

@inline matrix(p::SparseInversePreconditioner)  = p.Minv

function simplified_inverse_preconditioner(A::AbstractMatrix)
    
    arch                  = architecture(A)
    constr                = deepcopy(constructors(arch, A)) 
    colptr, rowval, nzval = copy_unpack_constructors(arch, constr)
    dev                   = device(arch)
    
    M = size(A, 1)

    diag = arch_array(arch, zeros(eltype(nzval), M))

    loop! = _get_diag!(dev, 256, M)
    event = loop!(diag, colptr, rowval, nzval; dependencies=Event(dev))
    wait(dev, event)

    loop! = _initialize_simplified_inverse_preconditioner!(dev, 256, M)
    event = loop!(nzval, colptr, rowval, diag; dependencies=Event(dev))
    wait(dev, event)
    
    constr_new = (colptr, rowval, nzval)

    Minv = arch_sparse_matrix(arch, constructors(arch, M, constr_new))

    return SparseInversePreconditioner(Minv)
end

@kernel function _initialize_simplified_inverse_preconditioner!(nzval, colptr, rowval, diag)
    col = @index(Global, Linear)

    for idx = colptr[col] : colptr[col+1] - 1
        if rowval[idx] == col
            nzval[idx] = diag[col]
        else
            nzval[idx] = - nzval[idx] * diag[rowval[idx]]
        end
    end
end

function sparse_inverse_preconditioner(A::AbstractMatrix; ε, nzrel)

   # let's choose an initial sparsity => diagonal
   A_cpu    = arch_sparse_matrix(CPU(), A)
   Minv_cpu = spai_preconditioner(A_cpu, ε = ε, nzrel = nzrel)
   
   Minv = arch_sparse_matrix(architecture(A), Minv_cpu)
   return SparseInversePreconditioner(Minv)
end
