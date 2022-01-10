using Oceananigans.Architectures
using Oceananigans.Architectures: device
import Oceananigans.Architectures: architecture
using CUDA, CUDA.CUSPARSE
using KernelAbstractions: @kernel, @index

using LinearAlgebra, SparseArrays, IncompleteLU
using SparseArrays: nnz

import LinearAlgebra.ldiv!

"""
`ILUFactorization`
    stores two sparse lower and upper trianguilar matrices `L` and `U` such that `LU ≈ A`
    is applied to `r` with `forward_substitution!(L, r)` followed by `backward_substitution!(U, r)`
    constructed with `ilu(A, τ = drop_tolerance)`
    
`SparseInversePreconditioner`
    stores a sparse matrix `M` such that `M ≈ A⁻¹` 
    is applied to `r` with a matrix multiplication `M * r`
    constructed with
    `heuristic_inverse_preconditioner(A)`
        -> same formulation as Marshall J. et al., "Finite-volume, incompressible Navier Stokes model for studies of the ocean on parallel computers"
        -> assumes that the sparsity of `M` is the same as the sparsity of `A`, no additional settings needed
    `sparse_approximate_preconditioner(A, ε = tolerance, nzrel = relative_maximum_number_of_elements)`
        -> same formulation as Grote M. J. & Huckle T, "Parallel Preconditioning with sparse approximate inverses" 
        -> starts constructing the sparse inverse of A from identity matrix until, either a tolerance (ε) is met or nnz(M) = nzrel * nnz(A) 

The suggested preconditioners are

on the `CPU`
`ilu()` (superior to everything always and in every situation!)

on the `GPU`
`heuristic_inverse_preconditioner()` (if `Δt` is variable or large problem_sizes)
`sparse_inverse_preconditioner()` (if `Δt` is constant and problem_size is not too large)

as a rule of thumb, for poisson solvers:
`sparse_inverse_preconditioner` is better performing than `heuristic_inverse_preconditioner` only if `nzrel >= 2.0`
As such, we urge to use `sparse_inverse_preconditioner` only when
- Δt is constant (we don't have to recalculate the preconditioner during the simulation)
- it is feasible to choose `nzrel = 2.0` (for not too large problem sizes)

`ilu()` cannot be used on the GPU because preconditioning the solver with a direct LU (or Choleski) type 
of preconditioner would require too much computation for the `ldiv!(P, r)` step completely hindering the performances
"""

validate_settings(T, arch, settings)                                 = settings
validate_settings(::Val{:Default}, arch, settings)                   = arch isa CPU ? (τ = 0.001, ) : nothing 
validate_settings(::Val{:SparseInverse}, arch, settings::Nothing)    = (ε = 0.2, nzrel = 2.0)
validate_settings(::Val{:ILUFactorization}, arch, settings::Nothing) = (τ = 0.001, ) 

validate_settings(::Val{:ILUFactorization}, arch, settings) = haskey(settings, :τ) ? 
                                                                     settings :
                                                                     throw(ArgumentError("τ has to be specified for ILUFactorization"))
validate_settings(::Val{:SparseInverse}, arch, settings)    = haskey(settings, :ε) && haskey(settings, :nzrel) ?
                                                                     settings :
                                                                     throw(ArgumentError("both ε and nzrel have to be specified for SparseInverse"))

function build_preconditioner(::Val{:Default}, matrix, settings)
    default_method = architecture(matrix) isa CPU ? :ILUFactorization : :HeuristicInverse
    return build_preconditioner(Val(default_method), matrix, settings)
end

build_preconditioner(::Val{nothing},            A, settings) = Identity()
build_preconditioner(::Val{:SparseInverse},     A, settings) = sparse_inverse_preconditioner(A, ε = settings.ε, nzrel = settings.nzrel)
build_preconditioner(::Val{:HeuristicInverse}, A, settings)  = heuristic_inverse_preconditioner(A)

function build_preconditioner(::Val{:ILUFactorization},  A, settings) 
    if architecture(A) isa GPU 
        throw(ArgumentError("the ILU factorization is not available on the GPU! choose another method"))
    else
        return ilu(A, τ = settings.τ)
    end
end

@inline architecture(::CuSparseMatrixCSC) = GPU()
@inline architecture(::SparseMatrixCSC)   = CPU()

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

function heuristic_inverse_preconditioner(A::AbstractMatrix)
    
    arch                  = architecture(A)
    constr                = deepcopy(constructors(arch, A)) 
    colptr, rowval, nzval = copy_unpack_constructors(arch, constr)
    dev                   = device(arch)
    
    M = size(A, 1)

    invdiag = arch_array(arch, zeros(eltype(nzval), M))

    loop! = _get_inv_diag!(dev, 256, M)
    event = loop!(invdiag, colptr, rowval, nzval; dependencies=Event(dev))
    wait(dev, event)

    loop! = _initialize_heuristic_inverse_preconditioner!(dev, 256, M)
    event = loop!(nzval, colptr, rowval, invdiag; dependencies=Event(dev))
    wait(dev, event)
    
    constr_new = (colptr, rowval, nzval)

    Minv = arch_sparse_matrix(arch, constructors(arch, M, constr_new))

    return SparseInversePreconditioner(Minv)
end

@kernel function _initialize_heuristic_inverse_preconditioner!(nzval, colptr, rowval, invdiag)
    col = @index(Global, Linear)

    for idx = colptr[col] : colptr[col+1] - 1
        if rowval[idx] == col
            nzval[idx] = invdiag[col]
        else
            nzval[idx] = - nzval[idx] * 2 / ( 1 / invdiag[rowval[idx]] + 1 / invdiag[col] ) * invdiag[col]
        end
    end
end

function sparse_inverse_preconditioner(A::AbstractMatrix; ε, nzrel)

   # let's choose an initial sparsity => diagonal
   A_cpu    = arch_sparse_matrix(CPU(), A)
   Minv_cpu = sparse_approximate_inverse(A_cpu, ε = ε, nzrel = nzrel)
   
   Minv = arch_sparse_matrix(architecture(A), Minv_cpu)
   return SparseInversePreconditioner(Minv)
end
