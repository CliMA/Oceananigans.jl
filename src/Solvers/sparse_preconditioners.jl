using Oceananigans.Architectures
using Oceananigans.Architectures: device
import Oceananigans.Architectures: architecture
using CUDA, CUDA.CUSPARSE
using KernelAbstractions: @kernel, @index

using LinearAlgebra, SparseArrays, IncompleteLU
using SparseArrays: nnz

import LinearAlgebra.ldiv!

"""
`ILUFactorization` (`preconditioner_method = :ILUFactorization`)
    stores two sparse lower and upper trianguilar matrices `L` and `U` such that `LU ≈ A`
    is applied to `r` with `forward_substitution!(L, r)` followed by `backward_substitution!(U, r)`
    constructed with `ilu(A, τ = drop_tolerance)`
    
`SparseInversePreconditioner` (`preconditioner_method = :SparseInverse` and `:AsymptoticInverse`)
    stores a sparse matrix `M` such that `M ≈ A⁻¹` 
    is applied to `r` with a matrix multiplication `M * r`
    constructed with
    `asymptotic_diagonal_inverse_preconditioner(A)`
        -> is an asymptotic expansion of the inverse of A assuming that A is diagonally dominant
        -> it is possible to choose order 0 (Jacobi), 1 or 2
    `sparse_approximate_preconditioner(A, ε = tolerance, nzrel = relative_maximum_number_of_elements)`
        -> same formulation as Grote M. J. & Huckle T, "Parallel Preconditioning with sparse approximate inverses" 
        -> starts constructing the sparse inverse of A from identity matrix until, either a tolerance (ε) is met or nnz(M) = nzrel * nnz(A) 

The suggested preconditioners are

on the `CPU`
`ilu()` (superior to everything always and in every situation!)

on the `GPU`
`aymptotic_diagonal_inverse_preconditioner()` (if `Δt` is variable or large problem_sizes)
`sparse_inverse_preconditioner()` (if `Δt` is constant and problem_size is not too large)

as a rule of thumb, for poisson solvers:
`sparse_inverse_preconditioner` is better performing than `asymptotic_diagonal_inverse_preconditioner` only if `nzrel >= 2.0`
As such, we urge to use `sparse_inverse_preconditioner` only when
- Δt is constant (we don't have to recalculate the preconditioner during the simulation)
- it is feasible to choose `nzrel = 2.0` (for not too large problem sizes)

Note that `asymptotic_diagonal_inverse_preconditioner` assumes the matrix to be diagonally dominant, for this reason it could 
be detrimental when used on non-diagonally dominant system (cases where Δt is very large). In this case it is better 
to use `sparse_inverse_preconditioner`

`ilu()` cannot be used on the GPU because preconditioning the solver with a direct LU (or Choleski) type 
of preconditioner would require too much computation for the `ldiv!(P, r)` step completely hindering the performances
"""

validate_settings(T, arch, settings)                                  = settings
validate_settings(::Val{:Default}, arch, settings)                    = arch isa CPU ? (τ = 0.001, ) : (order = 1, ) 
validate_settings(::Val{:SparseInverse}, arch, settings::Nothing)     = (ε = 0.1, nzrel = 2.0)
validate_settings(::Val{:ILUFactorization}, arch, settings::Nothing)  = (τ = 0.001, ) 
validate_settings(::Val{:AsymptoticInverse}, arch, settings::Nothing) = (order = 1, ) 

validate_settings(::Val{:ILUFactorization}, arch, settings)  = haskey(settings, :τ) ? 
                                                                      settings :
                                                                      throw(ArgumentError("τ has to be specified for ILUFactorization"))
validate_settings(::Val{:SparseInverse}, arch, settings)     = haskey(settings, :ε) && haskey(settings, :nzrel) ?
                                                                      settings :
                                                                      throw(ArgumentError("both ε and nzrel have to be specified for SparseInverse"))
validate_settings(::Val{:AsymptoticInverse}, arch, settings) = haskey(settings, :order) ?
                                                                      settings :
                                                                      throw(ArgumentError("and order ∈ [0, 1, 2] has to be specified for AsymptoticInverse"))


function build_preconditioner(::Val{:Default}, matrix, settings)
    default_method = architecture(matrix) isa CPU ? :ILUFactorization : :DiagonallyDominantInverse
    return build_preconditioner(Val(default_method), matrix, settings)
end

build_preconditioner(::Val{nothing},            A, settings)  = Identity()
build_preconditioner(::Val{:SparseInverse},     A, settings)  = sparse_inverse_preconditioner(A, ε = settings.ε, nzrel = settings.nzrel)
build_preconditioner(::Val{:AsymptoticInverse}, A, settings)  = asymptotic_diagonal_inverse_preconditioner(A, asymptotic_order = settings.order)

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

"""
The diagonally dominant inverse preconditioner is constructed with an asymptotic expansion of `A⁻¹` up to the second order
If `I` is the Identity matrix and `D` is the matrix containing the diagonal of `A`, then

the 0th order expansion is the Jacobi preconditioner `M = D⁻¹ ≈ A⁻¹` 

the 1st order expansion corresponds to `M = D⁻¹(I - (A - D)D⁻¹) ≈ A⁻¹` 

the 2nd order expansion corresponds to `M = D⁻¹(I - (A - D)D⁻¹ + (A - D)D⁻¹(A - D)D⁻¹) ≈ A⁻¹`

all preconditioners are calculated on CPU and then moved to the GPU. 
Additionally the first order expansion has a method to calculate the preconditioner directly on the GPU
`asymptotic_diagonal_inverse_preconditioner_first_order(A)` in case of variable time step where the preconditioner
has to be recalculated often
"""

function asymptotic_diagonal_inverse_preconditioner(A::AbstractMatrix; asymptotic_order)
    
    arch                  = architecture(A)
    constr                = deepcopy(constructors(arch, A)) 
    colptr, rowval, nzval = copy_unpack_constructors(arch, constr)
    dev                   = device(arch)
    
    M = size(A, 1)

    invdiag = arch_array(arch, zeros(eltype(nzval), M))

    loop! = _get_inv_diag!(dev, 256, M)
    event = loop!(invdiag, colptr, rowval, nzval; dependencies=Event(dev))
    wait(dev, event)

    if asymptotic_order == 0
        Minv_cpu = spdiagm(0=>arch_array(CPU(), invdiag))
        Minv     = arch_sparse_matrix(arch, Minv_cpu)
    elseif asymptotic_order == 1
        loop! = _initialize_asymptotic_diagonal_inverse_preconditioner_first_order!(dev, 256, M)
        event = loop!(nzval, colptr, rowval, invdiag; dependencies=Event(dev))
        wait(dev, event)
    
        constr_new = (colptr, rowval, nzval)
        Minv = arch_sparse_matrix(arch, constructors(arch, M, constr_new))
    else
        D   = spdiagm(0=>diag(arch_sparse_matrix(CPU(), A)))
        D⁻¹ = spdiagm(0=>arch_array(CPU(), invdiag))
        Minv_cpu = D⁻¹ * (I - (A - D) * D⁻¹ + (A - D) * D⁻¹ * (A - D) * D⁻¹)
        Minv = arch_sparse_matrix(arch, Minv_cpu)
    end

    return SparseInversePreconditioner(Minv)
end

@kernel function _initialize_asymptotic_diagonal_inverse_preconditioner_first_order!(nzval, colptr, rowval, invdiag)
    col = @index(Global, Linear)

    for idx = colptr[col] : colptr[col+1] - 1
        if rowval[idx] == col
            nzval[idx] = invdiag[col]
        else
            nzval[idx] = - nzval[idx] * invdiag[rowval[idx]] * invdiag[col]
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
