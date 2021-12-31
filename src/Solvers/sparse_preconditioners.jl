using Oceananigans.Architectures
using Oceananigans.Architectures: device
import Oceananigans.Architectures: architecture
using CUDA, CUDA.CUSPARSE
using KernelAbstractions: @kernel, @index

using LinearAlgebra, SparseArrays, IncompleteLU
using SparseArrays: nnz

import LinearAlgebra.ldiv!

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
validate_settings(T, arch, settings)                                 = settings
validate_settings(::Val{:Default}, arch, settings)                   = arch isa CPU ? (τ = 0.001,) : (ε = 0.1, nzrel = 1.0)
validate_settings(::Val{:SparseInverse}, arch, settings::Nothing)    = (ε = 0.1, nzrel = 1.0)
validate_settings(::Val{:ILUFactorization}, arch, settings::Nothing) = (τ = 0.001, ) 

validate_settings(::Val{:ILUFactorization}, arch, settings) = haskey(settings, :τ) ? 
                                                                     settings :
                                                                     throw(ArgumentError("τ has to be specified for ILUFactorization"))
validate_settings(::Val{:SparseInverse}, arch, settings)    = haskey(settings, :ε) && haskey(settings, :nzrel) ?
                                                                     settings :
                                                                     throw(ArgumentError("both ε and nzrel have to be specified for SparseInverse"))

function build_preconditioner(::Val{:Default}, matrix, settings)
    default_method = architecture(matrix) isa CPU ? :ILUFactorization : :SparseInverse
    return build_preconditioner(Val(default_method), matrix, settings)
end

build_preconditioner(::Val{:None},              A, settings) = Identity()
build_preconditioner(::Val{:Jacobi},            A, settings) = JacobiPreconditioner(A)
build_preconditioner(::Val{:SparseInverse},     A, settings) = sparse_inverse_preconditioner(A, ε = settings.ε, nzrel = settings.nzrel)
build_preconditioner(::Val{:SimplifiedInverse}, A, settings) = simplified_inverse_preconditioner(A)
build_preconditioner(::Val{:ILUFactorization},  A, settings) = ilu(A, τ = settings.τ)

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
