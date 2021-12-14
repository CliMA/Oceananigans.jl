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
Extending the ILUFactorization methods to a ILUFactorizationGPU type which lives on the GPU
This allow us to create an incomplete LU preconditioner on the GPU and use it within the IterativeSolvers
directly on the GPU

Next step should be to make an efficient parallel backward - forward substitution on the GPU
At the moment it is done with threads == blocks == 1. 

This is extremely unefficient!

choices of preconditioners on the CPU
Identity()
ilu()

"""

@inline arch_preconditioner(::Val{false}, args...)      = Identity()
@inline arch_preconditioner(::Val{true}, ::CPU, matrix) = ilu(matrix, τ = 0.01) # ILUinverse(constr, τ = 0.01, tolerance = 1.5) # MITGCMPreconditioner(constructors(CPU(), matrix)) #ILUinverse(matrix, τ = 0.001, tolerance = 1.5) #ilu(matrix, τ=0.001) 
@inline arch_preconditioner(::Val{true}, ::GPU, matrix) = Identity() #sparse_inverse_preconditioner(matrix) # Identity()            # ILUinverse(constr, τ = 0.01, tolerance = 1.5) # Identity()           

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

function MITGCMPreconditioner(constr::Tuple)
    
    arch                  = architecture(constr[3])
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

struct ILUinverse{M} <: AbstractInversePreconditioner{M}
    Minv :: M
end

function ILUinverse(constr::Tuple; τ = 0.01, tolerance)

    arch  = architecture(constr[3])
    A_cpu = arch_sparse_matrix(CPU(), constr)
    M = size(A_cpu, 1)

    @show "starting the ilu decomposition"
    P = ilu(A_cpu, τ = τ)

    @show "starting the incomplete inverse procedure"
    Linv = sparse_tri_inverse(P.L, tolerance = tolerance)
    Uinv = sparse_tri_inverse(P.U, tolerance = tolerance)

    M = arch_sparse_matrix(arch, Linv * sparse(Uinv'))

    return ILUinverse(M)
end

function sparse_tri_inverse(A::AbstractMatrix; tolerance = 1.5)
    
    (istril(A)) || throw(ArgumentError("A not lower triangular!!"))
    
    M = A.n

    # substitute diagonal with one
    A = A + spdiagm(0 => - diag(A) + ones(M))
    rowptr, colval, nzval = A.colptr, A.rowval, A.nzval

    newnzval  = []
    newcolval = []
    newrowptr = ones(Int64, M+1)

    @inbounds for i = 1 : M - 1
        tmp_val = - deepcopy(nzval[rowptr[i]+1:rowptr[i+1]-1])
        tmp_col =   deepcopy(colval[rowptr[i]+1:rowptr[i+1]-1])

        j = 1
        while j <= length(tmp_val)
            @inbounds α    = - tmp_val[j]
            if abs(α) > tolerance 
                for k = rowptr[j]+1:rowptr[j+1]-1
                    idx = findfirst(tmp_col .== colval[k]) 
                    if idx === nothing
                        @inbounds tmp_val = [tmp_val..., α * nzval[k]]
                        @inbounds tmp_col = [tmp_col..., colval[k]]
                    else
                        @inbounds tmp_val[idx] += α * nzval[k]
                    end
                end 
            end
            j += 1
        end
        println("finished with index $i")

        perm = sortperm(tmp_col)
        @inbounds newnzval  = [newnzval...,  tmp_val[perm]]
        @inbounds newcolval = [newcolval..., tmp_col[perm]]
        @inbounds newrowptr[i+1] = newrowptr[i] + length(tmp_val)
    end

    return SparseMatrixCSC(M, M, newrowptr, newcolval, newnzval)
end

function sparse_tri_inverse_old(A::AbstractMatrix; tolerance = 1.5)
    
    (istril(A)) || throw(ArgumentError("A not lower triangular!!"))
    
    M = A.n
    @show "Unrolling rows from matrix"

    rows, colidx = unroll_rows_from_matrix(A)
    @show "All unrolled"

    new_rows    = ()
    new_colidxs = ()

    for i = 1 : M - 1
        tmp_row = - deepcopy(rows[i][2:end])
        tmp_col =   deepcopy(colidx[i][2:end])

        j = 1
        while j <= length(tmp_row)
            α    = - tmp_row[j]
            row  = tmp_col[j]
            if abs(α) > tolerance 
                for k = 2:length(rows[row])
                    idx = findfirst(tmp_col .== colidx[row][k]) 
                    if idx === nothing
                        tmp_row = [tmp_row..., α * rows[row][k]]
                        tmp_col = [tmp_col..., colidx[row][k]]
                    else
                        tmp_row[idx] += α * rows[row][k]
                    end
                    @show i, j, idx
                end 
            end
            j += 1
        end


        perm = sortperm(tmp_col)
        new_rows    = (new_rows...,    tmp_row[perm])
        new_colidxs = (new_colidxs..., tmp_col[perm])
    end

    Ainv = reroll_rows_into_matrix(new_rows, new_colidxs, M)

    return Ainv
end

function tri_inverse(A::AbstractMatrix)

    (istril(A)) || "A not lower triangular!!"
    

    Ainv = zeros(size(A))
    Am   = zeros(size(A))
    
    M = size(A, 1)

    for i = 1 : M
        Am[i, i+1:M] = A[i, i+1:M]
    end
    for i = 1 : M 
        Ainv[i, :] = - A[i,:] 

        for j = i + 1 : M 
            α   = - Ainv[i, j]
            if abs(α) > 0.0
                Ainv[i, :] = Ainv[i, :] + α * Am[j, :] 
            end
        end
    end

    return sparse(Ainv)
end

@inline matrix(p::MITGCMPreconditioner)         = p.Minv
@inline matrix(p::ILUinverse)                   = p.Minv
# @inline matrix(p::SparseInversePreconditioner)  = p.Minv

# struct SparseInversePreconditioner{M} <: AbstractInversePreconditioner{M}
#    Minv :: M
# end

#function sparse_inverse_preconditioner(A::AbstractMatrix)
#
#    # let's choose an initial sparsity => diagonal
#    A_cpu    = arch_sparse_matrix(CPU(), A)
#    Minv_cpu = spdiagm(0 => diag(A_cpu) ./ diag(A_cpu))
#    
#    Minv = arch_sparse_matrix(architecture(A), Minv_cpu)
#    return SparseInversePreconditioner(Minv)
#end

# % Loop creating each column of M.
# for k = 1 : M,
# % Find the non-zero pattern of the kth column of M
#    Pj = P(k-colIdx(1)+1,:)’;
# % Consider the subset of A that contains the columns matching
# % the positions of non-zeros in mk, leaving only the columns J.
# % Then remove any rows that have all zeros, leaving only the rows I.
# % This is the subset of A that we are going to work with.
# % Handle communication for a bit.
# % - req_wait_list will be updated to reflect any new requests.
# % - newCols will contain the data of the non-zero columns of
# % A(:,J) that are not local to this processor.
# % - Jremote is the subset of J that identifies columns on other
# % processors
# [newCols req_wait_list Jremote] = handleCommunication(A,colIdx(1),...
# colIdx(end),true,J,...
# colCnt,req_wait_list);
# % Now, compute Aij(I,J) to be the A(I,J) mentioned in the
# % algorithm (in the algorithm "A" is all of A).
# Jloc = setdiff(J,Jremote);
# Aij = newCols;
# for i = 1 : length(Jloc),
# Aij(:,Jloc(i)) = A(:,Jloc(i)-colIdx(1)+1);
# end;

# [I, tmpJ, Ai] = find(Aij(:,J));
# % Eliminate repeat entries.
# I = unique(I);
# % Solve the least-squares problem Aij(I,J)*mhat = ehat
# % to find the value of mk with its original sparsity
# % pattern.
# % ehat is the jth column of the identity matrix, subsetted
# % to be the same size as A(I,J).
# ehat = spalloc(n,1,1);
# ehat(k) = 1;
# ehat = ehat(I,:);
# % Solve the least squares problem using QR decomposition.
# [Q R] = qr(Aij(I,J));
# mhat = R \ (Q’*ehat);
# % Update the current column of M
# % (well, do that but store it as a row instead).
# M(k-colIdx(1)+1,sortedJ) = mhat’;
# end; % end loop over columns
