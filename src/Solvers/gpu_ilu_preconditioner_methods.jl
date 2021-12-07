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
This allow us to create an incomplete LU preconditioner on the GPU and use it within the IterativeSolvers
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

function forward_substitution!(F::ILUFactorizationGPU, y::AbstractVector)
    L = F.L
    # for the moment just one thread does the substitution, maybe we wanna parallelize this
    @cuda threads=1 _forward_substitution!(L.dims[1], L.colPtr, L.rowVal, L.nzVal, y)  
    y
end

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

function _forward_substitution!(n, colptr, rowval, nzval, y)
    @inbounds begin
    for col = 1 : n - 1
            for idx = colptr[col] : colptr[col + 1] - 1
                y[rowval[idx]] -= nzval[idx] * y[col]
            end
        end
    end
end

function ilu(dA::CuSparseMatrix{Tv}; τ = 3.0) where {Tv}
    A = arch_sparse_matrix(CPU(), dA)
    p = ilu(A, τ = τ)

    return ILUFactorizationGPU(p)
end

using CUDA
using Statistics: dot
using BenchmarkTools
using Adapt


const MAX_THREADS_PER_BLOCK_TIMES_TWO = 2048

# Reduce a value across a warp
@inline function reduce_warp(op::F, val::T) where {F<:Function, T}
    offset = CUDA.warpsize() ÷ UInt32(2)
    # TODO: this can be unrolled if warpsize is known...
    while offset > 0
        val = op(val, CUDA.shfl_down_sync(5, val, offset))
        offset ÷= UInt32(2)
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::F, val::T) where {F<:Function, T}
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    # TODO: use fldmod1 (JuliaGPU/CUDAnative.jl#28)
    wid  = div(threadIdx().x-UInt32(1), CUDA.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x-UInt32(1), CUDA.warpsize()) + UInt32(1)

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDA.warpsize())) ? shared[lane] : zero(T)

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end

# Reduce an array across a complete grid
function reduce_grid(op::F, input, output::CuDeviceArray{T,1}, len::Integer) where {F<:Function, T}

    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= len
        @inbounds val = op(val, input[i])
        i += step
    end

    val = reduce_block(op, val)

    if threadIdx().x == UInt32(1)
        @inbounds output[blockIdx().x] = val
    end

    return
end

# Reduce an array across a complete grid
function reduce_grid_two(opr::F, opa::G, in1, in2, output::CuDeviceArray{T,1}, len::Integer) where {F<:Function, G, T}

    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= len
        @inbounds val = opr(val, opa(in1[i], in2[i]))
        i += step
    end

    val = reduce_block(opr, val)

    if threadIdx().x == UInt32(1)
        @inbounds output[blockIdx().x] = val
    end

    return
end

function gpu_reduce(opr::F, opa::G, in1::CuArray{T, N}, in2::CuArray{T, N}) where {F<:Function, G<:Function, T, N}
    len = length(in1)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 1024
    blocks = min((len + threads - 1) ÷ threads, 1024)

    tmp = CuArray{T}(undef, blocks)
    out = CuArray{T}(undef, 1)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    
    @cuda blocks=blocks threads=threads reduce_grid_two(opr, opa, in1, in2, tmp, Int32(len))
    @cuda blocks=1 threads=1024 reduce_grid(opr, tmp, out, Int32(blocks))

    return out
end

"""
My Reduce operation
"""

function reduce_multiply_one_block!(::Val{FT}, out, ain, bin, ::Val{totSize}, ::Val{block}) where {FT, totSize, block}

	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x * block
    
	glb   = tix + (bix -UInt32(1)) * block
	
    sum = FT(0.0)
    for i = glb:gdim:totSize
        @inbounds sum += ain[i] * bin[i]
    end
    
    shArr = @cuStaticSharedMem(FT, block)
	shArr[tix] = sum;

    sync_threads()

	iter = block ÷ UInt32(2)
    while iter > 0
		if tix < iter + UInt32(1)
			shArr[tix] += shArr[tix+iter]
        end
        sync_threads()
        iter = iter ÷ UInt32(2)
	end
	if tix == 1 
       out[bix] = shArr[1]
    end

    sync_threads()
end

function reduce_one_block!(::Val{FT}, out, ain, ::Val{totSize}, ::Val{block}) where {FT, totSize, block}
    
	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x * block

	glb   = tix + (bix -1) * block
	
    sum = 0.0;
    for i = glb:gdim:totSize
        @inbounds sum += ain[i] 
    end
    
    shArr = @cuStaticSharedMem(FT, block)
	shArr[tix] = sum;

    sync_threads()
    
    iter = block ÷ 2
    while iter > 0
		if tix < iter + 1
			shArr[tix] += shArr[tix+iter]
        end
        sync_threads()
        iter = iter ÷ 2
	end
    ain[bix] = shArr[1]
 
    sync_threads()
end

function parallel_dot(a::AbstractArray{FT, N}, b::AbstractArray{FT, N}) where {FT, N}

    len = length(a)

    block = UInt32(gcd(len, MAX_THREADS_PER_BLOCK_TIMES_TWO))
    grid  = UInt32(len / block)

    block = 2^floor(UInt32, log(2, block-1))

    tmp    = CuArray{FT}(undef, grid)
    output = CuArray{FT}(undef, 1)

    @cuda threads=block blocks=grid reduce_multiply_one_block!(Val(FT), tmp, a, b, Val(len), Val(block))
    @cuda threads=block blocks=1 reduce_one_block!(Val(FT), output, tmp, Val(grid), Val(block))
    
    return output
end


