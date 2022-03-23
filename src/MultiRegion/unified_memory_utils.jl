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
import LinearAlgebra: mul!

import Oceananigans.Solvers: constructors, arch_sparse_matrix

struct Unified end

@inline constructors(::Unified, A::SparseMatrixCSC)       = (unified_array(A.colptr), unified_array(A.rowval), unified_array(A.nzval),  (A.m, A.n))
@inline arch_sparse_matrix(::Unified, A::SparseMatrixCSC) = CuSparseMatrixCSC(constructors(Unified(), A)...)
@inline constructors(::GPU, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))

## Sparse Matrix Test

N  = 840000

b  = rand(N)
cb = CuArray(b)
ub = unified_array(b)
sb = unified_array(b)
r  = zeros(N)
cr = CuArray(r)
sr = unified_array(r)

A  = sprand(N, N, 0.00001)
cA = arch_sparse_matrix(GPU(), A)
sA = arch_sparse_matrix(Unified(), A)

num_dev = 4
M  = Int(N/num_dev)

mat = []
for i in 1:num_dev
    push!(mat, A[M*(i-1)+1:M*i, :])
end

uA = ()
ur = ()

for i in 1:num_dev
    CUDA.device!(i-1)
    uA = (uA..., arch_sparse_matrix(Unified(), mat[i]))
    ur = (ur..., unified_array(zeros(M)))
end

CUDA.device!(0)

function mul!(r::NTuple, A::NTuple, b)
    for i in 1:num_dev
        CUDA.device!(i-1)
        mul!(r[i], A[i], b)
    end
end

mul!(r ,  A,  b)
mul!(cr, cA, cb)
mul!(sr, sA, ub)
mul!(ur, uA, ub)

CUDA.device!(0)

# CPU matrix multiplication
trial = @benchmark begin
    CUDA.@sync blocking = true mul!(r, A, b)
end samples = 100

# 1 GPU matrix multiplication (Device buffer)
trial = @benchmark begin
    CUDA.@sync blocking = true  mul!(cr, cA, cb)
end samples = 100

# 1 GPU matrix multiplication (Unified buffer)
trial = @benchmark begin
    CUDA.@sync blocking = true  mul!(sr, sA, sb)
end samples = 100

# 3 GPUs matrix multiplication (Unified buffer)
trial = @benchmark begin
    CUDA.@sync blocking = true  mul!(ur, uA, ub)
end samples = 100
