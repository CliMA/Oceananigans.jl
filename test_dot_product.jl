
using Oceananigans
using Oceananigans.Operators: Δyᶠᶜᵃ, Δxᶜᶠᵃ, Δzᵃᵃᶜ
using CUDA
using Statistics: dot
using BenchmarkTools

const MAX_THREADS_PER_BLOCK = 1024

function reduce_multiply_one_block!(::Val{FT}, aout, ain, bin, ::Val{totSize}, ::Val{block}) where {FT, totSize, block}

	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x * block
    
	glb   = tix + (bix -1) * block
	
    sum = FT(0.0)
    for i = glb:gdim:totSize
        sum += ain[i] * bin[i]
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
	if tix == 1 
        aout[bix] = shArr[1]
    end

    sync_threads()
end

function reduce_one_block!(::Val{FT}, aout, ain, ::Val{totSize}, ::Val{block}) where {FT, totSize, block}
    
	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x * block

	glb   = tix + (bix -1) * block
	
    sum = 0.0;
    
    for i = glb:gdim:totSize
        sum += ain[i] 
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
	if tix == 1 
        aout[bix] = shArr[1]
    end

    sync_threads()
end

function parallel_dot(FT, a::AbstractArray, b::AbstractArray)

    block = Int(minimum([maximum([size(a)...]), MAX_THREADS_PER_BLOCK]))
    grid  = Int(prod(size(gpu_a6)) / block)

    wrk    = CuArray{FT}(undef, grid) 
    block = 2^floor(Int, log(2, block-1))

    @cuda threads=block blocks=grid reduce_multiply_one_block!(Val(FT), wrk, a, b, Val(grid*block*2), Val(block))
    if grid > 1 
        @cuda threads=block blocks=1 reduce_one_block!(Val(FT), wrk, wrk, Val(grid), Val(block))
    end

    return wrk
end

N = 512
cpu_grid = RectilinearGrid(architecture = CPU(), extent=(1, 1), size=(N, N), halo=(1, 1), topology=(Periodic, Periodic, Flat))
gpu_grid = RectilinearGrid(architecture = GPU(), extent=(1, 1), size=(N, N), halo=(1, 1), topology=(Periodic, Periodic, Flat))

gpu_f1 = CenterField(GPU(), gpu_grid)
gpu_f2 = CenterField(GPU(), gpu_grid)

cpu_f1 = CenterField(CPU(), cpu_grid)
cpu_f2 = CenterField(CPU(), cpu_grid)

cpu_a1 =   Array{Float64}(undef, (N, N))
cpu_a2 =   Array{Float64}(undef, (N, N))

gpu_a1 = CuArray{Float64}(undef, (N, N))
gpu_a2 = CuArray{Float64}(undef, (N, N))

gpu_a6 = CUDA.rand(Float64, N, N)
gpu_a7 = CUDA.rand(Float64, N, N)

FT = eltype(gpu_a6)

set!(gpu_f1, gpu_a6)
set!(gpu_f2, gpu_a7)

function dot_cpu(a, b)
    a_cpu = Array(a)
    b_cpu = Array(b)

    dot(a_cpu, b_cpu)

    a = CuArray(a_cpu)
    b = CuArray(b_cpu)

    return a, b
end



@benchmark a = parallel_dot(FT, gpu_a6, gpu_a7)
# @benchmark b = dot(gpu_a6, gpu_a7)
# @benchmark c = dot(gpu_f1, gpu_f2)

@info "CPU benchmarking"

# @info "benchmarking fields"
# @benchmark dot(cpu_f1, cpu_f2)
# @info "benchmarking arrays"
# @benchmark dot(cpu_a1, cpu_a2)
# @info "benchmarking fields as arrays"
# @benchmark dot(parent(cpu_f1.data), parent(cpu_f2.data))

# @info "GPU benchmarking"

# @info "benchmarking fields"
# @benchmark dot(gpu_f1, gpu_f2)
# @info "benchmarking arrays"
# @benchmark dot(gpu_a1, gpu_a2)
# @info "benchmarking fields as arrays"
# @benchmark dot(parent(gpu_f1.data), parent(gpu_f2.data))
