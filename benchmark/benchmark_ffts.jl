using Printf
using TimerOutputs

using FFTW

using Oceananigans

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

Nr = 10  # Number of repeats for benchmarking.

# Run benchmark across these parameters.
            Ns = [(256, 256, 256)]
   float_types = [Float64]  # Float types to benchmark.
         archs = [CPU()]             # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]      # Benchmark GPU on systems with CUDA-enabled GPUs.

#####
##### Run benchmarks
#####

function dim2xyz(d)
    d == 1 && return :x
    d == 2 && return :y
    d == 3 && return :z
end

for FT in float_types, N in Ns
    Nx, Ny, Nz = N

    #####
    ##### 1D FFT
    #####

    for dim in 1:3
        xyz = dim2xyz(dim)

        bn =  benchmark_name(N, "1D  FFT$xyz", CPU(), FT, npad=3)

        @info @sprintf("Planning transform: %s...\n", bn)
        R = rand(Complex{FT}, Nx, Ny, Nz)
        FFT! = FFTW.plan_fft!(R, dim, flags=FFTW.PATIENT)
        FFT! * R  # warmup

        @info @sprintf("Running benchmark:  %s...\n", bn)
        for _ in 1:Nr
            @timeit timer bn FFT! * R
        end
    end

    #####
    ##### 1D IFFT
    #####

    for dim in 1:3
        xyz = dim2xyz(dim)

        bn =  benchmark_name(N, "1D IFFT$xyz", CPU(), FT, npad=3)

        @info @sprintf("Planning transform: %s...\n", bn)
        R = rand(Complex{FT}, Nx, Ny, Nz)
        IFFT! = FFTW.plan_ifft!(R, 1, flags=FFTW.PATIENT)
        IFFT! * R  # warmup

        @info @sprintf("Running benchmark:  %s...\n", bn)
        for _ in 1:Nr
            @timeit timer bn IFFT! * R
        end
    end

    #####
    ##### 1D DCT
    #####

    bn = benchmark_name(N, "1D  DCTz", CPU(), FT, npad=3)

    @info @sprintf("Planning transform: %s...\n", bn)
    R = rand(FT, Nx, Ny, Nz)
    DCT! = FFTW.plan_r2r!(R, FFTW.REDFT10, 3, flags=FFTW.PATIENT)
    DCT! * R  # warmup

    @info @sprintf("Running benchmark:  %s...\n", bn)
    for _ in 1:Nr
        @timeit timer bn DCT! * R
    end

    #####
    ##### 1D IDCT
    #####

    bn = benchmark_name(N, "1D IDCTz", CPU(), FT, npad=3)

    @info @sprintf("Planning transform: %s...\n", bn)
    R = rand(FT, Nx, Ny, Nz)
    IDCT! = FFTW.plan_r2r!(R, FFTW.REDFT01, 3, flags=FFTW.PATIENT)
    IDCT! * R  # warmup

    @info @sprintf("Running benchmark:  %s...\n", bn)
    for _ in 1:Nr
        @timeit timer bn IDCT! * R
    end

    #####
    ##### 2D FFTxy
    #####

    bn =  benchmark_name(N, "2D  FFTxy ", CPU(), FT, npad=3)

    @info @sprintf("Planning transform: %s...\n", bn)
    R = rand(Complex{FT}, Nx, Ny, Nz)
    FFT! = FFTW.plan_fft!(R, [1, 2], flags=FFTW.PATIENT)
    FFT! * R  # warmup

    @info @sprintf("Running benchmark:  %s...\n", bn)
    for _ in 1:Nr
        @timeit timer bn FFT! * R
    end

    #####
    ##### 2D FFTxy (x then y)
    #####

    bn =  benchmark_name(N, "2D  FFTx,y", CPU(), FT, npad=3)

    @info @sprintf("Planning transform: %s...\n", bn)
    R = rand(Complex{FT}, Nx, Ny, Nz)
    FFTx! = FFTW.plan_fft!(R, 1, flags=FFTW.PATIENT)
    FFTy! = FFTW.plan_fft!(R, 2, flags=FFTW.PATIENT)
    FFTx! * R  # warmup
    FFTy! * R  # warmup

    @info @sprintf("Running benchmark:  %s...\n", bn)
    for _ in 1:Nr
        @timeit timer bn begin
            FFTx! * R
            FFTy! * R
        end
    end
end

#####
##### Print benchmark results
#####

println()
print_benchmark_info()
print_timer(timer, title="Transform benchmarks", sortby=:name)
println()
