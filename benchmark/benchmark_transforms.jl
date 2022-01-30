push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Printf
using BenchmarkTools
using FFTW
using CUDA
using Oceananigans
using Benchmarks

# Benchmark functions

function benchmark_fft(::Type{CPU}, N, dims; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(complex(FT), N, N, N)
    FFT! = FFTW.plan_fft!(A, dims, flags=planner_flag)
    trial = @benchmark ($FFT! * $A) samples=10
    return trial
end

function benchmark_fft(::Type{GPU}, N, dims; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(complex(FT), N, N, N) |> CuArray

    # Cannot do CUDA FFTs along non-batched dims so dim=2 must
    # be computed via tranposing.
    if dims == 2
        B = similar(A)
        FFT! = CUDA.CUFFT.plan_fft!(A, 1)

        trial = @benchmark begin
            CUDA.@sync begin
                permutedims!($B, $A, (2, 1, 3))
                $FFT! * $B
                permutedims!($A, $B, (2, 1, 3))
            end
        end samples=10
    else
        FFT! = CUDA.CUFFT.plan_fft!(A, dims)

        trial = @benchmark begin
            CUDA.@sync ($FFT! * $A)
        end samples=10
    end

    return trial
end

# Benchmark parameters

Architectures = CUDA.functional() ? [CPU, GPU] : [CPU]
Ns = [16, 64, 256]
dims = [1, 2, 3, (1, 2, 3)]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_fft; Architectures, Ns, dims)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :dims, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="FFT benchmarks")

println("3D FFT --> 3 × 1D FFTs slowdown:")
for Arch in Architectures, N in Ns
    fft_x_time = median(suite[(Arch, N, 1)]).time
    fft_y_time = median(suite[(Arch, N, 2)]).time
    fft_z_time = median(suite[(Arch, N, 3)]).time
    fft_xyz_time = median(suite[(Arch, N, (1, 2, 3))]).time
    slowdown = (fft_x_time + fft_y_time + fft_z_time) / fft_xyz_time
    @printf("%s, %3d: %.4fx\n", Arch, N, slowdown)
end
