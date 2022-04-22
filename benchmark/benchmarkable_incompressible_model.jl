push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Float_types = [Float32, Float64]
Ns = [32, 64, 128, 256]

# Define benchmarks

SUITE = BenchmarkGroup()

for Arch in Architectures, FT in Float_types, N in Ns
    @info "Setting up benchmark: ($Arch, $FT, $N)..."

    grid = RectilinearGrid(FT, size=(N, N, N), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=Arch(), grid=grid)

    time_step!(model, 1) # warmup

    benchmark = @benchmarkable begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    SUITE[(Arch, FT, N)] = benchmark
end
