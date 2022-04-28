push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]
Ns = [32, 64, 128, 256]

# Define benchmarks

SUITE = BenchmarkGroup()

for Arch in Architectures, N in Ns
    @info "Setting up benchmark: ($Arch, $N)..."

    grid = RectilinearGrid(FT, size=(N, N, N), extent=(1, 1, 1))
    model = NonhydrostaticModel(Arch(); grid)

    time_step!(model, 1) # warmup

    benchmark = @benchmarkable begin
        @sync_gpu time_step!($model, 1)
    end samples=10

    SUITE[(Arch, N)] = benchmark
end

