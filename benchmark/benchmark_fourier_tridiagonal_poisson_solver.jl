push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

using Oceananigans.Solvers

# Benchmark function

function benchmark_fourier_tridiagonal_poisson_solver(Arch, N, topo)
    grid = RectilinearGrid(architecture=Arch(), topology=topo, size=(N, N, N), x=(0, 1), y=(0, 1), z=collect(0:N))
    solver = FourierTridiagonalPoissonSolver(Arch(), grid)

    solve_poisson_equation!(solver) # warmup

    trial = @benchmark begin
        @sync_gpu solve_poisson_equation!($solver)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = CUDA.functional() ? [CPU, GPU] : [CPU]
Ns = [256]
PB = (Periodic, Bounded)
Topologies = collect(Iterators.product(PB, PB, (Bounded,)))[:]

# Run and summarize benchmarks

suite = run_benchmarks(benchmark_fourier_tridiagonal_poisson_solver; Architectures, Ns, Topologies)

df = benchmarks_dataframe(suite)
sort!(df, [:Architectures, :Topologies, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Fourier-tridiagonal Poisson solver benchmarks")

if GPU in Architectures
    df = gpu_speedups_suite(suite) |> speedups_dataframe
    sort!(df, [:Topologies, :Ns], by=(string, identity))
    benchmarks_pretty_table(df, title="Fourier-tridiagonal Poisson solver CPU to GPU speedup")
end

for Arch in Architectures
    suite_arch = speedups_suite(suite[@tagged Arch], base_case=(Arch, Ns[1], (Periodic, Periodic, Bounded)))
    df_arch = speedups_dataframe(suite_arch, slowdown=true)
    sort!(df_arch, [:Topologies, :Ns], by=string)
    benchmarks_pretty_table(df_arch, title="Fourier-tridiagonal Poisson solver relative performance ($Arch)")
end
