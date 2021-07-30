push!(LOAD_PATH, joinpath(@__DIR__, ".."))

#using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

# Benchmark parameters

timestepper = :QuasiAdamsBashforth2
Arch = CPU
FT = Float64
N = 128

print_system_info()

# Define benchmarks

@info "Setting up benchmark: ($Arch, $FT, $N)..."

grid = RegularRectilinearGrid(FT, size=(N, N, N), extent=(1, 1, 1))

model = NonhydrostaticModel(architecture = Arch(),
                            timestepper = timestepper,
                            grid = grid,
                            advection = WENO5(),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = IsotropicDiffusivity(ν=1, κ=1))

simulation = Simulation(model, Δt=1, stop_iteration=1)

@info "warming up"

run!(simulation)

simulation.stop_iteration += 10

if model.architecture isa GPU
    CUDA.@profile run!(simulation)
else
    @profile run!(simulation)
end

@info "done profiling ($Arch, $FT, $N)"
