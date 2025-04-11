push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Benchmarks

using Oceananigans.TimeSteppers: update_state!
using BenchmarkTools
using CUDA
using Oceananigans
using Statistics

# Problem size
Nx = 512
Ny = 256

function set_divergent_velocity!(model)
    # Create a divergent velocity
    grid = model.grid

    u, v, w = model.velocities
    η = model.free_surface.η

    u .= 0
    v .= 0
    η .= 0

    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u[imid, jmid, 1] = 1

    update_state!(model)

    return nothing
end

λ¹ₚ = 70
λ²ₚ = 70 + 180
φₚ  = 55

# We need a bottom height field that ``masks'' the singularities
bottom_height(λ, φ) = ((abs(λ - λ¹ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                      ((abs(λ - λ²ₚ) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78) ? 1 : 0

grids = Dict(
    (CPU, :RectilinearGrid)       => RectilinearGrid(CPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
    (CPU, :LatitudeLongitudeGrid) => LatitudeLongitudeGrid(CPU(), size=(Nx, Ny, 1), longitude=(-180, 180), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
    (CPU, :TripolarGrid)          => ImmersedBoundaryGrid(TripolarGrid(CPU(), size=(Nx, Ny, 1)), GridFittedBottom(bottom_height)), 
    (GPU, :RectilinearGrid)       => RectilinearGrid(GPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
    (GPU, :LatitudeLongitudeGrid) => LatitudeLongitudeGrid(GPU(), size=(Nx, Ny, 1), longitude=(-160, 160), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
    (GPU, :TripolarGrid)          => ImmersedBoundaryGrid(TripolarGrid(GPU(), size=(Nx, Ny, 1)), GridFittedBottom(bottom_height)) 
)

free_surfaces = Dict(
    :ExplicitFreeSurface => ExplicitFreeSurface(),
    :SplitExplicitFreeSurface => SplitExplicitFreeSurface(; substeps=50),
    :PCGImplicitFreeSurface => ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient), 
    :MatrixImplicitFreeSurface => ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver), 
)

function benchmark_hydrostatic_model(Arch, grid_type, free_surface_type)

    grid = grids[(Arch, grid_type)]

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection = VectorInvariant(),
                                          free_surface = free_surfaces[free_surface_type])

    set_divergent_velocity!(model)
    Δt = Oceananigans.Advection.cell_advection_timescale(grid, model.velocities) / 2
    time_step!(model, Δt) # warmup
    
    trial = @benchmark begin
        CUDA.@sync blocking = true time_step!($model, $Δt)
    end samples = 10

    return trial
end

# Benchmark parameters

architectures = has_cuda() ? [GPU, CPU] : [CPU]

grid_types = [
    :RectilinearGrid,
    :LatitudeLongitudeGrid,
    :TripolarGrid
]

free_surface_types = collect(keys(free_surfaces))
    
# Run and summarize benchmarks
print_system_info()
suite = run_benchmarks(benchmark_hydrostatic_model; architectures, grid_types, free_surface_types)

df = benchmarks_dataframe(suite)
benchmarks_pretty_table(df, title="Hydrostatic model benchmarks")
