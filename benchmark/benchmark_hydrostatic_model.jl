push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using DataDeps
using Oceananigans
using Benchmarks
using Statistics
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Diagnostics: accurate_cell_advection_timescale
# Need a grid

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_510_grid",
    "Conformal cubed sphere grid with 510×510 grid points on each face",
    "https://engaging-web.mit.edu/~alir/cubed_sphere_grids/cs510/cubed_sphere_510_grid.jld2"
)

DataDeps.register(dd)

# Benchmark function

Nx = 256
Ny = 128 

function set_simple_divergent_velocity!(model)
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


# All grids have 6 * 510^2 = 1,560,600 grid points.
grids = Dict(
     (CPU, :RectilinearGrid)          => RectilinearGrid(CPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
     (CPU, :LatitudeLongitudeGrid)    => LatitudeLongitudeGrid(CPU(), size=(Nx, Ny, 1), longitude=(-180, 180), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
    #  (CPU, :ConformalCubedSphereFaceGrid) => ConformalCubedSphereFaceGrid(size=(1445, 1080, 1), z=(-1, 0)),
    #  (CPU, :ConformalCubedSphereGrid)     => ConformalCubedSphereGrid(datadep"cubed_sphere_510_grid/cubed_sphere_510_grid.jld2", Nz=1, z=(-1, 0)),
     (GPU, :RectilinearGrid)          => RectilinearGrid(GPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
     (GPU, :LatitudeLongitudeGrid)    => LatitudeLongitudeGrid(GPU(), size=(Nx, Ny, 1), longitude=(-160, 160), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
    # Uncomment when ConformalCubedSphereFaceGrids of any size can be built natively without loading from file:
    #  (GPU, :ConformalCubedSphereFaceGrid) => ConformalCubedSphereFaceGrid(size=(1445, 1080, 1), z=(-1, 0), architecture=GPU()),
    #  (GPU, :ConformalCubedSphereGrid)     => ConformalCubedSphereGrid(datadep"cubed_sphere_510_grid/cubed_sphere_510_grid.jld2", Nz=1, z=(-1, 0), architecture=GPU()),
)

free_surfaces = Dict(
    :ExplicitFreeSurface => ExplicitFreeSurface(),
    :PCGImplicitFreeSurface => ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient), 
    :PCGImplicitFreeSurfaceNoPreconditioner => ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient, preconditioner_method = nothing), 
    :MatrixImplicitFreeSurface => ImplicitFreeSurface(solver_method = :MatrixIterativeSolver), 
    :MatrixImplicitFreeSurfaceNoPreconditioner => ImplicitFreeSurface(solver_method = :MatrixIterativeSolver, preconditioner_method = nothing),
    :MatrixImplicitFreeSurfaceSparsePreconditioner => ImplicitFreeSurface(solver_method = :MatrixIterativeSolver, preconditioner_method = :SparseInverse, preconditioner_settings = (ε = 0.05, nzrel = 2.0))
)

function benchmark_hydrostatic_model(Arch, grid_type, free_surface_type)

    grid = grids[(Arch, grid_type)]

    model = HydrostaticFreeSurfaceModel(
                      grid = grid,
        momentum_advection = VectorInvariant(),
              free_surface = free_surfaces[free_surface_type]
    )

    set_simple_divergent_velocity!(model)

    Δt = accurate_cell_advection_timescale(grid, model.velocities)

    time_step!(model, Δt) # warmup
    
    trial = @benchmark begin
        CUDA.@sync blocking = true time_step!($model, $Δt)
    end samples = 10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [GPU] : [CPU]

grid_types = [
    :RectilinearGrid,
    :LatitudeLongitudeGrid,
    # Uncomment when ConformalCubedSphereFaceGrids of any size can be built natively without loading from file:
    # :ConformalCubedSphereFaceGrid,
    # :ConformalCubedSphereGrid
]

free_surface_types = [
    :ExplicitFreeSurface,
    :MatrixImplicitFreeSurface,
    :MatrixImplicitFreeSurfaceNoPreconditioner,
    # :MatrixImplicitFreeSurfaceSparsePreconditioner,
    :PCGImplicitFreeSurface,
    :PCGImplicitFreeSurfaceNoPreconditioner
]

# Run and summarize benchmarks
print_system_info()
suite = run_benchmarks(benchmark_hydrostatic_model; Architectures, grid_types, free_surface_types)

df = benchmarks_dataframe(suite)
benchmarks_pretty_table(df, title="Hydrostatic model benchmarks")
