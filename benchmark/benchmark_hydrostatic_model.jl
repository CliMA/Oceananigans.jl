push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Benchmarks

using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Diagnostics: accurate_cell_advection_timescale

using BenchmarkTools
using CUDA
using Oceananigans
using Statistics

# Problem size
Nx = 256
Ny = 128

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

grids = Dict(
    (CPU, :RectilinearGrid)       => RectilinearGrid(CPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
    (CPU, :LatitudeLongitudeGrid) => LatitudeLongitudeGrid(CPU(), size=(Nx, Ny, 1), longitude=(-180, 180), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
    (GPU, :RectilinearGrid)       => RectilinearGrid(GPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
    (GPU, :LatitudeLongitudeGrid) => LatitudeLongitudeGrid(GPU(), size=(Nx, Ny, 1), longitude=(-160, 160), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
)

# Cubed sphere cases maybe worth considering eventually
#=
using DataDeps

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_510_grid",
    "Conformal cubed sphere grid with 510×510 grid points on each face",
    "https://engaging-web.mit.edu/~alir/cubed_sphere_grids/cs510/cubed_sphere_510_grid.jld2"
)

DataDeps.register(dd)

# (CPU, :ConformalCubedSphereGrid)     => ConformalCubedSphereGrid(datadep"cubed_sphere_510_grid/cubed_sphere_510_grid.jld2", Nz=1, z=(-1, 0)),
# (GPU, :ConformalCubedSphereGrid)     => ConformalCubedSphereGrid(datadep"cubed_sphere_510_grid/cubed_sphere_510_grid.jld2", Nz=1, z=(-1, 0), architecture=GPU()),
=#

free_surfaces = Dict(
    :ExplicitFreeSurface => ExplicitFreeSurface(),
    :PCGImplicitFreeSurface => ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient), 
    #:PCGImplicitFreeSurfaceNoPreconditioner => ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient, preconditioner_method = nothing), 
    :MatrixImplicitFreeSurfaceOrd2 => ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver), 
    #:MatrixImplicitFreeSurfaceOrd1 => ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver, preconditioner_settings= (order = 1,) ), 
    #:MatrixImplicitFreeSurfaceOrd0 => ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver, preconditioner_settings= (order = 0,) ), 
    #:MatrixImplicitFreeSurfaceNoPreconditioner => ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver, preconditioner_method = nothing),
    :MatrixImplicitFreeSurfaceSparsePreconditioner => ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver, preconditioner_method = :SparseInverse)
)

function benchmark_hydrostatic_model(Arch, grid_type, free_surface_type)

    grid = grids[(Arch, grid_type)]

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surfaces[free_surface_type])

    set_divergent_velocity!(model)
    Δt = accurate_cell_advection_timescale(grid, model.velocities) / 2
    time_step!(model, Δt) # warmup
    
    trial = @benchmark begin
        CUDA.@sync blocking = true time_step!($model, $Δt)
    end samples = 10

    return trial
end

# Benchmark parameters

#architectures = has_cuda() ? [GPU] : [CPU]
architectures = [CPU]

grid_types = [
    :RectilinearGrid,
    :LatitudeLongitudeGrid,
    # Uncomment when ConformalCubedSphereFaceGrids of any size can be built natively without loading from file:
    # :ConformalCubedSphereFaceGrid,
    # :ConformalCubedSphereGrid
]

free_surface_types = collect(keys(free_surfaces))
    
# Run and summarize benchmarks
print_system_info()
suite = run_benchmarks(benchmark_hydrostatic_model; architectures, grid_types, free_surface_types)

df = benchmarks_dataframe(suite)
benchmarks_pretty_table(df, title="Hydrostatic model benchmarks")
