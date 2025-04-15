push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Benchmarks

using Oceananigans.TimeSteppers: update_state!
using BenchmarkTools
using CUDA
using Oceananigans
using Statistics
using Oceananigans.Solvers

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
                     
random_vector = - 0.5 .* rand(Nx, Ny) .- 0.5
bottom_height(arch) = GridFittedBottom(Oceananigans.on_architecture(arch, random_vector))
rgrid(arch) = RectilinearGrid(arch, size=(Nx, Ny, 1), extent=(1, 1, 1), halo = (3, 3, 3))
lgrid(arch) = LatitudeLongitudeGrid(arch, size=(Nx, Ny, 1), longitude=(-180, 180), latitude=(-80, 80), z=(-1, 0), halo = (3, 3, 3))

grids = Dict(
   (CPU, :RectilinearGrid)       => rgrid(CPU()), 
   (CPU, :LatitudeLongitudeGrid) => lgrid(CPU()),
   (CPU, :ImmersedRecGrid)       => ImmersedBoundaryGrid(rgrid(CPU()), bottom_height(GPU())), 
   (CPU, :ImmersedLatGrid)       => ImmersedBoundaryGrid(lgrid(CPU()), bottom_height(GPU())),
   (GPU, :RectilinearGrid)       => rgrid(GPU()),
   (GPU, :LatitudeLongitudeGrid) => lgrid(GPU()),
   (GPU, :ImmersedRecGrid)       => ImmersedBoundaryGrid(rgrid(GPU()), bottom_height(GPU())), 
   (GPU, :ImmersedLatGrid)       => ImmersedBoundaryGrid(lgrid(CPU()), bottom_height(GPU()))
)

free_surfaces = Dict(
#    :ExplicitFreeSurface => ExplicitFreeSurface(),
#    :SplitExplicitFreeSurface => SplitExplicitFreeSurface(; substeps=50),
    :KrylovImplicitFreeSurface => ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, Solver=KrylovSolver), 
    :PCGImplicitFreeSurface    => ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient), 
    :MatrixImplicitFreeSurface => ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver), 
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
    :ImmersedRecGrid,
    :ImmersedLatGrid,
]

free_surface_types = collect(keys(free_surfaces))
    
# Run and summarize benchmarks
print_system_info()
suite = run_benchmarks(benchmark_hydrostatic_model; architectures, grid_types, free_surface_types)

df = benchmarks_dataframe(suite)
benchmarks_pretty_table(df2, title="Hydrostatic model benchmarks")
