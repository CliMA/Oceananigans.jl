using BenchmarkTools
using CUDA
using DataDeps
using Oceananigans
using Benchmarks

# Need a grid

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2"
)

DataDeps.register(dd)

# Benchmark function

# All grids have 6 * 32^2 = 6144 grid points.
grids = Dict(
    :RegularRectilinearGrid => RegularRectilinearGrid(size=(128, 48, 1), extent=(1, 1, 1)),
    :RegularLatitudeLongitudeGrid => RegularLatitudeLongitudeGrid(size=(128, 48, 1), longitude=(-180, 180), latitude=(-80, 80), z=(-1, 0)),
    :ConformalCubedSphereFaceGrid => ConformalCubedSphereFaceGrid(size=(128, 48, 1), z=(-1, 0)),
    :ConformalCubedSphereGrid => ConformalCubedSphereGrid(datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2", Nz=1, z=(-1, 0))
)

free_surfaces = Dict(
    :ExplicitFreeSurface => ExplicitFreeSurface(),
    :ImplicitFreeSurface => ImplicitFreeSurface()
)

function benchmark_hydrostatic_model(Arch, grid_type, free_surface_type)

    model = HydrostaticFreeSurfaceModel(
              architecture = Arch(),
                      grid = grids[grid_type],
        momentum_advection = VectorInvariant(),
              free_surface = free_surfaces[free_surface_type]
    )

    time_step!(model, 1) # warmup

    trial = @benchmark begin
        time_step!($model, 1)
    end samples=10

    return trial
end

# Benchmark parameters

Architectures = has_cuda() ? [CPU, GPU] : [CPU]

grid_types = [
    :RegularRectilinearGrid,
    :RegularLatitudeLongitudeGrid,
    # ConformalCubedSphereFaceGrid,
    :ConformalCubedSphereGrid
]

free_surface_types = [
    :ExplicitFreeSurface,
    # :ImplicitFreeSurface
]

# Run and summarize benchmarks

print_system_info()
suite = run_benchmarks(benchmark_hydrostatic_model; Architectures, grid_types, free_surface_types)

df = benchmarks_dataframe(suite)
# sort!(df, [:Architectures, :Float_types, :Ns], by=(string, string, identity))
benchmarks_pretty_table(df, title="Hydrostatic model benchmarks")
