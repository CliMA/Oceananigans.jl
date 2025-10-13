push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Benchmarks

using Oceananigans.TimeSteppers: update_state!
using BenchmarkTools
using CUDA
using Oceananigans
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures
using Statistics
using Oceananigans.Solvers
using SeawaterPolynomials.TEOS10
using NVTX 

CUDA.device!(2)

function ocean_benchmark(grid, closure)
    momentum_advection = nothing # WENOVectorInvariant()
    tracer_advection = WENO(order=7)
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
    coriolis = nothing # HydrostaticSphericalCoriolis()
    free_surface = nothing # SplitExplicitFreeSurface(grid; substeps=70)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection,
                                          tracer_advection,
                                          buoyancy,
                                          coriolis,
                                          closure,
                                          free_surface,
                                          tracers = (:T, :S, :e))

    @info "Model is built"
    return model
end

function benchmark_hydrostatic_model(Arch, grid_type, closure_type)

    grid  = grids[(Arch, grid_type)]
    model = ocean_benchmark(grid, closures[closure_type])

    T = 0.0001 .* rand(size(model.grid)) .+ 20
    S = 0.0001 .* rand(size(model.grid)) .+ 35
    
    set!(model.tracers.T, T)
    set!(model.tracers.S, S)

    set!(model.velocities.u, (x, y, z) -> 1e-6 * rand())
    set!(model.velocities.v, (x, y, z) -> 1e-6 * rand())

    Δt = 1
    for _ in 1:30
       time_step!(model, Δt) # warmup
    end

    # Make sure we do not have any NaN or Inf values anywhere
    fields = Oceananigans.fields(model)
    for (key, field) in zip(propertynames(fields), fields)
        arr = Array(interior(field))
        @assert all(isfinite.(arr)) || @show "Nan in $key"
        @assert all(Array(arr) .< 1e10)    || @show "Inf in $key"
        @show key, extrema(arr)
    end

    for _ in 1:10
        Oceananigans.TimeSteppers.compute_tendencies!(model, [])
    end

    # NVTX.@range "compute tendencies" begin
    trial = @benchmark begin
            Oceananigans.TimeSteppers.compute_tendencies!($model, [])
    end samples = 100

    return trial
end

# Problem size
Nx = 200
Ny = 200
Nz = 50

random_vector = - 5000 .* rand(Nx, Ny)

bottom_height(arch) = GridFittedBottom(Oceananigans.on_architecture(arch, random_vector))
lgrid(arch) = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), 
                                     longitude=(0, 360), 
                                      latitude=(-75, 75),
                                             z=collect(range(-5000, 0, length=51)), 
                                          halo=(7, 7, 7))

grids = Dict(
   (CPU, :LatitudeLongitudeGrid) => lgrid(CPU()),
   (CPU, :ImmersedLatGrid)       => ImmersedBoundaryGrid(lgrid(CPU()), bottom_height(CPU()); active_cells_map=true),
   (GPU, :LatitudeLongitudeGrid) => lgrid(GPU()),
   (GPU, :ImmersedLatGrid)       => ImmersedBoundaryGrid(lgrid(GPU()), bottom_height(GPU()); active_cells_map=true), 
)

closures = Dict(
   :DiffImplicit  => VerticalScalarDiffusivity(TurbulenceClosures.VerticallyImplicitTimeDiscretization(), ν=1e-5 , κ=1.0),
   :DiffExplicit  => VerticalScalarDiffusivity(ν=1e-5, κ=1e-5),
   :CATKEExplicit => Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities.CATKEVerticalDiffusivity(TurbulenceClosures.ExplicitTimeDiscretization()),
   :CATKEImplicit => Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities.CATKEVerticalDiffusivity(),
)

# Benchmark parameters

architectures = has_cuda() ? [GPU] : [CPU]

grid_types = [
#    :LatitudeLongitudeGrid,
   :ImmersedLatGrid,
]

closure_types = collect(keys(closures))
    
# Run and summarize benchmarks
# print_system_info()
suite = run_benchmarks(benchmark_hydrostatic_model; architectures, grid_types, closure_types)
df = benchmarks_dataframe(suite)
@show df2 = sort(df)

for _ in 1:5
    suite = run_benchmarks(benchmark_hydrostatic_model; architectures, grid_types, closure_types)
    @show df2 = sort(df)
end

benchmarks_pretty_table(df, title="Hydrostatic model benchmarks")
