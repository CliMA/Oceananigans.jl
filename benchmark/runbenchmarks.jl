using BenchmarkTools
using Oceananigans
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10
using Random
Random.seed!(1234)

function hydrostatic_model(grid, 
                           free_surface, 
                           momentum_advection, 
                           tracer_advection,
                           closure)

    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())

    model = HydrostaticFreeSurfaceModel(; grid, 
                                          free_surface, 
                                          momentum_advection,
                                          tracers = (:T, :S, :e),
                                          buoyancy,
                                          tracer_advection, 
                                          closure)

    set!(model, T=20, S=35)

    # Warm up
    time_step!(model, 0.00001)
    time_step!(model, 0.00001)

    return model
end

function nonhydrostatic_model(grid, advection)

    model = NonhydrostaticModel(; grid, advection)

    # Warm up
    time_step!(model, 0.00001)
    time_step!(model, 0.00001)

    return model
end

function run_model_benchmark(model)
    for i in 1:10
        time_step!(model, 0.00001)
    end
end

suite = BenchmarkGroup()

Nx = 20
Ny = 20
Nz = 20

rgrid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(7, 7, 7))
lgrid = LatitudeLongitudeGrid(size=(Nx, Ny, Nz), latitude=(-10, 10), longitude=(0, 360), z=(-1, 0), halo=(7, 7, 7))
tgrid = TripolarGrid(size=(Nx, Ny, Nz), z=(-1, 0), halo=(7, 7, 7))

bottom = 0.5 .* rand(Nx, Ny) .- 1

rigrid = ImmersedBoundaryGrid(rgrid, GridFittedBottom(bottom); active_cells_map=true)
ligrid = ImmersedBoundaryGrid(lgrid, GridFittedBottom(bottom); active_cells_map=true)
tigrid = ImmersedBoundaryGrid(tgrid, GridFittedBottom(bottom); active_cells_map=true)

# All grids we test
grids = [
    rgrid,
    lgrid,
    tgrid,
    rigrid,
    ligrid,
    tigrid
]

nonhydrostatic_grids = [rgrid, rigrid] 

free_surfaces = [
    ExplicitFreeSurface(),
    SplitExplicitFreeSurface(substeps=10),
]

momentum_advections = [
    WENOVectorInvariant()
]

tracer_advections = [
    Centered(order=4),
    WENO(order=9)
]

hydrostatic_closures = [
    nothing,
    CATKEVerticalDiffusivity()
]

suite["hydrostatic"]    = BenchmarkGroup(["grid", "free_surface", "momentum_advection", "tracer_advection", "closure"])
suite["nonhydrostatic"] = BenchmarkGroup(["grid", "advection"])

params(model::HydrostaticFreeSurfaceModel) = (params(model.grid), 
                                              params(model.free_surface), 
                                              params(model.advection.momentum), 
                                              params(model.advection.T), 
                                              params(model.closure))

params(model::NonhydrostaticModel) = (params(model.grid), 
                                      params(model.advection))
                                              
params(::Nothing)                  = "no closure"
params(::WENO)                     = "WENO"
params(::Centered)                 = "Centered"
params(::CATKEVerticalDiffusivity) = "CATKE"
params(::SplitExplicitFreeSurface) = "SplitExplicitFreeSurface"
params(::ExplicitFreeSurface)      = "ExplicitFreeSurface"
params(::VectorInvariant)          = "VectorInvariant"
params(i::ImmersedBoundaryGrid)    = "ImmersedBoundaryGrid on a " * params(i.underlying_grid) 
params(::RectilinearGrid)          = "RectilinearGrid"
params(::LatitudeLongitudeGrid)    = "LatitudeLongitudeGrid"
params(::TripolarGrid)             = "TripolarGrid"

for grid in grids, 
    free_surface in free_surfaces,
    momentum_advection in momentum_advections,
    tracer_advection in tracer_advections,
    closure in hydrostatic_closures
    
    model = hydrostatic_model(grid, free_surface, momentum_advection, tracer_advection, closure)
    suite["hydrostatic"][params(model)...] = @benchmarkable run_model_benchmark($model)
end

for grid in nonhydrostatic_grids, 
    advection in tracer_advections
    
    model = nonhydrostatic_model(grid, advection)
    suite["nonhydrostatic"][params(model)...] = @benchmarkable run_model_benchmark($model)
end

tune!(suite)
results = run(suite, verbose = true)

BenchmarkTools.save("output.json", median(results))