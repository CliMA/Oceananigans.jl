using Test
using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
const CAVD = ConvectiveAdjustmentVerticalDiffusivity

@testset "Ensembles of `HydrostaticFreeSurfaceModel` with different closures" begin

    Nz = 16
    Hz = 1 
    grid = RegularRectilinearGrid(size=Nz, z=(-10, 10), topology=(Flat, Flat, Bounded), halo=1)

    closures = [CAVD(background_κz=1.0) CAVD(background_κz=1.1)
                CAVD(background_κz=1.2) CAVD(background_κz=1.3)]
    
    @test size(closures) == (2, 2)
    @test closures[2, 1].background_κz == 1.2 

    Δt = 0.01 * grid.Δz^2

    model_kwargs = (; tracers=:c, buoyancy=nothing, coriolis=nothing)
    simulation_kwargs = (; Δt, stop_iteration=100)

    models = [HydrostaticFreeSurfaceModel(; grid, closure=closures[i, j], model_kwargs...) for i=1:2, j=1:2]

    set_ic!(model) = set!(model, c = (x, y, z) -> exp(-z^2)) 

    for model in models
        set_ic!(model)
        simulation = Simulation(model; simulation_kwargs...)
        run!(simulation)
    end 

    ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(2, 2), Hz=1)
    ensemble_grid = RegularRectilinearGrid(size=ensemble_size, z=(-10, 10), topology=(Flat, Flat, Bounded), halo=1)

    @test size(ensemble_grid) == (2, 2, Nz) 

    ensemble_model = HydrostaticFreeSurfaceModel(; grid=ensemble_grid, closure=closures, model_kwargs...)
    set_ic!(ensemble_model)

    @test size(parent(ensemble_model.tracers.c)) == (2, 2, Nz+2)

    ensemble_simulation = Simulation(ensemble_model; simulation_kwargs...)
    run!(ensemble_simulation)

    for i = 1:2, j = 1:2 
        @info "Testing IsotropicDiffusivity ensemble member ($i, $j)..."
        @test parent(ensemble_model.tracers.c)[i, j, :] == parent(models[i, j].tracers.c)[1, 1, :]
    end 

end

@testset "Ensembles of `HydrostaticFreeSurfaceModel` with different Coriolis parameters" begin

    Nz = 2 
    Hz = 1 
    grid = RegularRectilinearGrid(size=Nz, z=(-1, 0), topology=(Flat, Flat, Bounded), halo=1)

    #coriolises = [FPlane(f=0.0) FPlane(f=0.5)
    #              FPlane(f=1.0) FPlane(f=1.1)]

    coriolises = [FPlane(f=1.0) FPlane(f=1.0)
                  FPlane(f=1.0) FPlane(f=1.1)]
    
    Δt = 0.01

    @test size(coriolises) == (2, 2)
    @test coriolises[2, 1].f == 1.0 

    model_kwargs = (; tracers=nothing, buoyancy=nothing, closure=nothing)
    simulation_kwargs = (; Δt, stop_iteration=100)

    models = [HydrostaticFreeSurfaceModel(; grid, coriolis=coriolises[i, j], model_kwargs...) for i=1:2, j=1:2]

    set_ic!(model) = set!(model, u=sqrt(2), v=sqrt(2))

    for model in models
        set_ic!(model)
        simulation = Simulation(model; simulation_kwargs...)
        run!(simulation)
    end 

    ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(2, 2), Hz=1)
    ensemble_grid = RegularRectilinearGrid(size=ensemble_size, z=(-1, 0), topology=(Flat, Flat, Bounded), halo=1)
    ensemble_model = HydrostaticFreeSurfaceModel(; grid=ensemble_grid, coriolis=coriolises, model_kwargs...)
    set_ic!(ensemble_model)
    ensemble_simulation = Simulation(ensemble_model; simulation_kwargs...)
    run!(ensemble_simulation)

    for i = 1:2, j = 1:2 
        @info "Testing Coriolis ensemble member ($i, $j) with $(coriolises[i, j])..."
        @test ensemble_model.coriolis[i, j] == coriolises[i, j]
        @show parent(ensemble_model.velocities.u)[i, j, :]
        @show parent(ensemble_model.velocities.v)[i, j, :]
        @test parent(ensemble_model.velocities.u)[i, j, :] == parent(models[i, j].velocities.u)[1, 1, :]
        @test parent(ensemble_model.velocities.v)[i, j, :] == parent(models[i, j].velocities.v)[1, 1, :]
    end 

end
