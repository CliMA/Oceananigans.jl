include("reactant_test_utils.jl")

@testset "Reactant LatitudeLongitudeGrid Simulation Tests" begin
    @info "Performing Reactanigans LatitudeLongitudeGrid simulation tests..."
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    longitude = (0, 4)
    stretched_longitude = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.3, 2.5, 2.6, 3.5, 4.0]
    latitude = (0, 4)
    z = (-1, 0)
    lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude, latitude, z)
    stretched_lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude=stretched_longitude, latitude, z)

    @info "Testing LatitudeLongitudeGrid + SplitExplicitFreeSurface + HydrostaticFreeSurfaceModel Reactant correctness"
    hydrostatic_model_kw = (; momentum_advection=VectorInvariant(), free_surface=SplitExplicitFreeSurface(substeps=4))
    test_reactant_model_correctness(LatitudeLongitudeGrid,
                                    HydrostaticFreeSurfaceModel,
                                    lat_lon_kw,
                                    hydrostatic_model_kw)

    @info "Testing LatitudeLongitudeGrid + SplitExplicitFreeSurface + HydrostaticFreeSurfaceModel Reactant correctness"
    simulation = test_reactant_model_correctness(LatitudeLongitudeGrid,
                                                 HydrostaticFreeSurfaceModel,
                                                 lat_lon_kw,
                                                 hydrostatic_model_kw,
                                                 immersed_boundary_grid=true)
    η = simulation.model.free_surface.η
    η_grid = η.grid
    @test isnothing(η_grid.interior_active_cells)
    @test isnothing(η_grid.active_z_columns)

    #=
    hydrostatic_model_kw = (; momentum_advection=WENOVectorInvariant(), free_surface=SplitExplicitFreeSurface(substeps=4))
    @info "Testing LatitudeLongitudeGrid + WENO + SplitExplicitFreeSurface + HydrostaticFreeSurfaceModel Reactant correctness"
    simulation = test_reactant_model_correctness(LatitudeLongitudeGrid,
                                                 HydrostaticFreeSurfaceModel,
                                                 lat_lon_kw,
                                                 hydrostatic_model_kw,
                                                 immersed_boundary_grid=true)
    η = simulation.model.free_surface.η
    η_grid = η.grid
    @test isnothing(η_grid.interior_active_cells)
    @test isnothing(η_grid.active_z_columns)

    @info "Testing LatitudeLongitudeGrid + 'complicated HydrostaticFreeSurfaceModel' Reactant correctness"
    equation_of_state = TEOS10EquationOfState()
    hydrostatic_model_kw = (momentum_advection = WENOVectorInvariant(),
                            tracer_advection = WENO(),
                            tracers = (:T, :S, :e),
                            buoyancy = SeawaterBuoyancy(; equation_of_state),
                            closure = CATKEVerticalDiffusivity())

    test_reactant_model_correctness(LatitudeLongitudeGrid, HydrostaticFreeSurfaceModel, lat_lon_kw, hydrostatic_model_kw)
    =#
end

