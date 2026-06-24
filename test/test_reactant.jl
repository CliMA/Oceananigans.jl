include("reactant_test_utils.jl")

using CUDA

@kernel function _simple_tendency_kernel!(Gu, grid, advection, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = - Oceananigans.Advection.U_dot_∇u(i, j, k, grid, advection, velocities)
end

function simple_tendency!(model)
    grid = model.grid
    arch = grid.architecture
    Oceananigans.Utils.launch!(
        arch,
        grid,
        :xyz,
        _simple_tendency_kernel!,
        model.timestepper.Gⁿ.u,
        grid,
        model.advection.momentum,
        model.velocities)
    return nothing
end

@testset "Gu kernel" begin
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    longitude = (0, 4)
    latitude = (0, 4)
    z = (-1, 0)
    lat_lon_kw = (; size=(Nx, Ny, Nz), halo, longitude, latitude, z)
    hydrostatic_model_kw = (; momentum_advection=VectorInvariant(), free_surface=ExplicitFreeSurface())

    arch = Oceananigans.Architectures.ReactantState()
    grid = LatitudeLongitudeGrid(arch; lat_lon_kw...)
    model = HydrostaticFreeSurfaceModel(grid; hydrostatic_model_kw...)

    @test model.clock.stage == 1

    ui = randn(size(model.velocities.u)...)
    vi = randn(size(model.velocities.v)...)
    set!(model, u=ui, v=vi)

    @jit simple_tendency!(model)

    Gu = model.timestepper.Gⁿ.u
    Gv = model.timestepper.Gⁿ.v
    Gui = Array(interior(Gu))
    Gvi = Array(interior(Gv))

    carch = Oceananigans.Architectures.ReactantState()
    cgrid = LatitudeLongitudeGrid(carch; lat_lon_kw...)
    cmodel = HydrostaticFreeSurfaceModel(cgrid; hydrostatic_model_kw...)

    set!(cmodel, u=ui, v=vi)

    simple_tendency!(cmodel)
    @test all(Array(interior(model.timestepper.Gⁿ.u)) .≈ Array(interior(cmodel.timestepper.Gⁿ.u)))
end

@testset "Reactant RectilinearGrid Simulation Tests" begin
    @info "Performing Reactanigans RectilinearGrid simulation tests..."
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    z = (-1, 0)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    hydrostatic_model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))
    rungekutta3_kw = merge(hydrostatic_model_kw, (; timestepper=:SplitRungeKutta3))

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    hydrostatic_model_kw)

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel + SplitRungeKutta3 Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    rungekutta3_kw)

    @info "Testing immersed RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    hydrostatic_model_kw,
                                    immersed_boundary_grid=true)

    @info "Testing immersed RectilinearGrid + HydrostaticFreeSurfaceModel + SplitRungeKutta3 Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    rungekutta3_kw,
                                    immersed_boundary_grid=true)
end
