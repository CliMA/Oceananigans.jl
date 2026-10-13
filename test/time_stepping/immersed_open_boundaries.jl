include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.BoundaryConditions: NFBC, VBC
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

# A channel whose west end is an immersed wall with inflow `U` through its faces
function immersed_inflow_channel(arch, FT; wall_thickness = 2)
    grid = RectilinearGrid(arch, FT; size = (8, 4, 4), halo = (4, 4, 4), x = (0, 2), y = (0, 1), z = (0, 1),
                           topology = (Bounded, Periodic, Bounded))
    wall(x, y, z) = x < wall_thickness * 2 / 8
    return ImmersedBoundaryGrid(grid, GridFittedBoundary(wall))
end

function test_immersed_normal_flow_regularization(arch, FT)
    ibg = immersed_inflow_channel(arch, FT)
    u_bcs = FieldBoundaryConditions(immersed = NormalFlowBoundaryCondition(1))
    c_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(west = NormalFlowBoundaryCondition(1),
                                                                         east = ValueBoundaryCondition(2)))

    model = NonhydrostaticModel(ibg; tracers = :c, boundary_conditions = (; u = u_bcs, c = c_bcs),
                                pressure_solver = ConjugateGradientPoissonSolver(ibg))

    u_ibc = model.velocities.u.boundary_conditions.immersed
    @test u_ibc.west isa NFBC
    @test u_ibc.east isa NFBC
    @test isnothing(u_ibc.south) && isnothing(u_ibc.north) && isnothing(u_ibc.bottom) && isnothing(u_ibc.top)

    # Normal flow conditions apply only on the faces the field lives on
    c_ibc = model.tracers.c.boundary_conditions.immersed
    @test isnothing(c_ibc.west)
    @test c_ibc.east isa VBC

    return nothing
end

# Uniform inflow through an immersed wall into a channel with a prescribed outflow at the east:
# the flow stays uniform and divergence-free, and a tracer front enters with the inflow value.
function test_immersed_inflow_channel(arch, FT, Advection)
    ibg = immersed_inflow_channel(arch, FT)
    U = one(FT)

    u_bcs = FieldBoundaryConditions(immersed = NormalFlowBoundaryCondition(U), east = NormalFlowBoundaryCondition(U))
    c_bcs = FieldBoundaryConditions(immersed = ValueBoundaryCondition(one(FT)))

    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol = sqrt(eps(FT)), abstol = 0, maxiter = 500)
    model = NonhydrostaticModel(ibg; advection = Advection(), tracers = :c, pressure_solver,
                                boundary_conditions = (; u = u_bcs, c = c_bcs))
    set!(model, u = U)

    u, v, w = model.velocities
    c = model.tracers.c
    δ = Field(∂x(u) + ∂y(v) + ∂z(w))

    Δt = FT(0.2) * minimum_xspacing(ibg) / U
    stop_time = 1 / U
    simulation = Simulation(model; Δt, stop_time, verbose = false)
    run!(simulation)

    compute!(δ)
    @test maximum(abs, δ) < sqrt(eps(FT)) * 10
    @test all(Array(interior(u, 3:9, :, :)) .≈ U)

    # The tracer entering through the open faces is conserved in the fluid
    fluid_tracer = sum(Array(interior(c, 3:8, :, :))) * minimum_xspacing(ibg) * minimum_yspacing(ibg) * minimum_zspacing(ibg)
    inflow_area = 1
    @test fluid_tracer ≈ U * inflow_area * time(simulation) rtol = sqrt(eps(FT))

    # With upwinding the tracer front has advected a distance U * t into the channel
    if Advection !== Centered
        x = xnodes(c)
        front = 2 * 2 / 8 + U * time(simulation)
        c_interior = Array(interior(c, :, 1, 1))
        @test all(c_interior[x .< front - 2 * minimum_xspacing(ibg)] .≈ 1)
        @test all(abs.(c_interior[x .> front + 2 * minimum_xspacing(ibg)]) .< 1e-2)
    end

    return nothing
end

# Inflow through a bottom vent balanced by radiation out of the domain top boundary
function test_immersed_bottom_vent(arch, FT)
    grid = RectilinearGrid(arch, FT; size = (8, 8, 8), extent = (1, 1, 1), topology = (Bounded, Bounded, Bounded))
    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -3/4))
    W = one(FT)

    vent(i, j, k, grid, clock, fields) = ifelse((3 ≤ i ≤ 6) & (3 ≤ j ≤ 6), W, zero(W))
    w_bcs = FieldBoundaryConditions(immersed = NormalFlowBoundaryCondition(vent; discrete_form = true),
                                    top = NormalFlowBoundaryCondition(0; scheme = PerturbationAdvection(inflow_timescale = 1e-1)))

    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol = sqrt(eps(FT)), abstol = 0, maxiter = 500)
    model = NonhydrostaticModel(ibg; pressure_solver, boundary_conditions = (; w = w_bcs))

    u, v, w = model.velocities
    δ = Field(∂x(u) + ∂y(v) + ∂z(w))
    ∫w_top = Field(Integral(view(w, :, :, grid.Nz + 1), dims = (1, 2)))

    Δt = FT(0.1) * minimum_zspacing(ibg) / W
    simulation = Simulation(model; Δt, stop_iteration = 10, verbose = false)
    run!(simulation)

    compute!(δ)
    compute!(∫w_top)
    vent_area = 16 * minimum_xspacing(ibg) * minimum_yspacing(ibg)
    @test maximum(abs, δ) < sqrt(eps(FT)) * 10
    @test Array(interior(∫w_top))[1, 1, 1] ≈ W * vent_area

    return nothing
end

# A river entering a bay through the coastline raises the free surface by the inflow volume
function test_immersed_river_inflow(arch, FT, free_surface)
    grid = RectilinearGrid(arch, FT; size = (8, 8, 2), x = (0, 8), y = (0, 8), z = (-1, 0),
                           topology = (Bounded, Bounded, Bounded))
    ibg = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> x < 2))
    U = FT(0.1)

    river(i, j, k, grid, clock, fields) = ifelse((3 ≤ j) & (j ≤ 6), U, zero(U))
    u_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(west = NormalFlowBoundaryCondition(river; discrete_form = true)))
    S_bcs = FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(west = ValueBoundaryCondition(0)))

    model = HydrostaticFreeSurfaceModel(ibg; free_surface, tracers = :S, boundary_conditions = (; u = u_bcs, S = S_bcs))
    set!(model, S = 35)

    η = model.free_surface.displacement
    ∫η = Field(Integral(η, dims = (1, 2)))
    volume(∫η) = (compute!(∫η); Array(interior(∫η))[1, 1, 1])
    V₀ = volume(∫η)

    Δt = FT(0.01)
    simulation = Simulation(model; Δt, stop_iteration = 20, verbose = false)
    run!(simulation)

    Q = U * 4 * 1
    @test volume(∫η) - V₀ ≈ Q * 20 * Δt rtol = 1e-3
    @test all(Array(interior(model.velocities.u, 3, 3:6, :)) .≈ U)

    # Fresh water enters: the salinity next to the river mouth drops
    S = model.tracers.S
    @test all(Array(interior(S, 3, 3:6, :)) .< 35)
    @test all(isapprox.(Array(interior(S, 3, 1:2, :)), 35; rtol = 1e-4))

    return nothing
end

@testset "Immersed open boundaries" begin
    for arch in archs, FT in float_types
        @testset "Regularization [$(summary(arch)), $FT]" begin
            test_immersed_normal_flow_regularization(arch, FT)
        end

        @testset "Inflow through an immersed wall [$(summary(arch)), $FT, $Advection]" for Advection in (Centered, UpwindBiased, WENO)
            test_immersed_inflow_channel(arch, FT, Advection)
        end

        @testset "Inflow through a bottom vent [$(summary(arch)), $FT]" begin
            test_immersed_bottom_vent(arch, FT)
        end

        for free_surface in (SplitExplicitFreeSurface(substeps = 10), ImplicitFreeSurface(), ExplicitFreeSurface())
            @testset "River inflow through a coastline [$(summary(arch)), $FT, $(nameof(typeof(free_surface)))]" begin
                test_immersed_river_inflow(arch, FT, free_surface)
            end
        end
    end
end
