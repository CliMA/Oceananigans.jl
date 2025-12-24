include("dependencies_for_runtests.jl")

using Oceananigans.BoundaryConditions: ContinuousBoundaryFunction,
                                       fill_halo_regions!

using Oceananigans: prognostic_fields

function test_boundary_condition(arch, FT, Model, topo, side, field_name, boundary_condition)
    grid = RectilinearGrid(arch, FT, size=(1, 1, 1), extent=(1, π, 42), topology=topo)

    boundary_condition_kwarg = (; side => boundary_condition)
    field_boundary_conditions = FieldBoundaryConditions(; boundary_condition_kwarg...)
    bcs = (; field_name => field_boundary_conditions)
    model = Model(; grid, boundary_conditions=bcs,
                    buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

    success = try
        time_step!(model, 1e-16)
        true
    catch err
        @warn "test_boundary_condition errored with " * sprint(showerror, err)
        false
    end

    return success
end

function test_nonhydrostatic_flux_budget(grid, name, side, L)
    FT = eltype(grid)
    flux = FT(π)
    direction = side ∈ (:west, :south, :bottom, :immersed) ? 1 : -1
    bc_kwarg = Dict(side => BoundaryCondition(Flux(), flux * direction))
    field_bcs = FieldBoundaryConditions(; bc_kwarg...)
    boundary_conditions = (; name => field_bcs)

    model = NonhydrostaticModel(; grid, boundary_conditions, tracers=:c)

    is_velocity_field = name ∈ (:u, :v, :w)
    field = is_velocity_field ? getproperty(model.velocities, name) : getproperty(model.tracers, name)
    set!(field, 0)

    simulation = Simulation(model, Δt = 1.0, stop_iteration = 1)
    run!(simulation)

    mean_ϕ = mean(field)

    # budget: L * ∂<ϕ>/∂t = -Δflux = -flux / L (left) + flux / L (right)
    # therefore <ϕ> = flux * t / L
    #
    # Note \approx, because velocity budgets are off by machine precision (due to pressure solve?)
    return mean_ϕ ≈ flux * model.clock.time / L
end

function fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
    Lz = 1
    κ₀ = FT(exp(-3))
    bz = FT(π)
    flux = - κ₀ * bz

    grid = RectilinearGrid(arch, FT, size=(16, 16, 16), extent=(1, 1, Lz))

    buoyancy_bcs = FieldBoundaryConditions(bottom=GradientBoundaryCondition(bz))
    κₑ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()), bottom=ValueBoundaryCondition(κ₀))
    model_bcs = (b=buoyancy_bcs, κₑ=(b=κₑ_bcs,))

    model = NonhydrostaticModel(; grid,
                                timestepper = :QuasiAdamsBashforth2,
                                tracers = :b,
                                buoyancy = BuoyancyTracer(),
                                closure = AnisotropicMinimumDissipation(),
                                boundary_conditions = model_bcs)

    b₀(x, y, z) = z * bz
    set!(model, b=b₀)

    b = model.tracers.b
    mean_b₀ = mean(b)

    τκ = Lz^2 / κ₀  # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 10         # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    #
    # Use an atol of 1e-6 so test passes with Float32 as there's a big cancellation
    # error due to buoyancy order of magnitude.
    #
    # Float32:
    # mean_b₀ = -1.5707965f0
    # mean(interior(b)) = -1.5708286f0
    # mean(interior(b)) - mean_b₀ = -3.20673f-5
    # (flux * model.clock.time) / Lz = -3.141593f-5
    #
    # Float64
    # mean_b₀ = -1.5707963267949192
    # mean(interior(b)) = -1.57082774272148
    # mean(interior(b)) - mean_b₀ = -3.141592656086267e-5
    # (flux * model.clock.time) / Lz = -3.141592653589793e-5

    return isapprox(mean(b) - mean_b₀, flux * model.clock.time / Lz, atol=1e-6)
end



end_position(::Val{1}, grid) = (grid.Nx+1, 1, 1)
end_position(::Val{2}, grid) = (1, grid.Ny+1, 1)
end_position(::Val{3}, grid) = (1, 1, grid.Nz+1)

wall_normal_boundary_condition(::Val{1}, obc) = (; u = FieldBoundaryConditions(east = obc, west = obc))
wall_normal_boundary_condition(::Val{2}, obc) = (; v = FieldBoundaryConditions(south = obc, north = obc))
wall_normal_boundary_condition(::Val{3}, obc) = (; w = FieldBoundaryConditions(bottom = obc, top = obc))

normal_velocity(::Val{1}, model) = model.velocities.u
normal_velocity(::Val{2}, model) = model.velocities.v
normal_velocity(::Val{3}, model) = model.velocities.w

velocity_forcing(::Val{1}, forcing) = (; u = forcing)
velocity_forcing(::Val{2}, forcing) = (; v = forcing)
velocity_forcing(::Val{3}, forcing) = (; w = forcing)

function test_perturbation_advection_open_boundary_conditions(arch, FT)
    for orientation in 1:3
        topology = tuple(map(n -> ifelse(n == orientation, Bounded, Flat), 1:3)...)

        grid = RectilinearGrid(arch, FT; topology, size = (4, ), x = (0, 4), y = (0, 4), z = (0, 4), halo = (1, ))

        obc = OpenBoundaryCondition(-1, scheme = PerturbationAdvection(inflow_timescale = 10.0))
        boundary_conditions = wall_normal_boundary_condition(Val(orientation), obc)

        model = NonhydrostaticModel(; grid, boundary_conditions, timestepper = :QuasiAdamsBashforth2)
        u = normal_velocity(Val(orientation), model)
        fill!(u, -1)

        time_step!(model, 1)

        # nothing going on
        @test all(view(parent(u), :, :, :) .== -1)
        @test all(interior(u) .== -1)

        obc = OpenBoundaryCondition(t -> 0.1*t, scheme = PerturbationAdvection(inflow_timescale = 0.01, outflow_timescale = 0.5))
        forcing = velocity_forcing(Val(orientation), Forcing((x, t) -> 0.1))
        boundary_conditions = wall_normal_boundary_condition(Val(orientation), obc)

        model = NonhydrostaticModel(; grid,
                                      boundary_conditions,
                                      timestepper = :QuasiAdamsBashforth2,
                                      forcing)

        u = normal_velocity(Val(orientation), model)

        for _ in 1:100
            time_step!(model, 0.1)
        end

        @test all(map(u->isapprox(u, 1, atol=0.1), interior(u)))
    end
end

function test_open_boundary_condition_mass_conservation(arch, FT, boundary_conditions; N = 8)
    grid = RectilinearGrid(arch, FT, size=(N, N, N), extent=(1, 1, 1),
                           topology=(Bounded, Bounded, Bounded))

    model = NonhydrostaticModel(; grid, boundary_conditions, timestepper = :RungeKutta3)
    uᵢ(x, y, z) = 1 + 1e-2 * rand()
    set!(model, u = uᵢ)

    u, v, w = model.velocities
    Δt = 0.1 * minimum_zspacing(grid) / maximum(abs, u)
    simulation = Simulation(model; stop_time=1, Δt, verbose=false)

    ∇u = Field(∂x(u) + ∂y(v) + ∂z(w))
    ∫∇u = Field(Integral(∇u))

    run!(simulation)
    compute!(∫∇u)
    @test (@allowscalar ∫∇u[]) ≈ 0 atol=5*eps(FT)
end

test_boundary_conditions(C, FT, ArrayType) = (integer_bc(C, FT, ArrayType),
                                              float_bc(C, FT, ArrayType),
                                              irrational_bc(C, FT, ArrayType),
                                              array_bc(C, FT, ArrayType),
                                              simple_function_bc(C, FT, ArrayType),
                                              parameterized_function_bc(C, FT, ArrayType),
                                              field_dependent_function_bc(C, FT, ArrayType),
                                              parameterized_field_dependent_function_bc(C, FT, ArrayType),
                                              discrete_function_bc(C, FT, ArrayType),
                                              parameterized_discrete_function_bc(C, FT, ArrayType))

@testset "Boundary condition integration tests" begin
    @info "Testing boundary condition integration into NonhydrostaticModel..."

    @testset "Boundary condition regularization" begin
        @info "  Testing boundary condition regularization in NonhydrostaticModel constructor..."

        FT = Float64
        arch = first(archs)

        grid = RectilinearGrid(arch, FT, size=(1, 1, 1), extent=(1, π, 42), topology=(Bounded, Bounded, Bounded))

        u_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Value),
                                                        top    = simple_function_bc(Value),
                                                        north  = simple_function_bc(Value),
                                                        south  = simple_function_bc(Value),
                                                         east  = simple_function_bc(Open),
                                                         west  = simple_function_bc(Open))

        v_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Value),
                                                        top    = simple_function_bc(Value),
                                                        north  = simple_function_bc(Open),
                                                        south  = simple_function_bc(Open),
                                                         east  = simple_function_bc(Value),
                                                         west  = simple_function_bc(Value))


        w_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Open),
                                                        top    = simple_function_bc(Open),
                                                        north  = simple_function_bc(Value),
                                                        south  = simple_function_bc(Value),
                                                         east  = simple_function_bc(Value),
                                                         west  = simple_function_bc(Value))

        T_boundary_conditions = FieldBoundaryConditions(bottom = simple_function_bc(Value),
                                                        top    = simple_function_bc(Value),
                                                        north  = simple_function_bc(Value),
                                                        south  = simple_function_bc(Value),
                                                         east  = simple_function_bc(Value),
                                                         west  = simple_function_bc(Value))

        boundary_conditions = (u=u_boundary_conditions,
                               v=v_boundary_conditions,
                               w=w_boundary_conditions,
                               T=T_boundary_conditions)

        model = NonhydrostaticModel(grid = grid,
                                    boundary_conditions = boundary_conditions,
                                    buoyancy = SeawaterBuoyancy(),
                                    tracers = (:T, :S))

        @test location(model.velocities.u.boundary_conditions.bottom.condition) == (Face, Center, Nothing)
        @test location(model.velocities.u.boundary_conditions.top.condition)    == (Face, Center, Nothing)
        @test location(model.velocities.u.boundary_conditions.north.condition)  == (Face, Nothing, Center)
        @test location(model.velocities.u.boundary_conditions.south.condition)  == (Face, Nothing, Center)
        @test location(model.velocities.u.boundary_conditions.east.condition)   == (Nothing, Center, Center)
        @test location(model.velocities.u.boundary_conditions.west.condition)   == (Nothing, Center, Center)

        @test location(model.velocities.v.boundary_conditions.bottom.condition) == (Center, Face, Nothing)
        @test location(model.velocities.v.boundary_conditions.top.condition)    == (Center, Face, Nothing)
        @test location(model.velocities.v.boundary_conditions.north.condition)  == (Center, Nothing, Center)
        @test location(model.velocities.v.boundary_conditions.south.condition)  == (Center, Nothing, Center)
        @test location(model.velocities.v.boundary_conditions.east.condition)   == (Nothing, Face, Center)
        @test location(model.velocities.v.boundary_conditions.west.condition)   == (Nothing, Face, Center)

        @test location(model.velocities.w.boundary_conditions.bottom.condition) == (Center, Center, Nothing)
        @test location(model.velocities.w.boundary_conditions.top.condition)    == (Center, Center, Nothing)
        @test location(model.velocities.w.boundary_conditions.north.condition)  == (Center, Nothing, Face)
        @test location(model.velocities.w.boundary_conditions.south.condition)  == (Center, Nothing, Face)
        @test location(model.velocities.w.boundary_conditions.east.condition)   == (Nothing, Center, Face)
        @test location(model.velocities.w.boundary_conditions.west.condition)   == (Nothing, Center, Face)

        @test location(model.tracers.T.boundary_conditions.bottom.condition) == (Center, Center, Nothing)
        @test location(model.tracers.T.boundary_conditions.top.condition)    == (Center, Center, Nothing)
        @test location(model.tracers.T.boundary_conditions.north.condition)  == (Center, Nothing, Center)
        @test location(model.tracers.T.boundary_conditions.south.condition)  == (Center, Nothing, Center)
        @test location(model.tracers.T.boundary_conditions.east.condition)   == (Nothing, Center, Center)
        @test location(model.tracers.T.boundary_conditions.west.condition)   == (Nothing, Center, Center)
    end

    @testset "Boundary condition time-stepping works" begin
        for arch in archs, FT in (Float64,) #float_types

            topo = (Bounded, Bounded, Bounded)

            for C in (Gradient, Flux, Value), boundary_condition in test_boundary_conditions(C, FT, array_type(arch))
                @info "  Testing that time-stepping with $boundary_condition works [$(typeof(arch)), $FT]..."
                @test test_boundary_condition(arch, FT, NonhydrostaticModel, topo, :east, :T, boundary_condition)
                @test test_boundary_condition(arch, FT, NonhydrostaticModel, topo, :south, :T, boundary_condition)
                @test test_boundary_condition(arch, FT, NonhydrostaticModel, topo, :top, :T, boundary_condition)

                if (boundary_condition.condition isa ContinuousBoundaryFunction) && (arch isa GPU)
                    @info "Test skipped because of issue #4165"
                else
                    @test test_boundary_condition(arch, FT, HydrostaticFreeSurfaceModel, topo, :east, :T, boundary_condition)
                    @test test_boundary_condition(arch, FT, HydrostaticFreeSurfaceModel, topo, :south, :T, boundary_condition)
                    @test test_boundary_condition(arch, FT, HydrostaticFreeSurfaceModel, topo, :top, :T, boundary_condition)
                end
            end

            for boundary_condition in test_boundary_conditions(Open, FT, array_type(arch))
                @test test_boundary_condition(arch, FT, NonhydrostaticModel, topo, :east, :u, boundary_condition)
                @test test_boundary_condition(arch, FT, NonhydrostaticModel, topo, :south, :v, boundary_condition)
                @test test_boundary_condition(arch, FT, NonhydrostaticModel, topo, :top, :w, boundary_condition)

                if (boundary_condition.condition isa ContinuousBoundaryFunction) && (arch isa GPU)
                    @info "Test skipped because of issue #4165"
                else
                    @test test_boundary_condition(arch, FT, HydrostaticFreeSurfaceModel, topo, :east, :u, boundary_condition)
                    @test test_boundary_condition(arch, FT, HydrostaticFreeSurfaceModel, topo, :south, :v, boundary_condition)
                end
            end
        end
    end

    @testset "Budgets with Flux boundary conditions" begin
        for arch in archs
            A = typeof(arch)
            @info "  Testing budgets with Flux boundary conditions [$A]..."

            Lx = 0.3
            Ly = 0.4
            Lz = 0.5

            bottom(x, y) = 0
            ib = GridFittedBottom(bottom)
            grid_kw = (size = (2, 2, 2), x = (0, Lx), y = (0, Ly))

            rectilinear_grid(topology) = RectilinearGrid(arch; topology, z=(0, Lz), grid_kw...)
            immersed_rectilinear_grid(topology) = ImmersedBoundaryGrid(RectilinearGrid(arch; topology, z=(-Lz, Lz), grid_kw...), ib)
            immersed_active_rectilinear_grid(topology) = ImmersedBoundaryGrid(RectilinearGrid(arch; topology, z=(-Lz, Lz), grid_kw...), ib; active_cells_map = true)
            grids_to_test(topo) = [rectilinear_grid(topo), immersed_rectilinear_grid(topo), immersed_active_rectilinear_grid(topo)]

            for grid in grids_to_test((Periodic, Bounded, Bounded))
                for name in (:u, :c)
                    for (side, L) in zip((:north, :south, :top, :bottom), (Ly, Ly, Lz, Lz))
                        if grid isa ImmersedBoundaryGrid && side == :bottom
                            side = :immersed
                        end
                        @info "    Testing budgets with Flux boundary conditions [$(summary(grid)), $name, $side]..."
                        @test test_nonhydrostatic_flux_budget(grid, name, side, L)
                    end
                end
            end

            for grid in grids_to_test((Bounded, Periodic, Bounded))
                for name in (:v, :c)
                    for (side, L) in zip((:east, :west, :top, :bottom), (Lx, Lx, Lz, Lz))
                        if grid isa ImmersedBoundaryGrid && side == :bottom
                            side = :immersed
                        end
                        @info "    Testing budgets with Flux boundary conditions [$(summary(grid)), $name, $side]..."
                        @test test_nonhydrostatic_flux_budget(grid, name, side, L)
                    end
                end
            end

            # Omit ImmersedBoundaryGrid from vertically-periodic test
            grid = rectilinear_grid((Bounded, Bounded, Periodic))
            for name in (:w, :c)
                for (side, L) in zip((:east, :west, :north, :south), (Lx, Lx, Ly, Ly))
                    @info "    Testing budgets with Flux boundary conditions [$(summary(grid)), $name, $side]..."
                    @test test_nonhydrostatic_flux_budget(grid, name, side, L)
                end
            end
        end
    end

    @testset "Custom diffusivity boundary conditions" begin
        for arch in archs, FT in (Float64,) #float_types
            A = typeof(arch)
            @info "  Testing flux budgets with diffusivity boundary conditions [$A, $FT]..."
            @test fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
        end
    end

    @testset "Open boundary conditions" begin
        for arch in archs, FT in (Float64,) #float_types
            A = typeof(arch)
            @info "  Testing open boundary conditions [$A, $FT]..."
            test_perturbation_advection_open_boundary_conditions(arch, FT)

            # Only PerturbationAdvection OpenBoundaryCondition
            U₀ = 1
            inflow_timescale = 1e-1
            outflow_timescale = Inf

            u_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(U₀; scheme = PerturbationAdvection(inflow_timescale, outflow_timescale)),
                                            east = OpenBoundaryCondition(U₀; scheme = PerturbationAdvection(inflow_timescale, outflow_timescale)))
            boundary_conditions = (; u = u_bcs)
            test_open_boundary_condition_mass_conservation(arch, FT, boundary_conditions)
        end
    end
end
