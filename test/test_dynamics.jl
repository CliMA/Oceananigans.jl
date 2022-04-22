include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures: viscosity, ThreeDimensionalFormulation, HorizontalFormulation, VerticalFormulation
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom, mask_immersed_field!

function relative_error(u_num, u, time)
    u_ans = Field(location(u_num), u_num.grid)
    set!(u_ans, (x, y, z) -> u(x, y, z, time))
    return mean((interior(u_num) .- interior(u_ans)).^2 ) / mean(interior(u_ans).^2)
end

function test_diffusion_simple(fieldname, timestepper, time_discretization)
    model = NonhydrostaticModel(; timestepper,
                                  grid = RectilinearGrid(CPU(), size=(1, 1, 16), extent=(1, 1, 1)),
                                  closure = ScalarDiffusivity(time_discretization, ν=1, κ=1),
                                  coriolis = nothing,
                                  tracers = :c,
                                  buoyancy = nothing)

    value = π
    field = fields(model)[fieldname]
    interior(field) .= value
    update_state!(model)

    [time_step!(model, 1) for n = 1:10]

    field_data = interior(field)
    return !any(@. !isapprox(value, field_data))
end

function test_diffusion_budget(fieldname, field, model, κ, Δ, order=2)
    init_mean = mean(interior(field))
    update_state!(model)
    Δt = 1e-4 * Δ^order / κ # small to suppress non-conservative time-discretization error

    for n = 1:10
        time_step!(model, Δt)
    end

    final_mean = mean(interior(field))
    @info @sprintf("    Initial <%s>: %.16f, final <%s>: %.16f, final - initial: %.4e",
                   fieldname, init_mean, fieldname, final_mean, final_mean - init_mean)

    return isapprox(init_mean, final_mean)
end

function test_ScalarDiffusivity_budget(fieldname, model)
    set!(model; u=0, v=0, w=0, c=0)
    set!(model; Dict(fieldname => (x, y, z) -> rand())...)
    field = fields(model)[fieldname]
    ν = viscosity(model.closure, nothing)
    return test_diffusion_budget(fieldname, field, model, ν, model.grid.Δzᵃᵃᶜ)
end

function test_ScalarBiharmonicDiffusivity_budget(fieldname, model)
    set!(model; u=0, v=0, w=0, c=0)
    set!(model; Dict(fieldname => (x, y, z) -> rand())...)
    field = fields(model)[fieldname]
    return test_diffusion_budget(fieldname, field, model, model.closure.ν, model.grid.Δzᵃᵃᶜ, 4)
end

function test_diffusion_cosine(fieldname, grid, closure, ξ, tracers=:c) 
    κ, m = 1, 2 # diffusivity and cosine wavenumber
    model = NonhydrostaticModel(; grid, closure, tracers, buoyancy=nothing)
    field = fields(model)[fieldname]
    field .= cos.(m * ξ)
    update_state!(model)

    # Step forward with small time-step relative to diff. time-scale
    Δt = 1e-6 * grid.Lz^2 / κ
    for n = 1:5
        time_step!(model, Δt)
    end

    diffusing_cosine(ξ, t, κ, m) = exp(-κ * m^2 * t) * cos(m * ξ)
    analytical_solution = Field(location(field), grid)
    analytical_solution .= diffusing_cosine.(ξ, model.clock.time, κ, m)

    return isapprox(field, analytical_solution, atol=1e-6, rtol=1e-6)
end

function test_immersed_diffusion(Nz, z, time_discretization)
    closure         = ScalarDiffusivity(time_discretization, κ = 1)
    underlying_grid = RectilinearGrid(size=Nz, z=z, topology=(Flat, Flat, Bounded))
    grid            = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> 0))
    
    Δz_min = minimum(underlying_grid.Δzᵃᵃᶜ)
    model_kwargs = (tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

    full_model     = HydrostaticFreeSurfaceModel(; grid=underlying_grid, closure, model_kwargs...)
    immersed_model = HydrostaticFreeSurfaceModel(; grid=grid, closure, model_kwargs...)

    initial_temperature(x, y, z) = exp(-z^2 / 0.02)
    set!(full_model,     c=initial_temperature)
    set!(immersed_model, c=initial_temperature)

    Δt = Δz_min^2 / closure.κ * 1e-1

    for n = 1:100
        time_step!(full_model,     Δt)
        time_step!(immersed_model, Δt)
    end

    half = Int(grid.Nz/2 + 1)
    c_full     = interior(full_model.tracers.c)[1, 1, half:end]
    c_immersed = interior(immersed_model.tracers.c)[1, 1, half:end]

    return all(c_full .≈ c_immersed)
end

function test_3D_immersed_diffusion(Nz, z, time_discretization)
    closure = VerticalScalarDiffusivity(time_discretization, ν = 1, κ = 1)

    # Bathymetry
    b, l, m, u, t = -0.5, -0.2, 0, 0.2, 0.5

    bathymetry = [b b b b b b b b b 
                  b l l l l l l l b
                  b l m m m m m l b
                  b l m u u u m l b
                  b l m u t u m l b
                  b l m u u u m l b
                  b l m m m m m l b
                  b l l l l l l l b
                  b b b b b b b b b]

    underlying_grid = RectilinearGrid(size=(9, 9, Nz), x=(0, 1), y=(0, 1), z=z, topology=(Periodic, Periodic, Bounded))
    grid            = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))
    
    Δz_min = minimum(grid.grid.Δzᵃᵃᶜ)
    model_kwargs = (tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

    full_model     = HydrostaticFreeSurfaceModel(; grid=underlying_grid, closure=closure, model_kwargs...)
    immersed_model = HydrostaticFreeSurfaceModel(; grid=grid, closure=closure, model_kwargs...)

    initial_temperature(x, y, z) = exp(-z^2 / 0.02)
    set!(full_model,     c=initial_temperature)
    set!(immersed_model, c=initial_temperature)

    Δt = Δz_min^2 / closure.κ * 1e-1

    for n = 1:100
        time_step!(full_model    , Δt)
        time_step!(immersed_model, Δt)
    end

    half   = Int(grid.Nz/2 + 1)

    assesment = Array{Bool}(undef, 4)

    c_full       = interior(full_model.tracers.c)[3, 3:7, half:end]
    c_immersed   = interior(immersed_model.tracers.c)[3, 3:7, half:end]
    assesment[1] = all(c_full .≈ c_immersed)

    c_full       = interior(full_model.tracers.c)[3:7, 3, half:end]
    c_immersed   = interior(immersed_model.tracers.c)[3:7, 3, half:end]
    assesment[2] = all(c_full .≈ c_immersed)

    c_full       = interior(full_model.tracers.c)[7, 3:7, half:end]
    c_immersed   = interior(immersed_model.tracers.c)[7, 3:7, half:end]
    assesment[3] = all(c_full .≈ c_immersed)

    c_full       = interior(full_model.tracers.c)[3:7, 7, half:end]
    c_immersed   = interior(immersed_model.tracers.c)[3:7, 7, half:end]
    assesment[4] = all(c_full .≈ c_immersed)

    return all(assesment)
end

function passive_tracer_advection_test(timestepper; N=128, κ=1e-12, Nt=100, background_velocity_field=false)
    L, U, V = 1.0, 0.5, 0.8
    δ, x₀, y₀ = L/15, L/2, L/2

    Δt = 0.05 * L/N / sqrt(U^2 + V^2)

    T(x, y, z, t) = exp( -((x - U*t - x₀)^2 + (y - V*t - y₀)^2) / (2*δ^2) )
    u₀(x, y, z) = U
    v₀(x, y, z) = V
    T₀(x, y, z) = T(x, y, z, 0)
    background_fields = Dict()

    if background_velocity_field
        background_fields[:u] = (x, y, z, t) -> U
        background_fields[:v] = (x, y, z, t) -> V
        u₀ = 0
        v₀ = 0
    end

    background_fields = NamedTuple{Tuple(keys(background_fields))}(values(background_fields))

    grid = RectilinearGrid(size=(N, N, 2), extent=(L, L, L))
    closure = ScalarDiffusivity(ν=κ, κ=κ)
    model = NonhydrostaticModel(; grid, closure, timestepper,
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S),
                                background_fields=background_fields)

    set!(model, u=u₀, v=v₀, T=T₀)
    [time_step!(model, Δt) for n = 1:Nt]

    # Error tolerance is a bit arbitrary
    return relative_error(model.tracers.T, T, model.clock.time) < 1e-4
end

"""
Taylor-Green vortex test
See: https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex#Taylor%E2%80%93Green_vortex_solution
     and p. 310 of "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Application"
     by Hesthaven & Warburton.
"""
function taylor_green_vortex_test(arch, timestepper, time_discretization; FT=Float64, N=64, Nt=10)
    Nx, Ny, Nz = N, N, 2
    Lx, Ly, Lz = 1, 1, 1
    ν = 1

    # Choose a very small time step as we are diffusion-limited in this test: Δt ≤ Δx² / 2ν
    Δx = Lx / Nx
    Δt = (1/10π) * Δx^2 / ν

    # Taylor-Green vortex analytic solution.
    @inline u(x, y, z, t) = -sin(2π*y) * exp(-4π^2 * ν * t)
    @inline v(x, y, z, t) =  sin(2π*x) * exp(-4π^2 * ν * t)

    model = NonhydrostaticModel(
         timestepper = timestepper,
                grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)),
             closure = ScalarDiffusivity(time_discretization, ThreeDimensionalFormulation(), FT, ν=1),
             tracers = nothing,
            buoyancy = nothing)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    set!(model, u=u₀, v=v₀)

    for n in 1:Nt
        time_step!(model, Δt)
    end

    xF, yC, zC = nodes(model.velocities.u, reshape=true)
    xC, yF, zC = nodes(model.velocities.v, reshape=true)

    t = model.clock.time
    i = model.clock.iteration

    # Calculate relative error between model and analytic solutions for u and v.
    u_rel_err = abs.((interior(model.velocities.u) .- u.(xF, yC, zC, t)) ./ u.(xF, yC, zC, t))
    u_rel_err_avg = mean(u_rel_err)
    u_rel_err_max = maximum(u_rel_err)

    v_rel_err = abs.((interior(model.velocities.v) .- v.(xC, yF, zC, t)) ./ v.(xC, yF, zC, t))
    v_rel_err_avg = mean(v_rel_err)
    v_rel_err_max = maximum(v_rel_err)

    @info "Taylor-Green vortex test [$arch, $FT, Nx=Ny=$N, Nt=$Nt]: " *
          @sprintf("Δu: (avg=%6.3g, max=%6.3g), Δv: (avg=%6.3g, max=%6.3g)",
                   u_rel_err_avg, u_rel_err_max, v_rel_err_avg, v_rel_err_max)

    return u_rel_err_max < 5e-6 && v_rel_err_max < 5e-6
end

function stratified_fluid_remains_at_rest_with_tilted_gravity_buoyancy_tracer(arch, FT; N=32, L=2000, θ=60, N²=1e-5)
    topo = (Periodic, Bounded, Bounded)
    grid = RectilinearGrid(arch, FT, topology=topo, size=(1, N, N), extent=(L, L, L))

    g̃ = (0, sind(θ), cosd(θ))
    buoyancy = Buoyancy(model=BuoyancyTracer(), gravity_unit_vector=g̃)

    y_bc = GradientBoundaryCondition(N² * g̃[2])
    z_bc = GradientBoundaryCondition(N² * g̃[3])
    b_bcs = FieldBoundaryConditions(bottom=z_bc, top=z_bc, south=y_bc, north=y_bc)

    model = NonhydrostaticModel(; grid, buoyancy,
                                tracers = :b,
                                closure = nothing,
                                boundary_conditions = (; b=b_bcs))

    b₀(x, y, z) = N² * (x*g̃[1] + y*g̃[2] + z*g̃[3])
    set!(model, b=b₀)

    simulation = Simulation(model, Δt=10minutes, stop_time=1hour)
    run!(simulation)

    @compute ∂y_b = Field(∂y(model.tracers.b))
    @compute ∂z_b = Field(∂z(model.tracers.b))

    mean_∂y_b = mean(∂y_b)
    mean_∂z_b = mean(∂z_b)

    Δ_y = N² * g̃[2] - mean_∂y_b
    Δ_z = N² * g̃[3] - mean_∂z_b

    @info "N² * g̃[2] = $(N² * g̃[2]), mean(∂y_b) = $(mean_∂y_b), Δ = $Δ_y at t = $(prettytime(model.clock.time)) with θ=$(θ)°"
    @info "N² * g̃[3] = $(N² * g̃[3]), mean(∂z_b) = $(mean_∂z_b), Δ = $Δ_z at t = $(prettytime(model.clock.time)) with θ=$(θ)°"

    @test N² * g̃[2] ≈ mean(∂y_b)
    @test N² * g̃[3] ≈ mean(∂z_b)

    CUDA.@allowscalar begin
        @test all(N² * g̃[2] .≈ interior(∂y_b))
        @test all(N² * g̃[3] .≈ interior(∂z_b))
    end

    return nothing
end

function stratified_fluid_remains_at_rest_with_tilted_gravity_temperature_tracer(arch, FT; N=32, L=2000, θ=60, N²=1e-5)
    topo = (Periodic, Bounded, Bounded)
    grid = RectilinearGrid(arch, FT, topology=topo, size=(1, N, N), extent=(L, L, L))

    g̃ = (0, sind(θ), cosd(θ))
    buoyancy = Buoyancy(model=SeawaterBuoyancy(), gravity_unit_vector=g̃)

    α  = buoyancy.model.equation_of_state.thermal_expansion
    g₀ = buoyancy.model.gravitational_acceleration
    ∂T∂z = N² / (g₀ * α)

    y_bc = GradientBoundaryCondition(∂T∂z * g̃[2])
    z_bc = GradientBoundaryCondition(∂T∂z * g̃[3])
    T_bcs = FieldBoundaryConditions(bottom=z_bc, top=z_bc, south=y_bc, north=y_bc)

    model = NonhydrostaticModel(; grid, buoyancy,
                                tracers = (:T, :S),
                                closure = nothing,
                                boundary_conditions = (; T=T_bcs))

    T₀(x, y, z) = ∂T∂z * (x*g̃[1] + y*g̃[2] + z*g̃[3])
    set!(model, T=T₀)

    simulation = Simulation(model, Δt=10minute, stop_time=1hour)
    run!(simulation)

    @compute ∂y_T = Field(∂y(model.tracers.T))
    @compute ∂z_T = Field(∂z(model.tracers.T))

    mean_∂y_T = mean(∂y_T)
    mean_∂z_T = mean(∂z_T)

    Δ_y = ∂T∂z * g̃[2] - mean_∂y_T
    Δ_z = ∂T∂z * g̃[3] - mean_∂z_T

    @info "∂T∂z * g̃[2] = $(∂T∂z * g̃[2]), mean(∂y_T) = $(mean_∂y_T), Δ = $Δ_y at t = $(prettytime(model.clock.time)) with θ=$(θ)°"
    @info "∂T∂z * g̃[3] = $(∂T∂z * g̃[3]), mean(∂z_T) = $(mean_∂z_T), Δ = $Δ_z at t = $(prettytime(model.clock.time)) with θ=$(θ)°"

    @test ∂T∂z * g̃[2] ≈ mean(∂y_T)
    @test ∂T∂z * g̃[3] ≈ mean(∂z_T)

    CUDA.@allowscalar begin
        @test all(∂T∂z * g̃[2] .≈ interior(∂y_T))
        @test all(∂T∂z * g̃[3] .≈ interior(∂z_T))
    end

    return nothing
end

function inertial_oscillations_work_with_rotation_in_different_axis(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(), topology=(Flat, Flat, Flat))
    f₀ = 1
    ū = 1
    Δt = 1e-3
    T_inertial = 2π/f₀
    stop_time = T_inertial / 2
    zcoriolis = FPlane(f=f₀)
    xcoriolis = ConstantCartesianCoriolis(f=f₀, rotation_axis=(1,0,0))

    model_x =  NonhydrostaticModel(; grid, buoyancy=nothing, tracers=nothing, closure=nothing,
                                   timestepper = :RungeKutta3, coriolis = xcoriolis)
    set!(model_x, v=ū)
    simulation_x = Simulation(model_x, Δt=Δt, stop_time=stop_time)
    run!(simulation_x)

    model_z =  NonhydrostaticModel(; grid, buoyancy=nothing, tracers=nothing, closure=nothing,
                                   timestepper = :RungeKutta3, coriolis = zcoriolis)
    set!(model_z, u=ū)
    simulation_z = Simulation(model_z, Δt=Δt, stop_time=stop_time)
    run!(simulation_z)

    u_x = model_x.velocities.u[1, 1, 1]
    v_x = model_x.velocities.v[1, 1, 1]
    w_x = model_x.velocities.w[1, 1, 1]

    u_z = model_z.velocities.u[1, 1, 1]
    v_z = model_z.velocities.v[1, 1, 1]
    w_z = model_z.velocities.w[1, 1, 1]

    @test w_z == 0
    @test u_x == 0

    @test √(v_x^2 + w_x^2) ≈ 1
    @test √(u_z^2 + v_z^2) ≈ 1

    @test u_z ≈ v_x
    @test v_z ≈ w_x

    return nothing
end

timesteppers = (:QuasiAdamsBashforth2, :RungeKutta3)

@testset "Dynamics" begin
    @info "Testing dynamics..."

    @testset "Simple diffusion" begin
        @info "  Testing simple diffusion..."
        for fieldname in (:u, :v, :c), timestepper in timesteppers
            for time_discretization in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                @test test_diffusion_simple(fieldname, timestepper, time_discretization)
            end
        end
    end

    @testset "Budgets in isotropic diffusion" begin
        @info "  Testing model budgets with isotropic diffusion..."
        for timestepper in timesteppers
            for topology in ((Periodic, Periodic, Periodic),
                             (Periodic, Periodic, Bounded),
                             (Periodic, Bounded, Bounded),
                             (Bounded, Bounded, Bounded))
                
                # Can't use implicit time-stepping in vertically-periodic domains right now
                if topology !== (Periodic, Periodic, Periodic)
                    time_discretizations = (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                else
                    time_discretizations = tuple(ExplicitTimeDiscretization())
                end

                for time_discretization in time_discretizations
                    for closurename in [ScalarDiffusivity, VerticalScalarDiffusivity, HorizontalScalarDiffusivity]

                        # VerticallyImplicitTimeDiscretization is not supported for HorizontalScalarDiffusivity
                        (closurename == HorizontalScalarDiffusivity && time_discretization == VerticallyImplicitTimeDiscretization()) && continue
                        
                        closure = closurename(time_discretization, ν=1, κ=1)
                    
                        fieldnames = [:c]
                        topology[1] === Periodic && push!(fieldnames, :u)
                        topology[2] === Periodic && push!(fieldnames, :v)
                        topology[3] === Periodic && push!(fieldnames, :w)

                        grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=topology)

                        model = NonhydrostaticModel(timestepper = timestepper,
                                                    grid = grid,
                                                    closure = closure,
                                                    tracers = :c,
                                                    coriolis = nothing,
                                                    buoyancy = nothing)

                        td = typeof(time_discretization).name.wrapper

                        for fieldname in fieldnames
                            @info "    [$timestepper, $td, $closurename] " *
                                  "Testing $fieldname budget in a $topology domain with scalar diffusion..."
                            @test test_ScalarDiffusivity_budget(fieldname, model)
                        end
                    end
                end
            end
        end
    end

    @testset "Budgets in biharmonic diffusion" begin
        @info "  Testing model budgets with biharmonic diffusion..."
        for timestepper in timesteppers
            for topology in ((Periodic, Periodic, Periodic),
                             (Periodic, Periodic, Bounded),
                             (Periodic, Bounded, Bounded),
                             (Bounded, Bounded, Bounded))

                fieldnames = [:c]

                topology[1] === Periodic && push!(fieldnames, :u)
                topology[2] === Periodic && push!(fieldnames, :v)
                topology[3] === Periodic && push!(fieldnames, :w)

                grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1), topology=topology)

                for formulation in (ThreeDimensionalFormulation(), HorizontalFormulation(), VerticalFormulation())
                    model = NonhydrostaticModel(timestepper = timestepper,
                                                grid = grid,
                                                closure = ScalarBiharmonicDiffusivity(formulation, ν=1, κ=1),
                                                coriolis = nothing,
                                                tracers = :c,
                                                buoyancy = nothing)

                    for fieldname in fieldnames
                        @info "    [$timestepper] Testing $fieldname budget in a $topology domain " *
                              "with biharmonic diffusion and $formulation..."
                        @test test_ScalarBiharmonicDiffusivity_budget(fieldname, model)
                    end
                end
            end
        end
    end

    @testset "Diffusion of a cosine" begin
        for arch in [CPU()] # Need some work to make these run on GPU
            N, L = 128, π/2
            grid = RectilinearGrid(arch, size=N, x=(0, L), topology=(Bounded, Flat, Flat))

            # Derive coordinates
            x = xnodes(Center, grid, reshape=true)
            y = permutedims(x, (2, 1, 3))
            z = permutedims(x, (2, 3, 1))

            # Build a huge list of tests.
            grids = []
            coords = []
            closures = []
            fieldnames = []

            scalar_diffusivity = ScalarDiffusivity(ν=1, κ=1) 
            implicit_scalar_diffusivity = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1, κ=1) 
            vertical_scalar_diffusivity = VerticalScalarDiffusivity(ν=1, κ=1)
            implicit_vertical_scalar_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1, κ=1)
            
            # Cases on Non-Flat grids with `ScalarDiffusivity`
            append!(coords, [x, y, z])
            append!(fieldnames, [(:v, :w, :c), (:u, :w, :c), (:u, :v, :c)])
            append!(closures, [scalar_diffusivity for i = 1:3])
            append!(grids, [RectilinearGrid(arch, size=(N, 1, 1), x=(0, L), y=(0, 1), z=(0, 1), topology=(Bounded, Periodic, Periodic)),
                            RectilinearGrid(arch, size=(1, N, 1), x=(0, 1), y=(0, L), z=(0, 1), topology=(Periodic, Bounded, Periodic)),
                            RectilinearGrid(arch, size=(1, 1, N), x=(0, 1), y=(0, 1), z=(0, L), topology=(Periodic, Periodic, Bounded))])

            # Basic cases on Flat grids with alternate formulations of `ScalarDiffusivity`
            append!(coords, [x, y, z])
            append!(fieldnames, [(:v, :w, :c), (:u, :w, :c), (:u, :v, :c)])
            append!(closures, [HorizontalScalarDiffusivity(ν=1, κ=1),
                               HorizontalScalarDiffusivity(ν=1, κ=1),
                               VerticalScalarDiffusivity(ν=1, κ=1)])
            append!(grids, [RectilinearGrid(arch, size=N, x=(0, L), topology=(Bounded, Flat, Flat)),
                            RectilinearGrid(arch, size=N, y=(0, L), topology=(Flat, Bounded, Flat)),
                            RectilinearGrid(arch, size=N, z=(0, L), topology=(Flat, Flat, Bounded))])

            # VerticallyImplicitTimeDiscretization cases
            append!(coords, [z, z])
            append!(fieldnames, [(:u, :v, :c) for i = 1:2])
            append!(closures, [ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1, κ=1),
                               VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1, κ=1)])
            append!(grids, [RectilinearGrid(arch, size=N, z=(0, L), topology=(Flat, Flat, Bounded)),
                            RectilinearGrid(arch, size=N, z=(0, L), topology=(Flat, Flat, Bounded))])

            # Closure tuple.
            closure_tuple = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1/2, κ=1/2),
                             VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1/2, κ=1/2))
            push!(coords, z)
            push!(fieldnames, fieldnames[end])
            push!(closures, closure_tuple)
            push!(grids, grids[end])
                               
            # Immersed grid cases
            immersed_vertical_grid = ImmersedBoundaryGrid(RectilinearGrid(arch,
                                                                          size = (1, 1, 2N),
                                                                          x = (0, 1),
                                                                          y = (0, 1),
                                                                          z = (0, 2L),
                                                                          topology = (Periodic, Periodic, Bounded)),
                                                          GridFittedBottom((x, y) -> L))

            z_immersed = znodes(Center, immersed_vertical_grid, reshape=true)

            append!(coords, [z_immersed, z_immersed, z_immersed, z_immersed])
            append!(fieldnames, [(:u, :v, :c) for i = 1:4])
            append!(closures, [scalar_diffusivity,
                               implicit_scalar_diffusivity,
                               vertical_scalar_diffusivity,
                               implicit_vertical_scalar_diffusivity])
            append!(grids, [immersed_vertical_grid for i = 1:4])

            # Stretched grid cases
            stretched_z_grid = RectilinearGrid(arch, size=N, z=center_clustered_coord(N, L, 0), topology=(Flat, Flat, Bounded))
            stretched_immersed_z_grid =
            ImmersedBoundaryGrid(RectilinearGrid(arch,
                                                 size = (1, 1, 2N),
                                                 x = (0, 1),
                                                 y = (0, 1),
                                                 z = center_clustered_coord(2N, 2L, 0),
                                                 topology = (Periodic, Periodic, Bounded)),
                                     GridFittedBottom((x, y) -> L))

            stretched_grids = [stretched_z_grid, stretched_z_grid, stretched_immersed_z_grid, stretched_immersed_z_grid]
            append!(coords, [znodes(Center, grid, reshape=true) for grid in stretched_grids])
            append!(fieldnames, [(:u, :v, :c) for i = 1:4])
            append!(closures, [vertical_scalar_diffusivity,
                               implicit_vertical_scalar_diffusivity,
                               vertical_scalar_diffusivity,
                               implicit_vertical_scalar_diffusivity])
            append!(grids, stretched_grids)
                            
            # Run the tests
            for case = 1:length(grids)
                closure = closures[case]
                grid = grids[case]
                coord = coords[case]

                for fieldname in fieldnames[case]
                    @info "  Testing diffusion of a cosine [$fieldname, $(summary(closure)), $(summary(grid))]..."
                    @test test_diffusion_cosine(fieldname, grid, closure, coord)
                end
            end
        end
    end

    @testset "Gaussian immersed diffusion" begin
        for time_discretization in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())

            Nz, Lz, z₀ = 128, 1, -0.5
            z_regular = (z₀, Lz + z₀)
            z_stretch = center_clustered_coord(Nz, Lz, z₀)

            for z_coord = (z_regular, z_stretch)
                @info "  Testing gaussian immersed diffusion for " *
                      "[$time_discretization, $(z_coord isa Tuple ? "regular" : "stretched")]..."
                @test test_immersed_diffusion(Nz, z_coord, time_discretization)
                @test test_3D_immersed_diffusion(Nz, z_coord, time_discretization)
            end
        end
    end

    @testset "Passive tracer advection" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            @info "  Testing passive tracer advection [$timestepper]..."
            @test passive_tracer_advection_test(timestepper)
        end
    end

    @testset "Internal wave" begin
        include("test_internal_wave_dynamics.jl")

        Nx = Nz = 128
        Lx = Lz = 2π

        # Regular grid with no flat dimension
        y_periodic_regular_grid = RectilinearGrid(topology=(Periodic, Periodic, Bounded),
                                                         size=(Nx, 1, Nz), x=(0, Lx), y=(0, Lx), z=(-Lz, 0))

        # Regular grid with a flat y-dimension
        y_flat_regular_grid = RectilinearGrid(topology=(Periodic, Flat, Bounded),
                                                     size=(Nx, Nz), x=(0, Lx), z=(-Lz, 0))

        # Vertically stretched grid with regular spacing and no flat dimension
        z_faces = collect(znodes(Face, y_periodic_regular_grid))
        y_periodic_regularly_spaced_vertically_stretched_grid = RectilinearGrid(topology=(Periodic, Periodic, Bounded),
                                                                                size=(Nx, 1, Nz),
                                                                                x=(0, Lx), y=(0, Lx), z=z_faces)

        # Vertically stretched grid with regular spacing and no flat dimension
        y_flat_regularly_spaced_vertically_stretched_grid = RectilinearGrid(topology=(Periodic, Flat, Bounded),
                                                                            size=(Nx, Nz), x=(0, Lx), z=z_faces)

        solution, kwargs, background_fields, Δt, σ = internal_wave_solution(L=Lx, background_stratification=false)

        test_grids = (y_periodic_regular_grid,
                      y_flat_regular_grid,
                      y_periodic_regularly_spaced_vertically_stretched_grid,
                      y_flat_regularly_spaced_vertically_stretched_grid)

        @testset "Internal wave with HydrostaticFreeSurfaceModel" begin
            for grid in test_grids
                grid_name = typeof(grid).name.wrapper
                topo = topology(grid)

                # Choose gravitational acceleration so that σ_surface = sqrt(g * Lx) = 10σ
                gravitational_acceleration = (10σ)^2 / Lx
                free_surface = ImplicitFreeSurface(; gravitational_acceleration)
                model = HydrostaticFreeSurfaceModel(; free_surface, grid, kwargs...)

                @info "  Testing internal wave [HydrostaticFreeSurfaceModel, $grid_name, $topo]..."
                internal_wave_dynamics_test(model, solution, Δt)
            end
        end

        @testset "Internal wave with NonhydrostaticModel" begin
            for grid in test_grids
                grid_name = typeof(grid).name.wrapper
                topo = topology(grid)

                model = NonhydrostaticModel(; grid=grid, kwargs...)

                @info "  Testing internal wave [NonhydrostaticModel, $grid_name, $topo]..."
                internal_wave_dynamics_test(model, solution, Δt)
            end
        end
    end

    @testset "Taylor-Green vortex" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            for time_discretization in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
                td = typeof(time_discretization).name.wrapper
                @info "  Testing Taylor-Green vortex [$timestepper, $td]..."
                @test taylor_green_vortex_test(CPU(), timestepper, time_discretization)
            end
        end
    end

    @testset "Background fields" begin
        for timestepper in (:QuasiAdamsBashforth2,) #timesteppers
            @info "  Testing dynamics with background fields [$timestepper]..."
            @test_skip passive_tracer_advection_test(timestepper, background_velocity_field=true)
                        
            Nx = Nz = 128
            Lx = Lz = 2π

            # Regular grid with no flat dimension
            y_periodic_regular_grid = RectilinearGrid(topology=(Periodic, Periodic, Bounded),
                                                             size=(Nx, 1, Nz), x=(0, Lx), y=(0, Lx), z=(-Lz, 0))
                        
            solution, kwargs, background_fields, Δt, σ = internal_wave_solution(L=Lx, background_stratification=true)

            model = NonhydrostaticModel(; grid=y_periodic_regular_grid, background_fields=background_fields, kwargs...)
            internal_wave_dynamics_test(model, solution, Δt)
        end
    end

    @testset "Background rotation about arbitrary axis" begin
        for arch in archs
            @info "  Testing background rotation about arbitrary axis [$(typeof(arch))]..."
            inertial_oscillations_work_with_rotation_in_different_axis(arch, Float64)
        end
    end
end

