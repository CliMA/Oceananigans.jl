using Oceananigans.BoundaryConditions: ContinuousBoundaryFunction

function test_boundary_condition(arch, FT, topo, side, field_name, boundary_condition)
    grid = RegularCartesianGrid(FT, size=(1, 1, 1), extent=(1, π, 42), topology=topo)

    boundary_condition_kwarg = Dict(side => boundary_condition)

    field_boundary_conditions = TracerBoundaryConditions(grid; boundary_condition_kwarg...)

    bcs = NamedTuple{(field_name,)}((field_boundary_conditions,))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT,
                                boundary_conditions=bcs)

    success = try
        time_step!(model, 1e-16, euler=true)
        true
    catch err
        @warn "test_boundary_condition errored with " * sprint(showerror, err)
        false
    end

    return success
end

function test_flux_budget(arch, FT, fldname)
    N, κ, Lz = 16, 1, 0.7
    grid = RegularCartesianGrid(FT, size=(N, N, N), extent=(1, 1, Lz))

    bottom_flux = FT(0.3)
    flux_bc = BoundaryCondition(Flux, bottom_flux)

    if fldname == :u
        field_bcs = UVelocityBoundaryConditions(grid, bottom=flux_bc)
    elseif fldname == :v
        field_bcs = VVelocityBoundaryConditions(grid, bottom=flux_bc)
    else
        field_bcs = TracerBoundaryConditions(grid, bottom=flux_bc)
    end

    model_bcs = NamedTuple{(fldname,)}((field_bcs,))

    closure = IsotropicDiffusivity(FT, ν=κ, κ=κ)
    model = IncompressibleModel(grid=grid, closure=closure, architecture=arch, tracers=(:T, :S),
                                float_type=FT, buoyancy=nothing, boundary_conditions=model_bcs)

    field = get_model_field(fldname, model)
    @. field.data = 0

    τκ = Lz^2 / κ   # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 10         # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    return mean(interior(field)) ≈ bottom_flux * model.clock.time / Lz
end

function fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
    Lz = 1
    κ₀ = FT(exp(-3))
    bz = FT(π)
    flux = - κ₀ * bz

    grid = RegularCartesianGrid(FT, size=(16, 16, 16), extent=(1, 1, Lz))

    buoyancy_bcs = TracerBoundaryConditions(grid, bottom=BoundaryCondition(Gradient, bz))
    κₑ_bcs = DiffusivityBoundaryConditions(grid, bottom=BoundaryCondition(Value, κ₀))
    model_bcs = (b=buoyancy_bcs, κₑ=(b=κₑ_bcs,))

    model = IncompressibleModel(
        grid=grid, architecture=arch, float_type=FT, tracers=:b, buoyancy=BuoyancyTracer(),
        closure=AnisotropicMinimumDissipation(), boundary_conditions=model_bcs
    )

    b₀(x, y, z) = z * bz
    set!(model, b=b₀)

    b = model.tracers.b
    mean_b₀ = mean(interior(b))

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
    return isapprox(mean(interior(b)) - mean_b₀, flux * model.clock.time / Lz, atol=1e-6)
end

discrete_func(i, j, grid, clock, model_fields) = - model_fields.u[i, j, grid.Nz]
parameterized_discrete_func(i, j, grid, clock, model_fields, p) = - p.μ * model_fields.u[i, j, grid.Nz]

parameterized_fun(ξ, η, t, p) = p.μ * cos(p.ω * t)
field_dependent_fun(ξ, η, t, u, v, w) = - w * sqrt(u^2 + v^2) 
exploding_fun(ξ, η, t, T, S, p) = - p.μ * cosh(S - p.S0) * exp((T - p.T0) / p.λ)

# Many, many bc
test_boundary_conditions(C, FT=Float64, ArrayType=Array) = [
    BoundaryCondition(C, 1),
    BoundaryCondition(C, FT(π)),
    BoundaryCondition(C, π),
    BoundaryCondition(C, ArrayType(rand(FT, 1, 1))),
    BoundaryCondition(C, (ξ, η, t) -> exp(ξ) * cos(η) * sin(t)),
    BoundaryCondition(C, parameterized_fun, field_dependencies=(:u, :v, :w)),
    BoundaryCondition(C, field_dependent, parameters=(μ=0.1, ω=2π)),
    BoundaryCondition(C, exploding_fun, field_dependencies=(:T, :S), parameters=(S0=35, T0=100, μ=2π, λ=FT(2))),
    BoundaryCondition(C, discrete_func, discrete_form=true),
    BoundaryCondition(C, parameterized_discrete_func, discrete_form=true, parameters=(μ=0.1,))
   ]

@testset "Time stepping with boundary conditions" begin
    @info "Testing stepping with boundary conditions..."

    @testset "Boundary condition instatiation and time-stepping" begin
        for arch in archs
    	    ArrayType = array_type(arch)

            for FT in (Float64,) #float_types

                @info "  Testing boundary condition instantiation and time-stepping [$(typeof(arch)), $FT]..."

                if arch isa GPU
                    topo = (Periodic, Bounded, Bounded)
                    sides = (:south, :north, :top, :bottom)
                else
                    topo = (Bounded, Bounded, Bounded)
                    sides = (:east, :west, :south, :north, :top, :bottom)
                end

                for C in (Gradient, Flux, Value), boundary_condition in test_boundary_conditions(C, FT, ArrayType)
                    for side in sides
                        @test test_boundary_condition(arch, FT, topo, side, :T, boundary_condition)
                    end
                end

                for boundary_condition in test_boundary_conditions(NormalFlow, FT, ArrayType)
                    @test test_boundary_condition(arch, FT, (Periodic, Periodic, Bounded), :top,    :w, boundary_condition)
                    @test test_boundary_condition(arch, FT, (Periodic, Periodic, Bounded), :bottom, :w, boundary_condition)
                end

                for field_name in (:u, :v, :T, :S)
                    @test test_flux_budget(arch, FT, field_name)
                end
            end
        end
    end

    @testset "Custom diffusivity boundary conditions" begin
        for arch in archs, FT in (Float64,) #float_types
            @info "  Testing flux budgets with custom diffusivity boundary conditions [$(typeof(arch)), $FT]..."
            @test fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
        end
    end
end
