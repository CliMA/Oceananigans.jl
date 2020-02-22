function test_z_boundary_condition_simple(arch, FT, fldname, bctype, bc, Nx, Ny)
    Nz = 16
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(0.1, 0.2, 0.3))

    bc = BoundaryCondition(bctype, bc)
    field_bcs = TracerBoundaryConditions(grid, top=bc)
    model_bcs = NamedTuple{(fldname,)}((field_bcs,))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT,
                                boundary_conditions=model_bcs)

    time_step!(model, 1e-16, euler=true)

    return model isa IncompressibleModel
end

function test_z_boundary_condition_top_bottom_alias(arch, FT, fldname)
    N, val = 16, 1.0
    grid = RegularCartesianGrid(FT, size=(N, N, N), length=(0.1, 0.2, 0.3))

    top_bc    = BoundaryCondition(Value,  val)
    bottom_bc = BoundaryCondition(Value, -val)
    field_bcs = TracerBoundaryConditions(grid, top=top_bc, bottom=bottom_bc)
    model_bcs = NamedTuple{(fldname,)}((field_bcs,))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT,
                                boundary_conditions=model_bcs)

    time_step!(model, 1e-16, euler=true)

    field = get_model_field(fldname, model)
    bcs = field.boundary_conditions
    return getbc(bcs.z.top) == val && getbc(bcs.z.bottom) == -val
end

function test_z_boundary_condition_array(arch, FT, fldname)
    Nx = Ny = Nz = 16

    bcarray = rand(FT, Nx, Ny)

    if arch == GPU()
        bcarray = CuArray(bcarray)
    end

    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(0.1, 0.2, 0.3))

    value_bc = BoundaryCondition(Value, bcarray)
    field_bcs = TracerBoundaryConditions(grid, top=value_bc)
    model_bcs = NamedTuple{(fldname,)}((field_bcs,))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT,
                                boundary_conditions=model_bcs)

    time_step!(model, 1e-16, euler=true)

    field = get_model_field(fldname, model)
    bcs = field.boundary_conditions
    return bcs.z.top[1, 2] == bcarray[1, 2]
end

function test_flux_budget(arch, FT, fldname)
    N, κ, Lz = 16, 1, 0.7
    grid = RegularCartesianGrid(FT, size=(N, N, N), length=(1, 1, Lz))

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

    closure = ConstantIsotropicDiffusivity(FT, ν=κ, κ=κ)
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

    grid = RegularCartesianGrid(FT, size=(16, 16, 16), length=(1, 1, Lz))

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

@testset "Time stepping with boundary conditions" begin
    @info "Testing stepping with boundary conditions..."

    funbc(args...) = π
    boundaryfunbc = BoundaryFunction{:z, Face, Cell}((ξ, η, t) -> exp(ξ) * cos(η) * sin(t))

    @testset "Boundary condition instatiation and time-stepping" begin
        Nx = Ny = 16
        for arch in archs
            for FT in float_types
                @info "  Testing boundary condition instantiation and time-stepping [$(typeof(arch)), $FT]..."

                for fld in (:u, :v, :T, :S)
                    for bctype in (Gradient, Flux, Value)

                        arraybc = rand(FT, Nx, Ny)
                        if arch == GPU()
                            arraybc = CuArray(arraybc)
                        end

                        for bc in (FT(0.6), arraybc, funbc, boundaryfunbc)
                            @test test_z_boundary_condition_simple(arch, FT, fld, bctype, bc, Nx, Ny)
                        end
                    end

                    @test test_z_boundary_condition_top_bottom_alias(arch, FT, fld)
                    @test test_z_boundary_condition_array(arch, FT, fld)
                    @test test_flux_budget(arch, FT, fld)
                end
            end
        end
    end

    @testset "Custom diffusivity boundary conditions" begin
        for arch in archs, FT in float_types
            @info "  Testing flux budgets with custom diffusivity boundary conditions [$(typeof(arch)), $FT]..."
            @test fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
        end
    end
end
