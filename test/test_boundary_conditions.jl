using Oceananigans.BoundaryConditions: PBC, NFBC, NPBC

function test_boundary_function(B, X1, X2, func)
    boundary_function = BoundaryFunction{B, X1, X2}(func)
    return true
end

function test_z_boundary_condition_simple(arch, FT, fldname, bctype, bc, Nx, Ny)
    Nz = 16
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(0.1, 0.2, 0.3))

    bc = BoundaryCondition(bctype, bc)
    fieldbcs = TracerBoundaryConditions(grid, top=bc)
    modelbcs = SolutionBoundaryConditions(grid; Dict(fldname=>fieldbcs)...)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, boundary_conditions=modelbcs)

    time_step!(model, 1e-16, euler=true)

    return model isa IncompressibleModel
end

function test_z_boundary_condition_top_bottom_alias(arch, FT, fldname)
    N, val = 16, 1.0
    grid = RegularCartesianGrid(FT, size=(N, N, N), length=(0.1, 0.2, 0.3))

    top_bc    = BoundaryCondition(Value,  val)
    bottom_bc = BoundaryCondition(Value, -val)
    fieldbcs = TracerBoundaryConditions(grid, top=top_bc, bottom=bottom_bc)
    modelbcs = SolutionBoundaryConditions(grid; Dict(fldname=>fieldbcs)...)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, boundary_conditions=modelbcs)

    bcs = getfield(model.boundary_conditions.solution, fldname)

    time_step!(model, 1e-16, euler=true)

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
    fieldbcs = TracerBoundaryConditions(grid, top=value_bc)
    modelbcs = SolutionBoundaryConditions(grid; Dict(fldname=>fieldbcs)...)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, boundary_conditions=modelbcs)

    bcs = getfield(model.boundary_conditions.solution, fldname)

    time_step!(model, 1e-16, euler=true)

    return bcs.z.top[1, 2] == bcarray[1, 2]
end

function test_flux_budget(arch, FT, fldname)
    N, κ, Lz = 16, 1, 0.7
    grid = RegularCartesianGrid(FT, size=(N, N, N), length=(1, 1, Lz))

    bottom_flux = FT(0.3)
    flux_bc = BoundaryCondition(Flux, bottom_flux)
    fieldbcs = TracerBoundaryConditions(grid, bottom=flux_bc)
    modelbcs = SolutionBoundaryConditions(grid; Dict(fldname=>fieldbcs)...)

    closure = ConstantIsotropicDiffusivity(FT, ν=κ, κ=κ)
    model = IncompressibleModel(grid=grid, closure=closure, architecture=arch,
                                float_type=FT, buoyancy=nothing, boundary_conditions=modelbcs)

    if fldname ∈ (:u, :v, :w)
        field = getfield(model.velocities, fldname)
    else
        field = getfield(model.tracers, fldname)
    end

    @. field.data = 0

    bcs = getfield(model.boundary_conditions.solution, fldname)
    mean_init = mean(interior(field))

    τκ = Lz^2 / κ   # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 100        # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    return isapprox(mean(interior(field)) - mean_init, bottom_flux * model.clock.time / Lz)
end

function fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)

    Lz = 1
    κ₀ = FT(exp(-3))
    bz = FT(π)
    flux = - κ₀ * bz

    tracer_names = (:b,)
    grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, Lz))
    closure = with_tracers(tracer_names, AnisotropicMinimumDissipation())
    diffusivities = TurbulentDiffusivities(arch, grid, tracer_names, closure)

    buoyancy_bcs = TracerBoundaryConditions(grid, bottom=BoundaryCondition(Gradient, bz))
    solution_bcs = SolutionBoundaryConditions(grid, b=buoyancy_bcs)

    νₑ_bcs = DiffusivityBoundaryConditions(solution_bcs)
    κₑ_bcs = TracerBoundaryConditions(grid, bottom=BoundaryCondition(Value, κ₀))
    diffusivities_bcs = (νₑ=νₑ_bcs, κₑ=(b=κₑ_bcs))

    model_bcs = ModelBoundaryConditions(tracer_names, diffusivities, solution_bcs;
                                        diffusivities=diffusivities_bcs)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, tracers=tracer_names,
                                buoyancy=BuoyancyTracer(), closure=closure, diffusivities=diffusivities,
                                boundary_conditions=model_bcs)

    b₀(x, y, z) = z * bz
    set!(model, b=b₀)

    b = model.tracers.b
    mean_b₀ = mean(interior(b))

    τκ = Lz^2 / κ₀     # Diffusion time-scale
    Δt = 1e-6 * τκ     # Time step much less than diffusion time-scale
    Nt = 10            # Number of time steps

    for n in 1:Nt
        time_step!(model, Δt, euler= n==1)
    end

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    return isapprox(mean(interior(b)) - mean_b₀, flux * model.clock.time / Lz)
end

@testset "Boundary conditions" begin
    @info "Testing boundary conditions..."

    @testset "Boundary functions" begin
        @info "  Testing boundary functions..."

        simple_bc(ξ, η, t) = exp(ξ) * cos(η) * sin(t)
        for B in (:x, :y, :z)
            for X1 in (:Face, :Cell)
                @test test_boundary_function(B, X1, Cell, simple_bc)
            end
        end
    end

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

    @testset "Field boundary conditions" begin
        @info "  Testing field boundary functions..."

        ppb_topology = (Periodic, Periodic, Bounded)
        ppb_grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=ppb_topology)

        u_bcs = UVelocityBoundaryConditions(ppb_grid)
        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa PBC
        @test u_bcs.y.right isa PBC
        @test u_bcs.z.left  isa NFBC
        @test u_bcs.z.right isa NFBC

        v_bcs = VVelocityBoundaryConditions(ppb_grid)
        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa PBC
        @test v_bcs.y.right isa PBC
        @test v_bcs.z.left  isa NFBC
        @test v_bcs.z.right isa NFBC

        w_bcs = WVelocityBoundaryConditions(ppb_grid)
        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa PBC
        @test w_bcs.y.right isa PBC
        @test w_bcs.z.left  isa NPBC
        @test w_bcs.z.right isa NPBC

        Tbcs = TracerBoundaryConditions(ppb_grid)
        @test Tbcs isa FieldBoundaryConditions
        @test Tbcs.x.left  isa PBC
        @test Tbcs.x.right isa PBC
        @test Tbcs.y.left  isa PBC
        @test Tbcs.y.right isa PBC
        @test Tbcs.z.left  isa NFBC
        @test Tbcs.z.right isa NFBC

        pbb_topology = (Periodic, Bounded, Bounded)
        pbb_grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=pbb_topology)

        u_bcs = UVelocityBoundaryConditions(pbb_grid)
        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa NFBC
        @test u_bcs.y.right isa NFBC
        @test u_bcs.z.left  isa NFBC
        @test u_bcs.z.right isa NFBC

        v_bcs = VVelocityBoundaryConditions(pbb_grid)
        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa NPBC
        @test v_bcs.y.right isa NPBC
        @test v_bcs.z.left  isa NFBC
        @test v_bcs.z.right isa NFBC

        w_bcs = WVelocityBoundaryConditions(pbb_grid)
        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa NFBC
        @test w_bcs.y.right isa NFBC
        @test w_bcs.z.left  isa NPBC
        @test w_bcs.z.right isa NPBC

        Tbcs = TracerBoundaryConditions(pbb_grid)
        @test Tbcs isa FieldBoundaryConditions
        @test Tbcs.x.left  isa PBC
        @test Tbcs.x.right isa PBC
        @test Tbcs.y.left  isa NFBC
        @test Tbcs.y.right isa NFBC
        @test Tbcs.z.left  isa NFBC
        @test Tbcs.z.right isa NFBC
    end

    @testset "Custom diffusivity boundary conditions" begin
        for arch in archs, FT in float_types
            @info "  Testing flux budgets with custom diffusivity boundary conditions [$(typeof(arch)), $FT]..."
            @test fluxes_with_diffusivity_boundary_conditions_are_correct(arch, FT)
        end
    end
end
