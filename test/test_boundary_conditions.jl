function test_z_boundary_condition_simple(arch, T, fldname, bctype, bc, Nx, Ny)
    Nz = 16
    bc = BoundaryCondition(bctype, bc)
    fieldbcs = HorizontallyPeriodicBCs(top=bc)
    modelbcs = BoundaryConditions(; Dict(fldname=>fieldbcs)...)

    model = BasicModel(N=(Nx, Ny, Nz), L=(0.1, 0.2, 0.3), architecture=arch,
                       float_type=T, boundary_conditions=modelbcs)

    time_step!(model, 1, 1e-16)

    typeof(model) <: Model
end

function test_z_boundary_condition_top_bottom_alias(arch, FT, fldname)
    N, val = 16, 1.0
    top_bc = BoundaryCondition(Value, val)
    bottom_bc = BoundaryCondition(Value, -val)
    fieldbcs = HorizontallyPeriodicBCs(top=top_bc, bottom=bottom_bc)
    modelbcs = BoundaryConditions(; Dict(fldname=>fieldbcs)...)

    model = BasicModel(N=(N, N, N), L=(0.1, 0.2, 0.3), architecture=arch,
                       float_type=FT, boundary_conditions=modelbcs)

    bcs = getfield(model.boundary_conditions, fldname)

    time_step!(model, 1, 1e-16)

    getbc(bcs.z.top) == val && getbc(bcs.z.bottom) == -val
end

function test_z_boundary_condition_array(arch, FT, fldname)
    Nx = Ny = Nz = 16

    bcarray = rand(FT, Nx, Ny)

    if arch == GPU()
        bcarray = CuArray(bcarray)
    end

    value_bc = BoundaryCondition(Value, bcarray)
    fieldbcs = HorizontallyPeriodicBCs(top=value_bc)
    modelbcs = BoundaryConditions(; Dict(fldname=>fieldbcs)...)

    model = BasicModel(N=(Nx, Ny, Nz), L=(0.1, 0.2, 0.3), architecture=arch,
                       float_type=FT, boundary_conditions=modelbcs)

    bcs = getfield(model.boundary_conditions, fldname)

    time_step!(model, 1, 1e-16)

    bcs.z.top[1, 2] == bcarray[1, 2]
end

function test_flux_budget(arch, FT, fldname)
    N, κ, Lz = 16, 1, 0.7

    bottom_flux = FT(0.3)
    flux_bc = BoundaryCondition(Flux, bottom_flux)
    fieldbcs = HorizontallyPeriodicBCs(bottom=flux_bc)
    modelbcs = BoundaryConditions(; Dict(fldname=>fieldbcs)...)

    model = BasicModel(N=(N, N, N), L=(1, 1, Lz), ν=κ, κ=κ, architecture=arch,
                       float_type=FT, eos=LinearEquationOfState(βS=0.0, βT=0.0), 
                       boundary_conditions=modelbcs)

    if fldname ∈ (:u, :v, :w)
        field = getfield(model.velocities, fldname)
    else
        field = getfield(model.tracers, fldname)
    end

    @. field.data = 0

    bcs = getfield(model.boundary_conditions, fldname)
    mean_init = mean(data(field))

    τκ = Lz^2 / κ   # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 100        # Number of time steps

    time_step!(model, Nt, Δt)

    # budget: Lz*∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right)
    # therefore <ϕ> = bottom_flux * t / Lz
    isapprox(mean(data(field)) - mean_init, bottom_flux * model.clock.time / Lz)
end

@testset "Boundary conditions" begin
    println("Testing boundary conditions...")

    funbc(args...) = π

    Nx = Ny = 16
    for arch in archs
        for FT in float_types
            for fld in (:u, :v, :T, :S)
                for bctype in (Gradient, Flux, Value)

                    arraybc = rand(FT, Nx, Ny)
                    if arch == GPU()
                        arraybc = CuArray(arraybc)
                    end

                    for bc in (FT(0.6), arraybc, funbc)
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
