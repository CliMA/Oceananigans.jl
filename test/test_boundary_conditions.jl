function test_z_boundary_condition_simple(arch, TF, field_name, bctype, bc, Nx, Ny, Nz)
    Lx, Ly, Lz = 0.1, 0.2, 0.3
    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=TF)

    bc = BoundaryCondition(bctype, bc)
    bcs = getfield(model.boundary_conditions, field_name)
    bcs.z.top = bc # just set a boundary condition somewhere

    time_step!(model, 1, 1e-16)

    typeof(model) <: Model
end

function test_z_boundary_condition_top_bottom_alias(arch, TF, field_name, Nx, Ny, Nz)
    Lx, Ly, Lz = 0.1, 0.2, 0.3
    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=TF)

    bcval = 1.0
    bcs = getfield(model.boundary_conditions, field_name)
    bcs.z.top = BoundaryCondition(Value, bcval)
    bcs.z.bottom = BoundaryCondition(Value, -bcval)

    time_step!(model, 1, 1e-16)

    getbc(bcs.z.top) == bcval && getbc(bcs.z.bottom) == -bcval
end

function test_z_boundary_condition_array(arch, TF, field_name, Nx, Ny, Nz)
    Lx, Ly, Lz = 0.1, 0.2, 0.3
    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=TF)

    bcarray = rand(Nx, Ny)
    bcs = getfield(model.boundary_conditions, field_name)
    bcs.z.top = BoundaryCondition(Value, bcarray)

    time_step!(model, 1, 1e-16)

    getbc(bcs.z.top, 1, 2) == bcarray[1, 2]
end

function test_flux_budget(arch, TF, field_name)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, 0.7
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos,
        arch=arch, float_type=TF)

    if field_name ∈ (:u, :v, :w)
        field = getfield(model.velocities, field_name)
    else
        field = getfield(model.tracers, field_name)
    end

    @. field.data = 0

    bottom_flux = TF(0.3)
    flux_bc = BoundaryCondition(Flux, bottom_flux)
    bcs = getfield(model.boundary_conditions, field_name)
    bcs.z.bottom = flux_bc # "right" = "bottom" in the convention where k=Nz is the bottom.

    mean_init = mean(field.data)

    τκ = Lz^2 / κ # diffusion time-scale
    Δt = 1e-6 * τκ # time-step much less than diffusion time-scale
    Nt = 100

    time_step!(model, Nt, Δt)

    # budget is Lz * ∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right);
    # therefore <ϕ> = bottom_flux * t / Lz
    isapprox(mean(field.data) - mean_init, bottom_flux*model.clock.time/Lz)
end
