function test_z_boundary_condition_simple(field_name, bctype, bc, Nx, Ny, Nz)
    Lx, Ly, Lz = 0.1, 0.2, 0.3
    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz))

    bc = BoundaryCondition(bctype, bc)
    bcs = getfield(model.boundary_conditions, field_name)
    bcs.z.right = bc # just set a boundary condition somewhere

    time_step!(model, 1, 1e-16)
    typeof(model) <: Model
end


function test_flux_budget(field_name)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, 0.7
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)

    if field_name ∈ (:u, :v, :w)
        field = getfield(model.velocities, field_name)
    else
        field = getfield(model.tracers, field_name)
    end

    @. field.data = 0

    bottom_flux = 0.3
    flux_bc = BoundaryCondition(Flux, bottom_flux)
    bcs = getfield(model.boundary_conditions, field_name)
    bcs.z.right = flux_bc # "right" = "bottom" in the convention where k=Nz is the bottom.

    mean_init = mean(field.data)

    τκ = Lz^2 / κ # diffusion time-scale
    Δt = 1e-6 * τκ # time-step much less than diffusion time-scale
    Nt = 100

    time_step!(model, Nt, Δt)

    # budget is Lz * ∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right);
    # therefore <ϕ> = bottom_flux * t / Lz
    isapprox(mean(field.data) - mean_init, bottom_flux*model.clock.time/Lz)
end
