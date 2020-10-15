@kernel function compute_velocities!(velocities, grid, momenta, total_density)
    i, j, k = @index(Global, NTuple)

    @inbounds velocities.u[i, j, k] = momenta.ρu[i, j, k] / ℑxᶠᵃᵃ(i, j, k, grid, total_density)
    @inbounds velocities.v[i, j, k] = momenta.ρv[i, j, k] / ℑyᵃᶠᵃ(i, j, k, grid, total_density)
    @inbounds velocities.w[i, j, k] = momenta.ρw[i, j, k] / ℑzᵃᵃᶠ(i, j, k, grid, total_density)
end

function cfl(model, Δt)
    velocities = (
        u = model.time_stepper.intermediate_fields.ρu,
        v = model.time_stepper.intermediate_fields.ρv,
        w = model.time_stepper.intermediate_fields.ρw
    )

    velocities, momenta, total_density =
        datatuples(velocities, model.momenta, model.total_density)

    velocity_event =
        launch!(model.architecture, model.grid, :xyz, compute_velocities!,
                velocities, model.grid, momenta, total_density,
                dependencies=Event(device(model.architecture)))
    
    wait(device(model.architecture), velocity_event)

    u_max = maximum(velocities.u)
    v_max = maximum(velocities.v)
    w_max = maximum(velocities.w)

    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz
    CFL = Δt / min(Δx/u_max, Δy/v_max, Δz/w_max)

    return CFL
end

function acoustic_cfl(model, Δt)
    grid = model.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz
    ρ, Θ = model.density.data, model.tracers.Θᵐ.data
    gas = model.buoyancy
    R = gas_constant
    M = molar_mass_dry_air

    c_max = 0
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        π = Π(i, j, k, grid, gas, Θ)
        T = π * (Θ[i, j, k] / ρ[i, j, k])
        c = √(gas.γ * R * T / M)
        c_max = max(c_max, c)
    end

    return Δt / min(Δx/c_max, Δy/c_max, Δz/c_max)
end
