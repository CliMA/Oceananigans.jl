function cfl(model, Δt)
    grid = model.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    ρ = model.density
    ρu, ρv, ρw = model.momenta

    ρ, ρu, ρv, ρw = ρ.data, ρu.data, ρv.data, ρw.data

    u_max = v_max = w_max = 0
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        u_max = max(u_max, abs(ρu[i, j, k] / ρ[i, j, k]))
        v_max = max(v_max, abs(ρv[i, j, k] / ρ[i, j, k]))
        w_max = max(w_max, abs(ρw[i, j, k] / ρ[i, j, k]))
    end

    return Δt / min(Δx/u_max, Δy/v_max, Δz/w_max)
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
