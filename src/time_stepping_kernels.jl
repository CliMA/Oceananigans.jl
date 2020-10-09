"""
Compute total density from densities of massive tracers
"""
function update_total_density!(ρ, grid, gases, ρc̃)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            ρ[i, j, k] = diagnose_ρ(i, j, k, grid, gases, ρc̃)
        end
    end
end

update_total_density!(model) =
    update_total_density!(model.total_density, model.grid, model.gases, model.tracers)

"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_forcings!(F̃, grid, tvar, gases, gravity, coriolis, closure, ρ, ρũ, ρc̃, K̃, forcing, clock)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            F̃.ρu[i, j, k] = FU(i, j, k, grid, coriolis, closure, ρ, ρũ, K̃) + forcing.u(i, j, k, grid, clock, nothing)
            F̃.ρv[i, j, k] = FV(i, j, k, grid, coriolis, closure, ρ, ρũ, K̃) + forcing.v(i, j, k, grid, clock, nothing)
            F̃.ρw[i, j, k] = FW(i, j, k, grid, coriolis, closure, ρ, ρũ, K̃) + forcing.w(i, j, k, grid, clock, nothing)
        end

        for (tracer_index, ρc_name) in enumerate(propertynames(ρc̃))
            ρc   = getproperty(ρc̃, ρc_name)
            F_ρc = getproperty(F̃.tracers, ρc_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                F_ρc[i, j, k] = FC(i, j, k, grid, closure, tracer_index, ρ, ρc, K̃)
            end
        end

        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            F̃.tracers[1].data[i, j, k] += FT(i, j, k, grid, closure, tvar, gases, gravity, ρ, ρũ, ρc̃, K̃)
        end

    end
end

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_right_hand_sides!(R̃, grid, tvar, gases, gravity, ρ, ρũ, ρc̃, F̃)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            R̃.ρu[i, j, k] = RU(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃, F̃.ρu)
            R̃.ρv[i, j, k] = RV(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃, F̃.ρv)
            R̃.ρw[i, j, k] = RW(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃, F̃.ρw)
        end

        for ρc_name in propertynames(ρc̃)
            ρc   = getproperty(ρc̃, ρc_name)
            R_ρc = getproperty(R̃.tracers, ρc_name)
            F_ρc = getproperty(F̃.tracers, ρc_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                R_ρc[i, j, k] = RC(i, j, k, grid, ρ, ρũ, ρc, F_ρc)
            end
        end

        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            R̃.tracers[1].data[i, j, k] += RT(i, j, k, grid, tvar, gases, gravity, ρ, ρũ, ρc̃)
        end

    end
end

"""
Updates variables according to the RK3 time step:
    1. Φ*      = Φᵗ + Δt/3 * R(Φᵗ)
    2. Φ**     = Φᵗ + Δt/2 * R(Φ*)
    3. Φ(t+Δt) = Φᵗ + Δt   * R(Φ**)
"""
function advance_variables!(Ĩ, grid, ρũᵗ, ρc̃ᵗ, R̃; Δt)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            Ĩ.ρu[i, j, k] = ρũᵗ.ρu[i, j, k] + Δt * R̃.ρu[i, j, k]
            Ĩ.ρv[i, j, k] = ρũᵗ.ρv[i, j, k] + Δt * R̃.ρv[i, j, k]
            Ĩ.ρw[i, j, k] = ρũᵗ.ρw[i, j, k] + Δt * R̃.ρw[i, j, k]
        end

        for ρc_name in propertynames(ρc̃ᵗ)
            ρcᵗ  = getproperty(ρc̃ᵗ, ρc_name)
            I_ρc = getproperty(Ĩ.tracers, ρc_name)
            R_ρc = getproperty(R̃.tracers, ρc_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                I_ρc[i, j, k] = ρcᵗ[i, j, k] + Δt * R_ρc[i, j, k]
            end
        end
    end
end
