"""
Compute total density from densities of massive tracers
"""
function update_total_density!(ρ, grid, ρ̃, C̃)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            ρ[i, j, k] = diagnose_ρ(i, j, k, grid, ρ̃, C̃)
        end
    end
end

"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_forcings!(F, grid, tvar, g, coriolis, closure, Ũ, ρ, ρ̃, C̃, K̃, tfields, forcing, time)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            # Use params = nothing for now until Oceananigans.jl stops using model.parameters.
            F.ρu[i, j, k] = FU(i, j, k, grid, coriolis, closure, ρ, Ũ, K̃) + forcing.u(i, j, k, grid, time, Ũ, C̃, nothing)
            F.ρv[i, j, k] = FV(i, j, k, grid, coriolis, closure, ρ, Ũ, K̃) + forcing.v(i, j, k, grid, time, Ũ, C̃, nothing)
            F.ρw[i, j, k] = FW(i, j, k, grid, coriolis, closure, ρ, Ũ, K̃) + forcing.w(i, j, k, grid, time, Ũ, C̃, nothing)
        end

        for (tracer_index, C_name) in enumerate(propertynames(C̃))
            C   = getproperty(C̃, C_name)
            F_C = getproperty(F, C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                F_C[i, j, k] = FC(i, j, k, grid, closure, ρ, C, tracer_index, K̃)
            end
        end

        for tfield in tfields
            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                tfield.F.data[i, j, k] += FT(i, j, k, grid, closure, tfield.variable, g, ρ, ρ̃, Ũ, C̃, K̃)
            end
        end
    end
end

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_right_hand_sides!(R, grid, tvar, g, ρ, ρ̃, Ũ, C̃, F, tfields)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            R.ρu[i, j, k] = RU(i, j, k, grid, tvar, g, ρ, ρ̃, Ũ, C̃, F.ρu)
            R.ρv[i, j, k] = RV(i, j, k, grid, tvar, g, ρ, ρ̃, Ũ, C̃, F.ρv)
            R.ρw[i, j, k] = RW(i, j, k, grid, tvar, g, ρ, ρ̃, Ũ, C̃, F.ρw)
        end

        for C_name in propertynames(C̃)
            C   = getproperty(C̃, C_name)
            R_C = getproperty(R, C_name)
            F_C = getproperty(F, C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                R_C[i, j, k] = RC(i, j, k, grid, ρ, Ũ, C, F_C)
            end
        end

        for tfield in tfields
            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                tfield.R.data[i, j, k] += RT(i, j, k, grid, tfield.variable, g, ρ, ρ̃, Ũ, C̃)
            end
        end
    end
end

"""
Updates variables according to the RK3 time step:
    1. Φ*      = Φᵗ + Δt/3 * R(Φᵗ)
    2. Φ**     = Φᵗ + Δt/2 * R(Φ*)
    3. Φ(t+Δt) = Φᵗ + Δt   * R(Φ**)
"""
function advance_variables!(I, grid, Ũᵗ, C̃ᵗ, R; Δt)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            I.ρu[i, j, k] = Ũᵗ.ρu[i, j, k] + Δt * R.ρu[i, j, k]
            I.ρv[i, j, k] = Ũᵗ.ρv[i, j, k] + Δt * R.ρv[i, j, k]
            I.ρw[i, j, k] = Ũᵗ.ρw[i, j, k] + Δt * R.ρw[i, j, k]
        end

        for C_name in propertynames(C̃ᵗ)
            Cᵗ  = getproperty(C̃ᵗ, C_name)
            I_C = getproperty(I,  C_name)
            R_C = getproperty(R,  C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                I_C[i, j, k] = Cᵗ[i, j, k] + Δt * R_C[i, j, k]
            end
        end
    end
end
