"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_forcings!(F, grid, coriolis, Ũ, ρᵈ, C̃)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            F.U[i, j, k] = FU(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
            F.V[i, j, k] = FV(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
            F.W[i, j, k] = FW(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
        end

        for C_name in propertynames(C̃)
            C   = getproperty(C̃, C_name)
            F_C = getproperty(F, C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                F_C[i, j, k] = FC(i, j, k, grid, κ, ρᵈ, C)
            end
        end
    end
end

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_right_hand_sides!(R, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            R.U[i, j, k] = RU(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F.U)
            R.V[i, j, k] = RV(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F.V)
            R.W[i, j, k] = RW(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F.W)

            R.ρ[i, j, k] = Rρ(i, j, k, grid, Ũ)
        end

        for C_name in propertynames(C̃)
            C   = getproperty(C̃, C_name)
            R_C = getproperty(R, C_name)
            F_C = getproperty(F, C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                R_C[i, j, k] = RC(i, j, k, grid, ρᵈ, Ũ, C, F_C)
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
function advance_variables!(I, grid, Ũᵗ, C̃ᵗ, ρᵗ, R; Δt)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            I.U[i, j, k] = Ũᵗ.U[i, j, k] + Δt * R.U[i, j, k]
            I.V[i, j, k] = Ũᵗ.V[i, j, k] + Δt * R.V[i, j, k]
            I.W[i, j, k] = Ũᵗ.W[i, j, k] + Δt * R.W[i, j, k]
            I.ρ[i, j, k] =   ρᵗ[i, j, k] + Δt * R.ρ[i, j, k]
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
