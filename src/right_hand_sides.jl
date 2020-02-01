####
#### Dirty hacks!
####

const grav = 9.80665

####
#### Element-wise forcing and right-hand-side calculations
####

@inline function FU(i, j, k, grid, coriolis, closure, ρᵈ, Ũ, K)
    @inbounds begin
        return (- x_f_cross_U(i, j, k, grid, coriolis, Ũ)
                + ρᵈ[i, j, k] * ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, ρᵈ, Ũ, K))
    end
end


@inline function FV(i, j, k, grid, coriolis, closure, ρᵈ, Ũ, K)
    @inbounds begin
        return (- y_f_cross_U(i, j, k, grid, coriolis, Ũ)
                + ρᵈ[i, j, k] * ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, ρᵈ, Ũ, K))
    end
end

@inline function FW(i, j, k, grid, coriolis, closure, ρᵈ, Ũ, K)
    @inbounds begin
        return (- z_f_cross_U(i, j, k, grid, coriolis, Ũ)
                + ρᵈ[i, j, k] * ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, ρᵈ, Ũ, K))
    end
end

@inline FC(i, j, k, grid, closure, ρᵈ, C, tracer_index, K̃) =
    @inbounds ρᵈ[i, j, k] * ∇_κ_∇c(i, j, k, grid, closure, ρᵈ, C, Val(tracer_index), K̃)

@inline function RU(i, j, k, grid, pt, b, pₛ, mp, ρᵈ, Ũ, C, FU)
    @inbounds begin
        return (- div_ρuũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, mp, ρᵈ, C) * ∂p∂x(i, j, k, grid, pt, b, ρᵈ, C)
                + FU[i, j, k])
    end
end

@inline function RV(i, j, k, grid, pt, b, pₛ, mp, ρᵈ, Ũ, C, FV)
    @inbounds begin
        return (- div_ρvũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, mp, ρᵈ, C) * ∂p∂y(i, j, k, grid, pt, b, ρᵈ, C)
                + FV[i, j, k])
    end
end

@inline function RW(i, j, k, grid, pt, b, pₛ, mp, ρᵈ, Ũ, C, FW)
    @inbounds begin
        return (- div_ρwũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, mp, ρᵈ, C) * (
                      ∂p∂z(i, j, k, grid, pt, b, ρᵈ, C)
                    + buoyancy_perturbation(i, j, k, grid, grav, mp, ρᵈ, C))
                + FW[i, j, k])
    end
end

@inline Rρ(i, j, k, grid, Ũ) =
    -divᶜᶜᶜ(i, j, k, grid, Ũ.ρu, Ũ.ρv, Ũ.ρw)

@inline RC(i, j, k, grid, ρᵈ, Ũ, C, FC) =
    @inbounds -div_flux(i, j, k, grid, ρᵈ, Ũ.ρu, Ũ.ρv, Ũ.ρw, C) + FC[i, j, k]
