####
#### Dirty hacks!
####

const grav = 9.80665
const μ = 1e-2
const κ = 1e-2

####
#### Element-wise forcing and right-hand-side calculations
####

@inline FU(i, j, k, grid, coriolis, μ, ρᵈ, Ũ) = -x_f_cross_U(i, j, k, grid, coriolis, Ũ) + div_μ∇u(i, j, k, grid, μ, ρᵈ, Ũ.U)
@inline FV(i, j, k, grid, coriolis, μ, ρᵈ, Ũ) = -y_f_cross_U(i, j, k, grid, coriolis, Ũ) + div_μ∇v(i, j, k, grid, μ, ρᵈ, Ũ.V)
@inline FW(i, j, k, grid, coriolis, μ, ρᵈ, Ũ) = -z_f_cross_U(i, j, k, grid, coriolis, Ũ) + div_μ∇w(i, j, k, grid, μ, ρᵈ, Ũ.W)

@inline FC(i, j, k, grid, κ, ρᵈ, C) = div_κ∇c(i, j, k, grid, κ, ρᵈ, C)

@inline function RU(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C, FU)
    @inbounds begin
        return (- div_ρuũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * ∂p∂x(i, j, k, grid, pt, b, pₛ, C)
                + FU[i, j, k])
    end
end

@inline function RV(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C, FV)
    @inbounds begin
        return (- div_ρvũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * ∂p∂y(i, j, k, grid, pt, b, pₛ, C)
                + FV[i, j, k])
    end
end

@inline function RW(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C, FW)
    @inbounds begin
        return (- div_ρwũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * (  ∂p∂z(i, j, k, grid, pt, b, pₛ, C)
                                                      + buoyancy_perturbation(i, j, k, grid, grav, ρᵈ, C))
                + FW[i, j, k])
    end
end

@inline Rρ(i, j, k, grid, Ũ) = -divᶜᶜᶜ(i, j, k, grid, Ũ.U, Ũ.V, Ũ.W)
@inline RC(i, j, k, grid, ρᵈ, Ũ, C, FC) = @inbounds -div_flux(i, j, k, grid, ρᵈ, Ũ.U, Ũ.V, Ũ.W, C) + FC[i, j, k]
