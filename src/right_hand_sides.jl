####
#### Element-wise forcing and right-hand-side calculations
####

@inline function FU(i, j, k, grid, coriolis, closure, ρ, Ũ, K)
    @inbounds begin
        return (- x_f_cross_U(i, j, k, grid, coriolis, Ũ)
                + ∂ⱼτ₁ⱼ(i, j, k, grid, closure, ρ, Ũ, K))
    end
end


@inline function FV(i, j, k, grid, coriolis, closure, ρ, Ũ, K)
    @inbounds begin
        return (- y_f_cross_U(i, j, k, grid, coriolis, Ũ)
                + ∂ⱼτ₂ⱼ(i, j, k, grid, closure, ρ, Ũ, K))
    end
end

@inline function FW(i, j, k, grid, coriolis, closure, ρ, Ũ, K)
    @inbounds begin
        return (- z_f_cross_U(i, j, k, grid, coriolis, Ũ)
                + ∂ⱼτ₃ⱼ(i, j, k, grid, closure, ρ, Ũ, K))
    end
end

@inline function FC(i, j, k, grid, closure, ρ, C, tracer_index, K̃)
    @inbounds begin
        return ∂ⱼDᶜⱼ(i, j, k, grid, closure, ρ, C, tracer_index, K̃)
    end
end

@inline function FS(i, j, k, grid, closure, tvar::Entropy, ρ, ρ̃, Ũ, C̃, K̃)
    @inbounds begin
        Ṡ = 0.0
        for ind_gas = 1:length(ρ̃)
            tracer_index = ind_gas + 1
            C = C̃[tracer_index]
            Ṡ += ∂ⱼsᶜDᶜⱼ(i, j, k, grid, closure, diagnose_ρs, tvar, tracer_index, ρ̃, C̃, ρ, C, K̃)
        end
        T = diagnose_T(i, j, k, grid, tvar, ρ̃, C̃)
        Ṡ += Q_dissipation(i, j, k, grid, closure, ρ, Ũ) / T
        return Ṡ
    end
end

@inline function RU(i, j, k, grid, tvar, ρ, ρ̃, Ũ, C, FU)
    @inbounds begin
        return (- div_ρuũ(i, j, k, grid, ρ, Ũ)
                - ∂p∂x(i, j, k, grid, tvar, ρ̃, C)
                + FU[i, j, k])
    end
end

@inline function RV(i, j, k, grid, tvar, ρ, ρ̃, Ũ, C, FV)
    @inbounds begin
        return (- div_ρvũ(i, j, k, grid, ρ, Ũ)
                - ∂p∂y(i, j, k, grid, tvar, ρ̃, C)
                + FV[i, j, k])
    end
end

@inline function RW(i, j, k, grid, tvar, gravity, ρ, ρ̃, Ũ, C, FW)
    @inbounds begin
        return (- div_ρwũ(i, j, k, grid, ρ, Ũ)
                - ∂p∂z(i, j, k, grid, tvar, ρ̃, C)
                - gravity*ρ[i,j,k]
                + FW[i, j, k])
    end
end

@inline function RC(i, j, k, grid, ρ, Ũ, C, FC)
    @inbounds begin
        return -div_flux(i, j, k, grid, ρ, Ũ.ρu, Ũ.ρv, Ũ.ρw, C) + FC[i, j, k]
    end
end
