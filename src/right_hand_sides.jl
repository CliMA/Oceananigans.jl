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

@inline FC(i, j, k, grid, closure, ρ, C, ::Val{tracer_index}, K̃) where tracer_index =
    @inbounds ∂ⱼDᶜⱼ(i, j, k, grid, closure, ρ, C, Val(tracer_index), K̃)

@inline function FS(i, j, k, grid, closure, tvar::Entropy, ρ, ρ̃, Ũ, C̃, K̃)
    @inbounds begin
        Ṡ = 0.0
        T = diagnose_T(i, j, k, grid, tvar, ρ̃, C̃)
        for key in keys(ρ̃)
            tracer_index = findall(keys(C̃) .== key)[1]
            gas = getproperty(ρ̃, key)
            ρᶜ = getproperty(C̃, key)
            sᶜ = diagnose_s(gas, ρᶜ[i, j, k], T)
            Ṡ += -sᶜ∂ⱼDᶜⱼ(i, j, k, grid, closure, ρ, ρᶜ, sᶜ, Val(tracer_index), K̃)
        end
        Ṡ += Q_dissipation(i, j, k, grid, closure, ρ, Ũ) / T
    end
    return Ṡ
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

@inline RC(i, j, k, grid, ρ, Ũ, C, FC) =
    @inbounds -div_flux(i, j, k, grid, ρ, Ũ.ρu, Ũ.ρv, Ũ.ρw, C) + FC[i, j, k]
