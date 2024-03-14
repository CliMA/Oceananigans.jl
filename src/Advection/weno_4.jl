for side in [:left, :right], (dir, val) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [1, 2, 3])
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir)
    biased_β     = Symbol(side, :_biased_β)
    biased_p     = Symbol(side, :_biased_p)
    coeff        = Symbol(:coeff_, side) 
    stencil      = Symbol(side, :_stencil_, dir)
    stencil_u    = Symbol(:tangential_, side, :_stencil_u)
    stencil_v    = Symbol(:tangential_, side, :_stencil_v)
    weno_substep = Symbol(side, :_weno_substep_, dir)

    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4}, 
                                            ψ, idx, loc, args...) 
        
            # Stencil S₀
            β, ψ̅, C, α = $weno_substep(i, j, k, 1, grid, scheme, $val, ψ, idx, loc, args...)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_substep(i, j, k, 2, grid, scheme, $val, ψ, idx, loc, args...)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₂
            β, ψ̅, C, α = $weno_substep(i, j, k, 3, grid, scheme, $val, ψ, idx, loc, args...)
            τ  += add_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₃
            β, ψ̅, C, α = $weno_substep(i, j, k, 4, grid, scheme, $val, ψ, idx, loc, args...)
            τ  += add_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end
    end
end