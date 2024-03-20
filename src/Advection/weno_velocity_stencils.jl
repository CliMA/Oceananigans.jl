#####
##### STENCILS IN X
#####

for (side, add) in zip([:left, :right], (1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_xᶠᵃᵃ)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_xᶠᵃᵃ)
    val = 1
    
    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, u, v)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i     - $add, j, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i + 1 - $add, j, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i     - $add, j, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i + 1 - $add, j, k, grid, v)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (u₀, u₁), (v₀, v₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₀ = ψ(i - 1 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₀ = ℑyᵃᶠᵃ(i - 1 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₀ = ℑxᶠᵃᵃ(i - 1 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (u₀, u₁), (v₀, v₁), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, u, v)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, u, v)
            ψ₂ = ψ(i + 2 - $add, j, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i     - $add, j, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i + 1 - $add, j, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i + 2 - $add, j, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i     - $add, j, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i + 1 - $add, j, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i + 2 - $add, j, k, grid, v)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (u₀, u₁, u₂), (v₀, v₁, v₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = ψ(i - 1 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₀ = ℑyᵃᶠᵃ(i - 1 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₀ = ℑxᶠᵃᵃ(i - 1 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (u₀, u₁, u₂), (v₀, v₁, v₂), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = ψ(i - 2 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₀ = ℑyᵃᶠᵃ(i - 2 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₀ = ℑxᶠᵃᵃ(i - 2 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (u₀, u₁, u₂), (v₀, v₁, v₂), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, u, v)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, u, v)
            ψ₂ = ψ(i + 2 - $add, j, k, grid, u, v)
            ψ₃ = ψ(i + 3 - $add, j, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i     - $add, j, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i + 1 - $add, j, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i + 2 - $add, j, k, grid, u)
            u₃ = ℑyᵃᶠᵃ(i + 3 - $add, j, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i     - $add, j, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i + 1 - $add, j, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i + 2 - $add, j, k, grid, v)
            v₃ = ℑxᶠᵃᵃ(i + 3 - $add, j, k, grid, v)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = ψ(i - 1 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₀ = ℑyᵃᶠᵃ(i - 1 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₀ = ℑxᶠᵃᵃ(i - 1 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = ψ(i - 2 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₀ = ℑyᵃᶠᵃ(i - 2 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₀ = ℑxᶠᵃᵃ(i - 2 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = ψ(i - 3 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₀ = ℑyᵃᶠᵃ(i - 3 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₀ = ℑxᶠᵃᵃ(i - 3 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, u, v)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, u, v)
            ψ₂ = ψ(i + 2 - $add, j, k, grid, u, v)
            ψ₃ = ψ(i + 3 - $add, j, k, grid, u, v)
            ψ₄ = ψ(i + 4 - $add, j, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i     - $add, j, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i + 1 - $add, j, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i + 2 - $add, j, k, grid, u)
            u₃ = ℑyᵃᶠᵃ(i + 3 - $add, j, k, grid, u)
            u₄ = ℑyᵃᶠᵃ(i + 4 - $add, j, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i     - $add, j, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i + 1 - $add, j, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i + 2 - $add, j, k, grid, v)
            v₃ = ℑxᶠᵃᵃ(i + 3 - $add, j, k, grid, v)
            v₄ = ℑxᶠᵃᵃ(i + 4 - $add, j, k, grid, v)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i - 1 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i - 1 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i - 1 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i - 2 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i - 2 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i - 2 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i - 3 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i - 3 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i - 3 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i - 4 - $add, j, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i - 4 - $add, j, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i - 4 - $add, j, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 5, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end
    end
end

#####
##### STENCILS IN Y
#####

for (side, add) in zip([:left, :right], (1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_yᵃᶠᵃ)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_yᵃᶠᵃ)
    val = 2

    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, u, v)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i, j     - $add, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i, j + 1 - $add, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i, j     - $add, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i, j + 1 - $add, k, grid, v)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (u₀, u₁), (v₀, v₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₀ = ψ(i, j - 1 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₀ = ℑyᵃᶠᵃ(i, j - 1 - $add, k, grid, u)
            
            v₁ = v₀
            v₀ = ℑxᶠᵃᵃ(i, j - 1 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (u₀, u₁), (v₀, v₁), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, u, v)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, u, v)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i, j     - $add, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i, j + 1 - $add, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i, j + 2 - $add, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i, j     - $add, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i, j + 1 - $add, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i, j + 2 - $add, k, grid, v)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (u₀, u₁, u₂), (v₀, v₁, v₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = ψ(i, j - 1 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₀ = ℑyᵃᶠᵃ(i, j - 1 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₀ = ℑxᶠᵃᵃ(i, j - 1 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (u₀, u₁, u₂), (v₀, v₁, v₂), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = ψ(i, j - 2 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₀ = ℑyᵃᶠᵃ(i, j - 2 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₀ = ℑxᶠᵃᵃ(i, j - 2 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (u₀, u₁, u₂), (v₀, v₁, v₂), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, u, v)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, u, v)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, u, v)
            ψ₃ = ψ(i, j + 3 - $add, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i, j     - $add, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i, j + 1 - $add, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i, j + 2 - $add, k, grid, u)
            u₃ = ℑyᵃᶠᵃ(i, j + 3 - $add, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i, j     - $add, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i, j + 1 - $add, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i, j + 2 - $add, k, grid, v)
            v₃ = ℑxᶠᵃᵃ(i, j + 3 - $add, k, grid, v)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = ψ(i, j - 1 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₀ = ℑyᵃᶠᵃ(i, j - 1 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₀ = ℑxᶠᵃᵃ(i, j - 1 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = ψ(i, j - 2 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₀ = ℑyᵃᶠᵃ(i, j - 2 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₀ = ℑxᶠᵃᵃ(i, j - 2 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = ψ(i, j - 3 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₀ = ℑyᵃᶠᵃ(i, j - 3 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₀ = ℑxᶠᵃᵃ(i, j - 3 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (u₀, u₁, u₂, u₃), (v₀, v₁, v₂, v₃), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, u, v)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, u, v)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, u, v)
            ψ₃ = ψ(i, j + 3 - $add, k, grid, u, v)
            ψ₄ = ψ(i, j + 4 - $add, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i, j     - $add, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i, j + 1 - $add, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i, j + 2 - $add, k, grid, u)
            u₃ = ℑyᵃᶠᵃ(i, j + 3 - $add, k, grid, u)
            u₄ = ℑyᵃᶠᵃ(i, j + 4 - $add, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i, j     - $add, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i, j + 1 - $add, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i, j + 2 - $add, k, grid, v)
            v₃ = ℑxᶠᵃᵃ(i, j + 3 - $add, k, grid, v)
            v₄ = ℑxᶠᵃᵃ(i, j + 4 - $add, k, grid, v)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i, j - 1 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i, j - 1 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i, j - 1 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i, j - 2 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i, j - 2 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i, j - 2 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i, j - 3 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i, j - 3 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i, j - 3 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = ψ(i, j - 4 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₀ = ℑyᵃᶠᵃ(i, j - 4 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₀ = ℑxᶠᵃᵃ(i, j - 4 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (u₀, u₁, u₂, u₃, u₄), (v₀, v₁, v₂, v₃, v₄), 5, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{6, FT, XT, YT, ZT},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, u, v)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, u, v)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, u, v)
            ψ₃ = ψ(i, j + 3 - $add, k, grid, u, v)
            ψ₄ = ψ(i, j + 4 - $add, k, grid, u, v)
            ψ₅ = ψ(i, j + 5 - $add, k, grid, u, v)

            u₀ = ℑyᵃᶠᵃ(i, j     - $add, k, grid, u)
            u₁ = ℑyᵃᶠᵃ(i, j + 1 - $add, k, grid, u)
            u₂ = ℑyᵃᶠᵃ(i, j + 2 - $add, k, grid, u)
            u₃ = ℑyᵃᶠᵃ(i, j + 3 - $add, k, grid, u)
            u₄ = ℑyᵃᶠᵃ(i, j + 4 - $add, k, grid, u)
            u₅ = ℑyᵃᶠᵃ(i, j + 5 - $add, k, grid, u)

            v₀ = ℑxᶠᵃᵃ(i, j     - $add, k, grid, v)
            v₁ = ℑxᶠᵃᵃ(i, j + 1 - $add, k, grid, v)
            v₂ = ℑxᶠᵃᵃ(i, j + 2 - $add, k, grid, v)
            v₃ = ℑxᶠᵃᵃ(i, j + 3 - $add, k, grid, v)
            v₄ = ℑxᶠᵃᵃ(i, j + 4 - $add, k, grid, v)
            v₅ = ℑxᶠᵃᵃ(i, j + 5 - $add, k, grid, v)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅), (u₀, u₁, u₂, u₃, u₄, u₅), (v₀, v₁, v₂, v₃, v₄, v₅), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₅ = ψ₄
            ψ₀ = ψ(i, j - 1 - $add, k, grid, u, v)

            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₅ = u₄
            u₀ = ℑyᵃᶠᵃ(i, j - 1 - $add, k, grid, u)

            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₅ = v₄
            v₀ = ℑxᶠᵃᵃ(i, j - 1 - $add, k, grid, v)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅), (u₀, u₁, u₂, u₃, u₄, u₅), (v₀, v₁, v₂, v₃, v₄, v₅), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(6), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₅ = ψ₄
            ψ₀ = ψ(i, j - 2 - $add, k, grid, u, v)

            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₅ = u₄
            u₀ = ℑyᵃᶠᵃ(i, j - 2 - $add, k, grid, u)

            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₅ = v₄
            v₀ = ℑxᶠᵃᵃ(i, j - 2 - $add, k, grid, v)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅), (u₀, u₁, u₂, u₃, u₄, u₅), (v₀, v₁, v₂, v₃, v₄, v₅), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(6), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₅ = ψ₄
            ψ₀ = ψ(i, j - 3 - $add, k, grid, u, v)

            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₅ = u₄
            u₀ = ℑyᵃᶠᵃ(i, j - 3 - $add, k, grid, u)

            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₅ = v₄
            v₀ = ℑxᶠᵃᵃ(i, j - 3 - $add, k, grid, v)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅), (u₀, u₁, u₂, u₃, u₄, u₅), (v₀, v₁, v₂, v₃, v₄, v₅), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(6), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₅ = ψ₄
            ψ₀ = ψ(i, j - 4 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₅ = u₄
            u₀ = ℑyᵃᶠᵃ(i, j - 4 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₅ = v₄
            v₀ = ℑxᶠᵃᵃ(i, j - 4 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅), (u₀, u₁, u₂, u₃, u₄, u₅), (v₀, v₁, v₂, v₃, v₄, v₅), 5, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(6), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₅ = ψ₄
            ψ₀ = ψ(i, j - 5 - $add, k, grid, u, v)
            
            u₁ = u₀
            u₂ = u₁
            u₃ = u₂
            u₄ = u₃
            u₅ = u₄
            u₀ = ℑyᵃᶠᵃ(i, j - 5 - $add, k, grid, u)
            
            v₁ = v₀
            v₂ = v₁
            v₃ = v₂
            v₄ = v₃
            v₅ = v₄
            v₀ = ℑxᶠᵃᵃ(i, j - 5 - $add, k, grid, v)
            
            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅), (u₀, u₁, u₂, u₃, u₄, u₅), (v₀, v₁, v₂, v₃, v₄, v₅), 6, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(6), Val(5))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end
    end
end