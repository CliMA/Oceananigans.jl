#####
##### STENCILS IN X
#####

for (side, add) in zip([:left, :right], (-1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_xᶠᵃᵃ)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_xᶠᵃᵃ)
    val = 1
    
    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, args...)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, args...)

            ϕ₀ = VI.func(i     - $add, j, k, grid, args...)
            ϕ₁ = VI.func(i + 1 - $add, j, k, grid, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i - 1 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            
            ϕ₀ = VI.func(i - 1 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, args...)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, args...)
            ψ₂ = ψ(i + 2 - $add, j, k, grid, args...)

            ϕ₀ = VI.func(i     - $add, j, k, grid, args...)
            ϕ₁ = VI.func(i + 1 - $add, j, k, grid, args...)
            ϕ₂ = VI.func(i + 2 - $add, j, k, grid, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i - 1 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁

            ϕ₀ = VI.func(i - 1 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i - 2 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁

            ϕ₀ = VI.func(i - 2 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, args...)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, args...)
            ψ₂ = ψ(i + 2 - $add, j, k, grid, args...)
            ψ₃ = ψ(i + 3 - $add, j, k, grid, args...)

            ϕ₀ = VI.func(i     - $add, j, k, grid, args...)
            ϕ₁ = VI.func(i + 1 - $add, j, k, grid, args...)
            ϕ₂ = VI.func(i + 2 - $add, j, k, grid, args...)
            ϕ₃ = VI.func(i + 3 - $add, j, k, grid, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i - 1 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            ϕ₀ = VI.func(i - 1 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i - 2 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            ϕ₀ = VI.func(i - 2 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i - 3 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            ϕ₀ = VI.func(i - 3 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α


            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i     - $add, j, k, grid, args...)
            ψ₁ = ψ(i + 1 - $add, j, k, grid, args...)
            ψ₂ = ψ(i + 2 - $add, j, k, grid, args...)
            ψ₃ = ψ(i + 3 - $add, j, k, grid, args...)
            ψ₄ = ψ(i + 4 - $add, j, k, grid, args...)

            ϕ₀ = VI.func(i     - $add, j, k, grid, args...)
            ϕ₁ = VI.func(i + 1 - $add, j, k, grid, args...)
            ϕ₂ = VI.func(i + 2 - $add, j, k, grid, args...)
            ϕ₃ = VI.func(i + 3 - $add, j, k, grid, args...)
            ϕ₄ = VI.func(i + 4 - $add, j, k, grid, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i - 1 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i - 1 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i - 2 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i - 2 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i - 3 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i - 3 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i - 4 - $add, j, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i - 4 - $add, j, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end
    end
end

#####
##### STENCILS IN Y
#####

for (side, add) in zip([:left, :right], (-1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_yᵃᶠᵃ)
    weno_interpolant   = Symbol(side, :_weno_interpolant_yᵃᶠᵃ)
    val = 2
    
    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, args...)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, args...)

            ϕ₀ = VI.func(i, j     - $add, k, grid, args...)
            ϕ₁ = VI.func(i, j + 1 - $add, k, grid, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i, j - 1 - $add, k, grid, args...)
            ψ₁ = ψ₀
            
            ϕ₀ = VI.func(i, j - 1 - $add, k, grid, args...)
            ϕ₁ = ϕ₀

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, args...)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, args...)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, args...)

            ϕ₀ = VI.func(i, j     - $add, k, grid, args...)
            ϕ₁ = VI.func(i, j + 1 - $add, k, grid, args...)
            ϕ₂ = VI.func(i, j + 2 - $add, k, grid, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i, j - 1 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁

            ϕ₀ = VI.func(i, j - 1 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i, j - 2 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁

            ϕ₀ = VI.func(i, j - 2 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, args...)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, args...)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, args...)
            ψ₃ = ψ(i, j + 3 - $add, k, grid, args...)

            ϕ₀ = VI.func(i, j     - $add, k, grid, args...)
            ϕ₁ = VI.func(i, j + 1 - $add, k, grid, args...)
            ϕ₂ = VI.func(i, j + 2 - $add, k, grid, args...)
            ϕ₃ = VI.func(i, j + 3 - $add, k, grid, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i, j - 1 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            ϕ₀ = VI.func(i, j - 1 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i, j - 2 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            ϕ₀ = VI.func(i, j - 2 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i, j - 3 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            ϕ₀ = VI.func(i, j - 3 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α


            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = ψ(i, j     - $add, k, grid, args...)
            ψ₁ = ψ(i, j + 1 - $add, k, grid, args...)
            ψ₂ = ψ(i, j + 2 - $add, k, grid, args...)
            ψ₃ = ψ(i, j + 3 - $add, k, grid, args...)
            ψ₄ = ψ(i, j + 4 - $add, k, grid, args...)

            ϕ₀ = VI.func(i, j     - $add, k, grid, args...)
            ϕ₁ = VI.func(i, j + 1 - $add, k, grid, args...)
            ϕ₂ = VI.func(i, j + 2 - $add, k, grid, args...)
            ϕ₃ = VI.func(i, j + 3 - $add, k, grid, args...)
            ϕ₄ = VI.func(i, j + 4 - $add, k, grid, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = ψ(i, j - 1 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i, j - 1 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i, j - 2 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i, j - 2 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i, j - 3 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i, j - 3 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = ψ(i, j - 4 - $add, k, grid, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃

            ϕ₀ = VI.func(i, j - 4 - $add, k, grid, args...)
            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ^2

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end
    end
end