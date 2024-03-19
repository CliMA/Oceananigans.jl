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
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)

            ϕ₀ = getvalue(i     - $add, j, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i + 1 - $add, j, k, grid, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)
            
            ϕ₁ = ϕ₀
            ϕ₀ = getvalue(i - 1 - $add, j, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)
            ψ₂ = getvalue(i + 2 - $add, j, k, grid, ψ, args...)

            ϕ₀ = getvalue(i     - $add, j, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i + 1 - $add, j, k, grid, VI.func, args...)
            ϕ₂ = getvalue(i + 2 - $add, j, k, grid, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₀ = getvalue(i - 1 - $add, j, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i - 2 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₀ = getvalue(i - 2 - $add, j, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)
            ψ₂ = getvalue(i + 2 - $add, j, k, grid, ψ, args...)
            ψ₃ = getvalue(i + 3 - $add, j, k, grid, ψ, args...)

            ϕ₀ = getvalue(i     - $add, j, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i + 1 - $add, j, k, grid, VI.func, args...)
            ϕ₂ = getvalue(i + 2 - $add, j, k, grid, VI.func, args...)
            ϕ₃ = getvalue(i + 3 - $add, j, k, grid, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₀ = getvalue(i - 1 - $add, j, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i - 2 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₀ = getvalue(i - 2 - $add, j, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i - 3 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₀ = getvalue(i - 3 - $add, j, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)
            ψ₂ = getvalue(i + 2 - $add, j, k, grid, ψ, args...)
            ψ₃ = getvalue(i + 3 - $add, j, k, grid, ψ, args...)
            ψ₄ = getvalue(i + 4 - $add, j, k, grid, ψ, args...)

            ϕ₀ = getvalue(i     - $add, j, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i + 1 - $add, j, k, grid, VI.func, args...)
            ϕ₂ = getvalue(i + 2 - $add, j, k, grid, VI.func, args...)
            ϕ₃ = getvalue(i + 3 - $add, j, k, grid, VI.func, args...)
            ϕ₄ = getvalue(i + 4 - $add, j, k, grid, VI.func, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i - 1 - $add, j, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i - 2 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃            
            ϕ₀ = getvalue(i - 2 - $add, j, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i - 3 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i - 3 - $add, j, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i - 4 - $add, j, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i - 4 - $add, j, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)

            ϕ₀ = getvalue(i, j     - $add, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i, j + 1 - $add, k, grid, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)
            
            ϕ₁ = ϕ₀
            ϕ₀ = getvalue(i, j - 1 - $add, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)
            ψ₂ = getvalue(i, j + 2 - $add, k, grid, ψ, args...)

            ϕ₀ = getvalue(i, j     - $add, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i, j + 1 - $add, k, grid, VI.func, args...)
            ϕ₂ = getvalue(i, j + 2 - $add, k, grid, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₀ = getvalue(i, j - 1 - $add, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i, j - 2 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₀ = getvalue(i, j - 2 - $add, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)
            ψ₂ = getvalue(i, j + 2 - $add, k, grid, ψ, args...)
            ψ₃ = getvalue(i, j + 3 - $add, k, grid, ψ, args...)

            ϕ₀ = getvalue(i, j     - $add, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i, j + 1 - $add, k, grid, VI.func, args...)
            ϕ₂ = getvalue(i, j + 2 - $add, k, grid, VI.func, args...)
            ϕ₃ = getvalue(i, j + 3 - $add, k, grid, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₀ = getvalue(i, j - 1 - $add, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j - 2 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₀ = getvalue(i, j - 2 - $add, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j - 3 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₀ = getvalue(i, j - 3 - $add, k, grid, VI.func, args...)

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
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)
            ψ₂ = getvalue(i, j + 2 - $add, k, grid, ψ, args...)
            ψ₃ = getvalue(i, j + 3 - $add, k, grid, ψ, args...)
            ψ₄ = getvalue(i, j + 4 - $add, k, grid, ψ, args...)

            ϕ₀ = getvalue(i, j     - $add, k, grid, VI.func, args...)
            ϕ₁ = getvalue(i, j + 1 - $add, k, grid, VI.func, args...)
            ϕ₂ = getvalue(i, j + 2 - $add, k, grid, VI.func, args...)
            ϕ₃ = getvalue(i, j + 3 - $add, k, grid, VI.func, args...)
            ϕ₄ = getvalue(i, j + 4 - $add, k, grid, VI.func, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i, j - 1 - $add, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j - 2 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i, j - 2 - $add, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j - 3 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i, j - 3 - $add, k, grid, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j - 4 - $add, k, grid, ψ, args...)

            ϕ₁ = ϕ₀
            ϕ₂ = ϕ₁
            ϕ₃ = ϕ₂
            ϕ₄ = ϕ₃
            ϕ₀ = getvalue(i, j - 4 - $add, k, grid, VI.func, args...)

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