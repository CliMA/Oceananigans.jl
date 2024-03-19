@inline getvalue(i, j, k, grid, ψ, args...) = @inbounds ψ[i, j, k]
@inline getvalue(i, j, k, grid, ψ::Function, args...) = ψ(i, j, k, grid, args...)

#####
##### STENCILS IN X
#####

for (side, add) in zip([:left, :right], (-1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_xᶠᵃᵃ)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_xᶠᵃᵃ)
    val = 1
    
    @eval begin
        # Fallback for DefaultStencil formulations and disambiguation
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{2}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{3}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{4}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{5}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{6}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{2, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)
            ψ₁ = ψ₀

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(2), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{3, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)
            ψ₂ = getvalue(i + 2 - $add, j, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = getvalue(i - 2 - $add, j, k, grid, ψ, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{4, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)
            ψ₂ = getvalue(i + 2 - $add, j, k, grid, ψ, args...)
            ψ₃ = getvalue(i + 3 - $add, j, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₀ = getvalue(i - 1 - $add, j, k, grid, ψ, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = getvalue(i - 2 - $add, j, k, grid, ψ, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₀ = getvalue(i - 3 - $add, j, k, grid, ψ, args...)
            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{5, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i     - $add, j, k, grid, ψ, args...)
            ψ₁ = getvalue(i + 1 - $add, j, k, grid, ψ, args...)
            ψ₂ = getvalue(i + 2 - $add, j, k, grid, ψ, args...)
            ψ₃ = getvalue(i + 3 - $add, j, k, grid, ψ, args...)
            ψ₄ = getvalue(i + 4 - $add, j, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 1, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 2, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 3, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 4, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end
    end
end

#####
##### STENCILS IN Y
#####

for (side, add) in zip([:left, :right], (-1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_yᵃᶠᵃ)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_yᵃᶠᵃ)
    val = 2
    
    @eval begin
        # Fallback for DefaultStencil formulations and disambiguation
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{2}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{3}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{4}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{5}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{6}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{2, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(2), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{3, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)
            ψ₂ = getvalue(i, j + 2 - $add, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i, j - 2 - $add, k, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{4, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)
            ψ₂ = getvalue(i, j + 2 - $add, k, grid, ψ, args...)
            ψ₃ = getvalue(i, j + 3 - $add, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j - 1 - $add, k, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j - 2 - $add, k, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j - 3 - $add, k, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{5, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j     - $add, k, grid, ψ, args...)
            ψ₁ = getvalue(i, j + 1 - $add, k, grid, ψ, args...)
            ψ₂ = getvalue(i, j + 2 - $add, k, grid, ψ, args...)
            ψ₃ = getvalue(i, j + 3 - $add, k, grid, ψ, args...)
            ψ₄ = getvalue(i, j + 4 - $add, k, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 1, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 2, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 3, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 4, scheme, $val, idx, loc)
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

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end
    end
end

#####
##### STENCILS IN Z
#####

for (side, add) in zip([:left, :right], (-1, 0))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_zᵃᵃᶠ)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_zᵃᵃᶠ)
    val = 3
    
    @eval begin
        # Fallback for DefaultStencil formulations and disambiguation
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{2}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{3}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{4}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{5}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
        @inline $biased_interpolate(i, j, k, grid, scheme::WENO{6}, ψ, idx, loc, ::DefaultStencil, args...) = 
                                    $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{2, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j, k     - $add, grid, ψ, args...)
            ψ₁ = getvalue(i, j, k + 1 - $add, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₀ = getvalue(i, j, k - 1 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(2), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{3, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j, k     - $add, grid, ψ, args...)
            ψ₁ = getvalue(i, j, k + 1 - $add, grid, ψ, args...)
            ψ₂ = getvalue(i, j, k + 2 - $add, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i, j, k - 1 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₀ = getvalue(i, j, k - 2 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{4, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j, k     - $add, grid, ψ, args...)
            ψ₁ = getvalue(i, j, k + 1 - $add, grid, ψ, args...)
            ψ₂ = getvalue(i, j, k + 2 - $add, grid, ψ, args...)
            ψ₃ = getvalue(i, j, k + 3 - $add, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j, k - 1 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j, k - 2 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₀ = getvalue(i, j, k - 3 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{5, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = getvalue(i, j, k     - $add, grid, ψ, args...)
            ψ₁ = getvalue(i, j, k + 1 - $add, grid, ψ, args...)
            ψ₂ = getvalue(i, j, k + 2 - $add, grid, ψ, args...)
            ψ₃ = getvalue(i, j, k + 3 - $add, grid, ψ, args...)
            ψ₄ = getvalue(i, j, k + 4 - $add, grid, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j, k - 1 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j, k - 2 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j, k - 3 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            ψ₁ = ψ₀
            ψ₂ = ψ₁
            ψ₃ = ψ₂
            ψ₄ = ψ₃
            ψ₀ = getvalue(i, j, k - 4 - $add, grid, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = τ * τ

            return (ψ̂₁ + ψ̂₂ * τ) * rcp(w₁ + w₂ * τ)
        end
    end
end
