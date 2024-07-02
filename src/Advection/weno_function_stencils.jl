#####
##### STENCILS IN X
#####


for (side, add) in zip([:left, :right], (1, 0)), (dir, loc, val) in zip((:x, :y, :z), (:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (1, 2, 3))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir, loc)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_, dir, loc)
    get_shifted_value  = Symbol(:get_shifted_value_, dir)
    
    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,     - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, ψ, args...)

            ϕ₀ = $get_shifted_value(i, j, k, grid,     - $add, VI.func, args...)
            ϕ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)
            
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), (ϕ₀, ϕ₁), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(2), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ^2

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,     - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, ψ, args...)
            ψ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, ψ, args...)

            ϕ₀ = $get_shifted_value(i, j, k, grid,     - $add, VI.func, args...)
            ϕ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, VI.func, args...)
            ϕ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), (ϕ₀, ϕ₁, ϕ₂), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(3))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ^2

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,     - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, ψ, args...)
            ψ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, ψ, args...)
            ψ₃ = $get_shifted_value(i, j, k, grid, + 3 - $add, ψ, args...)

            ϕ₀ = $get_shifted_value(i, j, k, grid,     - $add, VI.func, args...)
            ϕ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, VI.func, args...)
            ϕ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, VI.func, args...)
            ϕ₃ = $get_shifted_value(i, j, k, grid, + 3 - $add, VI.func, args...)
        
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 3 - $add, ψ, args...)
            
            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 3 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), (ϕ₀, ϕ₁, ϕ₂, ϕ₃), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(4))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ^2

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,     - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, ψ, args...)
            ψ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, ψ, args...)
            ψ₃ = $get_shifted_value(i, j, k, grid, + 3 - $add, ψ, args...)
            ψ₄ = $get_shifted_value(i, j, k, grid, + 4 - $add, ψ, args...)

            ϕ₀ = $get_shifted_value(i, j, k, grid,     - $add, VI.func, args...)
            ϕ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, VI.func, args...)
            ϕ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, VI.func, args...)
            ϕ₃ = $get_shifted_value(i, j, k, grid, + 3 - $add, VI.func, args...)
            ϕ₄ = $get_shifted_value(i, j, k, grid, + 4 - $add, VI.func, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            ϕ₄ = ϕ₃
            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

            ϕ₄ = ϕ₃
            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 3 - $add, ψ, args...)

            ϕ₄ = ϕ₃
            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 3 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 4 - $add, ψ, args...)

            ϕ₄ = ϕ₃
            ϕ₃ = ϕ₂
            ϕ₂ = ϕ₁
            ϕ₁ = ϕ₀
            ϕ₀ = $get_shifted_value(i, j, k, grid, - 4 - $add, VI.func, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), (ϕ₀, ϕ₁, ϕ₂, ϕ₃, ϕ₄), 5, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(5))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ^2

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end
    end
end
