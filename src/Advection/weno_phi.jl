@inline function ϕ_reconstruction_stencil(buffer, shift, dir)
    N = buffer * 2
    order = shift == :symmetric ? N : N - 1
    if shift != :symmetric
        N = N .- 1
    end
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    stencil_full = Vector(undef, N)
    coeff = Symbol(:coeff, order, :_, shift)
    for (idx, n) in enumerate(rng)
        c = n - buffer - 1
        stencil_full[idx] = dir == :x ?
                            :(VI.func(i + $c, j, k, grid, args...)) :
                            dir == :y ?
                            :(VI.func(i, j + $c, k, grid, args...)) :
                            :(VI.func(i, j, k + $c, grid, args...))
    end
    return :($(stencil_full...),)
end

for side in [:left, :right], (dir, val, CT) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [1, 2, 3], [:XT, :YT, :ZT])
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir)
    biased_β           = Symbol(side, :_biased_β)
    biased_p           = Symbol(side, :_biased_p)
    coeff              = Symbol(:coeff_, side) 
    stencil            = Symbol(side, :_stencil_, dir)
    stencil_u          = Symbol(:tangential_, side, :_stencil_u)
    stencil_v          = Symbol(:tangential_, side, :_stencil_v)
    new_stencil        = Symbol(:new_stencil_, side, :_, dir)
    weno_interpolant   = Symbol(side, :_weno_interpolant_, dir)

    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}

            # All stencils
            ϕs = $(ϕ_reconstruction_stencil(2, side, dir))
            ψs = $(ψ_reconstruction_stencil(2, side, dir, true))
            
            β, ψ̅, C, α = $weno_interpolant(ψs[2:3], ϕs[2:3], 1, scheme, $val, idx, loc, args...)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:2], ϕs[1:2], 2, scheme, $val, idx, loc, args...)
            τ  += add_global_smoothness(β, Val(2), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = abs(τ)

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ϕs = $(ϕ_reconstruction_stencil(3, side, dir))
            ψs = $(ψ_reconstruction_stencil(3, side, dir, true))

            β, ψ̅, C, α = $weno_interpolant(ψs[3:5], ϕs[3:5], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:4], ϕs[2:4], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:3], ϕs[1:3], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = abs(τ)

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                    scheme::WENO{4, FT, XT, YT, ZT},
                                    ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ϕs = $(ϕ_reconstruction_stencil(4, side, dir))
            ψs = $(ψ_reconstruction_stencil(4, side, dir, true))

            β, ψ̅, C, α = $weno_interpolant(ψs[4:7], ϕs[4:7], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[3:6], ϕs[3:6], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:5], ϕs[2:5], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:4], ϕs[1:4], 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = abs(τ)

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ϕs = $(ϕ_reconstruction_stencil(5, side, dir))
            ψs = $(ψ_reconstruction_stencil(5, side, dir, true))

            β, ψ̅, C, α = $weno_interpolant(ψs[5:9], ϕs[5:9], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[4:8], ϕs[4:8], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[3:7], ϕs[3:7], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:6], ϕs[2:6], 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:5], ϕs[1:5], 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = abs(τ)

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{6, FT, XT, YT, ZT},
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ϕs = $(ϕ_reconstruction_stencil(6, side, dir))
            ψs = $(ψ_reconstruction_stencil(6, side, dir, true))

            β, ψ̅, C, α = $weno_interpolant(ψs[6:11], ϕs[6:11], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[5:10], ϕs[5:10], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[4:9], ϕs[4:9], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[3:8], ϕs[3:8], 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:7], ϕs[2:7], 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:6], ϕs[1:6], 6, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(5))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            τ = abs(τ)

            return (ψ̂₁ + ψ̂₂ * τ) / (w₁ + w₂ * τ)
        end
    end
end