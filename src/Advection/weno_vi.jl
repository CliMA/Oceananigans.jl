@inline function u_reconstruction_stencil(buffer, shift, dir)
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
                            :(ℑyᵃᶠᵃ(i + $c, j, k, grid, u)) :
                            dir == :y ?
                            :(ℑyᵃᶠᵃ(i, j + $c, k, grid, u)) :
                            :(ℑyᵃᶠᵃ(i, j, k + $c, grid, u))
    end
    return :($(stencil_full...),)
end

@inline function v_reconstruction_stencil(buffer, shift, dir)
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
                            :(ℑxᶠᵃᵃ(i + $c, j, k, grid, v)) :
                            dir == :y ?
                            :(ℑxᶠᵃᵃ(i, j + $c, k, grid, v)) :
                            :(ℑxᶠᵃᵃ(i, j, k + $c, grid, v))
    end
    return :($(stencil_full...),)
end

@inline function ζ_reconstruction_stencil(buffer, shift, dir)
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
                            :(ψ(i + $c, j, k, grid, u, v)) :
                            dir == :y ?
                            :(ψ(i, j + $c, k, grid, u, v)) :
                            :(ψ(i, j, k + $c, grid, u, v))
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
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}

            # All stencils
            us = $(u_reconstruction_stencil(2, side, dir))
            vs = $(v_reconstruction_stencil(2, side, dir))
            ψs = $(ζ_reconstruction_stencil(2, side, dir))
            
            β, ψ̅, C, α = $weno_interpolant(ψs[2:3], us[2:3], vs[2:3], 1, scheme, $val, idx, loc, args...)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:2], us[1:2], vs[1:2], 2, scheme, $val, idx, loc, args...)
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
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            us = $(u_reconstruction_stencil(3, side, dir))
            vs = $(v_reconstruction_stencil(3, side, dir))
            ψs = $(ζ_reconstruction_stencil(3, side, dir))

            β, ψ̅, C, α = $weno_interpolant(ψs[3:5], us[3:5], vs[3:5], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:4], us[2:4], vs[2:4], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:3], us[1:3], vs[1:3], 3, scheme, $val, idx, loc)
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
                                    ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            us = $(u_reconstruction_stencil(4, side, dir))
            vs = $(v_reconstruction_stencil(4, side, dir))
            ψs = $(ζ_reconstruction_stencil(4, side, dir))

            β, ψ̅, C, α = $weno_interpolant(ψs[4:7], us[4:7], vs[4:7], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[3:6], us[3:6], vs[3:6], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:5], us[2:5], vs[2:5], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:4], us[1:4], vs[1:4], 4, scheme, $val, idx, loc)
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
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            us = $(u_reconstruction_stencil(5, side, dir))
            vs = $(v_reconstruction_stencil(5, side, dir))
            ψs = $(ζ_reconstruction_stencil(5, side, dir))

            β, ψ̅, C, α = $weno_interpolant(ψs[5:9], us[5:9], vs[5:9], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[4:8], us[4:8], vs[4:8], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[3:7], us[3:7], vs[3:7], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:6], us[2:6], vs[2:6], 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:5], us[1:5], vs[1:5], 5, scheme, $val, idx, loc)
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
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            us = $(u_reconstruction_stencil(6, side, dir))
            vs = $(v_reconstruction_stencil(6, side, dir))
            ψs = $(ζ_reconstruction_stencil(6, side, dir))

            β, ψ̅, C, α = $weno_interpolant(ψs[6:11], us[6:11], vs[6:11], 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * C
            w₁ = C
            ψ̂₂ = ψ̅ * α  
            w₂ = α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[5:10], us[5:10], vs[5:10], 2, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(1))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[4:9], us[4:9], vs[4:9], 3, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(2))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[3:8], us[3:8], vs[3:8], 4, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(3))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[2:7], us[2:7], vs[2:7], 5, scheme, $val, idx, loc)
            τ  += add_global_smoothness(β, Val(6), Val(4))
            ψ̂₁ += ψ̅ * C
            w₁ += C
            ψ̂₂ += ψ̅ * α  
            w₂ += α

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant(ψs[1:6], us[1:6], vs[1:6], 6, scheme, $val, idx, loc)
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