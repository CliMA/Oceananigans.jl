@inline function ψ_reconstruction_stencil(buffer, shift, dir, func::Bool = false)
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
        if func
            stencil_full[idx] = dir == :x ?
                                :(ψ(i + $c, j, k, grid, args...)) :
                                dir == :y ?
                                :(ψ(i, j + $c, k, grid, args...)) :
                                :(ψ(i, j, k + $c, grid, args...))
        else
            stencil_full[idx] = dir == :x ?
                                :(ψ[i + $c, j, k]) :
                                dir == :y ?
                                :(ψ[i, j + $c, k]) :
                                :(ψ[i, j, k + $c])
        end
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
            ψs = $(ψ_reconstruction_stencil(2, side, dir))
            
            # Calculate x-velocity smoothness at stencil `s`
            β₀ = $biased_β(ψs[2:3], scheme, Val(0))
            β₁ = $biased_β(ψs[1:2], scheme, Val(1))
                
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs[2:3], $CT, Val($val), idx, loc) 
            ψ₁ = $biased_p(scheme, Val(1), ψs[1:2], $CT, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(2), (β₀, β₁))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
        
            return (ψ₀ * α₀ + ψ₁ * α₁) / (α₀ + α₁)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3, FT, XT, YT, ZT},
                                            ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            ψs = $(ψ_reconstruction_stencil(3, side, dir))
            
            # Calculate x-velocity smoothness at stencil `s`
            β₀ = $biased_β(ψs[3:5], scheme, Val(0))
            β₁ = $biased_β(ψs[2:4], scheme, Val(1))
            β₂ = $biased_β(ψs[1:3], scheme, Val(2))
            
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs[3:5], $CT, Val($val), idx, loc) 
            ψ₁ = $biased_p(scheme, Val(1), ψs[2:4], $CT, Val($val), idx, loc) 
            ψ₂ = $biased_p(scheme, Val(2), ψs[1:3], $CT, Val($val), idx, loc) 
            
            τ = global_smoothness_indicator(Val(3), (β₀, β₁, β₂))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂) / (α₀ + α₁ + α₂)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                    scheme::WENO{4, FT, XT, YT, ZT},
                                    ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            ψs = $(ψ_reconstruction_stencil(4, side, dir))
            
            # Calculate x-velocity smoothness at stencil `s`
            β₀ = $biased_β(ψs[4:7], scheme, Val(0))
            β₁ = $biased_β(ψs[3:6], scheme, Val(1))
            β₂ = $biased_β(ψs[2:5], scheme, Val(2))
            β₃ = $biased_β(ψs[1:4], scheme, Val(3))
            
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs[4:7], $CT, Val($val), idx, loc) 
            ψ₁ = $biased_p(scheme, Val(1), ψs[3:6], $CT, Val($val), idx, loc) 
            ψ₂ = $biased_p(scheme, Val(2), ψs[2:5], $CT, Val($val), idx, loc) 
            ψ₃ = $biased_p(scheme, Val(3), ψs[1:4], $CT, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(4), (β₀, β₁, β₂, β₃))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)
            α₃ = FT($coeff(scheme, Val(3))) * (1 + τ / (β₃ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂ + ψ₃ * α₃) / (α₀ + α₁ + α₂ + α₃)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5, FT, XT, YT, ZT},
                                            ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            ψs = $(ψ_reconstruction_stencil(5, side, dir))
            
            # Calculate x-velocity smoothness at stencil `s`
            β₀ = $biased_β(ψs[5:9], scheme, Val(0))
            β₁ = $biased_β(ψs[4:8], scheme, Val(1))
            β₂ = $biased_β(ψs[3:7], scheme, Val(2))
            β₃ = $biased_β(ψs[2:6], scheme, Val(3))
            β₄ = $biased_β(ψs[1:5], scheme, Val(4))
            
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs[5:9], $CT, Val($val), idx, loc) 
            ψ₁ = $biased_p(scheme, Val(1), ψs[4:8], $CT, Val($val), idx, loc) 
            ψ₂ = $biased_p(scheme, Val(2), ψs[3:7], $CT, Val($val), idx, loc) 
            ψ₃ = $biased_p(scheme, Val(3), ψs[2:6], $CT, Val($val), idx, loc) 
            ψ₄ = $biased_p(scheme, Val(4), ψs[1:5], $CT, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(5), (β₀, β₁, β₂, β₃, β₄))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)
            α₃ = FT($coeff(scheme, Val(3))) * (1 + τ / (β₃ + FT(ε))^2)
            α₄ = FT($coeff(scheme, Val(4))) * (1 + τ / (β₄ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂ + ψ₃ * α₃ + ψ₄ * α₄) / (α₀ + α₁ + α₂ + α₃ + α₄)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{6, FT, XT, YT, ZT},
                                            ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            ψs = $(ψ_reconstruction_stencil(6, side, dir))
            
            # Calculate x-velocity smoothness at stencil `s`
            β₀ = $biased_β(ψs[6:11], scheme, Val(0))
            β₁ = $biased_β(ψs[5:10], scheme, Val(1))
            β₂ = $biased_β(ψs[4:9],  scheme, Val(2))
            β₃ = $biased_β(ψs[3:8],  scheme, Val(3))
            β₄ = $biased_β(ψs[2:7],  scheme, Val(4))
            β₅ = $biased_β(ψs[1:6],  scheme, Val(5))

            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs[6:11], $CT, Val($val), idx, loc) 
            ψ₁ = $biased_p(scheme, Val(1), ψs[5:10], $CT, Val($val), idx, loc) 
            ψ₂ = $biased_p(scheme, Val(2), ψs[4:9],  $CT, Val($val), idx, loc) 
            ψ₃ = $biased_p(scheme, Val(3), ψs[3:8],  $CT, Val($val), idx, loc) 
            ψ₄ = $biased_p(scheme, Val(4), ψs[2:7],  $CT, Val($val), idx, loc) 
            ψ₅ = $biased_p(scheme, Val(5), ψs[1:6],  $CT, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(6), (β₀, β₁, β₂, β₃, β₄, β₅))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)
            α₃ = FT($coeff(scheme, Val(3))) * (1 + τ / (β₃ + FT(ε))^2)
            α₄ = FT($coeff(scheme, Val(4))) * (1 + τ / (β₄ + FT(ε))^2)
            α₅ = FT($coeff(scheme, Val(5))) * (1 + τ / (β₅ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂ + ψ₃ * α₃ + ψ₄ * α₄ + ψ₅ * α₅) / (α₀ + α₁ + α₂ + α₃ + α₄ + α₅)
        end
    end
end