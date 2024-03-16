@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-2, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-3, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-4, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-5, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-6, j, k, grid, args...), ψs[1:N-1]...)

@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-2, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-3, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-4, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-5, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-6, k, grid, args...), ψs[1:N-1]...)

@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-2, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-3, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-4, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-5, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-6, grid, args...), ψs[1:N-1]...)

@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i-2, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i-3, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i-4, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i-5, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i-6, j, k], ψs[1:N-1]...)

@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-2, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-3, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-4, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-5, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-6, k], ψs[1:N-1]...)

@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψargs...) where N = @inbounds (ψ[i, j, k-2], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψargs...) where N = @inbounds (ψ[i, j, k-3], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψargs...) where N = @inbounds (ψ[i, j, k-4], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψargs...) where N = @inbounds (ψ[i, j, k-5], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψargs...) where N = @inbounds (ψ[i, j, k-6], ψs[1:N-1]...)

for side in [:left, :right], (dir, val) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [1, 2, 3])
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir)
    biased_β           = Symbol(side, :_biased_β)
    biased_p           = Symbol(side, :_biased_p)
    coeff              = Symbol(:coeff_, side) 
    stencil            = Symbol(side, :_stencil_, dir)
    stencil_u          = Symbol(:tangential_, side, :_stencil_u)
    stencil_v          = Symbol(:tangential_, side, :_stencil_v)
    new_stencil        = Symbol(:new_stencil_, side, dir)

    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) 
        
            # Stencil S₀
            us = $stencil_u(i, j, k, scheme, Val(1), Val($val), grid, u)
            vs = $stencil_v(i, j, k, scheme, Val(1), Val($val), grid, v)
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(0))
            βv = $biased_β(vs, scheme, Val(0))

            # total smoothness
            β₀ = (βu + βv) / 2
        
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            
            # Stencil S₁
            us = $new_stencil(i, j, k, scheme, Val(2), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(2), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(2), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(1))
            βv = $biased_β(vs, scheme, Val(1))

            # total smoothness
            β₁ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₁ = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(2), (β₀, β₁))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁) / (α₀ + α₁)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{3},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) 
        
            # Stencil S₀
            us = $stencil_u(i, j, k, scheme, Val(1), Val($val), grid, u)
            vs = $stencil_v(i, j, k, scheme, Val(1), Val($val), grid, v)
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(0))
            βv = $biased_β(vs, scheme, Val(0))

            # total smoothness
            β₀ = (βu + βv) / 2
        
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            
            # Stencil S₁
            us = $new_stencil(i, j, k, scheme, Val(2), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(2), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(2), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(1))
            βv = $biased_β(vs, scheme, Val(1))

            # total smoothness
            β₁ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₁ = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 

            # Stencil S₂
            us = $new_stencil(i, j, k, scheme, Val(3), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(3), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(3), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(2))
            βv = $biased_β(vs, scheme, Val(2))

            # total smoothness
            β₂ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₂ = $biased_p(scheme, Val(2), ψs, Nothing, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(3), (β₀, β₁, β₂))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂) / (α₀ + α₁ + α₂)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                    scheme::WENO{4},
                                    ψ, idx, loc, ::VelocityStencil, u, v, args...) 
        
            # Stencil S₀
            us = $stencil_u(i, j, k, scheme, Val(1), Val($val), grid, u)
            vs = $stencil_v(i, j, k, scheme, Val(1), Val($val), grid, v)
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(0))
            βv = $biased_β(vs, scheme, Val(0))

            # total smoothness
            β₀ = (βu + βv) / 2
        
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            
            # Stencil S₁
            us = $new_stencil(i, j, k, scheme, Val(2), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(2), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(2), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(1))
            βv = $biased_β(vs, scheme, Val(1))

            # total smoothness
            β₁ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₁ = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 

            # Stencil S₂
            us = $new_stencil(i, j, k, scheme, Val(3), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(3), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(3), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(2))
            βv = $biased_β(vs, scheme, Val(2))

            # total smoothness
            β₂ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₂ = $biased_p(scheme, Val(2), ψs, Nothing, Val($val), idx, loc) 

            # Stencil S₃
            us = $new_stencil(i, j, k, scheme, Val(4), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(4), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(4), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(3))
            βv = $biased_β(vs, scheme, Val(3))

            # total smoothness
            β₃ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₃ = $biased_p(scheme, Val(3), ψs, Nothing, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(4), (β₀, β₁, β₂, β₃))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)
            α₃ = FT($coeff(scheme, Val(3))) * (1 + τ / (β₃ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂ + ψ₃ * α₃) / (α₀ + α₁ + α₂ + α₃)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{5},
                                            ψ, idx, loc, ::VelocityStencil, u, v, args...) 
        
            # Stencil S₀
            us = $stencil_u(i, j, k, scheme, Val(1), Val($val), grid, u)
            vs = $stencil_v(i, j, k, scheme, Val(1), Val($val), grid, v)
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(0))
            βv = $biased_β(vs, scheme, Val(0))

            # total smoothness
            β₀ = (βu + βv) / 2
        
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₀ = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            
            # Stencil S₁
            us = $new_stencil(i, j, k, scheme, Val(2), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(2), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(2), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(1))
            βv = $biased_β(vs, scheme, Val(1))

            # total smoothness
            β₁ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₁ = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 

            # Stencil S₂
            us = $new_stencil(i, j, k, scheme, Val(3), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(3), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(3), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(2))
            βv = $biased_β(vs, scheme, Val(2))

            # total smoothness
            β₂ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₂ = $biased_p(scheme, Val(2), ψs, Nothing, Val($val), idx, loc) 

            # Stencil S₃
            us = $new_stencil(i, j, k, scheme, Val(4), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(4), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(4), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(3))
            βv = $biased_β(vs, scheme, Val(3))

            # total smoothness
            β₃ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₃ = $biased_p(scheme, Val(3), ψs, Nothing, Val($val), idx, loc) 

            # Stencil S₄
            us = $new_stencil(i, j, k, scheme, Val(5), us, ℑyᵃᶠᵃ, grid, u)
            vs = $new_stencil(i, j, k, scheme, Val(5), vs, ℑxᶠᵃᵃ, grid, v)
            ψs = $new_stencil(i, j, k, scheme, Val(5), ψs, ψ, grid, u, v, args...)

            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(4))
            βv = $biased_β(vs, scheme, Val(4))

            # total smoothness
            β₄ = (βu + βv) / 2
    
            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ₄ = $biased_p(scheme, Val(4), ψs, Nothing, Val($val), idx, loc) 

            τ = global_smoothness_indicator(Val(5), (β₀, β₁, β₂, β₃, β₄))

            α₀ = FT($coeff(scheme, Val(0))) * (1 + τ / (β₀ + FT(ε))^2)
            α₁ = FT($coeff(scheme, Val(1))) * (1 + τ / (β₁ + FT(ε))^2)
            α₂ = FT($coeff(scheme, Val(2))) * (1 + τ / (β₂ + FT(ε))^2)
            α₃ = FT($coeff(scheme, Val(3))) * (1 + τ / (β₃ + FT(ε))^2)
            α₄ = FT($coeff(scheme, Val(4))) * (1 + τ / (β₄ + FT(ε))^2)

            return (ψ₀ * α₀ + ψ₁ * α₁ + ψ₂ * α₂ + ψ₃ * α₃ + ψ₄ * α₄) / (α₀ + α₁ + α₂ + α₃ + α₄)
        end
    end
end