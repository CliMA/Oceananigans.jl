using Printf

@inline get_shifted_value_x(i, j, k, grid, shift, ψ, args...) = @inbounds ψ[i + shift, j, k]
@inline get_shifted_value_x(i, j, k, grid, shift, ψ::Function, args...) = ψ(i + shift, j, k, grid, args...)

@inline get_shifted_value_y(i, j, k, grid, shift, ψ, args...) = @inbounds ψ[i, j + shift, k]
@inline get_shifted_value_y(i, j, k, grid, shift, ψ::Function, args...) = ψ(i, j + shift, k, grid, args...)

@inline get_shifted_value_z(i, j, k, grid, shift, ψ, args...) = @inbounds ψ[i, j, k + shift]
@inline get_shifted_value_z(i, j, k, grid, shift, ψ::Function, args...) = ψ(i, j, k + shift, grid, args...)

#####
##### STENCILS IN X
#####

for (side, add) in zip([:left, :right], (1, 0)), (dir, loc, val) in zip((:x, :y, :z), (:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (1, 2, 3))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir, loc)
    coeff              = Symbol(:coeff_, side) 
    weno_interpolant   = Symbol(side, :_weno_interpolant_, dir, loc)
    get_shifted_value  = Symbol(:get_shifted_value_, dir)

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
                                             scheme::WENO{2},
                                             ψ, idx, loc, args...) 
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,   - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, 1 - $add, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 1, scheme, $val, idx, loc)
            τ  = add_to_global_smoothness(β, Val(2), Val(1))
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(2), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ * τ

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{3, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,   - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, 1 - $add, ψ, args...)
            ψ₂ = $get_shifted_value(i, j, k, grid, 2 - $add, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 1, scheme, $val, idx, loc)
            τ  = add_to_global_smoothness(β, Val(3), Val(1))
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(3), Val(3))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ * τ

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1) 
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{4, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,   - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, 1 - $add, ψ, args...)
            ψ₂ = $get_shifted_value(i, j, k, grid, 2 - $add, ψ, args...)
            ψ₃ = $get_shifted_value(i, j, k, grid, 3 - $add, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 1, scheme, $val, idx, loc)
            τ  = add_to_global_smoothness(β, Val(4), Val(1))
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(3))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k,  grid, - 3 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(4), Val(4))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ * τ

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{5, FT, XT, YT, ZT},
                                             ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
            # All stencils
            ψ₀ = $get_shifted_value(i, j, k, grid,     - $add, ψ, args...)
            ψ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, ψ, args...)
            ψ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, ψ, args...)
            ψ₃ = $get_shifted_value(i, j, k, grid, + 3 - $add, ψ, args...)
            ψ₄ = $get_shifted_value(i, j, k, grid, + 4 - $add, ψ, args...)

            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 1, scheme, $val, idx, loc)
            τ  = β
            ψ̂₁ = ψ̅ * α  
            ψ̂₂ = ψ̅ * C
            w₁ = α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 2, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(2))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 3, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(3))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 3 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 4, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(4))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            ψ₄ = ψ₃
            ψ₃ = ψ₂
            ψ₂ = ψ₁
            ψ₁ = ψ₀
            ψ₀ = $get_shifted_value(i, j, k, grid, - 4 - $add, ψ, args...)

            # Stencil S₁
            β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 5, scheme, $val, idx, loc)
            τ  += add_to_global_smoothness(β, Val(5), Val(5))
            ψ̂₁ += ψ̅ * α  
            ψ̂₂ += ψ̅ * C
            w₁ += α

            τ = τ * τ

            return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
        end
    end
end


#= Metaprogrammed implementation (to finish)

@inline function $biased_interpolate(i, j, k, grid, 
                                        scheme::WENO{$N},
                                        ψ, idx, loc, args...) 

    # All stencils
    $(retrieve_initial_stencil(N, dir, add))

    β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 1, scheme, $val, idx, loc)
    τ  = β
    ψ̂₁ = ψ̅ * α  
    ψ̂₂ = ψ̅ * C
    w₁ = α

    @unroll for s in 2:$N
        $(update_stencil(N, dir, - s + 1, add))  
        β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), s, scheme, $val, idx, loc)
        τ  += add_to_global_smoothness(β, Val($N), Val(s))
        ψ̂₁ += ψ̅ * α  
        ψ̂₂ += ψ̅ * C
        w₁ += α
    end

    τ = τ * τ

    return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
end

# TO finish
stencil(N) = :($(Symbol(:ψ, i) for i in 1:N))

function retrieve_initial_stencil(N, dir, add)
    expr = []
    get_shifted_value = Symbol(:get_shifted_value_, dir)    
    for i in 0:N-1
        ψ = Symbol(:ψ, i)
        push!(expr, :($ψ = $get_shifted_value(i, j, k, grid, $i - $add, ψ, args...)))
    end
    eblk = Expr(:block, expr...)

    return eblk
end

function update_stencil(N, dir, newval, add)
    expr = []
    get_shifted_value = Symbol(:get_shifted_value_, dir)    
    for i in N-1:-1:1
        ψ  = Symbol(:ψ, i)
        ψp = Symbol(:ψ, i-1)
        push!(expr, :($ψ = $ψp))
    end
    ψ  = :ψ0
    push!(expr, :($ψ = $get_shifted_value(i, j, k, grid, $newval - $add, ψ, args...)))
    eblk = Expr(:block, expr...)

    return eblk
end

=#