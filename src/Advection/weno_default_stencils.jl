@inline get_shifted_value_x(i, j, k, grid, shift, ψ, args...) = @inbounds ψ[i + shift, j, k]
@inline get_shifted_value_x(i, j, k, grid, shift, ψ::Function, args...) = ψ(i + shift, j, k, grid, args...)

@inline get_shifted_value_y(i, j, k, grid, shift, ψ, args...) = @inbounds ψ[i, j + shift, k]
@inline get_shifted_value_y(i, j, k, grid, shift, ψ::Function, args...) = ψ(i, j + shift, k, grid, args...)

@inline get_shifted_value_z(i, j, k, grid, shift, ψ, args...) = @inbounds ψ[i, j, k + shift]
@inline get_shifted_value_z(i, j, k, grid, shift, ψ::Function, args...) = ψ(i, j, k + shift, grid, args...)

#####
##### STENCILS IN X
#####

# for (side, add) in zip([:left, :right], (1, 0)), (dir, loc, val) in zip((:x, :y, :z), (:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (1, 2, 3))
#     biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir, loc)
#     coeff              = Symbol(:coeff_, side) 
#     weno_interpolant   = Symbol(side, :_weno_interpolant_, dir, loc)
#     get_shifted_value  = Symbol(:get_shifted_value_, dir)

#     @eval begin
#         # Fallback for DefaultStencil formulations and disambiguation
#         @inline $biased_interpolate(i, j, k, grid, scheme::WENO{2}, ψ, idx, loc, ::DefaultStencil, args...) = 
#                                     $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
#         @inline $biased_interpolate(i, j, k, grid, scheme::WENO{3}, ψ, idx, loc, ::DefaultStencil, args...) = 
#                                     $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
#         @inline $biased_interpolate(i, j, k, grid, scheme::WENO{4}, ψ, idx, loc, ::DefaultStencil, args...) = 
#                                     $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
#         @inline $biased_interpolate(i, j, k, grid, scheme::WENO{5}, ψ, idx, loc, ::DefaultStencil, args...) = 
#                                     $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)
#         @inline $biased_interpolate(i, j, k, grid, scheme::WENO{6}, ψ, idx, loc, ::DefaultStencil, args...) = 
#                                     $biased_interpolate(i, j, k, grid, scheme, ψ, idx, loc, args...)

#         @inline function $biased_interpolate(i, j, k, grid, 
#                                              scheme::WENO{2},
#                                              ψ, idx, loc, args...) 
        
#             # All stencils
#             ψ₀ = $get_shifted_value(i, j, k, grid,   - $add, ψ, args...)
#             ψ₁ = $get_shifted_value(i, j, k, grid, 1 - $add, ψ, args...)

#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 1, scheme, $val, idx, loc)
#             τ  = add_to_global_smoothness(β, Val(2), Val(1))
#             ψ̂₁ = ψ̅ * α  
#             ψ̂₂ = ψ̅ * C
#             w₁ = α

#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁), 2, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(2), Val(2))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             τ = τ * τ

#             return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
#         end

#         @inline function $biased_interpolate(i, j, k, grid, 
#                                              scheme::WENO{3, FT, XT, YT, ZT},
#                                              ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
#             # All stencils
#             ψ₀ = $get_shifted_value(i, j, k, grid,   - $add, ψ, args...)
#             ψ₁ = $get_shifted_value(i, j, k, grid, 1 - $add, ψ, args...)
#             ψ₂ = $get_shifted_value(i, j, k, grid, 2 - $add, ψ, args...)

#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 1, scheme, $val, idx, loc)
#             τ  = add_to_global_smoothness(β, Val(3), Val(1))
#             ψ̂₁ = ψ̅ * α  
#             ψ̂₂ = ψ̅ * C
#             w₁ = α

#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 2, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(3), Val(2))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂), 3, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(3), Val(3))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             τ = τ * τ

#             return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1) 
#         end

#         @inline function $biased_interpolate(i, j, k, grid, 
#                                              scheme::WENO{4, FT, XT, YT, ZT},
#                                              ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
#             # All stencils
#             ψ₀ = $get_shifted_value(i, j, k, grid,   - $add, ψ, args...)
#             ψ₁ = $get_shifted_value(i, j, k, grid, 1 - $add, ψ, args...)
#             ψ₂ = $get_shifted_value(i, j, k, grid, 2 - $add, ψ, args...)
#             ψ₃ = $get_shifted_value(i, j, k, grid, 3 - $add, ψ, args...)

#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 1, scheme, $val, idx, loc)
#             τ  = add_to_global_smoothness(β, Val(4), Val(1))
#             ψ̂₁ = ψ̅ * α  
#             ψ̂₂ = ψ̅ * C
#             w₁ = α

#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 2, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(4), Val(2))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 3, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(4), Val(3))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k,  grid, - 3 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃), 4, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(4), Val(4))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             τ = τ * τ

#             return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
#         end

#         @inline function $biased_interpolate(i, j, k, grid, 
#                                              scheme::WENO{5, FT, XT, YT, ZT},
#                                              ψ, idx, loc, args...) where {FT, XT, YT, ZT}
        
#             # All stencils
#             ψ₀ = $get_shifted_value(i, j, k, grid,     - $add, ψ, args...)
#             ψ₁ = $get_shifted_value(i, j, k, grid, + 1 - $add, ψ, args...)
#             ψ₂ = $get_shifted_value(i, j, k, grid, + 2 - $add, ψ, args...)
#             ψ₃ = $get_shifted_value(i, j, k, grid, + 3 - $add, ψ, args...)
#             ψ₄ = $get_shifted_value(i, j, k, grid, + 4 - $add, ψ, args...)

#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 1, scheme, $val, idx, loc)
#             τ  = β
#             ψ̂₁ = ψ̅ * α  
#             ψ̂₂ = ψ̅ * C
#             w₁ = α

#             ψ₄ = ψ₃
#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 1 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 2, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(5), Val(2))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             ψ₄ = ψ₃
#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 2 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 3, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(5), Val(3))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             ψ₄ = ψ₃
#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 3 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 4, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(5), Val(4))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             ψ₄ = ψ₃
#             ψ₃ = ψ₂
#             ψ₂ = ψ₁
#             ψ₁ = ψ₀
#             ψ₀ = $get_shifted_value(i, j, k, grid, - 4 - $add, ψ, args...)

#             # Stencil S₁
#             β, ψ̅, C, α = $weno_interpolant((ψ₀, ψ₁, ψ₂, ψ₃, ψ₄), 5, scheme, $val, idx, loc)
#             τ  += add_to_global_smoothness(β, Val(5), Val(5))
#             ψ̂₁ += ψ̅ * α  
#             ψ̂₂ += ψ̅ * C
#             w₁ += α

#             τ = τ * τ

#             return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
#         end
#     end
# end

# Metaprogrammed implementation of WENO reconstruction of order N
function stencil(::Val{2}, v)
    v0 = Symbol(v, 0)
    v1 = Symbol(v, 1)
    return :(($v0, $v1))
end

function stencil(::Val{3}, v)
    v0 = Symbol(v, 0)
    v1 = Symbol(v, 1)
    v2 = Symbol(v, 2)
    return :(($v0, $v1, $v2))
end

function stencil(::Val{4}, v)
    v0 = Symbol(v, 0)
    v1 = Symbol(v, 1)
    v2 = Symbol(v, 2)
    v3 = Symbol(v, 3)
    return :(($v0, $v1, $v2, $v3))
end

function stencil(::Val{5}, v)
    v0 = Symbol(v, 0)
    v1 = Symbol(v, 1)
    v2 = Symbol(v, 2)
    v3 = Symbol(v, 3)
    v4 = Symbol(v, 4)
    return :(($v0, $v1, $v2, $v3, $v4))
end

function stencil(::Val{6}, v)
    v0 = Symbol(v, 0)
    v1 = Symbol(v, 1)
    v2 = Symbol(v, 2)
    v3 = Symbol(v, 3)
    v4 = Symbol(v, 4)
    v5 = Symbol(v, 5)
    return :(($v0, $v1, $v2, $v3, $v4, $v5))
end

function retrieve_initial_stencil(N, dir, add, variable)
    expr = []
    get_shifted_value = Symbol(:get_shifted_value_, dir)    
    for i in 0:N-1
        ψ = Symbol(variable, i)
        push!(expr, :($ψ = $get_shifted_value(i, j, k, grid,  $i - $add, $variable, args...)))
    end
    eblk = Expr(:block, expr...)

    return eblk
end

function update_stencil(N, dir, add, variable)
    expr = []
    get_shifted_value = Symbol(:get_shifted_value_, dir)    
    for i in N-1:-1:1
        ψ  = Symbol(variable, i)
        ψp = Symbol(variable, i-1)
        push!(expr, :($ψ = $ψp))
    end
    ψ  = Symbol(variable, 0)
    push!(expr, :($ψ = $get_shifted_value(i, j, k, grid,  - s + 1 - $add, $variable, args...)))
    eblk = Expr(:block, expr...)

    return eblk
end

function retrieve_velocity_stencil(N, dir, add, variable)
    expr = []
    get_shifted_value = Symbol(:get_shifted_value_, dir)    
    for i in 0:N-1
        ψ = Symbol(variable, i)
        push!(expr, :($ψ = $get_shifted_value(i, j, k, grid, $i - $add, $variable, u, v, args...)))
    end
    eblk = Expr(:block, expr...)

    return eblk
end

function update_velocity_stencil(N, dir, add, variable)
    expr = []
    get_shifted_value = Symbol(:get_shifted_value_, dir)    
    for i in N-1:-1:1
        ψ  = Symbol(variable, i)
        ψp = Symbol(variable, i-1)
        push!(expr, :($ψ = $ψp))
    end
    ψ  = Symbol(variable, 0)
    push!(expr, :($ψ = $get_shifted_value(i, j, k, grid, - s + 1 - $add, $variable, u, v, args...)))
    eblk = Expr(:block, expr...)

    return eblk
end

@inline ℑyᵃᶠᵃ_u(i, j, k, grid, u, v, args...) = ℑyᵃᶠᵃ(i, j, k, grid, u)
@inline ℑxᶠᵃᵃ_v(i, j, k, grid, u, v, args...) = ℑxᶠᵃᵃ(i, j, k, grid, v)

for (side, add) in zip([:left, :right], (1, 0)), (dir, loc, val) in zip((:x, :y, :z), (:ᶠᵃᵃ, :ᵃᶠᵃ, :ᵃᵃᶠ), (1, 2, 3))
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir, loc)
    weno_interpolant   = Symbol(side, :_weno_interpolant_, dir, loc)

    for N in [2, 3, 4, 5, 6]
        @eval begin
            @inline function $biased_interpolate(i, j, k, grid, 
                                                    scheme::WENO{$N},
                                                    ψ, idx, loc, args...) 

                # All stencils
                $(retrieve_initial_stencil(N, dir, add, :ψ))

                β, ψ̅, C, α = $weno_interpolant($(stencil(Val(N), :ψ)), 1, scheme, $val, idx, loc)
                τ  = β
                ψ̂₁ = ψ̅ * α  
                ψ̂₂ = ψ̅ * C
                w₁ = α

                @unroll for s in 2:$N
                    $(update_stencil(N, dir, add, :ψ))  
                    β, ψ̅, C, α = $weno_interpolant($(stencil(Val(N), :ψ)), s, scheme, $val, idx, loc)
                    τ  += add_to_global_smoothness(β, Val($N), Val(s))
                    ψ̂₁ += ψ̅ * α  
                    ψ̂₂ += ψ̅ * C
                    w₁ += α
                end

                τ = τ * τ

                return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
            end

            @inline function $biased_interpolate(i, j, k, grid, 
                                                    scheme::WENO{$N},
                                                    ψ, idx, loc, VI::FunctionStencil, args...) 

                ϕ = VI.func

                # All stencils
                $(retrieve_initial_stencil(N, dir, add, :ψ))
                $(retrieve_initial_stencil(N, dir, add, :ϕ))

                β, ψ̅, C, α = $weno_interpolant($(stencil(Val(N), :ψ)), 
                                               $(stencil(Val(N), :ϕ)), 
                                               1, scheme, $val, idx, loc)
                τ  = β
                ψ̂₁ = ψ̅ * α  
                ψ̂₂ = ψ̅ * C
                w₁ = α

                @unroll for s in 2:$N
                    $(update_stencil(N, dir, add, :ψ))  
                    $(update_stencil(N, dir, add, :ϕ))  

                    β, ψ̅, C, α = $weno_interpolant($(stencil(Val(N), :ψ)), 
                                                   $(stencil(Val(N), :ϕ)), 
                                                   s, scheme, $val, idx, loc)

                    τ  += add_to_global_smoothness(β, Val($N), Val(s))
                    ψ̂₁ += ψ̅ * α  
                    ψ̂₂ += ψ̅ * C
                    w₁ += α
                end

                τ = τ * τ

                return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
            end

            @inline function $biased_interpolate(i, j, k, grid, 
                                                scheme::WENO{$N},
                                                ψ, idx, loc, ::VelocityStencil, u, v, args...) 
            
                𝓊 = ℑyᵃᶠᵃ_u
                𝓋 = ℑxᶠᵃᵃ_v

                # All stencils
                $(retrieve_velocity_stencil(N, dir, add, :ψ))
                $(retrieve_velocity_stencil(N, dir, add, :𝓊))
                $(retrieve_velocity_stencil(N, dir, add, :𝓋))
            
                β, ψ̅, C, α = $weno_interpolant($(stencil(Val(N), :ψ)), 
                                               $(stencil(Val(N), :𝓊)), 
                                               $(stencil(Val(N), :𝓋)), 1, scheme, $val, idx, loc)
                τ  = β
                ψ̂₁ = ψ̅ * α  
                ψ̂₂ = ψ̅ * C
                w₁ = α

                @unroll for s in 2:$N
                    $(update_velocity_stencil(N, dir, add, :ψ))
                    $(update_velocity_stencil(N, dir, add, :𝓊))
                    $(update_velocity_stencil(N, dir, add, :𝓋))

                    β, ψ̅, C, α = $weno_interpolant($(stencil(Val(N), :ψ)), 
                                                   $(stencil(Val(N), :𝓊)), 
                                                   $(stencil(Val(N), :𝓋)), s, scheme, $val, idx, loc)

                    τ  += add_to_global_smoothness(β, Val($N), Val(s))
                    ψ̂₁ += ψ̅ * α  
                    ψ̂₂ += ψ̅ * C
                    w₁ += α
                end

                τ = τ^2

                return (ψ̂₁ * τ + ψ̂₂) / (w₁ * τ + 1)
            end
        end
    end
end