for side in [:left, :right], (dir, val) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [1, 2, 3])
    biased_interpolate = Symbol(:inner_, side, :_biased_interpolate_, dir)
    biased_β  = Symbol(side, :_biased_β)
    biased_p  = Symbol(side, :_biased_p)
    coeff     = Symbol(:coeff_, side) 
    stencil   = Symbol(side, :_stencil_, dir)
    stencil_u = Symbol(:tangential_, side, :_stencil_u)
    stencil_v = Symbol(:tangential_, side, :_stencil_v)

    @eval begin
        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT}, 
                                            ψ, idx, loc, args...) where {FT}
        
            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 1, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob = β
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 2, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob += add_global_smoothness(β, Val(4), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 3, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob += add_global_smoothness(β, Val(4), Val(2))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 4, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob += add_global_smoothness(β, Val(4), Val(3))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{4, FT}, 
                                            ψ, idx, loc, ::AbstractSmoothnessStencil, args...) where {FT}

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 1, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob = β
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 2, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob += add_global_smoothness(β, Val(4), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 3, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob += add_global_smoothness(β, Val(4), Val(2))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 4, i, j, k, grid, scheme, ψ, idx, loc, args...)
            glob += add_global_smoothness(β, Val(4), Val(3))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{4, FT}, 
                                             ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT}

            β, ψ̅, C, α = weno_substep($stencil, $stencil_u, $stencil_v, $biased_β, $biased_p, $coeff, $val, 1, i, j, k, grid, scheme, ψ, idx, loc, u, v, args...)
            glob = β
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            β, ψ̅, C, α = weno_substep($stencil, $stencil_u, $stencil_v, $biased_β, $biased_p, $coeff, $val, 2, i, j, k, grid, scheme, ψ, idx, loc, u, v, args...)
            glob += add_global_smoothness(β, Val(4), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $stencil_u, $stencil_v,  $biased_β, $biased_p, $coeff, $val, 3, i, j, k, grid, scheme, ψ, idx, loc, u, v, args...)
            glob += add_global_smoothness(β, Val(4), Val(2))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $stencil_u, $stencil_v, $biased_β, $biased_p, $coeff, $val, 4, i, j, k, grid, scheme, ψ, idx, loc, u, v, args...)
            glob += add_global_smoothness(β, Val(4), Val(3))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α
            
            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{4, FT},
                                             ψ, idx, loc, VI::FunctionStencil, args...) where {FT}

           
            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 1, i, j, k, grid, scheme, ψ, idx, loc, VI,args...)
            glob = β
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 2, i, j, k, grid, scheme, ψ, idx, loc, VI,args...)
            glob += add_global_smoothness(β, Val(4), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 3, i, j, k, grid, scheme, ψ, idx, loc, VI,args...)
            glob += add_global_smoothness(β, Val(4), Val(2))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            β, ψ̅, C, α = weno_substep($stencil, $biased_β, $biased_p, $coeff, $val, 4, i, j, k, grid, scheme, ψ, idx, loc, VI, args...)
            glob += add_global_smoothness(β, Val(4), Val(3))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end
    end
end