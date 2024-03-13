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
                                            scheme::WENO{2, FT}, tid, wrk,
                                            ψ, idx, loc, args...) where {FT}
        
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, args...)
            β  = $biased_β(ψs, scheme, Val(0))
            C  = FT($coeff(scheme, Val(0)))
            α  = @fastmath C / (β + FT(ε))^2
            ψ̅  = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            glob = β
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            ψs = $stencil(i, j, k, scheme, Val(2), ψ, grid, args...)
            β  = $biased_β(ψs, scheme, Val(1))
            C  = FT($coeff(scheme, Val(1)))
            α  = @fastmath C / (β + FT(ε))^2
            ψ̅  = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 
            glob += add_global_smoothness(β, Val(2), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                            scheme::WENO{2, FT}, tid, wrk,
                                            ψ, idx, loc, ::AbstractSmoothnessStencil, args...) where {FT}
        
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, args...)
            β  = $biased_β(ψs, scheme, Val(0))
            C  = FT($coeff(scheme, Val(0)))
            α  = @fastmath C / (β + FT(ε))^2
            ψ̅  = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            glob = β
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            ψs = $stencil(i, j, k, scheme, Val(2), ψ, grid, args...)
            β  = $biased_β(ψs, scheme, Val(1))
            C  = FT($coeff(scheme, Val(1)))
            α  = @fastmath C / (β + FT(ε))^2
            ψ̅  = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 
            glob += add_global_smoothness(β, Val(2), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{2, FT}, tid, wrk,
                                             ψ, idx, loc, ::VelocityStencil, u, v, args...) where {FT}


            ψs = $stencil_u(i, j, k, scheme, Val(1), Val($val), grid, u)
            βu = $biased_β(ψs, scheme, Val(0))
            ψs = $stencil_v(i, j, k, scheme, Val(1), Val($val), grid, v)
            βv = $biased_β(ψs, scheme, Val(0))
            βU = 0.5 * (βu + βv)
            C  = FT($coeff(scheme, Val(0)))
            α  = @fastmath C / (βU + FT(ε))^2
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, u, v, args...)
            ψ̅  = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            glob = βU
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            ψs = $stencil_u(i, j, k, scheme, Val(2), Val($val), grid, u)
            βu = $biased_β(ψs, scheme, Val(1))
            ψs = $stencil_v(i, j, k, scheme, Val(2), Val($val), grid, v)
            βv = $biased_β(ψs, scheme, Val(1))
            βU = 0.5 * (βu + βv)
            C  = FT($coeff(scheme, Val(1)))
            α  = @fastmath C / (βU + FT(ε))^2
            ψs = $stencil(i, j, k, scheme, Val(2), ψ, grid, u, v, args...)
            ψ̅  = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 
            glob += add_global_smoothness(βU, Val(2), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end

        @inline function $biased_interpolate(i, j, k, grid, 
                                             scheme::WENO{2, FT}, tid, wrk,
                                             ψ, idx, loc, VI::FunctionStencil, args...) where {FT}

            ψs = $stencil(i, j, k, scheme, Val(1), VI.func, grid, args...)
            βϕ = $biased_β(ψs, scheme, Val(0))
            C  = FT($coeff(scheme, Val(0)))
            α  = @fastmath C / (βϕ + FT(ε))^2
            ψs = $stencil(i, j, k, scheme, Val(1), ψ, grid, args...)
            ψ̅  = $biased_p(scheme, Val(0), ψs, Nothing, Val($val), idx, loc) 
            glob = βϕ
            sol1 = ψ̅ * C
            wei1 = C
            sol2 = ψ̅ * α  
            wei2 = α

            ψs = $stencil(i, j, k, scheme, Val(2), VI.func, grid, args...)
            βϕ = $biased_β(ψs, scheme, Val(1))
            C  = FT($coeff(scheme, Val(1)))
            α  = @fastmath C / (βϕ + FT(ε))^2
            ψs = $stencil(i, j, k, scheme, Val(2), ψ, grid, args...)
            ψ̅  = $biased_p(scheme, Val(1), ψs, Nothing, Val($val), idx, loc) 
            glob += add_global_smoothness(βϕ, Val(2), Val(1))
            sol1 += ψ̅ * C
            wei1 += C
            sol2 += ψ̅ * α  
            wei2 += α

            # Is glob squared here?
            return (sol1 + sol2 * glob) / (wei1 + wei2 * glob)
        end
    end
end