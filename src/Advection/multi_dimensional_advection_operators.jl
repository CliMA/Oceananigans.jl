#####
##### Multi Dimensional advection operators
#####

for buffer in (2, 3)
    coeff = Symbol(:coeff, buffer*2, :_multi_F)
    
    @eval begin
        @inline function div_𝐯u(i, j, k, grid, scheme::MDS{$buffer}, U, u)
            return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_momentum_flux_Uu, scheme, U[1], u) +
                                            δyᵃᶜᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_momentum_flux_Vu, scheme, U[2], u) +
                                            δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme.scheme_1d, U[3], u))
        end

        @inline function div_𝐯v(i, j, k, grid, scheme::MDS{$buffer}, U, v)
            return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_momentum_flux_Uv, scheme, U[1], v) +
                                            δyᵃᶠᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_momentum_flux_Vv, scheme, U[2], v) +
                                            δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme.scheme_1d, U[3], v))
        end

        @inline function div_𝐯w(i, j, k, grid, scheme::MDS{$buffer}, U, w)
            return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_momentum_flux_Uw, scheme, U[1], w) +
                                            δyᵃᶜᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_momentum_flux_Vw, scheme, U[2], w) +
                                            δzᵃᵃᶠ(i, j, k, grid, _advective_momentum_flux_Ww, scheme.scheme_1d, U[3], w))
        end        

        @inline function div_Uc(i, j, k, grid, scheme::MDS{$buffer}, U, c)
            1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_tracer_flux_x, scheme, U.u, c) +
                                     δyᵃᶜᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_tracer_flux_y, scheme, U.v, c) +
                                     δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, scheme.scheme_1d, U.w, c))
        end

        # Higher order Multi dimensions have to interpolate the horizontal High Order fluxes!
        @inline U_dot_∇u(i, j, k, grid, scheme::MDSWENOVectorInvariant{$buffer}, U) = (
            + _multi_dimensional_interpolate_x(i, j, k, grid, scheme, $coeff, vertical_vorticity_U, scheme, U.u, U.v)  
            + vertical_advection_U(i, j, k, grid, scheme.scheme_1d, U.u, U.w)  
            + bernoulli_head_U(i, j, k, grid, scheme.scheme_1d, U.u, U.v))     
    
        @inline U_dot_∇v(i, j, k, grid, scheme::MDSWENOVectorInvariant{$buffer}, U) = (
            + _multi_dimensional_interpolate_y(i, j, k, grid, scheme, $coeff, vertical_vorticity_V, scheme, U.u, U.v)    
            + vertical_advection_V(i, j, k, grid, scheme.scheme_1d, U.v, U.w)                                                 
            + bernoulli_head_V(i, j, k, grid, scheme.scheme_1d, U.u, U.v))                                             
    end
end


