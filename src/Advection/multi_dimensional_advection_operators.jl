#####
##### Multi Dimensional advection operators
#####

for buffer in (2, 3)
    coeff = Symbol(:coeff, buffer*2, :_multi_F)
    
    @eval begin
        @inline function div_𝐯u(i, j, k, grid, scheme::MDS{$buffer}, U, u)
            return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_momentum_flux_Uu, U[1], u) +
                                            δyᵃᶜᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_momentum_flux_Vu, U[2], u) +
                                            δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme.one_dimensional_scheme, U[3], u))
        end

        @inline function div_𝐯v(i, j, k, grid, scheme::MDS{$buffer}, U, v)
            return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_momentum_flux_Uv, U[1], v) +
                                            δyᵃᶠᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_momentum_flux_Vv, U[2], v) +
                                            δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme.one_dimensional_scheme, U[3], v))
        end

        @inline function div_𝐯w(i, j, k, grid, scheme::MDS{$buffer}, U, w)
            return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_momentum_flux_Uw, U[1], w) +
                                            δyᵃᶜᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_momentum_flux_Vw, U[2], w) +
                                            δzᵃᵃᶠ(i, j, k, grid, _advective_momentum_flux_Ww, scheme.one_dimensional_scheme, U[3], w))
        end        

        @inline function div_Uc(i, j, k, grid, scheme::MDS{$buffer}, U, c)
            1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _multi_dimensional_interpolate_y, scheme, $coeff, _advective_tracer_flux_x, U.u, c) +
                                     δyᵃᶜᵃ(i, j, k, grid, _multi_dimensional_interpolate_x, scheme, $coeff, _advective_tracer_flux_y, U.v, c) +
                                     δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, scheme.one_dimensional_scheme, U.w, c))
        end
    end
end


