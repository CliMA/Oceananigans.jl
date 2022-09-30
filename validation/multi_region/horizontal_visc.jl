using Oceananigans.TurbulenceClosures
using Oceananigans.Grids: min_Δx, min_Δy
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, ℑxyᶜᶜᵃ, ζ₃ᶠᶠᶜ, div_xyᶜᶜᶜ
using Oceananigans.Operators: Δx, Δy
using CUDA: @allowscalar

@inline Dₜ(i, j, k, grid, u, v) = ∂xᶠᶠᶜ(i, j, k, grid, v) + ∂yᶠᶠᶜ(i, j, k, grid, u)
@inline Dₛ(i, j, k, grid, u, v) = ∂xᶜᶜᶜ(i, j, k, grid, u) - ∂yᶜᶜᶜ(i, j, k, grid, v)
@inline Δ²ᶜᶜᶜ(i, j, k, grid)    = (1 / (1 / Δxᶜᶜᶜ(i, j, k, grid)^2 + 1 / Δyᶜᶜᶜ(i, j, k, grid)^2))
@inline Δ⁶ᶜᶜᶜ(i, j, k, grid)    = (1 / (1 / Δxᶜᶜᶜ(i, j, k, grid)^6 + 1 / Δyᶜᶜᶜ(i, j, k, grid)^6))

@inline Δ⁵ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) = 
				(1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^5 + 1 / Δy(i, j, k, grid, lx, ly, lz)^5))^5

@inline function νhb_smagorinski_final(i, j, k, grid, clock, fields, C₄) 
   δ₁ = Dₛ(i, j, k, grid, fields.u, fields.v)    
   δ₂ = ℑxyᶜᶜᵃ(i, j, k, grid, Dₜ, fields.u, fields.v)    
   return Δ²ᶜᶜᶜ(i, j, k, grid)^2 * C₄ * sqrt(δ₁^2 + δ₂^2)
end

function smagorinsky_viscosity(formulation, grid; Cₛₘ = 4.0)

    dx_min = min_Δx(grid.underlying_grid)
    dy_min = min_Δy(grid.underlying_grid)
    dx_max = @allowscalar grid.Δxᶠᶜᵃ[Int(grid.Ny / 2)]
    dy_max = @allowscalar grid.Δxᶠᶜᵃ[Int(grid.Ny / 2)]
    timescale_max = 100days
    timescale_min = 0.2days

    @show C₄    = (Cₛₘ / π)^2 / 8
    @show min_ν = (1 / (1 / dx_min^2 + 1 / dy_min^2))^2 / timescale_max
    @show max_ν = (1 / (1 / dx_max^2 + 1 / dy_max^2))^2 / timescale_min

    loc = (Center, Center, Center)

    return ScalarBiharmonicDiffusivity(formulation; 
                                       ν=νhb_smagorinski_final, discrete_form=true, loc, 
				       parameters = C₄)
end

using Oceananigans.Operators: ℑxyz

@inline function νhb_leith_final(i, j, k, grid, lx, ly, lz, clock, fields, p)
    
    location = (lx, ly, lz)
    from_∂xζ = (Center(), Face(), Center()) 
    from_∂yζ = (Face(), Center(), Center()) 
    from_∂xδ = (Face(), Center(), Center()) 
    from_∂yδ = (Center(), Face(), Center()) 
	
    ∂xζ = ℑxyz(i, j, k, grid, from_∂xζ, location, ∂xᶜᶠᶜ, ζ₃ᶠᶠᶜ, fields.u, fields.v)
    ∂yζ = ℑxyz(i, j, k, grid, from_∂yζ, location, ∂yᶠᶜᶜ, ζ₃ᶠᶠᶜ, fields.u, fields.v)
    ∂xδ = ℑxyz(i, j, k, grid, from_∂xδ, location, ∂xᶠᶜᶜ, div_xyᶜᶜᶜ, fields.u, fields.v)
    ∂yδ = ℑxyz(i, j, k, grid, from_∂yδ, location, ∂yᶜᶠᶜ, div_xyᶜᶜᶜ, fields.u, fields.v)
   
    dynamic_visc = sqrt( p.C₄ * (∂xζ^2 + ∂yζ^2) + p.C₄ₙ * (∂xδ^2 + ∂yδ^2) )

    return Δ⁵ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) * dynamic_visc
end

function leith_viscosity(formulation, grid; C_vort = 3.0, C_div = 3.0)

    @show C₄  = (C_vort / π)^6 / 8
    @show C₄ₙ = (C_div  / π)^6 / 8

    visc = ScalarBiharmonicDiffusivity(formulation; 
                                       ν=νhb_leith_final, discrete_form=true,  
                                       parameters = (; C₄, C₄ₙ))

    @show typeof(visc.ν)

    return visc
end


@inline νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) =
		(1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))^2 / λ

geometric_viscosity(formulation, timescale) = ScalarBiharmonicDiffusivity(formulation, ν=νhb, discrete_form=true, parameters = timescale) 
