using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ 
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractTimeDiscretization

const ATC = AbstractTurbulenceClosure
const ATD = AbstractTimeDiscretization

"""
    conditional_flux(i, j, k, ibg::IBG, ℓx, ℓy, ℓz, qᴮ, qᴵ, nc)

Return either

    i) The boundary flux `qᴮ` if `immersed_peripheral_node`
    ii) The interior flux `qᴵ` otherwise.

This can be used either to condition intrinsic flux functions, or immersed boundary flux functions.
"""
@inline conditional_flux(i, j, k, ibg, ℓx, ℓy, ℓz, qᴮ, qᴵ) = ifelse(immersed_peripheral_node(ℓx, ℓy, ℓz, i, j, k, ibg), qᴮ, qᴵ)

#####
##### "Boundary-aware" reconstruction
#####
##### Don't reconstruct with immersed cells!
#####

const AAS{N} = AbstractAdvectionScheme{N} where N

@inline near_x_boundary(i, j, k, ibg, ::AAS{0}) = false
@inline near_y_boundary(i, j, k, ibg, ::AAS{0}) = false
@inline near_z_boundary(i, j, k, ibg, ::AAS{0}) = false

@inline near_x_boundary(i, j, k, ibg, ::AAS{1}) = inactive_cell(i-1, j, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i+1, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AAS{1}) = inactive_cell(i, j-1, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j+1, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AAS{1}) = inactive_cell(i, j, k-1, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j, k+1, ibg)

@inline near_x_boundary(i, j, k, ibg, ::AAS{2}) = inactive_cell(i-2, j, k, ibg) | inactive_cell(i-1, j, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i+1, j, k, ibg) | inactive_cell(i+2, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AAS{2}) = inactive_cell(i, j-2, k, ibg) | inactive_cell(i, j-1, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j+1, k, ibg) | inactive_cell(i, j+2, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AAS{2}) = inactive_cell(i, j, k-2, ibg) | inactive_cell(i, j, k-1, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j, k+1, ibg) | inactive_cell(i, j, k+2, ibg)

using Oceananigans.Advection: WENOVectorInvariantVel, VorticityStencil, VelocityStencil

@inline function near_horizontal_boundary_x(i, j, k, ibg, scheme::WENOVectorInvariantVel) 
    return inactive_node(i,   j,   k, ibg, c, f, c) |       
           inactive_node(i-3, j,   k, ibg, c, f, c) | inactive_node(i+3,   j, k, ibg, c, f, c) |
           inactive_node(i-2, j,   k, ibg, c, f, c) | inactive_node(i+2,   j, k, ibg, c, f, c) |
           inactive_node(i-1, j,   k, ibg, c, f, c) | inactive_node(i+1,   j, k, ibg, c, f, c) | 
           inactive_node(i,   j,   k, ibg, f, c, c) |                                        
           inactive_node(i-2, j,   k, ibg, f, c, c) | inactive_node(i+2,   j, k, ibg, f, c, c) |
           inactive_node(i-1, j,   k, ibg, f, c, c) | inactive_node(i+1,   j, k, ibg, f, c, c) |
           inactive_node(i-2, j+1, k, ibg, f, c, c) | inactive_node(i+2, j+1, k, ibg, f, c, c) |
           inactive_node(i-1, j+1, k, ibg, f, c, c) | inactive_node(i+1, j+1, k, ibg, f, c, c) |
           inactive_node(i,   j+1, k, ibg, f, c, c) 
end

@inline function near_horizontal_boundary_y(i, j, k, ibg, scheme::WENOVectorInvariantVel) 
    return inactive_node(i,     j, k, ibg, f, c, c) | 
           inactive_node(i,   j+3, k, ibg, f, c, c) | inactive_node(i,   j+3, k, ibg, f, c, c) |
           inactive_node(i,   j+2, k, ibg, f, c, c) | inactive_node(i,   j+2, k, ibg, f, c, c) |
           inactive_node(i,   j+1, k, ibg, f, c, c) | inactive_node(i,   j+1, k, ibg, f, c, c) |     
           inactive_node(i,     j, k, ibg, c, f, c) |                                        
           inactive_node(i,   j-2, k, ibg, c, f, c) | inactive_node(i,   j+2, k, ibg, c, f, c) |
           inactive_node(i,   j-1, k, ibg, c, f, c) | inactive_node(i,   j+1, k, ibg, c, f, c) | 
           inactive_node(i+1, j-2, k, ibg, c, f, c) | inactive_node(i+1, j+2, k, ibg, c, f, c) |
           inactive_node(i+1, j-1, k, ibg, c, f, c) | inactive_node(i+1, j+1, k, ibg, c, f, c) |
           inactive_node(i+1, j,   k, ibg, c, f, c) 
end

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            near_boundary = Symbol(:near_, ξ, :_boundary)

            # Conditional high-order interpolation in Bounded directions
            @eval begin
                import Oceananigans.Advection: $alt_interp
                using Oceananigans.Advection: $interp

                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme, ψ) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $second_order_interp(i, j, k, ibg.underlying_grid, ψ),
                           $interp(i, j, k, ibg.underlying_grid, scheme, ψ))
            end
            if ξ == :z
                @eval begin
                    import Oceananigans.Advection: $alt_interp
                    using Oceananigans.Advection: $interp
    
                    @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariant, ∂z, VI, u) =
                        ifelse($near_boundary(i, j, k, ibg, scheme),
                            $second_order_interp(i, j, k, ibg.underlying_grid, ∂z, u),
                            $interp(i, j, k, ibg.underlying_grid, scheme, ∂z, VI, u))
                end
            else    
                @eval begin
                    import Oceananigans.Advection: $alt_interp
                    using Oceananigans.Advection: $interp
    
                    @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariant, ζ, VI, u, v) =
                        ifelse($near_boundary(i, j, k, ibg, scheme),
                                $second_order_interp(i, j, k, ibg.underlying_grid, ζ, u, v),
                                $interp(i, j, k, ibg.underlying_grid, scheme, ζ, VI, u, v))
                end    
            end
        end
    end
end

for bias in (:left_biased, :right_biased)
    for (d, dir) in zip((:x, :y), (:xᶜᵃᵃ, :yᵃᶜᵃ))
        interp     = Symbol(bias, :_interpolate_, dir)
        alt_interp = Symbol(:_, interp)

        near_horizontal_boundary = Symbol(:near_horizontal_boundary_, d)
        @eval begin
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariantVel, ζ, ::Type{VelocityStencil}, u, v) =
            ifelse($near_horizontal_boundary(i, j, k, ibg, scheme),
               $alt_interp(i, j, k, ibg, scheme, ζ, VorticityStencil, u, v),
               $interp(i, j, k, ibg.underlying_grid, scheme, ζ, VelocityStencil, u, v))
        end
    end
end
