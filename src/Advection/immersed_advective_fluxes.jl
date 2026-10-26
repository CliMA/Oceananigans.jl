using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, inactive_node
using Oceananigans.Fields: ZeroField

const IBG = ImmersedBoundaryGrid

const c = Center()
const f = Face()

"""
$(TYPEDSIGNATURES)

Return either

    i) The boundary flux `qᴮ` if the node condition `nc` is true (default: `nc = immersed_peripheral_node`), or
    ii) The interior flux `qᴵ` otherwise.

This can be used either to condition intrinsic flux functions, or immersed boundary flux functions.
"""
@inline function conditional_flux(i, j, k, ibg, ℓx, ℓy, ℓz, q_boundary, q_interior)
    on_immersed_periphery = immersed_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz)
    return ifelse(on_immersed_periphery, q_boundary, q_interior)
end

# Conveniences
@inline conditional_flux_ccc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, c, qᴮ, qᴵ)
@inline conditional_flux_ffc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, f, c, qᴮ, qᴵ)
@inline conditional_flux_fcf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, f, qᴮ, qᴵ)
@inline conditional_flux_cff(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, f, qᴮ, qᴵ)

@inline conditional_flux_fcc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, c, qᴮ, qᴵ)
@inline conditional_flux_cfc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, c, qᴮ, qᴵ)
@inline conditional_flux_ccf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, f, qᴮ, qᴵ)

#####
##### Immersed Advective fluxes
#####

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)

# Disambiguation for `FluxForm` tracer fluxes....
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, c) =
        _advective_tracer_flux_x(i, j, k, ibg, scheme.x, U, c)

@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, c) =
        _advective_tracer_flux_y(i, j, k, ibg, scheme.y, V, c)

@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, c) =
        _advective_tracer_flux_z(i, j, k, ibg, scheme.z, W, c)

#####
##### "Boundary-aware" reconstruct
#####
##### Don't reconstruct with immersed cells!
#####

"""
    inside_immersed_boundary(rng, buffer, dir, side)

`inactive_node` expressions for the cells `rng` spanned by the reconstruction stencil along `dir`. A stencil of
centers (`side = :ᶠ`) spans its `2buffer` cells and a stencil of faces (`side = :ᶜ`) the `2buffer - 1` cells between
its faces. A symmetric reconstruction spans all of them; a left-biased one all but the last and a right-biased one all
but the first. Cells come back ordered outwards from the center of the stencil, which makes a narrower buffer's span a
prefix of a wider one's.

Example
=======

```
julia> inside_immersed_boundary(1:3, 2, :z, :ᶜ)
3-element Vector{Any}:
 :(inactive_node(i, j, k + 0,  ibg, c, c, c))
 :(inactive_node(i, j, k + 1,  ibg, c, c, c))
 :(inactive_node(i, j, k + -1, ibg, c, c, c))

julia> inside_immersed_boundary(1:5, 3, :x, :ᶠ)
5-element Vector{Any}:
 :(inactive_node(i + 0,  j, k, ibg, c, c, c))
 :(inactive_node(i + -1, j, k, ibg, c, c, c))
 :(inactive_node(i + 1,  j, k, ibg, c, c, c))
 :(inactive_node(i + -2, j, k, ibg, c, c, c))
 :(inactive_node(i + -3, j, k, ibg, c, c, c))
```
"""
@inline function inside_immersed_boundary(rng, buffer, dir, side)

    inactive_cells  = Vector(undef, length(rng))

    center_offset(n) = side == :ᶠ ? n - buffer - 1 : n - buffer
    distance_from_center(n) = abs(center_offset(n))
    deepest_side_first(n) = side == :ᶠ ? center_offset(n) : -center_offset(n)

    ordered_cells = sort(collect(rng), by = n -> (distance_from_center(n), deepest_side_first(n)))

    for (idx, n) in enumerate(ordered_cells)
        offset = center_offset(n)
        inactive_cells[idx] =  dir == :x ?
                               :(inactive_node(i + $offset, j, k, ibg, c, c, c)) :
                               dir == :y ?
                               :(inactive_node(i, j + $offset, k, ibg, c, c, c)) :
                               :(inactive_node(i, j, k + $offset, ibg, c, c, c))
    end

    return inactive_cells
end

for side in (:ᶜ, :ᶠ)
    near_x_boundary_symm = Symbol(:near_x_immersed_boundary_symmetric, side)
    near_y_boundary_symm = Symbol(:near_y_immersed_boundary_symmetric, side)
    near_z_boundary_symm = Symbol(:near_z_immersed_boundary_symmetric, side)

    near_x_boundary_bias = Symbol(:near_x_immersed_boundary_biased, side)
    near_y_boundary_bias = Symbol(:near_y_immersed_boundary_biased, side)
    near_z_boundary_bias = Symbol(:near_z_immersed_boundary_biased, side)

    @eval begin
        @inline $near_x_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_y_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_z_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false

        @inline $near_x_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}, bias) = false
        @inline $near_y_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}, bias) = false
        @inline $near_z_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}, bias) = false
    end

    # Of the N cells of a symmetric stencil the left-biased one spans 1:N-1 and the right-biased one 2:N, so both share
    # 2:N-1 and the bias picks only which end cell joins it.
    for buffer in advection_buffers
        N        = side == :ᶜ ? 2buffer - 1 : 2buffer
        full     = 1:N
        shared   = 2:N-1
        westmost = 1:1
        eastmost = N:N

        @eval begin
            @inline $near_x_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(full, buffer, :x, side)...))
            @inline $near_y_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(full, buffer, :y, side)...))
            @inline $near_z_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(full, buffer, :z, side)...))

            @inline $near_x_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, bias) =
                (|)($(inside_immersed_boundary(shared, buffer, :x, side)...),
                    ifelse(bias == LeftBias, $(only(inside_immersed_boundary(westmost, buffer, :x, side))),
                                             $(only(inside_immersed_boundary(eastmost, buffer, :x, side)))))
            @inline $near_y_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, bias) =
                (|)($(inside_immersed_boundary(shared, buffer, :y, side)...),
                    ifelse(bias == LeftBias, $(only(inside_immersed_boundary(westmost, buffer, :y, side))),
                                             $(only(inside_immersed_boundary(eastmost, buffer, :y, side)))))
            @inline $near_z_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, bias) =
                (|)($(inside_immersed_boundary(shared, buffer, :z, side)...),
                    ifelse(bias == LeftBias, $(only(inside_immersed_boundary(westmost, buffer, :z, side))),
                                             $(only(inside_immersed_boundary(eastmost, buffer, :z, side)))))
        end
    end
end

for bias in (:symmetric, :biased)
    for (d, ξ) in enumerate((:x, :y, :z))
        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ), (alt1, alt2) in zip((:_, :__, :___, :____, :_____), (:_____, :_, :__, :___, :____))
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt1_interp = Symbol(alt1, interp)
            alt2_interp = Symbol(alt2, interp)

            near_boundary = Symbol(:near_, ξ, :_immersed_boundary_, bias, loc)

            b = bias == :biased ? (:bias,) : ()

            @eval begin
                # Fallback for low order interpolation
                @inline $alt1_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, args...) = $interp(i, j, k, ibg, scheme, args...)

                # Conditional high-order interpolation in Bounded directions
                @inline $alt1_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, $(b...), args...) =
                    ifelse($near_boundary(i, j, k, ibg, scheme, $(b...)),
                           $alt2_interp(i, j, k, ibg, scheme.buffer_scheme, $(b...), args...),
                           $interp(i, j, k, ibg, scheme, $(b...), args...))
            end
        end
    end
end
