using Oceananigans.ImmersedBoundaries: IBG, immersed_peripheral_node, inactive_node
using Oceananigans.Fields: ZeroField

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

# On immersed grids the stencil must also avoid inactive cells. Of the N cells of a symmetric stencil the left-biased one
# spans 1:N-1 and the right-biased one 2:N, so both share 2:N-1 and the bias picks only which end cell joins it. The halos beyond
# a Closed side are inactive and the halos beyond an Open side are active, so only directions with an Open side also test the
# bounds of the sides.
for side in (:ᶜ, :ᶠ), (d, ξ) in enumerate((:x, :y, :z))
    index = (:i, :j, :k)[d]
    open_topology = :(Union{Sides{Open}, Sides{<:Any, Open}})
    open_grid = (:(IBG{<:Any, <:$open_topology}), :(IBG{<:Any, <:Any, <:$open_topology}), :(IBG{<:Any, <:Any, <:Any, <:$open_topology}))[d]

    near_boundary_symmetric = Symbol(:near_, ξ, :_boundary_symmetric, side)
    near_boundary_biased    = Symbol(:near_, ξ, :_boundary_biased, side)
    within_bounds_symmetric = Symbol(:within_bounds_symmetric_, ξ, side)
    within_bounds_biased    = Symbol(:within_bounds_biased_, ξ, side)

    for buffer in advection_buffers
        N = side == :ᶜ ? 2buffer - 1 : 2buffer
        cells(range) = inside_immersed_boundary(range, buffer, ξ, side)
        westmost, eastmost = only(cells(1:1)), only(cells(N:N))
        symmetric_cells = :((|)($(cells(1:N)...)))
        biased_cells    = :((|)($(cells(2:N-1)...), ifelse(bias == LeftBias, $westmost, $eastmost)))

        @eval begin
            @inline $near_boundary_symmetric(i, j, k, ibg::IBG, scheme::AbstractAdvectionScheme{$buffer}) = $symmetric_cells
            @inline $near_boundary_biased(i, j, k, ibg::IBG, scheme::AbstractAdvectionScheme{$buffer}, bias) = $biased_cells

            @inline $near_boundary_symmetric(i, j, k, ibg::$open_grid, scheme::AbstractAdvectionScheme{$buffer}) = !$within_bounds_symmetric($index, ibg, scheme) | $symmetric_cells
            @inline $near_boundary_biased(i, j, k, ibg::$open_grid, scheme::AbstractAdvectionScheme{$buffer}, bias) = !$within_bounds_biased($index, ibg, scheme, bias) | $biased_cells
        end
    end
end
