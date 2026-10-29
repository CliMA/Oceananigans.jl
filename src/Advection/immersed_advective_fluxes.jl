using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, inactive_node
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: Open

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
    inside_immersed_boundary(buffer, shift, dir, side)

Check if the stencil required for reconstruction contains immersed nodes

Example
=======

```
julia> inside_immersed_boundary(2, :none, :z, :ᶜ)
3-element Vector{Any}:
 :(inactive_node(i, j, k + -1, ibg, c, c, c))
 :(inactive_node(i, j, k + 0,  ibg, c, c, c))
 :(inactive_node(i, j, k + 1,  ibg, c, c, c))

julia> inside_immersed_boundary(3, :left, :x, :ᶠ)
5-element Vector{Any}:
 :(inactive_node(i + -3, j, k, ibg, c, c, c))
 :(inactive_node(i + -2, j, k, ibg, c, c, c))
 :(inactive_node(i + -1, j, k, ibg, c, c, c))
 :(inactive_node(i + 0,  j, k, ibg, c, c, c))
 :(inactive_node(i + 1,  j, k, ibg, c, c, c))
```
"""
@inline function inside_immersed_boundary(buffer, shift, dir, side)

    N = buffer * 2
    if shift != :none
        N -=1
    end

    if shift == :interior
        rng = 1:N+1
    elseif shift == :right
        rng = 2:N+1
    else
        rng = 1:N
    end

    # A stencil of faces is checked through the cells between its faces
    cells = side == :ᶜ ? rng[1:end-1] : rng
    inactive_cells  = Vector(undef, length(cells))

    for (idx, n) in enumerate(cells)
        offset = side == :ᶠ ? n - buffer - 1 : n - buffer
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

        @inline $near_x_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_y_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_z_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
    end

    for buffer in advection_buffers
        @eval begin
            @inline $near_x_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :none, :x, side)...))
            @inline $near_y_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :none, :y, side)...))
            @inline $near_z_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :none, :z, side)...))

            @inline $near_x_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :interior, :x, side)...))
            @inline $near_y_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :interior, :y, side)...))
            @inline $near_z_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :interior, :z, side)...))
        end
    end

    # The halos beyond an Open side are active, so order reduction there comes from the topology
    for (ξ, d) in ((:x, 1), (:y, 2), (:z, 3))
        open_topology = :(Union{TwoSided{Open}, TwoSided{<:Any, Open}})
        open_grid = d == 1 ? :(IBG{<:Any, <:$open_topology}) :
                    d == 2 ? :(IBG{<:Any, <:Any, <:$open_topology}) :
                             :(IBG{<:Any, <:Any, <:Any, <:$open_topology})

        N = (:Nx, :Ny, :Nz)[d]
        index = (:i, :j, :k)[d]

        for bias in (:symmetric, :biased)
            near_boundary = Symbol(:near_, ξ, :_immersed_boundary_, bias, side)
            outside_halo  = Symbol(:outside_, bias, :_halo_, ξ, side)
            shift = bias == :symmetric ? :none : :interior

            for buffer in advection_buffers
                inactive_cells = inside_immersed_boundary(buffer, shift, ξ, side)
                @eval @inline $near_boundary(i, j, k, ibg::$open_grid, scheme::AbstractAdvectionScheme{$buffer}) =
                    (|)($(inactive_cells...)) | !$outside_halo($index, topology(ibg, $d), ibg.$N, scheme)
            end
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

            @eval begin
                # Fallback for low order interpolation
                @inline $alt1_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, args...) = $interp(i, j, k, ibg, scheme, args...)

                # Conditional high-order interpolation in Bounded directions
                @inline $alt1_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, args...) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $alt2_interp(i, j, k, ibg, scheme.buffer_scheme, args...),
                           $interp(i, j, k, ibg, scheme, args...))
            end
        end
    end
end
