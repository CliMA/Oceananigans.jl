# Generic reconstruction methods valid for all reconstruction schemes
# Unroll the functions to pass the coordinates in case of a stretched grid
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, args...) = inner_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, args...) = inner_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, args...) = inner_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ψ, args...) = inner_symmetric_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ψ, args...) = inner_symmetric_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ψ, args...) = inner_symmetric_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, args...)  = inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, args...)  = inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, args...)  = inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, args...) = inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, args...) = inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, args...) = inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ψ, args...)  = inner_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ψ, args...)  = inner_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ψ, args...)  = inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ψ, args...) = inner_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ψ, args...) = inner_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ψ, args...) = inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

struct FirstDerivative end
struct SecondDerivative end
struct Primitive end

num_prod(i, m, l, r, xr, xi, shift, op, order, args...)            = @inbounds prod(xr[i+shift] - xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::FirstDerivative)  = @inbounds 2*xr[i+shift] - sum(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::SecondDerivative) = 2

@inline function num_prod(i, m, l, r, xr, xi, shift, op, order, ::Primitive) 
    s = sum(xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
    p = prod(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))

    return xr[i+shift]^3 / 3 - s * xr[i+shift]^2 / 2 + p * xr[i+shift]
end

"""
    stencil_coefficients(i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)

Return coefficients for finite-volume polynomial reconstruction of order `order` at stencil `r`.

Positional Arguments
====================

- `xi`: the locations of the reconstructing value, i.e. either the center coordinate,
  for centered quantities or face coordinate for staggered
- `xr`: the opposite of the reconstruction location desired, i.e., if a recostruction at
  `Center`s is required xr is the face coordinate

On a uniform `grid`, the coefficients are independent of the `xr` and `xi` values.
"""
@inline function stencil_coefficients(i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)
    coeffs = zeros(order)
    @inbounds begin
        for j in 0:order-1
            for m in j+1:order
                numerator   = sum(num_prod(i, m, l, r, xr, xi, shift, op, order, der) for l=0:order if l != m)
                denominator = prod(xi[op(i, r-m+1)] - xi[op(i, r-l+1)] for l=0:order if l != m)
                coeffs[j+1] += numerator / denominator * (xi[op(i, r-j)] - xi[op(i, r-j+1)])
            end
        end
    end

    return tuple(coeffs...)
end

"""
    Coefficients for uniform centered and upwind schemes 

symmetric coefficients are for centered reconstruction (dispersive, even order), 
left and right are for upwind biased (diffusive, odd order)
examples:
julia> using Oceananigans.Advection: coeff2_symmetric, coeff3_left, coeff3_right, coeff4_symmetric, coeff5_left

julia> coeff2_symmetric
(0.5, 0.5)

julia> coeff3_left, coeff3_right
((0.33333333333333337, 0.8333333333333334, -0.16666666666666666), (-0.16666666666666669, 0.8333333333333333, 0.3333333333333333))

julia> coeff4_symmetric
(-0.08333333333333333, 0.5833333333333333, 0.5833333333333333, -0.08333333333333333)

julia> coeff5_left
(-0.049999999999999926, 0.45000000000000007, 0.7833333333333333, -0.21666666666666667, 0.03333333333333333)
"""
const coeff1_left  = 1.0
const coeff1_right = 1.0

# buffer in [1:6] allows up to Centered(order = 12) and UpwindBiased(order = 11)
for buffer in advection_buffers
    order_bias = 2buffer - 1
    order_symm = 2buffer

    coeff_symm  = Symbol(:coeff, order_symm, :_symmetric)
    coeff_left  = Symbol(:coeff, order_bias, :_left)
    coeff_right = Symbol(:coeff, order_bias, :_right)
    @eval begin
        const $coeff_symm  = stencil_coefficients(50, $(buffer - 1), collect(1:100), collect(1:100); order = $order_symm)
        if $order_bias > 1
            const $coeff_left  = stencil_coefficients(50, $(buffer - 2), collect(1:100), collect(1:100); order = $order_bias)
            const $coeff_right = stencil_coefficients(50, $(buffer - 1), collect(1:100), collect(1:100); order = $order_bias)
        end
    end
end

""" 
    calc_reconstruction_stencil(buffer, shift, dir, func::Bool = false)

Stencils for reconstruction calculations (note that WENO has its own reconstruction stencils)

The first argument is the `buffer`, not the `order`! 
- `order = 2 * buffer` for Centered reconstruction
- `order = 2 * buffer - 1` for Upwind reconstruction
   
Examples
========

```jldoctest
julia> using Oceananigans.Advection: calc_reconstruction_stencil

julia> calc_reconstruction_stencil(1, :right, :x)
:(+(convert(FT, coeff1_right[1]) * ψ[i + 0, j, k]))

julia> calc_reconstruction_stencil(1, :left, :x)
:(+(convert(FT, coeff1_left[1]) * ψ[i + -1, j, k]))

julia> calc_reconstruction_stencil(1, :symmetric, :x)
:(convert(FT, coeff2_symmetric[2]) * ψ[i + -1, j, k] + convert(FT, coeff2_symmetric[1]) * ψ[i + 0, j, k])

julia> calc_reconstruction_stencil(2, :symmetric, :x)
:(convert(FT, coeff4_symmetric[4]) * ψ[i + -2, j, k] + convert(FT, coeff4_symmetric[3]) * ψ[i + -1, j, k] + convert(FT, coeff4_symmetric[2]) * ψ[i + 0, j, k] + convert(FT, coeff4_symmetric[1]) * ψ[i + 1, j, k])

julia> calc_reconstruction_stencil(3, :left, :x)
:(convert(FT, coeff5_left[5]) * ψ[i + -3, j, k] + convert(FT, coeff5_left[4]) * ψ[i + -2, j, k] + convert(FT, coeff5_left[3]) * ψ[i + -1, j, k] + convert(FT, coeff5_left[2]) * ψ[i + 0, j, k] + convert(FT, coeff5_left[1]) * ψ[i + 1, j, k])
```
"""
@inline function calc_reconstruction_stencil(buffer, shift, dir, func::Bool = false)
    N = buffer * 2
    order = shift == :symmetric ? N : N - 1
    if shift != :symmetric
        N = N .- 1
    end
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    stencil_full = Vector(undef, N)
    coeff = Symbol(:coeff, order, :_, shift)
    for (idx, n) in enumerate(rng)
        c = n - buffer - 1
        if func
            stencil_full[idx] = dir == :x ?
                                :(convert(FT, $coeff[$(order - idx + 1)]) * ψ(i + $c, j, k, grid, args...)) :
                                dir == :y ?
                                :(convert(FT, $coeff[$(order - idx + 1)]) * ψ(i, j + $c, k, grid, args...)) :
                                :(convert(FT, $coeff[$(order - idx + 1)]) * ψ(i, j, k + $c, grid, args...))
        else
            stencil_full[idx] =  dir == :x ? 
                                :(convert(FT, $coeff[$(order - idx + 1)]) * ψ[i + $c, j, k]) :
                                dir == :y ?
                                :(convert(FT, $coeff[$(order - idx + 1)]) * ψ[i, j + $c, k]) :
                                :(convert(FT, $coeff[$(order - idx + 1)]) * ψ[i, j, k + $c])
        end
    end
    return Expr(:call, :+, stencil_full...)
end

#####
##### Shenanigans for stretched directions
#####

@inline function reconstruction_stencil(buffer, shift, dir, func::Bool = false)
    N = buffer * 2
    order = shift == :symmetric ? N : N - 1
    if shift != :symmetric
        N = N .- 1
    end
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    stencil_full = Vector(undef, N)
    coeff = Symbol(:coeff, order, :_, shift)
    for (idx, n) in enumerate(rng)
        c = n - buffer - 1
        if func
            stencil_full[idx] = dir == :x ?
                                :(ψ(i + $c, j, k, grid, args...)) :
                                dir == :y ?
                                :(ψ(i, j + $c, k, grid, args...)) :
                                :(ψ(i, j, k + $c, grid, args...))
        else
            stencil_full[idx] = dir == :x ?
                                :(ψ[i + $c, j, k]) :
                                dir == :y ?
                                :(ψ[i, j + $c, k]) :
                                :(ψ[i, j, k + $c])
        end
    end
    return :($(reverse(stencil_full)...),)
end

@inline function compute_reconstruction_coefficients(grid, FT, scheme; order)

    method = scheme == :Centered ? 1 : scheme == :Upwind ? 2 : 3

    if grid isa Nothing
        coeff_xᶠᵃᵃ = nothing
        coeff_xᶜᵃᵃ = nothing
        coeff_yᵃᶠᵃ = nothing
        coeff_yᵃᶜᵃ = nothing
        coeff_zᵃᵃᶠ = nothing
        coeff_zᵃᵃᶜ = nothing
    else
        arch       = architecture(grid)
        Hx, Hy, Hz = halo_size(grid)
        new_grid   = with_halo((Hx+1, Hy+1, Hz+1), grid)

        coeff_xᶠᵃᵃ = reconstruction_coefficients(FT, getproperty(metrics[1], new_grid), arch, new_grid.Nx, Val(method); order)
        coeff_xᶜᵃᵃ = reconstruction_coefficients(FT, getproperty(metrics[2], new_grid), arch, new_grid.Nx, Val(method); order)
        coeff_yᵃᶠᵃ = reconstruction_coefficients(FT, getproperty(metrics[3], new_grid), arch, new_grid.Ny, Val(method); order)
        coeff_yᵃᶜᵃ = reconstruction_coefficients(FT, getproperty(metrics[4], new_grid), arch, new_grid.Ny, Val(method); order)
        coeff_zᵃᵃᶠ = reconstruction_coefficients(FT, getproperty(metrics[5], new_grid), arch, new_grid.Nz, Val(method); order)
        coeff_zᵃᵃᶜ = reconstruction_coefficients(FT, getproperty(metrics[6], new_grid), arch, new_grid.Nz, Val(method); order)
    end

    return (coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ)
end

# Fallback for uniform directions
for val in [1, 2, 3]
    @eval begin
        @inline reconstruction_coefficients(FT, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N, ::Val{$val}; order) = nothing
        @inline reconstruction_coefficients(FT, coord::AbstractRange, arch, N, ::Val{$val}; order)                              = nothing
    end
end

# Stretched reconstruction coefficients for `Centered` schemes
@inline function reconstruction_coefficients(FT, coord, arch, N, ::Val{1}; order) 
    cpu_coord = on_architecture(CPU(), coord)
    r = ((order + 1) ÷ 2) - 1
    s = create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order)
    return s
end

# Stretched reconstruction coefficients for `UpwindBiased` schemes
@inline function reconstruction_coefficients(FT, coord, arch, N, ::Val{2}; order) 
    cpu_coord = on_architecture(CPU(), coord)
    rleft  = ((order + 1) ÷ 2) - 2
    rright = ((order + 1) ÷ 2) - 1
    s = []
    for r in [rleft, rright]
        push!(s, create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order))
    end
    return tuple(s...)
end

# Stretched reconstruction coefficients for `WENO` schemes
@inline function reconstruction_coefficients(FT, coord, arch, N, ::Val{3}; order) 

    cpu_coord = on_architecture(CPU(), coord)
    s = []
    for r in -1:order-1
        push!(s, create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order))
    end
    return tuple(s...)
end

# general reconstruction coefficients for order `order` and stencil `r` where r 
@inline function create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order)
    stencil = NTuple{order, FT}[]
    @inbounds begin
        for i = 0:N+1
            push!(stencil, stencil_coefficients(i, r, cpu_coord, cpu_coord; order))     
        end
    end
    return OffsetArray(on_architecture(arch, stencil), -1)
end
