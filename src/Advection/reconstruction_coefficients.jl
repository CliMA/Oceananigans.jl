struct FirstDerivative end
struct SecondDerivative end
struct Primitive end

num_prod(i, m, l, r, xr, xi, shift, op, order, args...)            = prod(xr[i+shift] - xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::FirstDerivative)  = 2*xr[i+shift] - sum(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::SecondDerivative) = 2

function num_prod(i, m, l, r, xr, xi, shift, op, order, ::Primitive) 
    s = sum(xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
    p = prod(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))

    return xr[i+shift]^3 / 3 - sum * xr[i+shift]^2 / 2 + prod * xr[i+shift]
end

"""
Coefficients for finite-volume polynomial reconstruction of order `order` at stencil r 
xi and xr are respectively:

- the locations of the reconstructing value 
   i.e. either the center coordinate, for centered quantities or face coordinate for staggered
- the opposite of the reconstruction location desired
   i.e., if a recostruction at `Center`s is required xr is the face coordinate

On a grid is uniform, coefficients are independent of the xr and xi values
"""
function stencil_coefficients(i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)
    coeffs = zeros(order)
    for j in 0:order-1
        for m in j+1:order
            numerator   = sum(num_prod(i, m, l, r, xr, xi, shift, op, order, der) for l=0:order if l != m)
            denominator = prod(xi[op(i, r-m+1)] - xi[op(i, r-l+1)] for l=0:order if l != m)
            coeffs[j+1] += numerator / denominator * (xi[op(i, r-j)] - xi[op(i, r-j+1)])
        end
    end

    return tuple(coeffs...)
end

"""
    Coefficients for uniform centered and upwind schemes 

symmetric coefficients are for centered reconstruction (dispersive, even order), 
left and right are for upwind biased (diffusive, odd order)
examples:
julia> using Oceananigans.Advection: coeff2_symm, coeff3_left, coeff3_right, coeff4_symm, coeff5_left

julia> coeff2_symm
(0.5, 0.5)

julia> coeff3_left, coeff3_right
((0.33333333333333337, 0.8333333333333334, -0.16666666666666666), (-0.16666666666666669, 0.8333333333333333, 0.3333333333333333))

julia> coeff4_symm
(-0.08333333333333333, 0.5833333333333333, 0.5833333333333333, -0.08333333333333333)

julia> coeff5_left
(-0.049999999999999926, 0.45000000000000007, 0.7833333333333333, -0.21666666666666667, 0.03333333333333333)
"""
const coeff1_left  = 1.0
const coeff1_right = 1.0

# buffer in [1:6] allows up to Centered(order = 12) and UpwindBiased(order = 11)
for buffer in [1, 2, 3, 4, 5, 6]
    order_bias = 2buffer - 1
    order_symm = 2buffer

    coeff_symm  = Symbol(:coeff, order_symm, :_symm)
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
    Stencils for reconstruction calculations (

The first argument is the buffer, not the order!! 

`order = 2 * buffer`   for Centered reconstruction
`order = 2 * buffer-1` for Upwind reconstruction

examples:
julia> calc_advection_stencil(1, :right, :x)
:(+(coeff1_right[1] * ψ[i + 0, j, k]))

julia> calc_advection_stencil(1, :left, :x)
:(+(coeff1_left[1] * ψ[i + -1, j, k]))

julia> calc_advection_stencil(1, :symm, :x)
:(coeff2_symm[1] * ψ[i + -1, j, k] + coeff2_symm[2] * ψ[i + 0, j, k])

julia> calc_advection_stencil(2, :symm, :x)
:(coeff4_symm[1] * ψ[i + -2, j, k] + coeff4_symm[2] * ψ[i + -1, j, k] + coeff4_symm[3] * ψ[i + 0, j, k] + coeff4_symm[4] * ψ[i + 1, j, k])

julia> calc_advection_stencil(3, :left, :x)
:(coeff5_left[1] * ψ[i + -3, j, k] + coeff5_left[2] * ψ[i + -2, j, k] + coeff5_left[3] * ψ[i + -1, j, k] + coeff5_left[4] * ψ[i + 0, j, k] + coeff5_left[5] * ψ[i + 1, j, k])

"""
function calc_advection_stencil(buffer, shift, dir, func::Bool = false) 
    N = buffer * 2
    order = shift == :symm ? N : N - 1
    if shift != :symm
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
                                :($coeff[$idx] * ψ(i + $c, j, k, grid, args...)) :
                                dir == :y ?
                                :($coeff[$idx] * ψ(i, j + $c, k, grid, args...)) :
                                :($coeff[$idx] * ψ(i, j, k + $c, grid, args...))
        else
            stencil_full[idx] =  dir == :x ? 
                                :($coeff[$idx] * ψ[i + $c, j, k]) :
                                dir == :y ?
                                :($coeff[$idx] * ψ[i, j + $c, k]) :
                                :($coeff[$idx] * ψ[i, j, k + $c])
        end
    end
    return Expr(:call, :+, stencil_full...)
end