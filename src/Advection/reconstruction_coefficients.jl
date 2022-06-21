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
- the opposite of the reconstruction desired
   i.e., if a recostruction at Centers is required xr is the face coordinate

If a grid is uniform, the coefficients are independent on the values of xr and xi and of their difference
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

# Coefficients for uniform centered and upwind schemes
const coeff1_left  = 1.0
const coeff1_right = 1.0

for buffer in [1, 2, 3, 4, 5]
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

function calc_advection_stencil(buffer, shift, dir, func) 
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