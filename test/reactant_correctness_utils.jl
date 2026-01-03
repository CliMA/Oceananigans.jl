using Printf
using KernelAbstractions: @kernel, @index

using Oceananigans.Advection: U_dot_∇u
using Oceananigans.Coriolis: x_f_cross_U
using Oceananigans.Utils: launch!
using Oceananigans.Grids: architecture

"""
    compare_interior(name, f₁, f₂; rtol=1e-8, atol=sqrt(eps(eltype(f₁))))

Compare the interior of two fields, returning `true` if they are approximately equal.
Prints a diagnostic line showing max difference and location.
"""
function compare_interior(name, f₁, f₂; rtol=1e-8, atol=sqrt(eps(eltype(f₁))))
    a₁ = Array(interior(f₁))
    a₂ = Array(interior(f₂))
    δ = a₁ .- a₂
    max_δ, idx = findmax(abs, δ)
    approx_equal = isapprox(a₁, a₂; rtol, atol)
    @printf("(%6s) interior: ψ₁ ≈ ψ₂: %-5s, max|ψ₁|=%.6e, max|ψ₂|=%.6e, max|δ|=%.6e at %s\n",
            name, approx_equal, maximum(abs, a₁), maximum(abs, a₂), max_δ, string(idx.I))
    return approx_equal
end

"""
    compare_parent(name, f₁, f₂; rtol=1e-8, atol=sqrt(eps(eltype(f₁))))

Compare the parent arrays of two fields, cropping to the common overlap
(to handle padding differences between vanilla and Reactant grids).
Returns `true` if they are approximately equal.
Prints a diagnostic line showing max difference and location.
"""
function compare_parent(name, f₁, f₂; rtol=1e-8, atol=sqrt(eps(eltype(f₁))))
    p₁ = Array(parent(f₁))
    p₂ = Array(parent(f₂))
    # Crop to the common overlap (smaller of the two in each dimension)
    sz₁ = size(p₁)
    sz₂ = size(p₂)
    common_sz = map(min, sz₁, sz₂)
    v₁ = view(p₁, Base.OneTo.(common_sz)...)
    v₂ = view(p₂, Base.OneTo.(common_sz)...)
    δ = v₁ .- v₂
    max_δ, idx = findmax(abs, δ)
    approx_equal = isapprox(v₁, v₂; rtol, atol)
    @printf("(%6s)   parent: ψ₁ ≈ ψ₂: %-5s, max|ψ₁|=%.6e, max|ψ₂|=%.6e, max|δ|=%.6e at %s (overlap %s)\n",
            name, approx_equal, maximum(abs, v₁), maximum(abs, v₂), max_δ, string(idx.I), string(common_sz))
    return approx_equal
end

#####
##### Simplified u-momentum tendency (advection + Coriolis only)
#####

"""
    simple_u_velocity_tendency(i, j, k, grid, advection, coriolis, velocities)

Compute a simplified u-velocity tendency with only advection and Coriolis terms:

    Gu = - U⋅∇u - f × U

This uses `U_dot_∇u` which is the general advection operator that works for
all advection schemes (flux form, vector invariant, etc.) on all grid types.
"""
@inline function simple_u_velocity_tendency(i, j, k, grid, advection, coriolis, velocities)
    return (- U_dot_∇u(i, j, k, grid, advection, velocities)
            - x_f_cross_U(i, j, k, grid, coriolis, velocities))
end

"""
Kernel to compute the simplified u-velocity tendency.
"""
@kernel function _compute_simple_Gu!(Gu, grid, advection, coriolis, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = simple_u_velocity_tendency(i, j, k, grid, advection, coriolis, velocities)
end

"""
    compute_simple_Gu!(Gu, advection, coriolis, velocities)

Compute the simplified u-velocity tendency (advection + Coriolis) and store in `Gu`.
"""
function compute_simple_Gu!(Gu, advection, coriolis, velocities)
    grid = Gu.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _compute_simple_Gu!, Gu, grid, advection, coriolis, velocities)
    return nothing
end
