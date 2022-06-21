#####
##### Centered fourth-order advection scheme
#####

"""
    struct CenteredFourthOrder <: AbstractCenteredAdvectionScheme{2}

Centered fourth-order advection scheme.
"""
struct Centered{N, CA} <: AbstractCenteredAdvectionScheme{N} 
    "advection scheme used near boundaries"
    boundary_scheme :: CA

    function Centered{N}(boundary_scheme::CA) where {N, CA}
        return new{N, CA}(boundary_scheme)
    end
end

function Centered(; order = 2) 

    N  = Int(order ÷ 2)

    if N > 1 
        boundary_scheme = Centered(order = order - 2)
    else
        boundary_scheme = nothing
    end

    return Centered{N}(boundary_scheme)
end

# Useful aliases
CenteredSecondOrder() = Centered(order = 2)
CenteredFourthOrder() = Centered(order = 4)

for (dir, ξ) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ), (:x, :y, :z))
    stencil = Symbol(:symmetric_interpolate_, dir)

    for buffer in [1, 2, 3, 4, 5]
        @eval begin
            @inline $stencil(i, j, k, grid, scheme::Centered{$buffer}, ψ, args...)           = @inbounds $(calc_advection_stencil(buffer, :symm, ξ, false))
            @inline $stencil(i, j, k, grid, scheme::Centered{$buffer}, ψ::Function, args...) = @inbounds $(calc_advection_stencil(buffer, :symm, ξ,  true))
        end
    end
end

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::Centered, u) = symmetric_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::Centered, v) = symmetric_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::Centered, w) = symmetric_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
