using Adapt
using CUDA: CuArray
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array

import Oceananigans.Operators: ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, 
                               ∂yᵃᶜᵃ, ∂yᵃᶠᵃ,
                               ∂zᵃᵃᶜ, ∂zᵃᵃᶠ

import Oceananigans.TurbulenceClosures: ivd_upper_diagonalᵃᵃᶜ,
                                        ivd_lower_diagonalᵃᵃᶜ,
                                        ivd_upper_diagonalᵃᵃᶠ,
                                        ivd_lower_diagonalᵃᵃᶠ


abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

const GFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBoundary}

#####
##### GridFittedBoundary
#####

struct GridFittedBoundary{S} <: AbstractGridFittedBoundary
    mask :: S
end

@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary) = ib.mask(node(c, c, c, i, j, k, underlying_grid)...)

#####
##### GridFittedBottom
#####

"""
    GridFittedBottom(bottom)

Return an immersed boundary...
"""
struct GridFittedBottom{B} <: AbstractGridFittedBoundary
    bottom :: B
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom)
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return z < ib.bottom(x, y)
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom{<:AbstractArray})
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return @inbounds z < ib.bottom[i, j]
end

const ArrayGridFittedBottom = GridFittedBottom{<:Array}
const CuArrayGridFittedBottom = GridFittedBottom{<:CuArray}

function ImmersedBoundaryGrid(grid, ib::Union{ArrayGridFittedBottom, CuArrayGridFittedBottom})
    # Wrap bathymetry in an OffsetArray with halos
    arch = grid.architecture
    bottom_field = Field{Center, Center, Nothing}(grid)
    bottom_data = arch_array(arch, ib.bottom)
    bottom_field .= bottom_data
    fill_halo_regions!(bottom_field, arch)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)
    new_ib = GridFittedBottom(offset_bottom_array)
    return ImmersedBoundaryGrid(grid, new_ib)
end

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

@inline Δzᵃᵃᶜ(i, j, k, ibg::GFBIBG) = ifelse(is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary),
                                             zero(eltype(ibg.grid)),
                                             Δzᵃᵃᶜ(i, j, k, ibg.grid))

@inline Δzᶠᶜᶜ(i, j, k, ibg::GFBIBG) = ifelse(solid_node(Face(), Center(), Center(), i, j, k, ibg),
                                             zero(eltype(ibg)),
                                             Δzᵃᵃᶜ(i, j, k, ibg.grid))

@inline Δzᶜᶠᶜ(i, j, k, ibg::GFBIBG) = ifelse(solid_node(Center(), Face(), Center(), i, j, k, ibg),
                                             zero(eltype(ibg)),
                                             Δzᵃᵃᶜ(i, j, k, ibg.grid))

@inline Δzᵃᵃᶠ(i, j, k, ibg::GFBIBG) = ifelse(is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary),
                                             zero(eltype(ibg.grid)),
                                             Δzᵃᵃᶠ(i, j, k, ibg.grid))

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom))     

#####
##### Implicit vertical diffusion
#####

for (xloc, xsup) in zip([Center, Face], [:ᶜ, :ᶠ])
    for (yloc, ysup) in zip([Center, Face], [:ᶜ, :ᶠ])
        for (zbias, zloc, zsup) in zip([1, 0], [Center, Face], [:ᶜ, :ᶠ])

            upper_diagonal = Symbol(:ivd_upper_diagonal, xsup, ysup, zsup)
            lower_diagonal = Symbol(:ivd_lower_diagonal, xsup, ysup, zsup)

            @eval begin
                @inline function $upper_diagonal(i, j, k, ibg::GFIBG, clock, Δt, interp_κ, κ)
                    return ifelse(solid_node($xloc(), $yloc(), $zloc(), i, j, k+$zbias, ibg), 
                                zero(eltype(ibg.grid)),
                                $upper_diagonal(i, j, k, ibg.grid, clock, Δt, interp_κ, κ))
                end
                
                @inline function $lower_diagonal(i, j, k, ibg::GFIBG, clock, Δt, interp_κ, κ)
                    return ifelse(solid_node($xloc(), $yloc(), $zloc(), i, j, k+$zbias, ibg),
                                zero(eltype(ibg.grid)),
                                $lower_diagonal(i, j, k, ibg.grid, clock, Δt, interp_κ, κ))
                end
            end 
        end
    end
end


# Tracers and horizontal velocities at cell centers in z

@inline function ivd_upper_diagonalᵃᵃᶜ(i, j, k, ibg::GFIBG, clock, Δt, κ⁻⁻ᶠ, κ)
    return ifelse(solid_node(Center(), Center(), Face(), i, j, k+1, ibg),
                  zero(eltype(ibg.grid)),
                  ivd_upper_diagonalᵃᵃᶜ(i, j, k, ibg.grid, clock, Δt, κ⁻⁻ᶠ, κ))
end

@inline function ivd_lower_diagonalᵃᵃᶜ(i, j, k, ibg::GFIBG, clock, Δt, κ⁻⁻ᶠ, κ)
    return ifelse(solid_node(Center(), Center(), Face(), i, j, k+1, ibg),
                  zero(eltype(ibg.grid)),
                  ivd_lower_diagonalᵃᵃᶜ(i, j, k, ibg.grid, clock, Δt, κ⁻⁻ᶠ, κ))
end

# Vertical velocitiy w at cell faces in z

@inline function ivd_upper_diagonalᵃᵃᶠ(i, j, k, ibg::GFIBG, clock, Δt, νᶜᶜᶜ, ν)
    return ifelse(solid_node(Center(), Center(), Center(), i, j, k, ibg),
                  zero(eltype(ibg.grid)),
                  ivd_upper_diagonalᵃᵃᶠ(i, j, k, ibg.grid, clock, Δt, νᶜᶜᶜ, ν))
end

@inline function ivd_lower_diagonalᵃᵃᶠ(i, j, k, ibg::GFIBG, clock, Δt, νᶜᶜᶜ, ν)
    return ifelse(solid_node(Center(), Center(), Center(), i, j, k, ibg),
                  zero(eltype(ibg.grid)),
                  ivd_lower_diagonalᵃᵃᶠ(i, j, k, ibg.grid, clock, Δt, νᶜᶜᶜ, ν))
end


# metrics are 0 inside the immersed boundaries. To avoid NaNs in the Nonhydrostatic pressure solver correction
# (we must be able to define derivatives also inside or across the immersed boundary)

derivative_operators = (:∂xᶜᵃᵃ, :∂xᶠᵃᵃ, 
                        :∂yᵃᶜᵃ, :∂yᵃᶠᵃ,
                        :∂zᵃᵃᶜ, :∂zᵃᵃᶠ)

for operator in derivative_operators
        @eval $operator(i, j, k, ibg::GFIBG, args...) = $operator(i, j, k, ibg.grid, args...)
end
