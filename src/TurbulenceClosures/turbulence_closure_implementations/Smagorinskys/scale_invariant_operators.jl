using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃²
using Oceananigans.Grids: BoundarySide, BoundaryTopology, Flat, TwoSided, XFlatGrid, YFlatGrid, ZFlatGrid, topology
using Oceananigans.Operators: volume, V⁻¹, ∂xᶜᶜᶜ, ∂xᶠᶜᶠ, ∂xᶠᶠᶜ, ∂yᶜᶜᶜ, ∂yᶜᶠᶠ, ∂yᶠᶠᶜ, ∂zᶜᶜᶜ, ∂zᶜᶠᶠ, ∂zᶠᶜᶠ

#####
##### Double dot product of strain on cell edges (currently unused)
#####

"Return the double dot product of strain at `ccc`."
@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `ffc`."
@inline ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w) =     ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 *   Σ₁₂²(i, j, k, grid, u, v, w) +
                                            2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `fcf`."
@inline ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w) =     ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 *   Σ₁₃²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `cff`."
@inline ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w) =     ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 *   Σ₂₃²(i, j, k, grid, u, v, w)

"Return the double dot product of strain at `ccf`."
@inline ΣᵢⱼΣᵢⱼᶜᶜᶠ(i, j, k, grid, u, v, w) =       ℑzᵃᵃᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑxyzᶜᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 *   ℑxᶜᵃᵃ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 *   ℑyᵃᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)


#####
##### Test filter
#####

# The test filter applies the three-point filter (¼, ½, ¼) along each direction. Away from
# boundaries it is the 27-point stencil ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, f, args...), the trapezoidal
# approximation of a box filter that is twice as wide as the grid spacing.
#
# Beyond a `Closed` or `Open` side the halos hold boundary data: `fill_halo_regions!` writes
# their first layer only, and nothing where two such sides meet. The three-point filter is
# therefore applied only where its stencil does not reach beyond such a side, which leaves cell
# centers next to the boundary and cell faces on it unfiltered in the boundary-normal direction.
# `ᾱ²` accounts for the directions that are not filtered.

const LeftBoundedTopology  = TwoSided{<:BoundarySide}
const RightBoundedTopology = TwoSided{<:Any, <:BoundarySide}

# Whether the three-point filter is applied at index `i` along a direction with topology `topo`
# and `N` cells, to a quantity located at cell centers (ᶜ) or cell faces (ᶠ) in that direction.
@inline test_filteredᶜ(i, topo, N) = true
@inline test_filteredᶠ(i, topo, N) = true

@inline test_filteredᶜ(i, ::Type{Flat}, N) = false
@inline test_filteredᶠ(i, ::Type{Flat}, N) = false

@inline test_filteredᶜ(i, ::Type{<:LeftBoundedTopology}, N) = i > 1
@inline test_filteredᶠ(i, ::Type{<:LeftBoundedTopology}, N) = i > 1

@inline test_filteredᶜ(i, ::Type{<:RightBoundedTopology}, N) = i < N
@inline test_filteredᶠ(i, ::Type{<:RightBoundedTopology}, N) = i < N + 1

@inline test_filteredᶜ(i, ::Type{<:BoundaryTopology}, N) = (i > 1) & (i < N)
@inline test_filteredᶠ(i, ::Type{<:BoundaryTopology}, N) = (i > 1) & (i < N + 1)

# Integer weights of the three-point filter, which sum to 4. Where the filter is not applied
# all of the weight goes to the central node.
@inline filter_weight(s, filtered) = ifelse(filtered, 2 - abs(s), 4 * (s == 0))

# Three-point filter of `f` along the direction with unit index vector `(δi, δj, δk)`. Where the
# filter is not applied the neighbors collapse onto the central node, so that nothing is read
# beyond a boundary. The single call to `f` in a loop matters: with one call per neighbor, each
# level of nested filters would triple the size of the kernels to compile.
@inline function three_point_filter(i, j, k, grid, (δi, δj, δk), filtered, f::F, args...) where F<:Function
    f̄ = zero(grid)
    for s in -1:1
        n = s * filtered
        f̄ += filter_weight(s, filtered) * f(i + n * δi, j + n * δj, k + n * δk, grid, args...)
    end
    return f̄ / 4
end

@inline filter_xᶜᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function =
    three_point_filter(i, j, k, grid, (1, 0, 0), test_filteredᶜ(i, topology(grid, 1), grid.Nx), f, args...)

@inline filter_xᶠᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function =
    three_point_filter(i, j, k, grid, (1, 0, 0), test_filteredᶠ(i, topology(grid, 1), grid.Nx), f, args...)

@inline filter_yᵃᶜᵃ(i, j, k, grid, f::F, args...) where F<:Function =
    three_point_filter(i, j, k, grid, (0, 1, 0), test_filteredᶜ(j, topology(grid, 2), grid.Ny), f, args...)

@inline filter_yᵃᶠᵃ(i, j, k, grid, f::F, args...) where F<:Function =
    three_point_filter(i, j, k, grid, (0, 1, 0), test_filteredᶠ(j, topology(grid, 2), grid.Ny), f, args...)

@inline filter_zᵃᵃᶜ(i, j, k, grid, f::F, args...) where F<:Function =
    three_point_filter(i, j, k, grid, (0, 0, 1), test_filteredᶜ(k, topology(grid, 3), grid.Nz), f, args...)

@inline filter_zᵃᵃᶠ(i, j, k, grid, f::F, args...) where F<:Function =
    three_point_filter(i, j, k, grid, (0, 0, 1), test_filteredᶠ(k, topology(grid, 3), grid.Nz), f, args...)

# There is nothing to filter along `Flat` directions
@inline filter_xᶜᵃᵃ(i, j, k, grid::XFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)
@inline filter_xᶠᵃᵃ(i, j, k, grid::XFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)
@inline filter_yᵃᶜᵃ(i, j, k, grid::YFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)
@inline filter_yᵃᶠᵃ(i, j, k, grid::YFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)
@inline filter_zᵃᵃᶜ(i, j, k, grid::ZFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)
@inline filter_zᵃᵃᶠ(i, j, k, grid::ZFlatGrid, f::F, args...) where F<:Function = f(i, j, k, grid, args...)

@inline filterᶜᶜᶜ(i, j, k, grid, f::F, args...) where F<:Function = filter_xᶜᵃᵃ(i, j, k, grid, filter_yᵃᶜᵃ, filter_zᵃᵃᶜ, f, args...)
@inline filterᶠᶜᶜ(i, j, k, grid, f::F, args...) where F<:Function = filter_xᶠᵃᵃ(i, j, k, grid, filter_yᵃᶜᵃ, filter_zᵃᵃᶜ, f, args...)
@inline filterᶜᶠᶜ(i, j, k, grid, f::F, args...) where F<:Function = filter_xᶜᵃᵃ(i, j, k, grid, filter_yᵃᶠᵃ, filter_zᵃᵃᶜ, f, args...)
@inline filterᶜᶜᶠ(i, j, k, grid, f::F, args...) where F<:Function = filter_xᶜᵃᵃ(i, j, k, grid, filter_yᵃᶜᵃ, filter_zᵃᵃᶠ, f, args...)

@inline node_value(i, j, k, grid, c) = @inbounds c[i, j, k]

@inline filterᶜᶜᶜ(i, j, k, grid, c::AbstractArray) = filterᶜᶜᶜ(i, j, k, grid, node_value, c)
@inline filterᶠᶜᶜ(i, j, k, grid, u::AbstractArray) = filterᶠᶜᶜ(i, j, k, grid, node_value, u)
@inline filterᶜᶠᶜ(i, j, k, grid, v::AbstractArray) = filterᶜᶠᶜ(i, j, k, grid, node_value, v)
@inline filterᶜᶜᶠ(i, j, k, grid, w::AbstractArray) = filterᶜᶜᶠ(i, j, k, grid, node_value, w)

#####
##### Velocity gradients
#####

# Diagonal
@inline ∂x_ū(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, filterᶠᶜᶜ, u)
@inline ∂y_v̄(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, filterᶜᶠᶜ, v)
@inline ∂z_w̄(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, filterᶜᶜᶠ, w)

# Off-diagonal
@inline ∂x_v̄(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, filterᶜᶠᶜ, v)
@inline ∂x_w̄(i, j, k, grid, w) = ∂xᶠᶜᶠ(i, j, k, grid, filterᶜᶜᶠ, w)

@inline ∂y_ū(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, filterᶠᶜᶜ, u)
@inline ∂y_w̄(i, j, k, grid, w) = ∂yᶜᶠᶠ(i, j, k, grid, filterᶜᶜᶠ, w)

@inline ∂z_ū(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, filterᶠᶜᶜ, u)
@inline ∂z_v̄(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, filterᶜᶠᶜ, v)

#####
##### Strain components
#####

const AG = AbstractGrid

# ccc strain components
@inline Σ̄₁₁(i, j, k, grid, u) = ∂x_ū(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, v) = ∂y_v̄(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, w) = ∂z_w̄(i, j, k, grid, w)

@inline tr_Σ̄(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u) + Σ̄₂₂(i, j, k, grid, v) + Σ̄₃₃(i, j, k, grid, w)
@inline tr_Σ̄²(ijk...) = Σ̄₁₁(ijk...)^2 + Σ̄₂₂(ijk...)^2 + Σ̄₃₃(ijk...)^2

# ffc
@inline Σ̄₁₂(i, j, k, grid::AG{FT}, u, v) where FT = FT(0.5) * (∂y_ū(i, j, k, grid, u) + ∂x_v̄(i, j, k, grid, v))
@inline Σ̄₁₂²(i, j, k, grid, u, v) = Σ̄₁₂(i, j, k, grid, u, v)^2

# fcf
@inline Σ̄₁₃(i, j, k, grid::AG{FT}, u, w) where FT = FT(0.5) * (∂z_ū(i, j, k, grid, u) + ∂x_w̄(i, j, k, grid, w))
@inline Σ̄₁₃²(i, j, k, grid, u, w) = Σ̄₁₃(i, j, k, grid, u, w)^2

# cff
@inline Σ̄₂₃(i, j, k, grid::AG{FT}, v, w) where FT = FT(0.5) * (∂z_v̄(i, j, k, grid, v) + ∂y_w̄(i, j, k, grid, w))
@inline Σ̄₂₃²(i, j, k, grid, v, w) = Σ̄₂₃(i, j, k, grid, v, w)^2

@inline Σ̄₁₁(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, u, v, w) = Σ̄₂₂(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, u, v, w) = Σ̄₃₃(i, j, k, grid, w)

@inline Σ̄₁₂(i, j, k, grid, u, v, w) = Σ̄₁₂(i, j, k, grid, u, v)
@inline Σ̄₁₃(i, j, k, grid, u, v, w) = Σ̄₁₃(i, j, k, grid, u, w)
@inline Σ̄₂₃(i, j, k, grid, u, v, w) = Σ̄₂₃(i, j, k, grid, v, w)

@inline Σ̄₁₂²(i, j, k, grid, u, v, w) = Σ̄₁₂²(i, j, k, grid, u, v)
@inline Σ̄₁₃²(i, j, k, grid, u, v, w) = Σ̄₁₃²(i, j, k, grid, u, w)
@inline Σ̄₂₃²(i, j, k, grid, u, v, w) = Σ̄₂₃²(i, j, k, grid, v, w)

#####
##### Double dot product of strain on cell edges
#####

"Return the double dot product of strain at `ccc` on a 2δ test grid."
@inline Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) = (tr_Σ̄²(i, j, k, grid, u, v, w)
                                             + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂², u, v, w)
                                             + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃², u, v, w)
                                             + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃², u, v, w))

# Here the notation ⟨A⟩ is equivalent to Ā: a filter of size 2Δᶠ, where Δᶠ is the grid scale.

@inline ΣΣ₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₁₁(i, j, k, grid, u, v, w)
@inline ΣΣ₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₂₂(i, j, k, grid, u, v, w)
@inline ΣΣ₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₃₃(i, j, k, grid, u, v, w)

@inline ΣΣ₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂, u, v, w)
@inline ΣΣ₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃, u, v, w)
@inline ΣΣ₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃, u, v, w)

@inline var"⟨ΣΣ₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filterᶜᶜᶜ(i, j, k, grid, ΣΣ₁₁ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filterᶜᶜᶜ(i, j, k, grid, ΣΣ₂₂ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filterᶜᶜᶜ(i, j, k, grid, ΣΣ₃₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)

@inline var"⟨ΣΣ₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filterᶜᶜᶜ(i, j, k, grid, ΣΣ₁₂ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filterᶜᶜᶜ(i, j, k, grid, ΣΣ₁₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filterᶜᶜᶜ(i, j, k, grid, ΣΣ₂₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)

@inline Σ̄Σ̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₁₁(i, j, k, grid, u, v, w)
@inline Σ̄Σ̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₂₂(i, j, k, grid, u, v, w)
@inline Σ̄Σ̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₃₃(i, j, k, grid, u, v, w)

@inline Σ̄Σ̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂, u, v, w)
@inline Σ̄Σ̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃, u, v, w)
@inline Σ̄Σ̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃, u, v, w)

const β = 1

# Square of the ratio α of the test-filter width to the grid-filter width Δᶠ. The three-point filter
# doubles the filter width along the directions it is applied in, so that α = 2^(n/d) where it is
# applied along n of the d directions that are not `Flat`. Away from boundaries n = d and α = 2.
@inline function ᾱ²(i, j, k, grid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    directions = (TX !== Flat) + (TY !== Flat) + (TZ !== Flat)

    filtered_x = test_filteredᶜ(i, TX, grid.Nx)
    filtered_y = test_filteredᶜ(j, TY, grid.Ny)
    filtered_z = test_filteredᶜ(k, TZ, grid.Nz)

    # Each filtered direction contributes a factor 4^(1/d), which is known at compile time
    directional_ᾱ² = convert(FT, 4)^(one(grid) / directions)

    partial_ᾱ² = ifelse(filtered_x, directional_ᾱ², one(grid)) *
                 ifelse(filtered_y, directional_ᾱ², one(grid)) *
                 ifelse(filtered_z, directional_ᾱ², one(grid))

    return ifelse(filtered_x + filtered_y + filtered_z == directions, convert(FT, 4), partial_ᾱ²)
end

@inline Δᶠ(i, j, k, grid) = ∛volume(i, j, k, grid, Center(), Center(), Center())
@inline Δᶠ⁻¹(i, j, k, grid) = ∛V⁻¹(i, j, k, grid, Center(), Center(), Center())
@inline M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ²(i, j, k, grid) * β * Σ̄Σ̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ²(i, j, k, grid) * β * Σ̄Σ̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ²(i, j, k, grid) * β * Σ̄Σ̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))

@inline M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ²(i, j, k, grid) * β * Σ̄Σ̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ²(i, j, k, grid) * β * Σ̄Σ̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ²(i, j, k, grid) * β * Σ̄Σ̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))

@inline uᵢ²(i, j, k, grid, uᵢ) = @inbounds uᵢ[i, j, k]^2
@inline u₁u₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, uᵢ², u)
@inline u₂u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, uᵢ², v)
@inline u₃u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑzᵃᵃᶜ(i, j, k, grid, uᵢ², w)

@inline ū²(i, j, k, grid, u) = filterᶠᶜᶜ(i, j, k, grid, u)^2
@inline v̄²(i, j, k, grid, v) = filterᶜᶠᶜ(i, j, k, grid, v)^2
@inline w̄²(i, j, k, grid, w) = filterᶜᶜᶠ(i, j, k, grid, w)^2

@inline ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ū², u)
@inline ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, v̄², v)
@inline ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑzᵃᵃᶜ(i, j, k, grid, w̄², w)

@inline u₁u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline u₁u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑzᵃᵃᶜ(i, j, k, grid, w)
@inline u₂u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, filterᶠᶜᶜ, u) * ℑyᵃᶜᵃ(i, j, k, grid, filterᶜᶠᶜ, v)
@inline ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, filterᶠᶜᶜ, u) * ℑzᵃᵃᶜ(i, j, k, grid, filterᶜᶜᶠ, w)
@inline ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, filterᶜᶠᶜ, v) * ℑzᵃᵃᶜ(i, j, k, grid, filterᶜᶜᶠ, w)

@inline L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = filterᶜᶜᶜ(i, j, k, grid, u₁u₁ᶜᶜᶜ, u, v, w) - ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = filterᶜᶜᶜ(i, j, k, grid, u₂u₂ᶜᶜᶜ, u, v, w) - ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = filterᶜᶜᶜ(i, j, k, grid, u₃u₃ᶜᶜᶜ, u, v, w) - ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

@inline L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = filterᶜᶜᶜ(i, j, k, grid, u₁u₂ᶜᶜᶜ, u, v, w) - ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = filterᶜᶜᶜ(i, j, k, grid, u₁u₃ᶜᶜᶜ, u, v, w) - ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = filterᶜᶜᶜ(i, j, k, grid, u₂u₃ᶜᶜᶜ, u, v, w) - ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
