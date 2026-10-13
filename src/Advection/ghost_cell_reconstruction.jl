using Oceananigans.Grids: topology
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Δx, Δy, Δz

"""
    GhostCells([FT=Oceananigans.defaults.FloatType])

Boundary scheme of a `WENO` reconstruction of order 5 or higher, `WENO(order = 7, boundary_scheme = GhostCells())`.
The inactive nodes of a stencil are filled by linear extrapolation of the active ones, which keeps the reconstruction
exact for linear data next to boundaries, and the reconstructed values next to boundaries are limited.
"""
struct GhostCells{S}
    scheme :: S
end

GhostCells(FT::DataType = Oceananigans.defaults.FloatType) = GhostCells(nothing)

Base.summary(::GhostCells) = "GhostCells()"

const GhostCellWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:GhostCells}

Adapt.adapt_structure(to, scheme::GhostCells) = GhostCells(Adapt.adapt(to, scheme.scheme))

Oceananigans.Architectures.on_architecture(to, scheme::GhostCells) = GhostCells(on_architecture(to, scheme.scheme))

@inline active_node(i, j, k, grid, ℓx, ℓy, ℓz) = !inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

# Distance between the node (i, j, k) and the previous node along the direction
@inline node_spacing(i, j, k, grid, ::Val{1}, ::Center, ℓy, ℓz) = Δx(i, j, k, grid, Face(), ℓy, ℓz)
@inline node_spacing(i, j, k, grid, ::Val{1}, ::Face,   ℓy, ℓz) = Δx(i - 1, j, k, grid, Center(), ℓy, ℓz)
@inline node_spacing(i, j, k, grid, ::Val{2}, ℓx, ::Center, ℓz) = Δy(i, j, k, grid, ℓx, Face(), ℓz)
@inline node_spacing(i, j, k, grid, ::Val{2}, ℓx, ::Face,   ℓz) = Δy(i, j - 1, k, grid, ℓx, Center(), ℓz)
@inline node_spacing(i, j, k, grid, ::Val{3}, ℓx, ℓy, ::Center) = Δz(i, j, k, grid, ℓx, ℓy, Face())
@inline node_spacing(i, j, k, grid, ::Val{3}, ℓx, ℓy, ::Face)   = Δz(i, j, k - 1, grid, ℓx, ℓy, Center())

@inline function step_outward(nodes, next_node)
    outer_value, outer_position, inner_value, inner_position = nodes
    next_value, next_position, next_is_active = next_node
    stepped_nodes = (next_value, next_position, outer_value, outer_position)
    return ifelse(next_is_active, stepped_nodes, nodes)
end

@inline function boundary_line(values, positions, active, side, extrapolable, ::Val{N}) where N
    upwind_nodes = @inbounds (values[N], positions[N], values[N - side], positions[N - side])

    side_nodes = ntuple(Val(N - 2)) do n
        @inline @inbounds values[N + n * side], positions[N + n * side], active[n]
    end

    outer_value, outer_position, inner_value, inner_position = foldl(step_outward, side_nodes; init = upwind_nodes)
    slope = ifelse(extrapolable, (outer_value - inner_value) / (outer_position - inner_position), zero(outer_value))

    return outer_value, outer_position, slope
end

@inline function ghost_cell_reconstruction(weno::WENO{N}, values, positions, active_below, active_above, ghosted_face, δˢ) where N
    extrapolable = @inbounds active_below[1] | active_above[1]

    below_value, below_position, below_slope = boundary_line(values, positions, active_below, -1, extrapolable, Val(N))
    above_value, above_position, above_slope = boundary_line(values, positions, active_above, +1, extrapolable, Val(N))

    below = ntuple(Val(N - 1)) do n
        @inline
        node = N - n
        ghost = @inbounds below_value + (positions[node] - below_position) * below_slope
        @inbounds ifelse(active_below[n], values[node], ghost)
    end

    above = ntuple(Val(N - 1)) do n
        @inline
        node = N + n
        ghost = @inbounds above_value + (positions[node] - above_position) * above_slope
        @inbounds ifelse(active_above[n], values[node], ghost)
    end

    completed_stencil = (reverse(below)..., @inbounds(values[N]), above...)

    δ = weno_differences(weno, completed_stencil)
    ω = biased_weno_weights(smoothness(ghosted_face, δ, δˢ), nothing, weno)

    return weno_reconstruction(weno, weno_anchor(weno, completed_stencil), δ, ω)
end

@inline smoothness(ghosted_face, δ,  ::Nothing) = δ
@inline smoothness(ghosted_face, δ, δˢ::NTuple) = ifelse(ghosted_face, δ, δˢ)
@inline smoothness(ghosted_face, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(ghosted_face, (δ, δ), δˢ)

# Neighbor of the stencil node n on the given side, or the boundary value between the two if the neighbor is inactive,
# extrapolated from node n and the three nodes behind it
@inline function neighbor_or_boundary_value(values, positions, active, n, side)
    n₁ = n - side
    n₂ = n - 2side
    n₃ = n - 3side

    ψ₀ = @inbounds values[n]
    ψ₁ = @inbounds values[n₁]
    ψ₂ = @inbounds values[n₂]
    ψ₃ = @inbounds values[n₃]

    x₀ = @inbounds positions[n]
    x₁ = @inbounds positions[n₁]
    x₂ = @inbounds positions[n₂]
    x₃ = @inbounds positions[n₃]

    # differences over the spacing between nodes n and n₁
    δ₁ = ψ₀ - ψ₁
    δ₂ = (ψ₁ - ψ₂) * (x₀ - x₁) / (x₁ - x₂)
    δ₃ = (ψ₂ - ψ₃) * (x₀ - x₁) / (x₂ - x₃)

    same_sign = (sign(δ₁) == sign(δ₂)) & (sign(δ₂) == sign(δ₃))
    minmod_difference = ifelse(same_sign, sign(δ₁) * min(abs(δ₁), abs(δ₂), abs(δ₃)), zero(δ₁))

    # Ratio of the harmonic to the arithmetic mean of the squared differences, scaled by the largest difference
    largest_difference = max(abs(δ₁), abs(δ₂), abs(δ₃))
    scale = ifelse(largest_difference > floatmin(largest_difference), 1 / largest_difference, zero(largest_difference))
    a = (δ₁ * scale)^2
    b = (δ₂ * scale)^2
    c = (δ₃ * scale)^2
    denominator = (a + b + c) * (a * b + b * c + c * a)
    slope_agreement = 9 * a * b * c / ifelse(denominator > 0, denominator, one(denominator))

    extrapolable   = @inbounds active[n₁] & active[n₂] & active[n₃]
    boundary_value = ψ₀ + ifelse(extrapolable, slope_agreement * minmod_difference / 2, zero(ψ₀))

    return @inbounds ifelse(active[n + side], values[n + side], boundary_value)
end

# The two neighbors of the node along a direction, each replaced by the node value if it is inactive
@inline neighbors_or_node(i, j, k, ψ₀, ::Val, ::Type{Flat}, grid, ℓx, ℓy, ℓz, ψ, args...) = ψ₀, ψ₀, false

@inline function neighbors_or_node(i, j, k, ψ₀, ::Val{D}, topology, grid, ℓx, ℓy, ℓz, ψ, args...) where D
    di, dj, dk = unit_vector(Val(D))

    forward_is_active  = active_node(i + di, j + dj, k + dk, grid, ℓx, ℓy, ℓz)
    backward_is_active = active_node(i - di, j - dj, k - dk, grid, ℓx, ℓy, ℓz)

    forward_value  = ifelse(forward_is_active,  stencil_value(i + di, j + dj, k + dk, grid, ψ, args...), ψ₀)
    backward_value = ifelse(backward_is_active, stencil_value(i - di, j - dj, k - dk, grid, ψ, args...), ψ₀)

    return forward_value, backward_value, !forward_is_active | !backward_is_active
end

# Extrema of the node and its active neighbors, the same whichever face asks for them
@inline function local_extrema(i, j, k, grid, ℓx, ℓy, ℓz, ψ, args...)
    ψ₀ = stencil_value(i, j, k, grid, ψ, args...)

    ψᵢ₊₁, ψᵢ₋₁, inactive_x_neighbor = neighbors_or_node(i, j, k, ψ₀, Val(1), topology(grid, 1), grid, ℓx, ℓy, ℓz, ψ, args...)
    ψⱼ₊₁, ψⱼ₋₁, inactive_y_neighbor = neighbors_or_node(i, j, k, ψ₀, Val(2), topology(grid, 2), grid, ℓx, ℓy, ℓz, ψ, args...)
    ψₖ₊₁, ψₖ₋₁, inactive_z_neighbor = neighbors_or_node(i, j, k, ψ₀, Val(3), topology(grid, 3), grid, ℓx, ℓy, ℓz, ψ, args...)

    ψ⁻ = min(ψ₀, ψᵢ₊₁, ψᵢ₋₁, ψⱼ₊₁, ψⱼ₋₁, ψₖ₊₁, ψₖ₋₁)
    ψ⁺ = max(ψ₀, ψᵢ₊₁, ψᵢ₋₁, ψⱼ₊₁, ψⱼ₋₁, ψₖ₊₁, ψₖ₋₁)
    near_boundary = inactive_x_neighbor | inactive_y_neighbor | inactive_z_neighbor

    return ψ⁻, ψ⁺, near_boundary
end

@inline function ghost_cell_interpolate(i, j, k, grid, scheme::GhostCells{<:WENO{N}}, bias, d, ℓx, ℓy, ℓz, δˢ, ψ, args...) where N
    weno = scheme.scheme
    e = unit_vector(d)

    values      = upwind_stencil(i, j, k, grid, weno, bias, e, ψ, args...)
    active_cell = upwind_stencil(i, j, k, grid, weno, bias, e, active_node, ℓx, ℓy, ℓz)

    # distances along the grid line, increasing with the index like the nodes of a LeftBias stencil
    distances = accumulate(+, upwind_stencil(i, j, k, grid, weno, LeftBias, e, node_spacing, d, ℓx, ℓy, ℓz))
    positions = ifelse(bias === LeftBias, distances, reverse(distances))

    active_cell_below = ntuple(Val(N - 1)) do n
        @inline
        @inbounds active_cell[N - n]
    end

    active_cell_above = ntuple(Val(N - 1)) do n
        @inline
        @inbounds active_cell[N + n]
    end

    active_below = accumulate(&, active_cell_below)
    active_above = accumulate(&, active_cell_above)
    ghosted_face = @inbounds !(active_below[N - 1] & active_above[N - 1])

    ψ̂ = ghost_cell_reconstruction(weno, values, positions, active_below, active_above, ghosted_face, δˢ)

    di, dj, dk = e
    left_extrema  = local_extrema(i - di, j - dj, k - dk, grid, ℓx, ℓy, ℓz, ψ, args...)
    right_extrema = local_extrema(i, j, k, grid, ℓx, ℓy, ℓz, ψ, args...)

    ψᵁ⁻, ψᵁ⁺, upwind_near_boundary   = ifelse(bias === LeftBias, left_extrema, right_extrema)
    ψᴰ⁻, ψᴰ⁺, downwind_near_boundary = ifelse(bias === LeftBias, right_extrema, left_extrema)

    # the extrema also contain the nodes beyond U and D along the line, or the boundary values if they are inactive
    ψᵁᵁ = neighbor_or_boundary_value(values, positions, active_cell, N, -1)
    ψᴰᴰ = neighbor_or_boundary_value(values, positions, active_cell, N + 1, +1)

    ψᵁ⁻ = min(ψᵁ⁻, ψᵁᵁ)
    ψᵁ⁺ = max(ψᵁ⁺, ψᵁᵁ)
    ψᴰ⁻ = min(ψᴰ⁻, ψᴰᴰ)
    ψᴰ⁺ = max(ψᴰ⁺, ψᴰᴰ)

    ψᵁ = @inbounds values[N]
    α  = convert(typeof(ψ̂), 3//2)
    ψ̂⁻ = max(ψᵁ - α * (ψᵁ⁺ - ψᵁ),  ψᴰ⁻)
    ψ̂⁺ = min(ψᵁ + α * (ψᵁ  - ψᵁ⁻), ψᴰ⁺)

    limited_face = ghosted_face | upwind_near_boundary | downwind_near_boundary

    return ifelse(limited_face, clamp(ψ̂, ψ̂⁻, ψ̂⁺), ψ̂)
end

for (dir, ξ) in enumerate((:x, :y, :z))
    e = unit_vector(Val(dir))
    d = Val(dir)
    ℓᶠ = ntuple(3) do n
        n == dir ? :(Face()) : :(Center())
    end
    ℓᶜ = (:(Center()), :(Center()), :(Center()))
    interpolateᶠ = Symbol(:biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶠ : :ᵃ, 3)...)
    interpolateᶜ = Symbol(:biased_interpolate_, ξ, ntuple(d -> d == dir ? :ᶜ : :ᵃ, 3)...)

    @eval begin
        @inline $interpolateᶠ(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = ghost_cell_interpolate(i, j, k,              grid, scheme, bias, $d, $(ℓᶜ...), nothing, ψ, args...)
        @inline $interpolateᶜ(i, j, k, grid, scheme::GhostCells, bias, ψ, args...) = ghost_cell_interpolate(((i, j, k) .+ $e)..., grid, scheme, bias, $d, $(ℓᶠ...), nothing, ψ, args...)

        @inline function $interpolateᶠ(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...)
            δˢ = smoothness_differences(i, j, k, grid, scheme.scheme, bias, $e, stencil, args...)
            return ghost_cell_interpolate(i, j, k, grid, scheme, bias, $d, $(ℓᶜ...), δˢ, ψ, args...)
        end

        @inline function $interpolateᶜ(i, j, k, grid, scheme::GhostCells, bias, ψ, stencil::AbstractSmoothnessStencil, args...)
            δˢ = smoothness_differences(((i, j, k) .+ $e)..., grid, scheme.scheme, bias, $e, stencil, args...)
            return ghost_cell_interpolate(((i, j, k) .+ $e)..., grid, scheme, bias, $d, $(ℓᶠ...), δˢ, ψ, args...)
        end
    end

    for location in (:ᶜ, :ᶠ)
        symmetric = Symbol(:symmetric_interpolate_, ξ, ntuple(d -> d == dir ? location : :ᵃ, 3)...)
        @eval @inline $symmetric(i, j, k, grid, ::GhostCells{<:WENO{<:Any, FT}}, args...) where FT = $symmetric(i, j, k, grid, Centered{1, FT}(nothing, ExplicitTimeDiscretization()), args...)
    end
end
