using Oceananigans.Grids: topology
using Oceananigans.ImmersedBoundaries: inactive_node
using Oceananigans.Operators: Δx, Δy, Δz
using Oceananigans.Utils: newton_div

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

# Selects instead of `min` and `max`, which order signed zeros and propagate NaN at the cost of several GPU instructions
@inline smallest(a, b) = ifelse(a < b, a, b)
@inline  largest(a, b) = ifelse(a > b, a, b)
@inline smallest(a, b, c...) = smallest(smallest(a, b), c...)
@inline  largest(a, b, c...) = largest(largest(a, b),   c...)
@inline ordered(a, b) = ifelse(a < b, (a, b), (b, a))

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

# The last two active nodes of the run that leaves the stencil node n on the given side
@inline function extrapolation_nodes(values, positions, run, n, side, ::Val{N}) where N
    first_nodes = @inbounds (values[n], positions[n], values[n - side], positions[n - side])

    side_nodes = ntuple(Val(N - 2)) do m
        @inline @inbounds values[n + m * side], positions[n + m * side], run[m]
    end

    return foldl(step_outward, side_nodes; init = first_nodes)
end

# WENO reconstruction of the face between the stencil nodes N and N + 1, given in index order,
# after the inactive nodes of each side are moved onto the line through the last two active nodes of that side
@inline @muladd function ghost_cell_reconstruction(weno::WENO{N, FT, WCT}, values, positions, active, bias, δˢ) where {N, FT, WCT}
    left_run = accumulate(&, ntuple(Val(N - 1)) do m
        @inline
        @inbounds active[N - m]
    end)

    right_run = accumulate(&, ntuple(Val(N - 1)) do m
        @inline
        @inbounds active[N + 1 + m]
    end)

    ψᴸ₀, xᴸ₀, ψᴸ₁, xᴸ₁ = extrapolation_nodes(values, positions, left_run,  N,     -1, Val(N))
    ψᴿ₀, xᴿ₀, ψᴿ₁, xᴿ₁ = extrapolation_nodes(values, positions, right_run, N + 1, +1, Val(N))

    # one division for the slopes of both lines
    inverse_spacings = newton_div(WCT, one(xᴸ₀), (xᴸ₀ - xᴸ₁) * (xᴿ₀ - xᴿ₁))
    left_slope  = (ψᴸ₀ - ψᴸ₁) * (xᴿ₀ - xᴿ₁) * inverse_spacings
    right_slope = (ψᴿ₀ - ψᴿ₁) * (xᴸ₀ - xᴸ₁) * inverse_spacings

    left = ntuple(Val(N - 1)) do m
        @inline
        node  = N - m
        ghost = @inbounds ψᴸ₀ + (positions[node] - xᴸ₀) * left_slope
        @inbounds ifelse(left_run[m], values[node], ghost)
    end

    right = ntuple(Val(N - 1)) do m
        @inline
        node  = N + 1 + m
        ghost = @inbounds ψᴿ₀ + (positions[node] - xᴿ₀) * right_slope
        @inbounds ifelse(right_run[m], values[node], ghost)
    end

    completed_stencil = (reverse(left)..., @inbounds(values[N]), @inbounds(values[N + 1]), right...)

    # the upwind stencil is made of the first 2N - 1 nodes for a LeftBias, of the last 2N - 1 reversed for a RightBias
    left_is_upwind = bias === LeftBias
    upwind_values = ntuple(Val(2N - 1)) do n
        @inline
        @inbounds ifelse(left_is_upwind, completed_stencil[n], completed_stencil[2N + 1 - n])
    end

    # nodes 1 to 2N - 1 form the stencil of a LeftBias reconstruction, nodes 2 to 2N that of a RightBias one
    left_bias_stencil_is_active  = @inbounds left_run[N - 1] & right_run[N - 2]
    right_bias_stencil_is_active = @inbounds left_run[N - 2] & right_run[N - 1]
    ghosted_face = !ifelse(left_is_upwind, left_bias_stencil_is_active, right_bias_stencil_is_active)

    δ = weno_differences(weno, upwind_values)
    ω = biased_weno_weights(smoothness(ghosted_face, δ, δˢ), nothing, weno)
    ψ̂ = weno_reconstruction(weno, weno_anchor(weno, upwind_values), δ, ω)

    return ψ̂, ghosted_face
end

@inline smoothness(ghosted_face, δ,  ::Nothing) = δ
@inline smoothness(ghosted_face, δ, δˢ::NTuple) = ifelse(ghosted_face, δ, δˢ)
@inline smoothness(ghosted_face, δ, δˢ::Tuple{Tuple, Tuple}) = ifelse(ghosted_face, (δ, δ), δˢ)

# Value at the boundary between the stencil nodes n and n - 1, extrapolated from node n and the three nodes after it
@inline @muladd function boundary_value(values, positions, active, n, WCT)
    ψ₀ = @inbounds values[n]
    ψ₁ = @inbounds values[n + 1]
    ψ₂ = @inbounds values[n + 2]
    ψ₃ = @inbounds values[n + 3]

    h₀ = @inbounds positions[n] - positions[n + 1]
    h₁ = @inbounds positions[n + 1] - positions[n + 2]
    h₂ = @inbounds positions[n + 2] - positions[n + 3]

    # differences over the spacing between nodes n and n + 1, multiplied by h₁ h₂ so that no spacing divides them
    δ₁ = (ψ₀ - ψ₁) * h₁ * h₂
    δ₂ = (ψ₁ - ψ₂) * h₀ * h₂
    δ₃ = (ψ₂ - ψ₃) * h₀ * h₁

    increasing = (δ₁ > 0) & (δ₂ > 0) & (δ₃ > 0)
    decreasing = (δ₁ < 0) & (δ₂ < 0) & (δ₃ < 0)
    smallest_difference = smallest(abs(δ₁), abs(δ₂), abs(δ₃))
    minmod_difference = ifelse(increasing, smallest_difference, ifelse(decreasing, -smallest_difference, zero(δ₁)))

    # Ratio of the harmonic to the arithmetic mean of the squared differences, scaled by the largest difference
    largest_difference = largest(abs(δ₁), abs(δ₂), abs(δ₃))
    scale = ifelse(largest_difference > floatmin(largest_difference), newton_div(WCT, one(largest_difference), largest_difference), zero(largest_difference))
    a = (δ₁ * scale)^2
    b = (δ₂ * scale)^2
    c = (δ₃ * scale)^2
    denominator = (a + b + c) * (a * b + b * c + c * a)

    # the agreement times half the minmod difference, divided back by h₁ h₂
    extension = newton_div(WCT, 9 * a * b * c * minmod_difference, 2 * h₁ * h₂ * ifelse(denominator > 0, denominator, one(denominator)))
    extrapolable = @inbounds active[n + 1] & active[n + 2] & active[n + 3]

    return ψ₀ + ifelse(extrapolable, extension, zero(ψ₀))
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

    ψˣ⁻, ψˣ⁺ = ordered(ψᵢ₊₁, ψᵢ₋₁)
    ψʸ⁻, ψʸ⁺ = ordered(ψⱼ₊₁, ψⱼ₋₁)
    ψᶻ⁻, ψᶻ⁺ = ordered(ψₖ₊₁, ψₖ₋₁)

    ψ⁻ = smallest(ψ₀, ψˣ⁻, ψʸ⁻, ψᶻ⁻)
    ψ⁺ =  largest(ψ₀, ψˣ⁺, ψʸ⁺, ψᶻ⁺)
    near_boundary = inactive_x_neighbor | inactive_y_neighbor | inactive_z_neighbor

    return ψ⁻, ψ⁺, near_boundary
end

@inline @muladd function ghost_cell_interpolate(i, j, k, grid, scheme::GhostCells{<:WENO{N, FT, WCT}}, bias, d, ℓx, ℓy, ℓz, δˢ, ψ, args...) where {N, FT, WCT}
    weno = scheme.scheme
    e = unit_vector(d)

    # The stencil in index order: node N is the cell L on the left of the face, node N + 1 the cell R on its right
    values    = upwind_stencil(i, j, k, grid, weno, LeftBias, e, ψ, args...)
    active    = upwind_stencil(i, j, k, grid, weno, LeftBias, e, active_node, ℓx, ℓy, ℓz)
    positions = accumulate(+, upwind_stencil(i, j, k, grid, weno, LeftBias, e, node_spacing, d, ℓx, ℓy, ℓz))

    ψ̂, ghosted_face = ghost_cell_reconstruction(weno, values, positions, active, bias, δˢ)

    di, dj, dk = e
    ψᴸ⁻, ψᴸ⁺, left_near_boundary  = local_extrema(i - di, j - dj, k - dk, grid, ℓx, ℓy, ℓz, ψ, args...)
    ψᴿ⁻, ψᴿ⁺, right_near_boundary = local_extrema(i, j, k, grid, ℓx, ℓy, ℓz, ψ, args...)

    # One boundary value serves the face: beyond L it needs node N + 2 active, beyond R it needs it inactive
    beyond_left = @inbounds active[N + 2]
    oriented_values    = ifelse(beyond_left, values,    reverse(values))
    oriented_positions = ifelse(beyond_left, positions, reverse(positions))
    oriented_activity  = ifelse(beyond_left, active,    reverse(active))
    ψᵇ = boundary_value(oriented_values, oriented_positions, oriented_activity, N, WCT)

    # the extrema also contain the nodes beyond L and R along the line, or the boundary value if they are inactive
    ψᴸᴸ = @inbounds ifelse(active[N - 1], values[N - 1], ifelse(beyond_left, ψᵇ, values[N]))
    ψᴿᴿ = @inbounds ifelse(active[N + 2], values[N + 2], ψᵇ)

    ψᴸ⁻ = smallest(ψᴸ⁻, ψᴸᴸ)
    ψᴸ⁺ = largest(ψᴸ⁺, ψᴸᴸ)
    ψᴿ⁻ = smallest(ψᴿ⁻, ψᴿᴿ)
    ψᴿ⁺ = largest(ψᴿ⁺, ψᴿᴿ)

    left_is_upwind = bias === LeftBias
    ψᵁ  = @inbounds ifelse(left_is_upwind, values[N], values[N + 1])
    ψᵁ⁻ = ifelse(left_is_upwind, ψᴸ⁻, ψᴿ⁻)
    ψᵁ⁺ = ifelse(left_is_upwind, ψᴸ⁺, ψᴿ⁺)
    ψᴰ⁻ = ifelse(left_is_upwind, ψᴿ⁻, ψᴸ⁻)
    ψᴰ⁺ = ifelse(left_is_upwind, ψᴿ⁺, ψᴸ⁺)

    α  = convert(typeof(ψ̂), 3//2)
    ψ̂⁻ =  largest(ψᵁ - α * (ψᵁ⁺ - ψᵁ),  ψᴰ⁻)
    ψ̂⁺ = smallest(ψᵁ + α * (ψᵁ  - ψᵁ⁻), ψᴰ⁺)

    limited_face = ghosted_face | left_near_boundary | right_near_boundary

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
