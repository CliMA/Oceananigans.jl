using Oceananigans.Grids: ξnodes, ηnodes, rnodes
using Oceananigans.ImmersedBoundaries: underlying_grid

#####
##### WENO on stretched directions: reconstruction coefficients, smoothness indicators and optimal weights computed
##### exactly from the cell widths and tabulated along each direction
#####

# The weights of the averages over the cells between `edges` that give the value at x★ of the polynomial with those averages
function cell_average_weights(edges, x★)
    n = length(edges) - 1
    averages = [(edges[j+1]^(q+1) - edges[j]^(q+1)) / ((q + 1) * (edges[j+1] - edges[j])) for q in 0:n-1, j in 1:n]
    return averages \ [x★^q for q in 0:n-1]
end

# The coefficients of the 2N - 1 cells between the 2N `edges`, reconstructing at the right edge of the anchor cell N
function biased_coefficients(FT, N, edges)
    R = Rational{BigInt}
    edges = R.(edges)
    ξ = (edges .- (edges[N] + edges[N+1]) / 2) ./ (edges[N+1] - edges[N])
    candidate(s) = ξ[N-s:2N-s]
    weights = [cell_average_weights(candidate(s), R(1, 2)) for s in 0:N-1]

    # β is invariant to a constant shift of the stencil, so it is a quadratic form in the first differences
    T = [R(i < m) for m in 1:N, i in 1:N-1]
    normalization = reference_smoothness_normalization[N - 1]
    β = Tuple(FT.(ldlt_smoothness_coefficients(T' * smoothness_matrix(candidate(s)) * T * normalization)) for s in 0:N-1)
    p = Tuple(FT.(Tuple(difference_coefficients(weights[s+1], N, s))) for s in 0:N-1)

    # The optimal weights combine the candidates into the reconstruction from the whole stencil
    A = zeros(R, 2N - 1, N)
    for s in 0:N-1, m in 1:N
        A[N - s + m - 1, s + 1] = weights[s + 1][m]
    end
    C★ = (A' * A) \ (A' * cell_average_weights(ξ, R(1, 2)))

    return (; p, β, C★ = Tuple(FT.(C★)))
end

# The coefficients at every point of `nodes`, the faces or the centers of a direction with its halos, from the cells
# between consecutive nodes. Stencils that leave the halos reuse the nearest complete one.
function coefficient_table(FT, N, nodes, arch)
    entries = map(firstindex(nodes):lastindex(nodes)) do p
        p = clamp(p, firstindex(nodes) + N, lastindex(nodes) - N)
        left = biased_coefficients(FT, N, [nodes[q] for q in p-N:p+N-1])
        right = biased_coefficients(FT, N, [-nodes[q] for q in p+N:-1:p-N+1]) # mirrored
        (; left, right)
    end
    return OffsetArray(on_architecture(arch, entries), firstindex(nodes) - 1)
end

materialize_coefficients(coefficients, scheme, grid) = coefficients

function materialize_coefficients(directions::Tuple, ::WENO{N, FT}, grid) where {N, FT}
    arch = architecture(grid)
    cpu_grid = on_architecture(CPU(), underlying_grid(grid))

    # Flat and uniform directions, and the curvilinear horizontal ones (without ξnodes and ηnodes), keep the constants
    function direction_coefficients(direction, nodes)
        faces = direction ∈ directions && applicable(nodes, cpu_grid, Face()) ? nodes(cpu_grid, Face(); with_halos = true) : nothing
        isnothing(faces) && return nothing
        Δ = diff(collect(faces))
        all(≈(first(Δ)), Δ) && return nothing
        centers = nodes(cpu_grid, Center(); with_halos = true)
        return (faces = coefficient_table(FT, N, faces, arch), centers = coefficient_table(FT, N, centers, arch))
    end

    return NamedTuple{(:x, :y, :z)}(map(direction_coefficients, (:x, :y, :z), (ξnodes, ηnodes, rnodes)))
end

#####
##### Reconstruction with tabulated coefficients
#####

@inline weno_coefficients(c::NamedTuple, ::Val{d}, location, n, bias) where d = weno_coefficients(getfield(c, d), location, n, bias)

@inline function weno_coefficients(c::NamedTuple, ::Val{location}, n, bias) where location
    entry = @inbounds getfield(c, location)[n]
    return ifelse(bias == LeftBias, entry.left, entry.right)
end

@inline coeff_p(scheme, ::Val{s}, C) where s = @inbounds C.p[s+1]
@inline      C★(scheme, ::Val{s}, C) where s = @inbounds C.C★[s+1]
@inline smoothness_indicator(δ, scheme, ::Val{s}, C) where s = smoothness_operation(scheme, δ, @inbounds C.β[s+1])
