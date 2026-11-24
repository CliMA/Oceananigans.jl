using Oceananigans.Grids: rnode, topology
using Oceananigans.Operators: Vᶜᶜᶜ, Δrᶜᶜᶜ, Δx⁻¹ᶜᶜᶜ, Δy⁻¹ᶜᶜᶜ, Δr⁻¹ᶜᶜᶜ

#####
##### First moments Mᵈ = ⟨ξᵈ c⟩ over the wet part of each cell, ξ ∈ [-1/2, 1/2]³ across its box (across its level for a cut cell). A face
##### carries the mean and moment mᵘ of its upwind cell's profile over its open part:
#####   V ∂ₜMᵈ = -[(F - c̄ q)⁺ + (F - c̄ q)⁻] / 2 + V (u⁺ - u⁻) / Δᵈ (Mᵈ - c̄ μᵈ) - Σₑ [q (m - c̄ μᶠᵈ)]⁺₋,   m = cᵘ μᶠᵈ + θ (mᵘ - cᵘ μᶠᵈ)
##### The profile of a cell, integrated at the centres of S² columns of the cell (S strips of a face), exact along each:
#####   switch  between two water masses where the 3³ range of the means exceeds 3/4 of their 5³ range, with the values below the bed
#####           mirrored across it, in an active cell with explicit vertical transport
#####   bounds  c⁻, c⁺, the extrema of the active 3³ means
#####   tanh    between two water masses, c⁻ + Δc H(β n(θ, φ) ⋅ ξ - γ), H(z) = (1 + tanh z) / 2, γ exact for the mean, (θ, φ, log β) by
#####           Gauss-Newton on the moments from the cell's last fit, log β ∈ [log 10⁻³, log 200], the step scaled by the first of 1, 1/2,
#####           1/4, 1/8 that lowers the residual; the cell keeps its moments
#####   reset   any other cell takes the moments of the linear profile with the gradient of the degree-6 polynomial through 7 means
#####   ramp    and carries the least-variance profile clamp(λ ⋅ (1, ξ), c⁻, c⁺) with its mean and moments: its linear profile, or Newton
#####           steps on the dual where that leaves the bounds
##### profiles = (c⁻, c⁺, w₁, w₂, w₃, w₄, state): w = (γ, q), q = β n(θ, φ), between two water masses, the ramp's λ otherwise
#####

"""
$(TYPEDSIGNATURES)

Names of the tracers that a model adds for a tracer `name` advected with `scheme`: its first moments, `name` with superscripts x, y
and z, for a `THINCBVD` scheme with `moments=true`. With a tuple of `names` and the tracer advection of a model, a scheme or a
`NamedTuple` of schemes, the tracers added for all `names`.
"""
advection_required_tracers(scheme, name::Symbol) = ()
advection_required_tracers(scheme::THINCBVD, name::Symbol) = scheme.moments ? Symbol.(name, (:ˣ, :ʸ, :ᶻ)) : ()
advection_required_tracers(advection, names::Tuple) = Tuple(m for name in names for m in advection_required_tracers(advection, name))

function advection_required_tracers(advection::NamedTuple, names::Tuple)
    return Tuple(m for name in names for m in advection_required_tracers(get(advection, name, nothing), name))
end

struct FirstMoment{D, S}
    scheme :: S
end

FirstMoment{D}(scheme) where D = FirstMoment{D, typeof(scheme)}(scheme)
Adapt.adapt_structure(to, s::FirstMoment{D}) where D = FirstMoment{D}(Adapt.adapt(to, s.scheme))
Base.summary(s::FirstMoment{D}) where D = string("FirstMoment{", D, "} of ", summary(s.scheme))

"""
$(SIGNATURES)

Return `advection` with `FirstMoment{1}`, `FirstMoment{2}` and `FirstMoment{3}` of the scheme of every tracer `T` whose first moments
`Tˣ`, `Tʸ` and `Tᶻ` it advects, and, with `tracers`, with the scheme of `T` updating these moments in `tracers` and its profiles.
"""
function with_moment_advection(advection::NamedTuple, tracers=nothing)
    moment_advected = filter(name -> !isempty(advection_required_tracers(advection[name], name)), keys(advection))
    listing = isnothing(tracers) || isempty(moment_advected) ? nothing : moment_listing(tracers[first(moment_advected)].grid)
    entries = map(keys(advection)) do name
        s = advection[name]
        moment_names = advection_required_tracers(s, name)
        (isempty(moment_names) || !(moment_names ⊆ keys(advection))) && return NamedTuple()

        if !isnothing(tracers)
            r = s.reconstructions
            moments = map(moment -> tracers[moment], moment_names)
            grid = first(moments).grid
            profiles = (ntuple(_ -> CenterField(grid), 6)..., CenterField(grid, UInt8))
            list, reach, band = listing
            positions, face_integrals = moment_places(grid, reach)
            reconstructions = Reconstructions(r.ghost_filled, r.x, r.y, r.z, r.wet_column, r.antidiffusive_fractions, moments, profiles,
                                              (positions, list, reach, band, face_integrals), r.extrema)
            s = THINCBVD{required_halo_size_x(s), eltype(s), typeof(TimeSteppers.time_discretization(s))}(s.buffer_scheme, s.steepnesses,
                                                                                                          s.base_steepness, s.moments,
                                                                                                          reconstructions)
        end

        return merge(NamedTuple{(name,)}((s,)), NamedTuple{moment_names}((FirstMoment{1}(s), FirstMoment{2}(s), FirstMoment{3}(s))))
    end

    return merge(advection, entries...)
end

#####
##### Geometry: the wet part of a cell on a CutCellBottom grid lies above the plane z = rᵇ - 2aˣ ξ₁ - 2aʸ ξ₂ of its column
#####

const CutCellBottomGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:CutCellBottom}

# columns per horizontal direction of the geometry, the faces and the tanh fit, and of the overshoot test and the ramp fit
const column_subsamples = 16
const ramp_column_subsamples = 4

@inline subsamples(grid, d, S=column_subsamples) = ifelse(topology(grid, d) === Flat, 1, S)
@inline subsample_offset(n, S, FT) = (2n - 1) * one(FT) / 2S - one(FT) / 2
@inline tangential_directions(e) = ifelse(e == 1, (2, 3), ifelse(e == 2, (1, 3), (1, 2)))

# The face normal to e in strips across r, each open along t: x and y faces along z, z faces along x
@inline strip_directions(e) = ifelse(e == 1, (2, 3), ifelse(e == 2, (1, 3), (2, 1)))

# The part of s ∈ [-1/2, 1/2] where d + 2a s > 0; the reciprocal of 2a is shared by all the columns (strips) of a cell (face)
@inline function open_interval(d, a)
    h = one(d) / 2
    b = -d * inv(ifelse(a == 0, one(a), 2a))
    s⁻ = ifelse(a > 0, clamp(b, -h, h), -h)
    s⁺ = ifelse(a < 0, clamp(b, -h, h), ifelse((a == 0) & (d ≤ 0), -h, h))
    return s⁻, s⁺
end

@inline function bottom_plane(i, j, ibg)
    ib = ibg.immersed_boundary
    rᵇ = @inbounds ib.bottom_height[i, j, 1]
    return rᵇ, (@inbounds ib.west_bottom_height[i, j, 1]) - rᵇ, (@inbounds ib.south_bottom_height[i, j, 1]) - rᵇ
end

# The interval of ξ₃ in the wet part of cell i, j, k at ξ₁, ξ₂
@inline wet_interval(i, j, k, grid, ξ, ζ) = (-one(ξ) / 2, one(ξ) / 2)

@inline function wet_interval(i, j, k, ibg::CutCellBottomGrid, ξ, ζ)
    rᵇ, aˣ, aʸ = bottom_plane(i, j, ibg)
    rᶜ = rnode(i, j, k, ibg.underlying_grid, Center(), Center(), Center())
    return open_interval(rᶜ - rᵇ + 2aˣ * ξ + 2aʸ * ζ, Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid) / 2)
end

# The open interval along t of the strip at s across r of the face at i, j, k normal to e: x and y faces are open above the higher
# plane of their two columns, z faces where the plane of their column lies below them
@inline face_interval(i, j, k, grid, ::Val, s) = (-one(s) / 2, one(s) / 2)

@inline function face_interval(i, j, k, ibg::CutCellBottomGrid, ::Val{1}, ζ)
    rᵇ⁺, aˣ⁺, aʸ⁺ = bottom_plane(i, j, ibg)
    rᵇ⁻, aˣ⁻, aʸ⁻ = bottom_plane(i - 1, j, ibg)
    rᶜ = rnode(i, j, k, ibg.underlying_grid, Center(), Center(), Center())
    return open_interval(rᶜ - max(rᵇ⁺ + aˣ⁺ - 2aʸ⁺ * ζ, rᵇ⁻ - aˣ⁻ - 2aʸ⁻ * ζ), Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid) / 2)
end

@inline function face_interval(i, j, k, ibg::CutCellBottomGrid, ::Val{2}, ξ)
    rᵇ⁺, aˣ⁺, aʸ⁺ = bottom_plane(i, j, ibg)
    rᵇ⁻, aˣ⁻, aʸ⁻ = bottom_plane(i, j - 1, ibg)
    rᶜ = rnode(i, j, k, ibg.underlying_grid, Center(), Center(), Center())
    return open_interval(rᶜ - max(rᵇ⁺ - 2aˣ⁺ * ξ + aʸ⁺, rᵇ⁻ - 2aˣ⁻ * ξ - aʸ⁻), Δrᶜᶜᶜ(i, j, k, ibg.underlying_grid) / 2)
end

@inline function face_interval(i, j, k, ibg::CutCellBottomGrid, ::Val{3}, ζ)
    rᵇ, aˣ, aʸ = bottom_plane(i, j, ibg)
    return open_interval(rnode(i, j, k, ibg.underlying_grid, Center(), Center(), Face()) - rᵇ + 2aʸ * ζ, aˣ)
end

# η spans the level of a cut cell
@inline Δη⁻¹(i, j, k, grid) = Δr⁻¹ᶜᶜᶜ(i, j, k, grid)
@inline Δη⁻¹(i, j, k, ibg::CutCellBottomGrid) = Δr⁻¹ᶜᶜᶜ(i, j, k, ibg.underlying_grid)

# ⟨·⟩ over the wet part of cell i, j, k from f(ξ, ζ, η⁻, η⁺), the integrals along the column at ξ, ζ over its wet interval [η⁻, η⁺], at
# the centres of S² columns; zero without a wet part
@inline function wet_column_mean(f, i, j, k, grid, Σ, S=column_subsamples)
    FT = eltype(grid)
    Sˣ, Sʸ = subsamples(grid, 1, S), subsamples(grid, 2, S)
    ΣL = zero(FT)
    for m in 1:Sˣ, n in 1:Sʸ
        ξ, ζ = subsample_offset(m, Sˣ, FT), subsample_offset(n, Sʸ, FT)
        η⁻, η⁺ = wet_interval(i, j, k, grid, ξ, ζ)
        Σ = Σ .+ f(ξ, ζ, η⁻, η⁺)
        ΣL += η⁺ - η⁻
    end
    return Σ ./ ifelse(ΣL > 0, ΣL, one(FT))
end

# ⟨ξ⟩ and ⟨ξ ξᵀ⟩ of the box: (1 - 1/S²) / 12 over the column centres along x and y, 1/12 along z, zero in a Flat direction
@inline function box_moments(grid)
    FT = eltype(grid)
    σ = map(d -> (1 - one(FT) / subsamples(grid, d)^2) / 12, (1, 2))
    σᶻ = ifelse(topology(grid, 3) === Flat, zero(FT), one(FT) / 12)
    return (zero(FT), zero(FT), zero(FT)), (σ..., σᶻ, zero(FT), zero(FT), zero(FT))
end

@inline wet_part_moments(i, j, k, grid) = box_moments(grid)

# those of the box in a cell without a wet part
@inline function wet_part_moments(i, j, k, ibg::CutCellBottomGrid)
    FT = eltype(ibg)
    Σ = wet_column_mean(i, j, k, ibg, ntuple(Returns(zero(FT)), Val(10))) do ξ, ζ, η⁻, η⁺
        L, Lη, Lηη = η⁺ - η⁻, (η⁺^2 - η⁻^2) / 2, (η⁺^3 - η⁻^3) / 3
        (L, ξ * L, ζ * L, Lη, ξ^2 * L, ζ^2 * L, Lηη, ξ * ζ * L, ξ * Lη, ζ * Lη)
    end
    μᵇ, Cᵇ = box_moments(ibg)
    wet = Σ[1] > 0
    return ifelse(wet, (Σ[2], Σ[3], Σ[4]), μᵇ), ifelse(wet, (Σ[5], Σ[6], Σ[7], Σ[8], Σ[9], Σ[10]), Cᵇ)
end

# ⟨ξ⟩ over the open part of the face at i, j, k normal to e, at the centres of its S strips
@inline function strip_face_moments(i, j, k, grid, ::Val{e}) where e
    FT = eltype(grid)
    r, t = strip_directions(e)
    S = subsamples(grid, r)
    Σr, Σt, ΣL = zero(FT), zero(FT), zero(FT)
    for m in 1:S
        s = subsample_offset(m, S, FT)
        t⁻, t⁺ = face_interval(i, j, k, grid, Val(e), s)
        L = t⁺ - t⁻
        Σr += s * L
        Σt += L * (t⁻ + t⁺) / 2
        ΣL += L
    end
    W = ifelse(ΣL > 0, ΣL, one(FT))
    μʳ, μᵗ = Σr / W, Σt / W
    return ntuple(d -> ifelse(d == r, μʳ, ifelse(d == t, μᵗ, zero(FT))), Val(3))
end

#####
##### Stored once: the wet part's (μ, C) of each cell and the tangential μᶠ of its west, south and bottom faces, per column in the
##### band of levels where they vary, between one slot for the levels below it and one for the levels above it
#####

@inline stored_geometry(r) = r.switched_cells[4]

@inline function band_slot(i, j, k, band)
    i′, j′ = i + band.reach[1], j + band.reach[2]
    k⁻, levels, offset = @inbounds band.first_level[i′, j′], band.levels[i′, j′], band.offsets[i′, j′]
    return offset + 1 + clamp(k - k⁻ + 1, 0, levels + 1)
end

@inline function stored_wet_part_moments(i, j, k, band)
    n = band_slot(i, j, k, band)
    v = band.values
    return @inbounds (v[1][n], v[2][n], v[3][n]), (v[4][n], v[5][n], v[6][n], v[7][n], v[8][n], v[9][n])
end

@inline stored_face_moment(i, j, k, band, m) = @inbounds band.values[9 + m][band_slot(i, j, k, band)]

# the geometry of each column from its first level that differs from the bottom level to its last that differs from the top level,
# computed on slabs of `rows` rows of j
function geometry_band(grid, reach; rows=16)
    FT = eltype(grid)
    arch = architecture(grid)
    is, js, ks = map((N, r) -> 1 - r:N + r, size(grid), reach)
    first_level, levels, offsets = (zeros(Int32, length(is), length(js)) for _ in 1:3)
    values = ntuple(_ -> FT[], 15)
    slab = ntuple(_ -> on_architecture(arch, zeros(FT, length(is), rows, length(ks))), 15)
    for j₀ in first(js):rows:last(js)
        origin = (first(is), j₀, first(ks))
        launch!(arch, grid, KernelParameters((length(is), min(rows, last(js) - j₀ + 1), length(ks)), origin .- 1), _compute_moment_geometry!, slab, grid, origin)
        host_slab = map(a -> on_architecture(CPU(), a), slab)
        for j in j₀:min(j₀ + rows - 1, last(js)), (i′, i) in enumerate(is)
            column = map(a -> view(a, i′, j - j₀ + 1, :), host_slab)
            values_at(k) = map(c -> @inbounds(c[k - first(ks) + 1]), column)
            bottom, top = values_at(first(ks)), values_at(last(ks))
            k⁻ = something(findfirst(k -> !isequal(values_at(k), bottom), ks), length(ks) + 1) + first(ks) - 1
            k⁺ = something(findlast(k -> !isequal(values_at(k), top), ks), 0) + first(ks) - 1
            j′ = j - first(js) + 1
            first_level[i′, j′], levels[i′, j′], offsets[i′, j′] = k⁻, max(k⁺ - k⁻ + 1, 0), length(values[1])
            for column_values in (bottom, (values_at(k) for k in k⁻:k⁺)..., top)
                foreach(push!, values, column_values)
            end
        end
    end
    return (; reach, first_level=on_architecture(arch, first_level), levels=on_architecture(arch, levels),
              offsets=on_architecture(arch, offsets), values=map(v -> on_architecture(arch, v), values))
end

# (μ, C, the face moments) of cell i, j, k, at its place in the slab from origin
@kernel function _compute_moment_geometry!(slab, grid, origin)
    i, j, k = @index(Global, NTuple)
    i′, j′, k′ = (i, j, k) .- origin .+ 1
    μ, C = wet_part_moments(i, j, k, grid)
    μˣ = strip_face_moments(i, j, k, grid, Val(1))
    μʸ = strip_face_moments(i, j, k, grid, Val(2))
    μᶻ = strip_face_moments(i, j, k, grid, Val(3))
    geometry = (μ..., C..., μˣ[2], μˣ[3], μʸ[1], μʸ[3], μᶻ[1], μᶻ[2])
    map((a, g) -> @inbounds(a[i′, j′, k′] = g), slab, geometry)
end

#####
##### The cells to fit, listed by linear index over the update range, and the values kept for each in vectors grown with the list
#####

# Vectors indexed by the place of a cell in the list of the cells to fit, grown with the list: the list itself, and the tanh integrals
# over the six faces of each cell fitted with a tanh
mutable struct ListedValues{V}
    values :: V
end

Adapt.adapt_structure(to, l::ListedValues) = map(v -> Adapt.adapt(to, v), l.values)
@inline listed_values(l::ListedValues) = l.values
@inline listed_values(values::Tuple) = values
grow!(l::ListedValues, n) = n > length(first(l.values)) && (l.values = map(v -> similar(v, 2n), l.values))

# n vectors of eltype T for the listed cells among `cells`, starting at a sixty-fourth of them
listed_vectors(arch, T, cells, n) = ListedValues(ntuple(_ -> on_architecture(arch, zeros(T, max(1024, cells ÷ 64))), n))

# the list of the cells to fit and the geometry band, shared by the tracers of the grid fitted one after another
function moment_listing(grid)
    reach = map(H -> min(H, 1), halo_size(grid))
    return listed_vectors(architecture(grid), Int32, prod(size(grid) .+ 2 .* reach), 1), reach, geometry_band(grid, reach)
end

# a tracer's running count of its listed cells, which keeps the place of each tanh-fitted cell, and the tanh integrals of their faces
function moment_places(grid, reach)
    arch = architecture(grid)
    positions = on_architecture(arch, zeros(Int32, size(grid) .+ 2 .* reach))
    return positions, listed_vectors(arch, eltype(grid), length(positions), 18)
end

@kernel function _mark_cells!(positions, state, bit, reach)
    i, j, k = @index(Global, NTuple)
    @inbounds positions[i + reach[1], j + reach[2], k + reach[3]] = (state[i, j, k] & bit) != 0
end

@kernel function _list_cells!(list, positions, state, bit, reach)
    i, j, k = @index(Global, NTuple)
    n = (i + reach[1], j + reach[2], k + reach[3])
    slot = @inbounds ifelse((state[i, j, k] & bit) != 0, positions[n...], length(list))
    @inbounds list[slot] = LinearIndices(size(positions))[n...] % Int32
end

@inline listed_cell((n,), (extent, reach)) = Tuple(CartesianIndices(extent)[n]) .- reach

# the cells with `bit` in their state, in order, after which `positions` holds the place of each in the list; returns how many
function list_cells!(list, positions, state, bit, reach, arch, grid, update_range)
    launch!(arch, grid, update_range, _mark_cells!, positions, state, bit, reach)
    cumsum!(vec(positions), vec(positions))
    listed = Int(@allowscalar positions[end])
    grow!(list, listed + 1)
    listed > 0 && launch!(arch, grid, update_range, _list_cells!, first(list.values), positions, state, bit, reach)
    return listed
end

#####
##### Means of tanh and of its derivative over segments, exact
#####

# tanh x and E = e^(-2|x|) - 1, from one expm1; log(2 + E) is the tail of log cosh x beyond |x| - log 2. Beyond |x| = 20, where E
# is -1 to round-off, |x| is clamped, which keeps expm1 off its underflow path
@inline function tanh_and_decay(x)
    E = expm1(-2 * min(abs(x), convert(typeof(x), 20)))
    return copysign(newton_div(BackendOptimizedDivision, -E, 2 + E), x), E
end

# Li₂(-e^(-2u)) from its tail l = log(1 + e^(-2u)) ≤ log 2: Landen's Li₂(-w) = -Li₂(w / (1 + w)) - l² / 2, and the Bernoulli series of
# Li₂(y) in -log(1 - y) = l
@inline function tail_dilogarithm(l)
    FT = typeof(l)
    l² = l^2
    c = map(b -> convert(FT, b), (1/36, -1/3600, 1/211680, -1/10886400, 1/526901760, -4.064761645144226e-11, 8.921691020456452e-13))
    odd = c[1] + l² * (c[2] + l² * (c[3] + l² * (c[4] + l² * (c[5] + l² * (c[6] + l² * c[7])))))
    return -(l - l² / 4 + l * l² * odd) - l² / 2
end

# ∫₀ᵘ log cosh, u ≥ 0, from the tail l of u
@inline function integrated_log_cosh(u, l)
    FT = typeof(u)
    return u^2 / 2 - u * log(convert(FT, 2)) + (tail_dilogarithm(l) + convert(FT, π^2 / 12)) / 2
end

# log(1 + u) to round-off from one log: w - 1 carries the rounding of w = 1 + u (Goldberg)
@inline function compensated_log1p(u)
    w = 1 + u
    return ifelse(w == 1, u, newton_div(BackendOptimizedDivision, log(w) * u, w - 1))
end

# tanh x by its Taylor series to x¹¹, to round-off for |x| ≤ 0.31
@inline function small_tanh(x)
    FT = typeof(x)
    x² = x^2
    c = map(b -> convert(FT, b), (-1/3, 2/15, -17/315, 62/2835, -1382/155925))
    return x * (1 + x² * (c[1] + x² * (c[2] + x² * (c[3] + x² * (c[4] + x² * c[5])))))
end

# The means of tanh, sech², σ tanh, σ sech² and σ² sech², σ = (x - x̄) / (x₂ - x₁), over [x₁, x₂]. Where the differences cancel, the
# expansions about the midpoint, switched at |x₂ - x₁| where the errors of the two forms are equal
@inline function segment_means(x₁, x₂)
    FT = typeof(x₁)
    Δ = x₂ - x₁
    τ₁, E₁ = tanh_and_decay(x₁)
    τ₂, E₂ = tanh_and_decay(x₂)
    # the tails l = log(2 + E) at both ends through their difference, one log1p where only ⟨tanh⟩ needs them
    δl = compensated_log1p(newton_div(BackendOptimizedDivision, E₂ - E₁, 2 + E₁))
    l₁ = log(2 + E₁)
    l₂ = l₁ + δl
    # tanh of the midpoint by the addition formula, exact to round-off wherever an expansion uses it (|Δ| < 0.62 in Float32)
    t = small_tanh(Δ / 2)
    τ = newton_div(BackendOptimizedDivision, τ₁ + t, 1 + τ₁ * t)
    Δ⁻¹ = newton_div(BackendOptimizedDivision, one(FT), ifelse(Δ == 0, one(FT), Δ))
    same_sign = signbit(x₁) == signbit(x₂)
    sign = ifelse(signbit(x₁ + x₂), -one(FT), one(FT))
    lead = ifelse(same_sign, sign, (abs(x₂) - abs(x₁)) * Δ⁻¹)
    s = 1 - τ^2
    δ = abs(Δ)
    δ², δ³ = δ^2, δ^3
    near = δ² * δ³ < 336 * eps(FT)
    mean_tanh = ifelse(near, τ - Δ^2 * convert(FT, 1 / 12) * τ * s, lead + δl * Δ⁻¹)
    mean_sech² = ifelse(near, s + Δ^2 * convert(FT, 1 / 12) * s * (2τ^2 - s), (τ₂ - τ₁) * Δ⁻¹)

    Λ₁, Λ₂ = tail_dilogarithm(l₁), tail_dilogarithm(l₂)
    log_cosh(x, l) = abs(x) - log(convert(FT, 2)) + l
    signed_integral(x, l) = copysign(integrated_log_cosh(abs(x), l), x)
    apart = (log_cosh(x₁, l₁) + log_cosh(x₂, l₂)) / 2 - (signed_integral(x₂, l₂) - signed_integral(x₁, l₁)) * Δ⁻¹
    together = (l₁ + l₂) / 2 - sign * (Λ₂ - Λ₁) * Δ⁻¹ / 2
    near_moment = δ * δ³ * δ³ < 10^4 * eps(FT)
    mean_σ_tanh = ifelse(near_moment, Δ * s * (20 + Δ^2 * (2τ^2 - s)) * convert(FT, 1 / 240), ifelse(same_sign, together, apart) * Δ⁻¹)

    τs = τ * s
    near_σ_sech² = δ³ * δ³ < 10^3 * eps(FT)
    mean_σ_sech² = ifelse(near_σ_sech², Δ * τs * (Δ^2 * (2s - τ^2) - 10) * convert(FT, 1 / 60), ((τ₁ + τ₂) / 2 - mean_tanh) * Δ⁻¹)
    near_σ²_sech² = δ³ * δ³ * δ³ < 10^5 * eps(FT)
    expansion = s * (560 + Δ^2 * (84 * (2τ^2 - s) + 5Δ^2 * (2s^2 - 11τ^2 * s + 2τ^4))) * convert(FT, 1 / 6720)
    mean_σ²_sech² = ifelse(near_σ²_sech², expansion, mean_sech² / 4 - 2 * mean_σ_tanh * Δ⁻¹)
    return mean_tanh, mean_sech², mean_σ_tanh, mean_σ_sech², mean_σ²_sech²
end

#####
##### Profiles over the faces
#####

# the bits of a cell's state: tanh fitted, ramp fitted, between two water masses
const tanh_fit_bit, ramp_fit_bit, switched_bit = 0x01, 0x02, 0x04

# P = (c⁻, Δc, γ, q, c⁺, λ) from the stored (c⁻, c⁺, w, state): w is (γ, q) between two water masses and λ elsewhere, Δc = 0 outside
@inline function tanh_profile(profiles, i, j, k)
    c⁻, c⁺, w₁, w₂, w₃, w₄, state = map(f -> @inbounds(f[i, j, k]), profiles)
    Δc = ifelse((state & switched_bit) != 0, c⁺ - c⁻, zero(c⁻))
    return (c⁻, Δc, w₁, w₂, w₃, w₄, c⁺, w₁, w₂, w₃, w₄)
end

# (⟨H⟩, ⟨ξ_d₁ H⟩, ⟨ξ_d₂ H⟩) of the tanh profile p over the open part of the face at i, j, k normal to e, at ξₑ = side, (d₁, d₂) its
# tangential directions: at the centres of S strips across r, exact along each strip
@inline function face_tanh_integrals(i, j, k, grid, p, ::Val{e}, side) where e
    FT = eltype(grid)
    r, t = strip_directions(e)
    d₁, d₂ = tangential_directions(e)
    S = subsamples(grid, r)
    ΣI, Σ₁, Σ₂, ΣL = zero(FT), zero(FT), zero(FT), zero(FT)
    for m in 1:S
        s = subsample_offset(m, S, FT)
        t⁻, t⁺ = face_interval(i, j, k, grid, Val(e), s)
        L, t̄ = t⁺ - t⁻, (t⁻ + t⁺) / 2
        α = p[1] + p[1 + e] * side + p[1 + r] * s
        mean_tanh, _, mean_σ_tanh = segment_means(α + p[1 + t] * t⁻, α + p[1 + t] * t⁺)
        IH = L * (1 + mean_tanh) / 2
        tIH = L * (t̄ * (1 + mean_tanh) + L * mean_σ_tanh) / 2
        ΣI += IH
        Σ₁ += ifelse(d₁ == t, tIH, s * IH)
        Σ₂ += ifelse(d₂ == t, tIH, s * IH)
        ΣL += L
    end
    W = ifelse(ΣL > 0, ΣL, one(FT))
    return ΣI / W, Σ₁ / W, Σ₂ / W
end

# those of face f = 2e - 1 (ξₑ = -1/2, the face at i, j, k) or 2e (ξₑ = 1/2, the next face along e) of the cell at place n
@inline function store_face_integrals!(values, n, i, j, k, grid, p, ::Val{f}) where f
    e, plus = (f + 1) ÷ 2, iseven(f)
    side = ifelse(plus, one(eltype(grid)), -one(eltype(grid))) / 2
    iᶠ, jᶠ, kᶠ = i + (plus & (e == 1)), j + (plus & (e == 2)), k + (plus & (e == 3))
    @inbounds values[3f - 2][n], values[3f - 1][n], values[3f][n] = face_tanh_integrals(iᶠ, jᶠ, kᶠ, grid, p, Val(e), side)
end

# (⟨H⟩, ⟨ξ_d₁ H⟩, ⟨ξ_d₂ H⟩) of the upwind cell iᵘ of the face at i, j, k normal to e: stored by its tanh fit, or those of the bound
# (H ≡ 0 or 1) of a cell on a plateau
@inline function upwind_face_integrals(i, j, k, r, iᵘ, P, ũ, ::Val{e}) where e
    positions, _, reach, band, store = r.switched_cells
    values = listed_values(store)
    n = @inbounds clamp(positions[iᵘ[1] + reach[1], iᵘ[2] + reach[2], iᵘ[3] + reach[3]], 1, length(first(values)))
    fitted = (@inbounds(r.profiles[7][iᵘ...]) & tanh_fit_bit) != 0
    f⁻, f⁺ = 2e - 1, 2e
    stored = @inbounds ifelse(ũ > 0, (values[3f⁺ - 2][n], values[3f⁺ - 1][n], values[3f⁺][n]), (values[3f⁻ - 2][n], values[3f⁻ - 1][n], values[3f⁻][n]))
    H = ifelse(P[3] < 0, one(P[3]), zero(P[3]))
    bound = (H, stored_face_moment(i, j, k, band, 2e - 1) * H, stored_face_moment(i, j, k, band, 2e) * H)
    return ifelse(fitted, stored, bound)
end

# clamp(a + c t, c⁻, c⁺) over [t⁻, t⁺]: the constant b⁻ up to p, the line up to q, the constant b⁺ after
@inline function clamp_pieces(a, c, c⁻, c⁺, t⁻, t⁺)
    FT = typeof(a)
    flat = c == 0
    c⁻¹ = 1 / ifelse(flat, one(FT), c)
    rising = c > 0
    t₁, t₂ = (c⁻ - a) * c⁻¹, (c⁺ - a) * c⁻¹
    p = clamp(ifelse(flat, t⁺, ifelse(rising, t₁, t₂)), t⁻, t⁺)
    q = clamp(ifelse(flat, t⁺, ifelse(rising, t₂, t₁)), p, t⁺)
    b⁻ = ifelse(flat, clamp(a, c⁻, c⁺), ifelse(rising, c⁻, c⁺))
    b⁺ = ifelse(rising, c⁺, c⁻)
    return p, q, b⁻, b⁺
end

# (∫ b, ∫ t b) of b = clamp(a + c t, c⁻, c⁺) over [t⁻, t⁺], from its pieces
@inline function clamped_line_integrals(a, c, t⁻, t⁺, (p, q, b⁻, b⁺))
    I₀ = b⁻ * (p - t⁻) + a * (q - p) + c * (q^2 - p^2) / 2 + b⁺ * (t⁺ - q)
    I₁ = b⁻ * (p^2 - t⁻^2) / 2 + a * (q^2 - p^2) / 2 + c * (q^3 - p^3) / 3 + b⁺ * (t⁺^2 - q^2) / 2
    return I₀, I₁
end

# ∫ (ℓ b - b² / 2) of ℓ = a + c t and b = clamp(ℓ, c⁻, c⁺) over [t⁻, t⁺], from its pieces
@inline function clamped_line_penalty(a, c, t⁻, t⁺, (p, q, b⁻, b⁺))
    line(u, v) = a * (v - u) + c * (v^2 - u^2) / 2
    return b⁻ * line(t⁻, p) - b⁻^2 * (p - t⁻) / 2 + (a^2 * (q - p) + a * c * (q^2 - p^2) + c^2 * (q^3 - p^3) / 3) / 2 +
           b⁺ * line(q, t⁺) - b⁺^2 * (t⁺ - q) / 2
end

# (⟨ξ_d₁ b⟩, ⟨ξ_d₂ b⟩) of the ramp b of P over the open part of the face at i, j, k normal to e, at ξₑ = side, (d₁, d₂) its
# tangential directions: at the centres of S strips across r, exact along each strip
@inline function face_ramp_moments(i, j, k, grid, P, ::Val{e}, side) where e
    FT = eltype(grid)
    λ = (P[8], P[9], P[10], P[11])
    r, t = strip_directions(e)
    S = subsamples(grid, r)
    Σᵗ, Σʳ, ΣL = zero(FT), zero(FT), zero(FT)
    for m in 1:S
        s = subsample_offset(m, S, FT)
        t⁻, t⁺ = face_interval(i, j, k, grid, Val(e), s)
        a = λ[1] + λ[1 + e] * side + λ[1 + r] * s
        Ib, tIb = clamped_line_integrals(a, λ[1 + t], t⁻, t⁺, clamp_pieces(a, λ[1 + t], P[1], P[7], t⁻, t⁺))
        Σᵗ += tIb
        Σʳ += s * Ib
        ΣL += t⁺ - t⁻
    end
    W = ifelse(ΣL > 0, ΣL, one(FT))
    ξᵗb, ξʳb = Σᵗ / W, Σʳ / W
    return map(d -> ifelse(d == t, ξᵗb, ξʳb), tangential_directions(e))
end

#####
##### The tanh fit: c⁻ + Δc H(β n(θ, φ) ⋅ ξ - γ), n = (cos φ cos θ, sin φ, cos φ sin θ)
#####

@inline symmetric(C, a, b) = @inbounds C[ifelse(a == b, a, a + b + 1)]
@inline covariances(μ, C) = (C[1] - μ[1]^2, C[2] - μ[2]^2, C[3] - μ[3]^2, C[4] - μ[1] * μ[2], C[5] - μ[1] * μ[3], C[6] - μ[2] * μ[3])
@inline symmetric_product(σ, g) = ntuple(a -> symmetric(σ, a, 1) * g[1] + symmetric(σ, a, 2) * g[2] + symmetric(σ, a, 3) * g[3], Val(3))

@inline function direction(θ, φ)
    sθ, cθ = sincos(θ)
    sφ, cφ = sincos(φ)
    return (cφ * cθ, sφ, cφ * sθ)
end

const tanh_fit_iterations = 1
const offset_iterations = 8
const line_search_steps = 4

# the sharpness a cell entering two water masses starts its fit from, along the direction of its moments
const fresh_sharpness = 10

# (⟨H⟩, ⟨H′⟩) at each q, γ of qs, γs over the wet part of cell i, j, k
@inline function tanh_means(i, j, k, grid, qs::NTuple{N}, γs) where N
    FT = eltype(grid)
    Σ = wet_column_mean(i, j, k, grid, ntuple(Returns(zero(FT)), Val(2N))) do ξ, ζ, η⁻, η⁺
        L = η⁺ - η⁻
        means = map((q, γ) -> segment_means(q[1] * ξ + q[2] * ζ + q[3] * η⁻ - γ, q[1] * ξ + q[2] * ζ + q[3] * η⁺ - γ), qs, γs)
        (map(m -> L * (1 + m[1]) / 2, means)..., map(m -> L * m[2] / 2, means)...)
    end
    return ntuple(n -> Σ[n], Val(N)), ntuple(n -> Σ[n + N], Val(N))
end

# γ of each q in qs with ⟨H(q ⋅ ξ - γ)⟩ = t₁, by Newton steps from γs on the log of the mass on the side of t₁ (⟨H⟩ below 1/2,
# 1 - ⟨H⟩ above), linear in γ where the interface leaves the cell, kept inside the bracket that bisection keeps
@inline function exact_mean_offsets(i, j, k, grid, t₁, qs, γs)
    FT = eltype(grid)
    β = map(q -> sqrt(q[1]^2 + q[2]^2 + q[3]^2), qs)
    lower, upper = map(b -> -(b + 40), β), map(b -> b + 40, β)
    done = map(Returns(false), qs)
    low = t₁ < one(FT) / 2
    side_target = ifelse(low, t₁, 1 - t₁)
    for _ in 1:offset_iterations
        H, H′ = tanh_means(i, j, k, grid, qs, γs)
        above = H .> t₁
        lower = map((d, a, l, γ) -> ifelse(d | !a, l, γ), done, above, lower, γs)
        upper = map((d, a, u, γ) -> ifelse(d | a, u, γ), done, above, upper, γs)
        done = map((d, h) -> d | (abs(h - t₁) < convert(FT, 1e-13)), done, H)
        newton = map(γs, H, H′) do γ, h, h′
            side = ifelse(low, h, 1 - h)
            γ + ifelse(low, one(FT), -one(FT)) * log(side / side_target) * side / h′
        end
        # the bracket is closed: a converged γ is itself an end of the bracket, and its Newton step returns it
        γs = map((d, γ, n, h′, l, u) -> ifelse(d, γ, ifelse((h′ > 0) & (l ≤ n) & (n ≤ u), n, (l + u) / 2)), done, γs, newton, H′, lower, upper)
        done = map((d, l, u) -> d | (u - l < convert(FT, 1e-12)), done, lower, upper)
    end
    return γs
end

# ⟨ξ H⟩ - tᴹ at each q, γ of qs, γs
@inline function moment_residuals(i, j, k, grid, t, qs::NTuple{N}, γs) where N
    FT = eltype(grid)
    Σ = wet_column_mean(i, j, k, grid, ntuple(Returns(zero(FT)), Val(3N))) do ξ, ζ, η⁻, η⁺
        L, η̄ = η⁺ - η⁻, (η⁻ + η⁺) / 2
        integrals = map(qs, γs) do q, γ
            α = q[1] * ξ + q[2] * ζ - γ
            mean_tanh, _, mean_σ_tanh = segment_means(α + q[3] * η⁻, α + q[3] * η⁺)
            IH = L * (1 + mean_tanh) / 2
            (IH, η̄ * IH + L^2 * mean_σ_tanh / 2)
        end
        (map(I -> ξ * I[1], integrals)..., map(I -> ζ * I[1], integrals)..., map(I -> I[2], integrals)...)
    end
    return ntuple(n -> (Σ[n] - t[2], Σ[n + N] - t[3], Σ[n + 2N] - t[4]), Val(N))
end

# (θ, φ, log β) a fraction s of the step δ away, for each s in scales, log β within bounds
@inline line_search_candidates(θ, φ, logβ, δ, scales, bounds) = map(s -> (θ + s * δ[1], φ + s * δ[2], clamp(logβ + s * δ[3], bounds...)), scales)

@inline residual_size(r) = sqrt(r[1]^2 + r[2]^2 + r[3]^2)

# The Gauss-Newton step (δθ, δφ, δ log β) for the residual r at (θ, φ, log β, γ): dr/dα = (A - b bᵀ / h) ∂q/∂α with A = ⟨ξ ξᵀ H′⟩,
# b = ⟨ξ H′⟩, h = ⟨H′⟩, γ following the mean (∂γ/∂α = b ⋅ ∂q/∂α / h), on the normal equations (JᵀJ + 10⁻¹² I) δ = -Jᵀ r: δ log β of
# the full system (by the adjugate) within the bounds of log β, then (δθ, δφ) with that δ log β. Returns δ and the centroid b / h of
# the interface
@inline function gauss_newton_step(i, j, k, grid, r, θ, φ, logβ, γ, bounds)
    FT = eltype(grid)
    β = exp(logβ)
    q = β .* direction(θ, φ)
    Σ = wet_column_mean(i, j, k, grid, ntuple(Returns(zero(FT)), Val(10))) do ξ, ζ, η⁻, η⁺
        L, η̄ = η⁺ - η⁻, (η⁻ + η⁺) / 2
        α = q[1] * ξ + q[2] * ζ - γ
        _, mean_sech², _, mean_σ_sech², mean_σ²_sech² = segment_means(α + q[3] * η⁻, α + q[3] * η⁺)
        I₀ = L * mean_sech² / 2
        I₁ = η̄ * I₀ + L^2 * mean_σ_sech² / 2
        I₂ = η̄^2 * I₀ + η̄ * L^2 * mean_σ_sech² + L^3 * mean_σ²_sech² / 2
        (I₀, ξ * I₀, ζ * I₀, I₁, ξ^2 * I₀, ζ^2 * I₀, I₂, ξ * ζ * I₀, ξ * I₁, ζ * I₁)
    end
    h = Σ[1]
    K = h .* covariances((Σ[2], Σ[3], Σ[4]) ./ h, (Σ[5], Σ[6], Σ[7], Σ[8], Σ[9], Σ[10]) ./ h)
    sθ, cθ = sincos(θ)
    sφ, cφ = sincos(φ)
    ∂q = (β .* (-cφ * sθ, zero(FT), cφ * cθ), β .* (-sφ * cθ, cφ, -sφ * sθ), q)
    J = map(v -> symmetric_product(K, v), ∂q)
    ε = convert(FT, 1e-12)
    G = (sum(J[1] .* J[1]) + ε, sum(J[2] .* J[2]) + ε, sum(J[3] .* J[3]) + ε, sum(J[1] .* J[2]), sum(J[1] .* J[3]), sum(J[2] .* J[3]))
    A = (G[2] * G[3] - G[6]^2, G[1] * G[3] - G[5]^2, G[1] * G[2] - G[4]^2, G[5] * G[6] - G[4] * G[3], G[4] * G[6] - G[5] * G[2], G[4] * G[5] - G[1] * G[6])
    g = map(Jₐ -> -sum(Jₐ .* r), J)
    δ₃ = clamp(logβ + symmetric_product(A, g)[3] / (G[1] * A[1] + G[4] * A[4] + G[5] * A[5]), bounds...) - logβ
    g₁, g₂ = g[1] - G[5] * δ₃, g[2] - G[6] * δ₃
    δ = (G[2] * g₁ - G[4] * g₂, G[1] * g₂ - G[4] * g₁) ./ (G[1] * G[2] - G[4]^2)
    return (δ..., δ₃), (Σ[2], Σ[3], Σ[4]) ./ h
end

@inline pick(n, xs) = foldr((s, y) -> ifelse(s == n, xs[s], y), ntuple(identity, Val(length(xs) - 1)); init=last(xs))

# the offset of each candidate q′ = β′ n′ that keeps the interface of (q, γ) through its centroid ξᶜ, inside its bracket
@inline predicted_offsets(qs, candidates, q, γ, ξᶜ) = map((q′, c) -> clamp(γ + sum((q′ .- q) .* ξᶜ), -exp(c[3]) - 40, exp(c[3]) + 40), qs, candidates)

# (γ, q) of the tanh with the targets t = (⟨H⟩, ⟨ξ H⟩) over the wet part of cell i, j, k, from P = (c⁻, Δc, γ, q); the steps in
# (θ, φ, log β) of q = β n(θ, φ), q = 0 for a cell entering two water masses
@inline function tanh_fit(i, j, k, grid, t, μ, P)
    FT = eltype(grid)
    γ, q = P[3], (P[4], P[5], P[6])
    β = sqrt(q[1]^2 + q[2]^2 + q[3]^2)
    θ = atan(q[3], q[1])
    φ = atan(q[2], sqrt(q[1]^2 + q[3]^2))
    a = (t[2] - t[1] * μ[1], t[3] - t[1] * μ[2], t[4] - t[1] * μ[3])
    fresh = (β == 0) & (sum(a .* a) > 0)
    logβ = ifelse(β > 0, log(β), ifelse(fresh, log(convert(FT, fresh_sharpness)), zero(FT)))
    θ = ifelse(fresh, atan(a[3], a[1]), θ)
    φ = ifelse(fresh, atan(a[2], sqrt(a[1]^2 + a[3]^2)), φ)
    γ = first(exact_mean_offsets(i, j, k, grid, t[1], (exp(logβ) .* direction(θ, φ),), (clamp(γ, -exp(logβ) - 40, exp(logβ) + 40),)))
    r = first(moment_residuals(i, j, k, grid, t, (exp(logβ) .* direction(θ, φ),), (γ,)))
    done = false
    scales = ntuple(s -> one(FT) / 2^(s - 1), Val(line_search_steps))
    bounds = (log(convert(FT, 1e-3)), log(convert(FT, 200)))
    for _ in 1:tanh_fit_iterations
        done = done | (residual_size(r) < convert(FT, 1e-10))
        δ, ξᶜ = gauss_newton_step(i, j, k, grid, r, θ, φ, logβ, γ, bounds)
        # a singular system gives a zero step, which the line search rejects
        δ, ξᶜ = ifelse(isfinite(sum(δ) + sum(ξᶜ)), (δ, ξᶜ), (map(zero, δ), map(zero, ξᶜ)))
        candidates = line_search_candidates(θ, φ, logβ, δ, scales, bounds)
        qs = map(c -> exp(c[3]) .* direction(c[1], c[2]), candidates)
        γs = exact_mean_offsets(i, j, k, grid, t[1], qs, predicted_offsets(qs, candidates, exp(logβ) .* direction(θ, φ), γ, ξᶜ))
        rs = moment_residuals(i, j, k, grid, t, qs, γs)
        lower = map(residual_size, rs) .< residual_size(r)
        n = foldr((s, m) -> ifelse(lower[s], s, m), ntuple(identity, Val(line_search_steps - 1)); init=line_search_steps)
        accepted = !done & reduce(|, lower)
        θ, φ, logβ = ifelse(accepted, pick(n, candidates), (θ, φ, logβ))
        γ, r = ifelse(accepted, (pick(n, γs), pick(n, rs)), (γ, r))
        done = done | !accepted
    end
    return (γ, (exp(logβ) .* direction(θ, φ))...)
end

@kernel function _fit_tanh_profiles!(listing, moments, profiles, grid, ĉ, geometry, (positions, face_integrals))
    i, j, k = listed_cell(@index(Global, NTuple), listing)
    reach = last(listing)
    n = @inbounds positions[i + reach[1], j + reach[2], k + reach[3]]
    c̄ = @inbounds ĉ[i, j, k]
    P = tanh_profile(profiles, i, j, k)
    μ, _ = stored_wet_part_moments(i, j, k, geometry)
    M = @inbounds (moments[1][i, j, k], moments[2][i, j, k], moments[3][i, j, k])
    fit = tanh_fit(i, j, k, grid, ((c̄ - P[1]) / P[2], ((M .- P[1] .* μ) ./ P[2])...), μ, P)
    @inbounds profiles[3][i, j, k], profiles[4][i, j, k], profiles[5][i, j, k], profiles[6][i, j, k] = fit
    p = (-fit[1], fit[2], fit[3], fit[4])
    foreach(f -> store_face_integrals!(listed_values(face_integrals), n, i, j, k, grid, p, f), (Val(1), Val(2), Val(3), Val(4), Val(5), Val(6)))
end

#####
##### The least-variance ramp: λ maximizing the concave dual λ ⋅ m - ⟨ℓ b - b²/2⟩, b = clamp(ℓ, c⁻, c⁺), ℓ = λ ⋅ φ, φ = (1, ξ),
##### by Newton steps with Hessian -⟨φ φᵀ⟩ over the unclipped part, each the first of 1, 1/2, 1/4, 1/8 of the step that raises
##### the dual enough (Armijo), the slopes within 10⁴ (c⁺ - c⁻)
#####

const ramp_iterations = 3
const backtracking_steps = 4

# δ with J δ = b for J = (J₁₁, J₁₂, J₁₃, J₁₄, J₂₂, J₃₃, J₄₄, J₂₃, J₂₄, J₃₄) symmetric positive definite, by Cholesky:
# NaN where J is singular to round-off, a step the line search rejects
@inline function cholesky_solve(J, b)
    J₁₁, J₁₂, J₁₃, J₁₄, J₂₂, J₃₃, J₄₄, J₂₃, J₂₄, J₃₄ = J
    z = zero(J₁₁)
    L₁₁ = sqrt(max(J₁₁, z))
    L₂₁, L₃₁, L₄₁ = J₁₂ / L₁₁, J₁₃ / L₁₁, J₁₄ / L₁₁
    L₂₂ = sqrt(max(J₂₂ - L₂₁^2, z))
    L₃₂, L₄₂ = (J₂₃ - L₃₁ * L₂₁) / L₂₂, (J₂₄ - L₄₁ * L₂₁) / L₂₂
    L₃₃ = sqrt(max(J₃₃ - L₃₁^2 - L₃₂^2, z))
    L₄₃ = (J₃₄ - L₄₁ * L₃₁ - L₄₂ * L₃₂) / L₃₃
    L₄₄ = sqrt(max(J₄₄ - L₄₁^2 - L₄₂^2 - L₄₃^2, z))
    y₁ = b[1] / L₁₁
    y₂ = (b[2] - L₂₁ * y₁) / L₂₂
    y₃ = (b[3] - L₃₁ * y₁ - L₃₂ * y₂) / L₃₃
    y₄ = (b[4] - L₄₁ * y₁ - L₄₂ * y₂ - L₄₃ * y₃) / L₄₄
    δ₄ = y₄ / L₄₄
    δ₃ = (y₃ - L₄₃ * δ₄) / L₃₃
    δ₂ = (y₂ - L₃₂ * δ₃ - L₄₂ * δ₄) / L₂₂
    return ((y₁ - L₂₁ * δ₂ - L₃₁ * δ₃ - L₄₁ * δ₄) / L₁₁, δ₂, δ₃, δ₄)
end

# (⟨φ b⟩, ⟨ℓ b - b²/2⟩, ⟨φ φᵀ⟩ where c⁻ < ℓ < c⁺) over the wet part of cell i, j, k, exact along each column
@inline function ramp_sums(i, j, k, grid, λ, c⁻, c⁺)
    FT = eltype(grid)
    return wet_column_mean(i, j, k, grid, ntuple(Returns(zero(FT)), Val(15)), ramp_column_subsamples) do ξ, ζ, η⁻, η⁺
        a = λ[1] + λ[2] * ξ + λ[3] * ζ
        pieces = clamp_pieces(a, λ[4], c⁻, c⁺, η⁻, η⁺)
        p, q = pieces[1], pieces[2]
        Ib, Iηb = clamped_line_integrals(a, λ[4], η⁻, η⁺, pieces)
        Iu, Iηu, Iη²u = q - p, (q^2 - p^2) / 2, (q^3 - p^3) / 3
        (Ib, ξ * Ib, ζ * Ib, Iηb, clamped_line_penalty(a, λ[4], η⁻, η⁺, pieces),
         Iu, ξ * Iu, ζ * Iu, Iηu, ξ^2 * Iu, ζ^2 * Iu, Iη²u, ξ * ζ * Iu, ξ * Iηu, ζ * Iηu)
    end
end

# ⟨ℓ b - b²/2⟩ at each λ of λs
@inline function ramp_penalties(i, j, k, grid, λs::NTuple{N}, c⁻, c⁺) where N
    FT = eltype(grid)
    return wet_column_mean(i, j, k, grid, ntuple(Returns(zero(FT)), Val(N)), ramp_column_subsamples) do ξ, ζ, η⁻, η⁺
        map(λs) do λ
            a = λ[1] + λ[2] * ξ + λ[3] * ζ
            clamped_line_penalty(a, λ[4], η⁻, η⁺, clamp_pieces(a, λ[4], c⁻, c⁺, η⁻, η⁺))
        end
    end
end

@inline function ramp_trials(λ, δ, steps, cap)
    return map(s -> (λ[1] + s * δ[1], clamp(λ[2] + s * δ[2], -cap, cap), clamp(λ[3] + s * δ[3], -cap, cap), clamp(λ[4] + s * δ[4], -cap, cap)), steps)
end

# λ of the least-variance ramp with mean and moments m = (c̄, M) over the wet part of cell i, j, k, from λ
@inline function least_variance_ramp(i, j, k, grid, m, λ, c⁻, c⁺)
    FT = eltype(grid)
    width = c⁺ - c⁻
    cap = convert(FT, 1e4) * width
    ε = convert(FT, 1e-14)
    flat = map(d -> ifelse(topology(grid, d) === Flat, one(FT), zero(FT)), (1, 2, 3))
    steps = ntuple(s -> one(FT) / 2^(s - 1), Val(backtracking_steps))
    done = false
    for _ in 1:ramp_iterations
        Σ = ramp_sums(i, j, k, grid, λ, c⁻, c⁺)
        r = m .- (Σ[1], Σ[2], Σ[3], Σ[4])
        done = done | (maximum(abs, r) < convert(FT, 1e-12) * width)
        D = sum(λ .* m) - Σ[5]
        J = (Σ[6] + ε, Σ[7], Σ[8], Σ[9], Σ[10] + ε + flat[1], Σ[11] + ε + flat[2], Σ[12] + ε + flat[3], Σ[13], Σ[14], Σ[15])
        δ = cholesky_solve(J, r)
        trials = ramp_trials(λ, δ, steps, cap)
        Ds = map((λₛ, pₛ) -> sum(λₛ .* m) - pₛ, trials, ramp_penalties(i, j, k, grid, trials, c⁻, c⁺))
        rise = sum(δ .* r)
        n = minimum(ntuple(s -> ifelse(Ds[s] ≥ D + convert(FT, 1e-4) * steps[s] * rise, s, backtracking_steps + 1), Val(backtracking_steps)))
        accepted = !done & (n ≤ backtracking_steps)
        chosen = reduce((a, b) -> a .+ b, ntuple(s -> ifelse(s == n, trials[s], map(zero, trials[s])), Val(backtracking_steps)))
        λ = ifelse(accepted, chosen, λ)
        done = done | !accepted
    end
    return λ
end

@kernel function _fit_least_variance_ramps!(listing, moments, profiles, grid, ĉ, geometry)
    i, j, k = listed_cell(@index(Global, NTuple), listing)
    P = tanh_profile(profiles, i, j, k)
    m = @inbounds (ĉ[i, j, k], moments[1][i, j, k], moments[2][i, j, k], moments[3][i, j, k])
    λ = least_variance_ramp(i, j, k, grid, m, (P[8], P[9], P[10], P[11]), P[1], P[7])
    @inbounds profiles[3][i, j, k], profiles[4][i, j, k], profiles[5][i, j, k], profiles[6][i, j, k] = λ
end

#####
##### Update: the switch, the bounds and the moments of every cell in the update range, and which cells fit a tanh or a ramp
#####

const polynomial_moment_weights = (-191, 1688, -7843, 0, 7843, -1688, 191) ./ 10080

@inline function cell_gradient(i, j, k, grid, ĉ, ::Val{d}) where d
    FT = eltype(grid)
    @inline cell(n) = d == 1 ? (i + n, j, k) : d == 2 ? (i, j + n, k) : (i, j, k + n)
    @inline wet(n) = !inactive_cell(cell(n)..., grid)
    @inline c(n) = @inbounds ĉ[cell(n)...]
    g⁶ = sum(ntuple(n -> convert(FT, polynomial_moment_weights[n]) * c(n - 4), Val(7)))
    g² = ifelse(wet(-1) & wet(1), (c(1) - c(-1)) / 2, ifelse(wet(1), c(1) - c(0), ifelse(wet(-1), c(0) - c(-1), zero(grid))))
    return ifelse(all(ntuple(n -> wet(n - 4), Val(7))), g⁶, g²)
end

@inline cell_gradient(i, j, k, grid::XFlatGrid, ĉ, ::Val{1}) = zero(grid)
@inline cell_gradient(i, j, k, grid::YFlatGrid, ĉ, ::Val{2}) = zero(grid)
@inline cell_gradient(i, j, k, grid::ZFlatGrid, ĉ, ::Val{3}) = zero(grid)

# The active cell's value and c̄ elsewhere, and the same with the value mirrored across the bed below the bed of an active column
@inline function active_and_mirrored_values(i, j, k, grid, ĉ, (iˢ, jˢ, kˢ), c̄)
    kᵇ = @inbounds kˢ[i, j, 1]
    mirror = (@inbounds(iˢ[i, j, 1]) == i) & (@inbounds(jˢ[i, j, 1]) == j) & (1 ≤ k) & (k < kᵇ)
    active = (1 ≤ k) & (k ≤ size(grid, 3)) & !inactive_cell(i, j, k, grid)
    c = @inbounds ĉ[i, j, ifelse(mirror, min(2kᵇ - 1 - k, size(grid, 3)), k)]
    return ifelse(active, c, c̄), ifelse(active | mirror, c, c̄)
end

# whether λ ⋅ (1, ξ) leaves (c⁻, c⁺) in the wet part of cell i, j, k: at the ends of the wet interval of each of its S² columns
@inline function leaves_bounds(i, j, k, grid, λ, c⁻, c⁺)
    FT = eltype(grid)
    Sˣ, Sʸ = subsamples(grid, 1, ramp_column_subsamples), subsamples(grid, 2, ramp_column_subsamples)
    outside = false
    for m in 1:Sˣ, n in 1:Sʸ
        ξ, ζ = subsample_offset(m, Sˣ, FT), subsample_offset(n, Sʸ, FT)
        η⁻, η⁺ = wet_interval(i, j, k, grid, ξ, ζ)
        a = λ[1] + λ[2] * ξ + λ[3] * ζ
        ℓ⁻, ℓ⁺ = a + λ[4] * η⁻, a + λ[4] * η⁺
        inside = (c⁻ < ℓ⁻) & (ℓ⁻ < c⁺) & (c⁻ < ℓ⁺) & (ℓ⁺ < c⁺)
        outside = outside | ((η⁻ < η⁺) & !inside)
    end
    return outside
end

# the extrema of the active values over the 3³ neighbourhood of cell i, j, k and of the mirrored values over its 3³ and its 5³, in one
# pass over the 5³
@inline function neighbourhood_extrema(i, j, k, grid, ĉ, wet_column, c̄)
    a, m = active_and_mirrored_values(i, j, k, grid, ĉ, wet_column, c̄)
    a⁻ = a⁺ = a
    m₃⁻ = m₃⁺ = m₅⁻ = m₅⁺ = m
    nx, ny, nz = map(d -> ifelse(topology(grid, d) === Flat, 0, 2), (1, 2, 3))
    for di in -nx:nx, dj in -ny:ny, dk in -nz:nz
        a, m = active_and_mirrored_values(i + di, j + dj, k + dk, grid, ĉ, wet_column, c̄)
        inner = (abs(di) ≤ 1) & (abs(dj) ≤ 1) & (abs(dk) ≤ 1)
        a⁻, a⁺ = ifelse(inner & (a < a⁻), a, a⁻), ifelse(inner & (a > a⁺), a, a⁺)
        m₃⁻, m₃⁺ = ifelse(inner & (m < m₃⁻), m, m₃⁻), ifelse(inner & (m > m₃⁺), m, m₃⁺)
        m₅⁻, m₅⁺ = ifelse(m < m₅⁻, m, m₅⁻), ifelse(m > m₅⁺, m, m₅⁺)
    end
    return (a⁻, a⁺), (m₃⁻, m₃⁺), (m₅⁻, m₅⁺)
end

# c̄ uniform within its bounds: on one of them, or between bounds that do not differ
@inline function on_plateau(c̄, c⁻, c⁺)
    FT = typeof(c̄)
    Δc = c⁺ - c⁻
    return (Δc < convert(FT, 1e-12) * max(one(FT), abs(c⁺))) | (min(c̄ - c⁻, c⁺ - c̄) < convert(FT, 1e-10) * Δc)
end

@kernel function _update_moments!(moments, profiles, grid, ĉ, scheme, w)
    i, j, k = @index(Global, NTuple)
    FT = eltype(grid)
    c̄ = @inbounds ĉ[i, j, k]
    (c⁻, c⁺), (m₃⁻, m₃⁺), (m₅⁻, m₅⁺) = neighbourhood_extrema(i, j, k, grid, ĉ, scheme.reconstructions.wet_column, c̄)
    td = TimeSteppers.time_discretization(scheme)
    s = min(explicit_velocity_scaleᶜᶜᶠ(i, j, k, grid, scheme, td, w), explicit_velocity_scaleᶜᶜᶠ(i, j, k + 1, grid, scheme, td, w))

    # between two water masses: the mirrored 3³ range exceeds 3/4 of the 5³ range, in an active cell with explicit vertical transport
    wet = !inactive_cell(i, j, k, grid)
    switched = (4 * (m₃⁺ - m₃⁻) > 3 * (m₅⁺ - m₅⁻)) & wet & (s == 1)

    μ, C = stored_wet_part_moments(i, j, k, stored_geometry(scheme.reconstructions))
    g = (cell_gradient(i, j, k, grid, ĉ, Val(1)), cell_gradient(i, j, k, grid, ĉ, Val(2)), cell_gradient(i, j, k, grid, ĉ, Val(3)))
    M⁰ = @inbounds (moments[1][i, j, k], moments[2][i, j, k], moments[3][i, j, k])
    M = ifelse(switched, M⁰, c̄ .* μ .+ symmetric_product(covariances(μ, C), g))

    # the least-variance ramp clamp(λ ⋅ φ, c⁻, c⁺) of any other cell: its linear profile, iterated where it leaves the bounds
    Δc = c⁺ - c⁻
    plateau = on_plateau(c̄, c⁻, c⁺)
    λ = ifelse(plateau, (c̄, zero(FT), zero(FT), zero(FT)), (c̄ - g[1] * μ[1] - g[2] * μ[2] - g[3] * μ[3], g...))
    overshoot = leaves_bounds(i, j, k, grid, λ, c⁻, c⁺)

    # the last stage's w: (γ, q) of a cell then between two water masses, its ramp otherwise
    was_switched = (@inbounds(profiles[7][i, j, k]) & switched_bit) != 0
    w⁰ = @inbounds (profiles[3][i, j, k], profiles[4][i, j, k], profiles[5][i, j, k], profiles[6][i, j, k])
    γ⁰ = ifelse(was_switched, w⁰[1], zero(FT))
    q⁰ = ifelse(was_switched, (w⁰[2], w⁰[3], w⁰[4]), (zero(FT), zero(FT), zero(FT)))

    # a switched cell on a plateau carries the bound it sits on (H ≡ 0 or 1) and keeps its last fit
    β = sqrt(q⁰[1]^2 + q⁰[2]^2 + q⁰[3]^2)
    γ = ifelse(plateau, ifelse(c̄ - c⁻ > c⁺ - c̄, -(β + 40), β + 40), γ⁰)

    # a cell whose ramp iterates starts from its last ramp, which has the same maximizer
    two_water_masses = switched & (Δc > 0)
    ramp = !switched & wet & !plateau & overshoot
    λʳ = ifelse(ramp & !was_switched, w⁰, λ)
    state = ifelse(two_water_masses, switched_bit, 0x00) | ifelse(two_water_masses & !plateau, tanh_fit_bit, 0x00) | ifelse(ramp, ramp_fit_bit, 0x00)

    @inbounds begin
        moments[1][i, j, k] = M[1]
        moments[2][i, j, k] = M[2]
        moments[3][i, j, k] = M[3]
        profiles[1][i, j, k] = c⁻
        profiles[2][i, j, k] = c⁺
        profiles[3][i, j, k], profiles[4][i, j, k], profiles[5][i, j, k], profiles[6][i, j, k] = ifelse(two_water_masses, (γ, q⁰...), λʳ)
        profiles[7][i, j, k] = state
    end
end

#####
##### Fluxes through the two faces normal to e
#####

@inline two_water_masses_face_value(ψ, ::Nothing) = ψ
@inline two_water_masses_face_value(ψ, (P, H)) = ifelse(P[2] > 0, P[1] + P[2] * H, ψ)

@inline upwind_face_profile(i, j, k, ::Nothing, r, ũ, ::Val) = nothing

for (FlatGrid, Δ⁻¹, e) in ((XFlatGrid, Δx⁻¹ᶜᶜᶜ, 1), (YFlatGrid, Δy⁻¹ᶜᶜᶜ, 2), (ZFlatGrid, Δη⁻¹, 3))
    minus_side = e == 1 ? (:(i - 1), :j, :k) : e == 2 ? (:i, :(j - 1), :k) : (:i, :j, :(k - 1))
    plus_side  = e == 1 ? (:(i + 1), :j, :k) : e == 2 ? (:i, :(j + 1), :k) : (:i, :j, :(k + 1))
    velocity(index) = e == 3 ? :(u[$(index...)] * explicit_velocity_scaleᶜᶜᶠ($(index...), grid, scheme, TimeSteppers.time_discretization(scheme), u)) :
                               :(u[$(index...)])

    @eval begin
        # the upwind profile P of the face at i, j, k, and the mean ⟨H⟩ of its tanh over the open part of the face
        @inline function upwind_face_profile(i, j, k, profiles::Tuple, r, ũ, ::Val{$e})
            iᵘ = ifelse(ũ > 0, ($(minus_side...),), (i, j, k))
            P = tanh_profile(profiles, iᵘ...)
            return P, first(upwind_face_integrals(i, j, k, r, iᵘ, P, ũ, Val($e)))
        end

        @inline function normal_moment_flux(i, j, k, grid, scheme, u, r, M, ::Val{$e})
            q⁻, F⁻, _ = limited_face_flux(i, j, k, grid, scheme, u, r, Val($e))
            q⁺, F⁺, _ = limited_face_flux($(plus_side...), grid, scheme, u, r, Val($e))
            c̄ = r[i, j, k]
            μ, _ = stored_wet_part_moments(i, j, k, stored_geometry(r))
            δu = @inbounds $(velocity(plus_side)) - $(velocity((:i, :j, :k)))
            return (F⁺ + F⁻ - c̄ * (q⁺ + q⁻)) / 2 - Vᶜᶜᶜ(i, j, k, grid) * $Δ⁻¹(i, j, k, grid) * δu * (@inbounds(M[i, j, k]) - c̄ * μ[$e])
        end

        # the fluxes of the moments along the two tangential directions through the face at i, j, k, relative to the mean c̄
        @inline function face_moment_fluxes(i, j, k, grid, scheme, u, r, c̄, ::Val{$e})
            ũ = @inbounds u[i, j, k]
            h = one(c̄) / 2
            iᵘ = ifelse(ũ > 0, ($(minus_side...),), (i, j, k))
            P = tanh_profile(r.profiles, iᵘ...)
            _, ξH₁, ξH₂ = upwind_face_integrals(i, j, k, r, iᵘ, P, ũ, Val($e))
            ξb₁, ξb₂ = face_ramp_moments(i, j, k, grid, P, Val($e), ifelse(ũ > 0, h, -h))
            q, _, θ = limited_face_flux(i, j, k, grid, scheme, u, r, Val($e))
            band = stored_geometry(r)
            cᵘ = r[iᵘ...]
            function flux(μᶠ, ξH, ξb)
                mᵘ = ifelse(P[2] > 0, P[1] * μᶠ + P[2] * ξH, ξb)
                m = cᵘ * μᶠ + θ * (mᵘ - cᵘ * μᶠ)
                return q * (m - c̄ * μᶠ)
            end
            return flux(stored_face_moment(i, j, k, band, $(2e - 1)), ξH₁, ξb₁), flux(stored_face_moment(i, j, k, band, $(2e)), ξH₂, ξb₂)
        end

        @inline normal_moment_flux(i, j, k, grid::$FlatGrid, scheme, u, r, M, ::Val{$e}) = zero(grid)
        @inline face_moment_fluxes(i, j, k, grid::$FlatGrid, scheme, u, r, c̄, ::Val{$e}) = (zero(grid), zero(grid))
    end
end

# The tendencies of the first moments M of a tracer advected with `scheme` by U, from the tracer's fluxes through the six faces; zero
# along a Flat direction
@inline function moment_tendencies(i, j, k, grid, scheme, U, M)
    r = scheme.reconstructions
    u = values(U)
    c̄ = r[i, j, k]
    Fˣ = face_moment_fluxes(i + 1, j, k, grid, scheme, u[1], r, c̄, Val(1)) .- face_moment_fluxes(i, j, k, grid, scheme, u[1], r, c̄, Val(1))
    Fʸ = face_moment_fluxes(i, j + 1, k, grid, scheme, u[2], r, c̄, Val(2)) .- face_moment_fluxes(i, j, k, grid, scheme, u[2], r, c̄, Val(2))
    Fᶻ = face_moment_fluxes(i, j, k + 1, grid, scheme, u[3], r, c̄, Val(3)) .- face_moment_fluxes(i, j, k, grid, scheme, u[3], r, c̄, Val(3))
    V⁻¹ = V⁻¹ᶜᶜᶜ(i, j, k, grid)
    divergences = (V⁻¹ * (normal_moment_flux(i, j, k, grid, scheme, u[1], r, M[1], Val(1)) + Fʸ[1] + Fᶻ[1]),
                   V⁻¹ * (normal_moment_flux(i, j, k, grid, scheme, u[2], r, M[2], Val(2)) + Fˣ[1] + Fᶻ[2]),
                   V⁻¹ * (normal_moment_flux(i, j, k, grid, scheme, u[3], r, M[3], Val(3)) + Fˣ[2] + Fʸ[2]))
    return ntuple(d -> -ifelse(topology(grid, d) === Flat, zero(grid), divergences[d]), Val(3))
end
