using Oceananigans.Grids: MultiEnvelopeGrid, LinearEnvelope, MultiEnvelope
using Oceananigans.Architectures: architecture, on_architecture, CPU

import Oceananigans.Grids: materialize_envelopes!

#####
##### Bathymetry-derived, shelf-safe envelopes
#####

"""
    shelf_safe_envelopes(bathymetry, target_depths; minimum_thickness=10)

Build a tuple of `n = length(target_depths) + 1` envelope-depth functions (shallowest → deepest) from a
`bathymetry(x, y)` depth function and intermediate `target_depths` (e.g. `(250, 1500)`). In the open ocean
each envelope sits at its target; where the bathymetry shoals onto a shelf the deeper envelopes are clipped
so they stay **strictly increasing** with at least `minimum_thickness` between successive envelopes (and
above the bottom). This guarantees σᵉ > 0 everywhere — the zones compress smoothly onto the shelf rather
than collapsing to zero thickness. The deepest returned envelope is the bathymetry itself.
"""
function shelf_safe_envelopes(bathymetry, target_depths; minimum_thickness=10)
    n = length(target_depths) + 1
    envelopes = ntuple(n) do i
        function envelope(x, y)
            bottom = bathymetry(x, y)
            if i == n
                return bottom
            else
                # clip the target so that the (n - i) deeper envelopes + the bottom still fit below it
                ceiling = bottom - (n - i) * minimum_thickness
                return min(target_depths[i], ceiling)
            end
        end
        envelope
    end
    return envelopes
end

"""
    smooth_envelope_field!(field, grid; passes=8)

Iterated 1-2-1 horizontal smoothing of a 2-D field in place (after Martinho & Batten 2006), to reduce the
envelope slope parameter r = |Hb - Ha|/(Hb + Ha) and hence the pressure-gradient error. `passes` controls
the cutoff. Halos are refreshed between passes.
"""
function smooth_envelope_field!(field, grid; passes=8)
    tmp = Field((Center(), Center(), nothing), grid)
    for _ in 1:passes
        fill_halo_regions!(field)
        parent(tmp) .= parent(field)
        launch!(architecture(grid), grid, (size(grid, 1), size(grid, 2)), _smooth_121!, field, tmp, grid)
    end
    fill_halo_regions!(field)
    return field
end

@kernel function _smooth_121!(out, in_field, grid)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j, 1] = (in_field[i-1, j, 1] + in_field[i+1, j, 1] +
                              in_field[i, j-1, 1] + in_field[i, j+1, 1] +
                              4 * in_field[i, j, 1]) / 8
end

# Fill the static envelope metric from the bottom-envelope depth. For a `LinearEnvelope` the stretch is
# linear (pure σ), so σᵉ = H/Lz is depth-independent: its k-halos come for free by broadcasting the
# halo-filled 2-D bottom field over k, and h = H. Evaluating the envelope at each stagger via `set!`
# (rather than interpolating) keeps σᵉ and h consistent stagger-by-stagger.
function materialize_envelopes!(grid::MultiEnvelopeGrid, bottom_height)
    Lz = grid.Lz

    bottomᶜᶜ = Field((Center(), Center(), nothing), grid); set!(bottomᶜᶜ, bottom_height); fill_halo_regions!(bottomᶜᶜ)
    bottomᶠᶜ = Field((Face(),   Center(), nothing), grid); set!(bottomᶠᶜ, bottom_height); fill_halo_regions!(bottomᶠᶜ)
    bottomᶜᶠ = Field((Center(), Face(),   nothing), grid); set!(bottomᶜᶠ, bottom_height); fill_halo_regions!(bottomᶜᶠ)
    bottomᶠᶠ = Field((Face(),   Face(),   nothing), grid); set!(bottomᶠᶠ, bottom_height); fill_halo_regions!(bottomᶠᶠ)

    parent(grid.z.σᶜᶜᵉ) .= parent(bottomᶜᶜ) ./ Lz
    parent(grid.z.σᶠᶜᵉ) .= parent(bottomᶠᶜ) ./ Lz
    parent(grid.z.σᶜᶠᵉ) .= parent(bottomᶜᶠ) ./ Lz
    parent(grid.z.σᶠᶠᵉ) .= parent(bottomᶠᶠ) ./ Lz

    parent(grid.z.hᶜᶜ) .= parent(bottomᶜᶜ)
    parent(grid.z.hᶠᶜ) .= parent(bottomᶠᶜ)
    parent(grid.z.hᶜᶠ) .= parent(bottomᶜᶠ)
    parent(grid.z.hᶠᶠ) .= parent(bottomᶠᶠ)

    if grid.z.formulation isa LinearEnvelope
        parent(grid.z.formulation.bottom) .= parent(bottomᶜᶜ)
    end

    return grid
end

#####
##### Monotone cubic Hermite (PCHIP) — smooth (C¹) transition between zones (Bruciaferri Eq. 5b).
##### Replaces the piecewise-constant (C⁰) σᵉ with a depth-smooth one that still passes exactly through the
##### envelope depths at zone interfaces. The Fritsch–Carlson slopes guarantee monotonicity ⇒ σᵉ > 0.
#####

# Knot slopes for a monotone cubic Hermite through (r, z), r strictly increasing.
function pchip_knot_slopes(r, z)
    n = length(r)
    δ = [(z[i+1] - z[i]) / (r[i+1] - r[i]) for i in 1:n-1]
    d = zeros(eltype(z), n)
    d[1] = δ[1]
    d[n] = δ[n-1]
    for i in 2:n-1
        if δ[i-1] * δ[i] > 0
            w1 = 2 * (r[i+1] - r[i]) + (r[i] - r[i-1])
            w2 = (r[i+1] - r[i]) + 2 * (r[i] - r[i-1])
            d[i] = (w1 + w2) / (w1 / δ[i-1] + w2 / δ[i])
        end
    end
    return d
end

# Evaluate the Hermite interpolant defined by knots (r, z) with slopes d at abscissa x.
function hermite_value(r, z, d, x)
    i = clamp(searchsortedlast(r, x), 1, length(r) - 1)
    h = r[i+1] - r[i]
    t = (x - r[i]) / h
    t² = t * t; t³ = t² * t
    return (2t³ - 3t² + 1) * z[i] + (t³ - 2t² + t) * h * d[i] +
           (-2t³ + 3t²) * z[i+1] + (t³ - t²) * h * d[i+1]
end

# Fill one stagger's σᵉ via the monotone spline. Operates on CPU parent arrays; `envelopes_parent` are the
# CPU envelope parents (shallowest → deepest), `rᶠ` the reference faces, `boundary_faces` the reference face
# indices at the zone interfaces (bottom → surface).
function fill_spline_stagger!(σ_parent, rᶠ, Hz, Nz, envelopes_parent, boundary_faces)
    n = length(envelopes_parent)
    r_knots = [rᶠ[boundary_faces[m]] for m in 1:n+1]
    z_knots = zeros(eltype(σ_parent), n + 1)
    ẑ = zeros(eltype(σ_parent), Nz + 1)
    for ci in axes(σ_parent, 1), cj in axes(σ_parent, 2)
        z_knots[1] = -envelopes_parent[n][ci, cj, 1]                 # bottom (deepest)
        for m in 2:n
            z_knots[m] = -envelopes_parent[n + 1 - m][ci, cj, 1]
        end
        z_knots[n+1] = 0                                             # surface
        d = pchip_knot_slopes(r_knots, z_knots)
        for k in 1:Nz+1
            ẑ[k] = hermite_value(r_knots, z_knots, d, rᶠ[k])
        end
        for k in 1:Nz
            σ_parent[ci, cj, k + Hz] = (ẑ[k+1] - ẑ[k]) / (rᶠ[k+1] - rᶠ[k])
        end
        for kp in 1:Hz
            σ_parent[ci, cj, kp] = σ_parent[ci, cj, Hz + 1]
            σ_parent[ci, cj, Nz + Hz + kp] = σ_parent[ci, cj, Nz + Hz]
        end
    end
    return σ_parent
end

# Multi-envelope: `envelope_heights` is a tuple of n depth functions (shallowest → deepest). Each zone i
# maps linearly between its bounding envelopes, so σᵉ = (physical zone thickness)/(reference zone thickness)
# is constant within the zone (surface zone occupies the top k-levels, since k increases upward). σᵉ and h
# are evaluated stagger-by-stagger from envelopes `set!` at each stagger (halo-safe), then the vertical
# halos extend the boundary zones. With `smooth_transitions=true` the piecewise-constant σᵉ is replaced by a
# monotone cubic Hermite (C¹) that still honours the envelope depths at every interface.
function materialize_envelopes!(grid::MultiEnvelopeGrid, envelope_heights::Tuple; smooth_transitions=false)
    f = grid.z.formulation
    f isa MultiEnvelope || throw(ArgumentError("a tuple of envelope_heights requires a MultiEnvelope formulation"))
    length(envelope_heights) == length(f.level_counts) ||
        throw(ArgumentError("number of envelope_heights must equal length(level_counts)"))
    sum(f.level_counts) == size(grid, 3) ||
        throw(ArgumentError("sum(level_counts) must equal Nz = $(size(grid, 3))"))

    Nz = size(grid, 3)
    Hz = grid.Hz
    rᶠ = on_architecture(CPU(), grid.z.cᵃᵃᶠ)

    # Reference face indices at the zone interfaces, bottom → surface (for the spline knots).
    L = f.level_counts
    boundary_faces = Vector{Int}(undef, length(L) + 1)
    boundary_faces[1] = 1
    accumulated = 0
    for m in 1:length(L)
        accumulated += L[length(L) + 1 - m]
        boundary_faces[m + 1] = 1 + accumulated
    end

    locations = ((Center(), Center(), nothing), (Face(),   Center(), nothing),
                 (Center(), Face(),   nothing), (Face(),   Face(),   nothing))
    σ_arrays  = (grid.z.σᶜᶜᵉ, grid.z.σᶠᶜᵉ, grid.z.σᶜᶠᵉ, grid.z.σᶠᶠᵉ)
    h_arrays  = (grid.z.hᶜᶜ,  grid.z.hᶠᶜ,  grid.z.hᶜᶠ,  grid.z.hᶠᶠ)

    for (loc, σ, h) in zip(locations, σ_arrays, h_arrays)
        envelopes = map(envelope_heights) do envelope_height
            envelope = Field(loc, grid)
            set!(envelope, envelope_height)
            fill_halo_regions!(envelope)
            envelope
        end

        if smooth_transitions
            envelopes_parent = map(e -> on_architecture(CPU(), parent(e)), envelopes)
            σ_cpu = on_architecture(CPU(), parent(σ))
            fill_spline_stagger!(σ_cpu, rᶠ, Hz, Nz, envelopes_parent, boundary_faces)
            parent(σ) .= on_architecture(architecture(grid), σ_cpu)
        else
            cum = 0
            for i in eachindex(envelope_heights)
                k_top = Nz - cum
                cum += f.level_counts[i]
                k_bot = Nz - cum + 1
                reference_thickness = rᶠ[k_top + 1] - rᶠ[k_bot]
                physical_thickness = i == 1 ? parent(envelopes[1]) :
                                              parent(envelopes[i]) .- parent(envelopes[i - 1])
                σ_zone = physical_thickness ./ reference_thickness
                for k in k_bot:k_top
                    @views parent(σ)[:, :, k + Hz] .= σ_zone[:, :, 1]
                end
            end
            for kp in 1:Hz
                @views parent(σ)[:, :, kp] .= parent(σ)[:, :, Hz + 1]
                @views parent(σ)[:, :, Nz + Hz + kp] .= parent(σ)[:, :, Nz + Hz]
            end
        end

        parent(h) .= parent(envelopes[end])
    end

    for (i, envelope_height) in enumerate(envelope_heights)
        envelope = Field((Center(), Center(), nothing), grid)
        set!(envelope, envelope_height)
        fill_halo_regions!(envelope)
        parent(f.envelopes[i]) .= parent(envelope)
    end

    return grid
end
