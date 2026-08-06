####
#### Averaging kernels for the SplitExplicitFreeSurface
####

# We provide several options to filter the barotropic mode during substepping.
# The choice of the filter impacts the stability, the accuracy and the order of the solution.
# All averaging kernels must be a function of `τ` (the barotropic substep)
#
# The conservation and consistency of the barotropic solution is ensured by the
# low order moments of the averaging weigths (wₘ) which should always obey
#
#  μ₀ = ∑wₘ   = 1
#  μ₁ = ∑wₘτₘ = 1
#
# The dissipation, dispersion, and order are determined by the higher moments of the weigths
#
#  μ₂ = ∑wₘ(τₘ - 1)²
#  μ₃ = ∑wₘ(τₘ - 1)³
#
# μ₂ > 0 implies diffusion of the barotropic mode and μ₃ ≠ 0 implies dispersion. To achieve second order
# (if the barotropic subtepping allows it, i.e. RK2 or RK3), necessarily μ₂ = 0, while
# for a third order barotropic substepping procedure (only using RK3), μ₃ = 0.
# The stability of kernels with μ₂ = 0 or μ₂ = μ₃ = 0 will be lower (at high stratification)
# than for kernels with μ₂ > 0, so we recommend to pair them only with a higher order
# barotropic timestepper where the order increase of the filter translates into an
# effective higher order barotropic substepping procedure.
#
# - `nothing`: no averaging, unstable but good for simple tests
# - `ConstantAveragingKernel()`:       μ₂ ≫ 0, very diffusive, very stable
# - `CosineAveragingKernel()`:         μ₂ > 0, μ₃ = 0, lower dispersion
# - `LowDissipationAveragingKernel()`: μ₂ = 0, μ₃ > 0, lowest diffusivity, 2nd order, best for both FB and RK2
# - `SymmetricTrigAveragingKernel()`:  μ₂ = 0, μ₃ = 0, 3rd order, cheapest μ₂ = μ₃ = 0 window
# - `WideTrig74AveragingKernel()`:     μ₂ = 0, μ₃ = 0, 3rd order, wider window (7/4), needs substeps % 8 == 0
# - `WideTrig2AveragingKernel()`:      μ₂ = 0, μ₃ = 0, 3rd order, widest window (2), needs substeps % 4 == 0
# - `OptimizedSymmetricAveragingKernel()`:  μ₂ = 0, μ₃ = 0, 3rd order, stratification-robust
# - `OptimizedAsymmetricAveragingKernel()`: μ₂ = 0, μ₃ = 0, 3rd order, stratification-robust, lowest TV — DEFAULT
#
# Stability of a μ₂ = μ₃ = 0 kernel is governed by two independent things, and it is a mistake to attribute
# either to the other:
#
#   1. the STOP BAND of the kernel, which controls the barotropic-barotropic resonance at μ₀ = πα₀ ≈ 3.2;
#   2. the DISSIPATION OF THE BAROTROPIC SUBSTEP itself, which controls a large-scale (μ₀ ≈ 0.4) residual.
#
# These act on disjoint parts of the spectrum, so neither a better kernel nor a better substep alone is
# sufficient. In particular a Runge-Kutta substep does NOT rescue a kernel with a weak stop band: measured in
# the Each-Stage coupling that `SplitRungeKuttaTimeStepper` uses, `WideTrig74`/`WideTrig2` are formally
# unstable at the μ₀ ≈ 3.2 resonance once the stratification reaches N ≈ 2.5e-2 s⁻¹ (max|λ|-1 = +3.2e-2 and
# +1.4e-2), and swapping a forward-backward substep for RK3 makes them *worse* (+3.5e-2, +1.7e-2), not better.
# Below N ≈ 2.0e-2 s⁻¹ every kernel here is fine, so this only bites at very strong stratification.
#
# The two `Optimized*` kernels come from a minimax search over the shape parameters, minimising the worst
# Each-Stage amplification over μ₀ and over N = 1.0…2.5e-2 s⁻¹ subject to μ₀ = 1 and μ₁ = μ₂ = μ₃ = 0. Both
# stay stable across that whole range (+1.0e-6 with a forward-backward substep, -2.8e-12 with RK3) at the same
# cost as `WideTrig74` (M★ = 42 at substeps = 48). `OptimizedAsymmetric` is the default: it matches
# `OptimizedSymmetric` in stability but has a much lower total variation (0.200 against 0.471), and since
# |Ŵ(x)| ≤ TV/|sin(x/2)| that is genuine high-wavenumber margin.
#
# The `WideTrig` shapes owe μ₃ = 0 to being symmetric about their own window centre, and that symmetry survives
# the discretisation only when the window edges land on the τ grid (τₖ = 2k/substeps). The window [1/2, Wend]
# needs both substeps/4 and Wend·substeps/2 integral, i.e.
#
#   WideTrig74 : substeps % 8 == 0
#   WideTrig2  : substeps % 4 == 0
#
# Off the grid the retained stencil is lopsided, μ₃ ≠ 0, and the kernel silently drops from third to second
# order (WideTrig74 at substeps = 36 gives μ₃ = 1.1e-2 rather than 0). `SymmetricTrigAveragingKernel` and
# `OptimizedSymmetricAveragingKernel` need only an even `substeps`: their windows are symmetric about τ = 1,
# which is then a grid point, so the trimmed stencil stays symmetric. `OptimizedAsymmetricAveragingKernel`
# needs no condition at all — it solves μ₀ = 1, μ₁ = μ₂ = μ₃ = 0 directly on whatever τ grid it is given, so
# the moments are exact for every `substeps` rather than inherited from a continuous symmetry.
#
# A kernel tuned at one `substeps` is not automatically valid at another: the sampling can mask a stop-band
# leak that survives refinement. Every kernel here was checked to be flat in `substeps` over 32…96 before
# being admitted, and any kernel added later should be checked the same way.

struct ConstantAveragingKernel <: Function end
struct CosineAveragingKernel   <: Function end
struct LowDissipationAveragingKernel <: Function end
struct SymmetricTrigAveragingKernel <: Function end
struct WideTrig74AveragingKernel <: Function end
struct WideTrig2AveragingKernel  <: Function end
struct OptimizedSymmetricAveragingKernel  <: Function end
struct OptimizedAsymmetricAveragingKernel <: Function end

# Generic weights from general `averaging_kernel`s
@inline function weights_from_substeps(FT, substeps, averaging_kernel)
    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]

    averaging_weights = map(averaging_kernel, τᶠ[2:end])
    # Find the latest allowable weight
    M★ = something(findlast(>(0), averaging_weights), firstindex(averaging_weights))

    trimmed_weights = averaging_weights[1:M★]
    trimmed_weights ./= sum(trimmed_weights)

    # Rescale the substep size so the trimmed weights' first moment lands exactly on the baroclinic step
    barycenter = sum(trimmed_weights .* (1:M★)) * Δτ
    Δτ = Δτ / barycenter

    transport_weights = [sum(trimmed_weights[i:M★]) for i in 1:M★] .* Δτ

    return FT(Δτ), map(FT, tuple(trimmed_weights...)), map(FT, tuple(transport_weights...))
end

# If we do not have an averaging kernel, we take the endpoint
@inline function weights_from_substeps(FT, substeps, ::Nothing)
    fractional_step   = one(FT) / substeps
    averaging_weights = ntuple(m -> m == substeps ? one(FT) : zero(FT), substeps)
    transport_weights = ntuple(_ -> one(FT) / substeps, substeps)
    return fractional_step, averaging_weights, transport_weights
end

@inline (::CosineAveragingKernel)(τ::FT) where FT = τ ≥ 0.5 && τ ≤ 1.5 ? convert(FT, 1 + cos(2π * (τ - 1))) : zero(FT)
@inline (::ConstantAveragingKernel)(τ::FT) where FT = convert(FT, 1)

# (p = 2, q = 4) minimize dispersion error from Shchepetkin and McWilliams (2005): https://doi.org/10.1016/j.ocemod.2004.08.002
@inline function weights_from_substeps(FT, substeps, ::LowDissipationAveragingKernel)
    r = low_dispersion_coefficient(FT, substeps)
    return weights_from_substeps(FT, substeps, τ -> averaging_shape_function(τ; p = 2, q = 4, r))
end

function averaging_second_moment(FT, substeps, r)
    Δτ, w, _ = weights_from_substeps(FT, substeps, τ -> averaging_shape_function(τ; r = convert(FT, r)))
    return sum(w[m] * (m * Δτ - 1)^2 for m in eachindex(w))
end

@inline function averaging_shape_function(τ::FT; p = 2, q = 4, r = FT(0.18927)) where FT
    τ₀ = (p + 2) * (p + q + 2) / (p + 1) / (p + q + 1)
    return (τ / τ₀)^p * (1 - (τ / τ₀)^q) - r * (τ / τ₀)
end

function low_dispersion_coefficient(FT, Ns)
    f(r) = averaging_second_moment(FT, Ns, r)
    rₗ, rₕ = FT(0.18927), FT(0.9)
    fₗ, fₕ = f(rₗ), f(rₕ)
    if fₗ * fₕ > 0
        return FT(0.285)
    end
    for _ in 1:80
        rₘ = (rₗ + rₕ) / 2
        fₘ = f(rₘ)
        if fₗ * fₘ ≤ 0
            rₕ, fₕ = rₘ, fₘ
        else
            rₗ, fₗ = rₘ, fₘ
        end
    end
    return (rₗ + rₕ) / 2
end

function weights_from_substeps(FT, substeps, ::SymmetricTrigAveragingKernel)
    a2 = FT(-27//20)   # -1.35
    a3 = FT(1//2)      # +0.5
    a1 = symmetric_trig_first_amplitude(FT, substeps; a2, a3)
    Δτ, w, M★ = symmetric_trig_weights(FT, substeps, a1, a2, a3)
    t = [Δτ * sum(@view w[m:M★]) for m in 1:M★]
    return FT(Δτ), map(FT, tuple(w...)), map(FT, tuple(t...))
end

@inline symmetric_trig_shape(τ::FT, a1, a2, a3) where FT =
    ifelse((τ ≥ FT(1//2)) & (τ ≤ FT(3//2)), 1 + a1 * cospi(2*(τ-1)) + a2 * cospi(4*(τ-1)) + a3 * cospi(6*(τ-1)), zero(FT))

function symmetric_trig_weights(FT, substeps, a1, a2, a3)
    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]
    raw = map(τ -> symmetric_trig_shape(τ, FT(a1), FT(a2), FT(a3)), τᶠ[2:end])
    M★ = findlast(!=(0), raw)
    w  = collect(raw[1:M★])
    w ./= sum(w)
    barycenter = sum(w .* (1:M★)) * Δτ
    Δτ = Δτ / barycenter
    return FT(Δτ), w, M★
end

function symmetric_trig_second_moment(FT, substeps, a1, a2, a3)
    Δτ, w, M★ = symmetric_trig_weights(FT, substeps, a1, a2, a3)
    return sum(w[m] * (m * Δτ - 1)^2 for m in 1:M★)
end

# Bisect `a1` so the (normalized, barycenter-centered) second moment vanishes; higher harmonics fixed.
function symmetric_trig_first_amplitude(FT, substeps; a2 = FT(-27//20), a3 = FT(1//2))
    f(a1) = symmetric_trig_second_moment(FT, substeps, a1, a2, a3)
    lo, hi = FT(1//5), FT(16//5)
    flo = f(lo)
    for _ in 1:70
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid ≤ 0
            hi = mid
        else
            lo, flo = mid, fmid
        end
    end
    return (lo + hi) / 2
end

# Warn (once per kernel and substep count) when the window edge misses the τ grid and μ₃ ≠ 0 silently.
function validate_wide_trig_substeps(substeps, name, required)
    if substeps % required != 0
        @warn string(name, " needs `substeps` to be a multiple of ", required, " so that its averaging ",
                     "window edge lands on the τ grid. With substeps = ", substeps, " the retained stencil ",
                     "is asymmetric about the window centre, so μ₃ ≠ 0 and the kernel is second- rather than ",
                     "third-order accurate. Use substeps = ", required * cld(substeps, required), ".") _id=Symbol(name, substeps) maxlog=1
    end
    return nothing
end

function weights_from_substeps(FT, substeps, ::WideTrig74AveragingKernel)
    validate_wide_trig_substeps(substeps, "WideTrig74AveragingKernel", 8)
    return wide_trig_weights_from_substeps(FT, substeps, 7//4, (FT(-1), FT(1//2), FT(-1), FT(1//2)))
end

function weights_from_substeps(FT, substeps, ::WideTrig2AveragingKernel)
    validate_wide_trig_substeps(substeps, "WideTrig2AveragingKernel", 4)
    return wide_trig_weights_from_substeps(FT, substeps, 2, (FT(-1), FT(1//2), FT(-1), FT(4//5)))
end

function wide_trig_weights_from_substeps(FT, substeps, Wend, higher)
    a1 = wide_trig_first_amplitude(FT, substeps, Wend, higher)
    Δτ, w, M★ = wide_trig_weights(FT, substeps, Wend, (a1, higher...))
    t = [Δτ * sum(@view w[m:M★]) for m in 1:M★]
    return FT(Δτ), map(FT, tuple(w...)), map(FT, tuple(t...))
end

@inline function wide_trig_shape(τ::FT, Wend, a) where FT
    c = (1//2 + Wend) / 2
    L = Wend - 1//2
    s = one(FT)
    for k in eachindex(a)
        s += a[k] * cospi(2k * (τ - c) / L)
    end
    return ifelse((τ ≥ FT(1//2)) & (τ ≤ FT(Wend)), s, zero(FT))
end

function wide_trig_weights(FT, substeps, Wend, a)
    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]
    raw = map(τ -> wide_trig_shape(τ, FT(Wend), map(FT, a)), τᶠ[2:end])
    M★ = findlast(!=(0), raw)
    w  = collect(raw[1:M★])
    w ./= sum(w)
    barycenter = sum(w .* (1:M★)) * Δτ
    Δτ = Δτ / barycenter
    return FT(Δτ), w, M★
end

function wide_trig_second_moment(FT, substeps, Wend, a)
    Δτ, w, M★ = wide_trig_weights(FT, substeps, Wend, a)
    return sum(w[m] * (m * Δτ - 1)^2 for m in 1:M★)
end

function wide_trig_first_amplitude(FT, substeps, Wend, higher)
    f(a1) = wide_trig_second_moment(FT, substeps, Wend, (a1, higher...))
    lo, hi = FT(-3), FT(4)
    flo = f(lo)
    for _ in 1:80
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid ≤ 0
            hi = mid
        else
            lo, flo = mid, fmid
        end
    end
    return (lo + hi) / 2
end

#####
##### Minimax-optimized μ₂ = μ₃ = 0 kernels
#####

# shape(s) = 1 + ∑ₖ aₖ cos(πks/R) on |s| ≤ R, with s = τ - 1, so the shape is symmetric about τ = 1 and
# μ₁ = μ₃ = 0 follow whenever τ = 1 is a grid point (any even `substeps`). a₅ is fixed by shape(±R) = 0, so
# the shape tapers smoothly to zero instead of being truncated — that taper is what gives the stop band its
# decay. a₁ is then bisected so that μ₂ = 0. R and a₂…a₄ come from the minimax search.
const optimized_symmetric_radius = 0.752661
const optimized_symmetric_higher_amplitudes = (-1.289381, -0.674748, 0.605401)

@inline function optimized_symmetric_amplitudes(a1, higher)
    a = (a1, higher...)
    return (a..., 1 + sum(a[k] * (-1)^k for k in 1:4))
end

@inline function optimized_symmetric_shape(τ::FT, R, a) where FT
    s = τ - 1
    v = one(FT)
    for k in eachindex(a)
        v += a[k] * cospi(k * s / R)
    end
    return ifelse(abs(s) ≤ FT(R), v, zero(FT))
end

function optimized_symmetric_weights(FT, substeps, R, a)
    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]
    raw = map(τ -> optimized_symmetric_shape(τ, FT(R), map(FT, a)), τᶠ[2:end])
    M★ = findlast(!=(0), raw)
    w  = collect(raw[1:M★])
    w ./= sum(w)
    barycenter = sum(w .* (1:M★)) * Δτ
    Δτ = Δτ / barycenter
    return FT(Δτ), w, M★
end

function optimized_symmetric_second_moment(FT, substeps, R, a)
    Δτ, w, M★ = optimized_symmetric_weights(FT, substeps, R, a)
    return sum(w[m] * (m * Δτ - 1)^2 for m in 1:M★)
end

function optimized_symmetric_first_amplitude(FT, substeps, R, higher)
    f(a1) = optimized_symmetric_second_moment(FT, substeps, R, optimized_symmetric_amplitudes(a1, higher))
    lo, hi = FT(-4), FT(4)
    flo = f(lo)
    for _ in 1:80
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid ≤ 0
            hi = mid
        else
            lo, flo = mid, fmid
        end
    end
    return (lo + hi) / 2
end

function weights_from_substeps(FT, substeps, ::OptimizedSymmetricAveragingKernel)
    R = optimized_symmetric_radius
    a1 = optimized_symmetric_first_amplitude(FT, substeps, R, optimized_symmetric_higher_amplitudes)
    a  = optimized_symmetric_amplitudes(a1, optimized_symmetric_higher_amplitudes)
    Δτ, w, M★ = optimized_symmetric_weights(FT, substeps, R, a)
    t = [Δτ * sum(@view w[m:M★]) for m in 1:M★]
    return FT(Δτ), map(FT, tuple(w...)), map(FT, tuple(t...))
end

# v(τ) = (1 - u²)^p ∑_{k=0}^{4} bₖ uᵏ with u = (τ - c)/R. The moments are LINEAR in b, so μ₀ = 1 and
# μ₁ = μ₂ = μ₃ = 0 are imposed directly as four equations on the five bₖ, on whatever τ grid is supplied.
# Nothing is inherited from a continuous symmetry, so the moments are exact for every `substeps` and no
# grid-alignment condition arises. The (1-u²)^p factor tapers the shape to zero at both ends of its support,
# which is what keeps the total variation — and hence the stop band, via |Ŵ(x)| ≤ TV/|sin(x/2)| — so low.
const optimized_asymmetric_radius = 0.864972
const optimized_asymmetric_center = 0.909165
const optimized_asymmetric_power  = 0.595115

function weights_from_substeps(FT, substeps, ::OptimizedAsymmetricAveragingKernel)
    τᶠ = range(FT(0), FT(2), length = substeps+1)
    Δτ = τᶠ[2] - τᶠ[1]
    τ  = collect(τᶠ[2:end])
    s  = τ .- 1

    R, c, p = FT(optimized_asymmetric_radius), FT(optimized_asymmetric_center), FT(optimized_asymmetric_power)
    u   = (τ .- c) ./ R
    tap = [abs(uᵐ) ≤ 1 ? (1 - uᵐ^2)^p : zero(FT) for uᵐ in u]
    B   = [tap .* u.^k for k in 0:4]

    A = zeros(FT, 4, 5)
    for k in 1:5, (j, sʲ) in enumerate((s.^0, s, s.^2, s.^3))
        A[j, k] = sum(B[k] .* sʲ)
    end
    b = A \ FT[1, 0, 0, 0]

    raw = sum(b[k] .* B[k] for k in 1:5)
    M★ = findlast(!=(0), raw)
    w  = collect(raw[1:M★])
    w ./= sum(w)
    barycenter = sum(w .* (1:M★)) * Δτ
    Δτ = Δτ / barycenter
    t  = [Δτ * sum(@view w[m:M★]) for m in 1:M★]
    return FT(Δτ), map(FT, tuple(w...)), map(FT, tuple(t...))
end
