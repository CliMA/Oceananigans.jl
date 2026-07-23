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
# - `SymmetricTrigAveragingKernel()`:  μ₂ = 0, μ₃ = 0, lowest diffusivity overall, 3rd order, best for RK3
# - `WideTrig74AveragingKernel()`:     μ₂ = 0, μ₃ = 0, 3rd order, stratification-robust (wider window, 7/4), best for RK3
# - `WideTrig2AveragingKernel()`:      μ₂ = 0, μ₃ = 0, 3rd order, most stratification-robust (widest window, 2), best for RK3
#
# `SymmetricTrigAveragingKernel` loses stability at strong stratification; the two `WideTrig` kernels keep its
# μ₂ = μ₃ = 0 (3rd order) while widening the averaging window, which deepens μ₄ (the low-frequency dissipation)
# enough to stay stable there — at the cost of more barotropic substeps (M★ = (Wend/2)·substeps).

struct ConstantAveragingKernel <: Function end
struct CosineAveragingKernel   <: Function end
struct LowDissipationAveragingKernel <: Function end
struct SymmetricTrigAveragingKernel <: Function end
struct WideTrig74AveragingKernel <: Function end
struct WideTrig2AveragingKernel  <: Function end

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

weights_from_substeps(FT, substeps, ::WideTrig74AveragingKernel) = wide_trig_weights_from_substeps(FT, substeps, 7//4, (FT(-1), FT(1//2), FT(-1), FT(1//2)))
weights_from_substeps(FT, substeps, ::WideTrig2AveragingKernel) = wide_trig_weights_from_substeps(FT, substeps, 2, (FT(-1), FT(1//2), FT(-1), FT(4//5)))

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
