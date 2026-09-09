####
#### Averaging kernels for the SplitExplicitFreeSurface
####
#
# An averaging kernel is a function of τ, the barotropic substep normalized by the baroclinic step.
# The moments of the resulting weights wₘ set the order of the filtered barotropic mode:
#
#   μ₀ = ∑ wₘ = 1 and μ₁ = ∑ wₘ τₘ = 1 (conservation and consistency, always enforced)
#   μ₂ = ∑ wₘ (τₘ - 1)²                (diffusion)
#   μ₃ = ∑ wₘ (τₘ - 1)³                (dispersion)
#
# μ₂ = 0 gives second order and μ₂ = μ₃ = 0 third order, realized only if the barotropic substep is at
# least as accurate (`RungeKutta3Scheme`). Kernels with μ₂ = 0 are less stable at strong stratification.
#
# - `nothing`: no averaging, only for simple tests
# - `ConstantAveragingKernel()`:            μ₂ ≫ 0
# - `CosineAveragingKernel()`:              μ₂ > 0, μ₃ = 0
# - `averaging_shape_function`:             μ₂ > 0, Shchepetkin and McWilliams (2005), the default
# - `LowDissipationAveragingKernel()`:      μ₂ = 0, the same shape with `r` retuned
# - `SymmetricTrigAveragingKernel()`:       μ₂ = μ₃ = 0, narrowest window
# - `WideTrigAveragingKernel()`:            μ₂ = μ₃ = 0, wider window, needs substeps % 8 == 0
# - `OptimizedSymmetricAveragingKernel()`:  μ₂ = μ₃ = 0, minimax shape, stratification robust
# - `OptimizedAsymmetricAveragingKernel()`: μ₂ = μ₃ = 0, minimax shape, lowest total variation

struct ConstantAveragingKernel       <: Function end
struct CosineAveragingKernel         <: Function end
struct LowDissipationAveragingKernel <: Function end
struct SymmetricTrigAveragingKernel  <: Function end
struct WideTrigAveragingKernel       <: Function end
struct OptimizedSymmetricAveragingKernel  <: Function end
struct OptimizedAsymmetricAveragingKernel <: Function end

#####
##### From a shape function to the substep weights
#####

# Sample `shape` on the substep grid, keep the leading weights that `retain` accepts, normalize them, and
# rescale the substep so that μ₁ = 1.
function trimmed_weights(FT, substeps, shape, retain = !=(0))
    τᶠ  = range(FT(0), FT(2), length = substeps+1)
    raw = map(shape, τᶠ[2:end])
    M★  = something(findlast(retain, raw), firstindex(raw))

    w = collect(raw[1:M★])
    w ./= sum(w)

    Δτ = τᶠ[2] - τᶠ[1]
    return FT(Δτ / (sum(w .* (1:M★)) * Δτ)), w
end

@inline kernel_moment(Δτ, w, p) = sum(w[m] * (m * Δτ - 1)^p for m in eachindex(w))

# The transport weights are the reversed cumulative sum of the averaging weights.
function weights_from_shape(FT, substeps, shape, retain = !=(0))
    Δτ, w = trimmed_weights(FT, substeps, shape, retain)
    t = [Δτ * sum(@view w[m:end]) for m in eachindex(w)]
    return Δτ, map(FT, tuple(w...)), map(FT, tuple(t...))
end

# Bisect the leading amplitude of `shape(a)` so that μ₂ vanishes; the higher amplitudes are fixed.
function vanishing_second_moment(FT, substeps, shape, lo, hi, retain = !=(0))
    μ₂(a) = kernel_moment(trimmed_weights(FT, substeps, shape(a), retain)..., 2)

    flo = μ₂(lo)
    for _ in 1:80
        mid  = (lo + hi) / 2
        fmid = μ₂(mid)
        if flo * fmid ≤ 0
            hi = mid
        else
            lo, flo = mid, fmid
        end
    end

    return (lo + hi) / 2
end

# A user-supplied kernel is trimmed where it turns negative, rather than at the end of its support.
weights_from_substeps(FT, substeps, averaging_kernel) = weights_from_shape(FT, substeps, averaging_kernel, >(0))

# Without an averaging kernel we take the endpoint.
@inline function weights_from_substeps(FT, substeps, ::Nothing)
    fractional_step   = one(FT) / substeps
    averaging_weights = ntuple(m -> m == substeps ? one(FT) : zero(FT), substeps)
    transport_weights = ntuple(_ -> one(FT) / substeps, substeps)
    return fractional_step, averaging_weights, transport_weights
end

#####
##### Shchepetkin and McWilliams (2005), and its μ₂ = 0 retuning
#####

@inline (::CosineAveragingKernel)(τ::FT) where FT = τ ≥ 0.5 && τ ≤ 1.5 ? convert(FT, 1 + cos(2π * (τ - 1))) : zero(FT)
@inline (::ConstantAveragingKernel)(τ::FT) where FT = convert(FT, 1)

# (p = 2, q = 4) minimize dispersion error from Shchepetkin and McWilliams (2005): https://doi.org/10.1016/j.ocemod.2004.08.002
@inline function averaging_shape_function(τ::FT; p = 2, q = 4, r = FT(0.18927)) where FT
    τ₀ = (p + 2) * (p + q + 2) / (p + 1) / (p + q + 1)
    return (τ / τ₀)^p * (1 - (τ / τ₀)^q) - r * (τ / τ₀)
end

function weights_from_substeps(FT, substeps, ::LowDissipationAveragingKernel)
    shape(r) = τ -> averaging_shape_function(τ; r = convert(FT, r))
    μ₂(r) = kernel_moment(trimmed_weights(FT, substeps, shape(r), >(0))..., 2)

    rₗ, rₕ = FT(0.18927), FT(0.9)
    r = μ₂(rₗ) * μ₂(rₕ) > 0 ? FT(0.285) : vanishing_second_moment(FT, substeps, shape, rₗ, rₕ, >(0))

    return weights_from_shape(FT, substeps, shape(r), >(0))
end

#####
##### Trigonometric μ₂ = μ₃ = 0 windows
#####

# Both owe μ₃ = 0 to being symmetric about their own centre, and that symmetry survives the discretisation
# only when the window edges land on the substep grid τₖ = 2k/substeps. The symmetric window is centred on
# τ = 1, a grid point for any even `substeps`; the wide one is centred on τ = 9/8 and is not.
required_substep_multiple(averaging_kernel) = 1
required_substep_multiple(::WideTrigAveragingKernel) = 8

@inline symmetric_trig_shape(τ::FT, a1, a2, a3) where FT =
    ifelse((τ ≥ FT(1//2)) & (τ ≤ FT(3//2)), 1 + a1 * cospi(2*(τ-1)) + a2 * cospi(4*(τ-1)) + a3 * cospi(6*(τ-1)), zero(FT))

function weights_from_substeps(FT, substeps, ::SymmetricTrigAveragingKernel)
    a2, a3 = FT(-27//20), FT(1//2)
    shape(a1) = τ -> symmetric_trig_shape(τ, FT(a1), a2, a3)
    a1 = vanishing_second_moment(FT, substeps, shape, FT(1//5), FT(16//5))
    return weights_from_shape(FT, substeps, shape(a1))
end

const wide_trig_window_end = 7//4

@inline function wide_trig_shape(τ::FT, a) where FT
    c = FT((1//2 + wide_trig_window_end) / 2)
    L = FT(wide_trig_window_end - 1//2)
    s = one(FT)
    for k in eachindex(a)
        s += a[k] * cospi(2k * (τ - c) / L)
    end
    return ifelse((τ ≥ FT(1//2)) & (τ ≤ FT(wide_trig_window_end)), s, zero(FT))
end

function weights_from_substeps(FT, substeps, ::WideTrigAveragingKernel)
    higher = (FT(-1), FT(1//2), FT(-1), FT(1//2))
    shape(a1) = τ -> wide_trig_shape(τ, (FT(a1), higher...))
    a1 = vanishing_second_moment(FT, substeps, shape, FT(-3), FT(4))
    return weights_from_shape(FT, substeps, shape(a1))
end

#####
##### Minimax-optimized μ₂ = μ₃ = 0 kernels
#####

# shape(s) = 1 + ∑ₖ aₖ cos(πks/R) on |s| ≤ R, with s = τ - 1. a₅ is fixed by shape(±R) = 0 so the shape
# tapers to zero rather than being truncated, and a₁ is bisected so that μ₂ = 0.
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

function weights_from_substeps(FT, substeps, ::OptimizedSymmetricAveragingKernel)
    R = FT(optimized_symmetric_radius)
    shape(a1) = τ -> optimized_symmetric_shape(τ, R, map(FT, optimized_symmetric_amplitudes(a1, optimized_symmetric_higher_amplitudes)))
    a1 = vanishing_second_moment(FT, substeps, shape, FT(-4), FT(4))
    return weights_from_shape(FT, substeps, shape(a1))
end

# v(τ) = (1 - u²)^p ∑_{k=0}^{4} bₖ uᵏ with u = (τ - c)/R. The moments are linear in b, so μ₀ = 1 and
# μ₁ = μ₂ = μ₃ = 0 are four equations on the five bₖ, solved on whatever substep grid is supplied: the
# moments are then exact for every `substeps`, with no grid-alignment condition.
const optimized_asymmetric_radius = 0.864972
const optimized_asymmetric_center = 0.909165
const optimized_asymmetric_power  = 0.595115

function weights_from_substeps(FT, substeps, ::OptimizedAsymmetricAveragingKernel)
    R, c, p = FT(optimized_asymmetric_radius), FT(optimized_asymmetric_center), FT(optimized_asymmetric_power)
    basis(τ, k) = (u = (τ - c) / R; abs(u) ≤ 1 ? (1 - u^2)^p * u^k : zero(FT))

    τᶠ = collect(range(FT(0), FT(2), length = substeps+1)[2:end])
    A  = FT[sum(basis.(τᶠ, k) .* (τᶠ .- 1).^(j-1)) for j in 1:4, k in 0:4]
    b  = A \ FT[1, 0, 0, 0]

    return weights_from_shape(FT, substeps, τ -> sum(b[k+1] * basis(τ, k) for k in 0:4))
end
