using Statistics

abstract type AbstractFeatureScaling end

#####
##### Zero-mean unit-variance feature scaling
#####

struct ZeroMeanUnitVarianceScaling{T} <: AbstractFeatureScaling
    μ :: T
    σ :: T
end

"""
    ZeroMeanUnitVarianceScaling(data)

Returns a feature scaler for `data` with zero mean and unit variance.
"""
function ZeroMeanUnitVarianceScaling(data)
    μ, σ = mean(data), std(data)
    return ZeroMeanUnitVarianceScaling(μ, σ)
end

scale(x, s::ZeroMeanUnitVarianceScaling) = (x .- s.μ) / s.σ
unscale(y, s::ZeroMeanUnitVarianceScaling) = s.σ * y .+ s.μ

#####
##### Min-max feature scaling
#####

struct MinMaxScaling{T} <: AbstractFeatureScaling
           a :: T
           b :: T
    data_min :: T
    data_max :: T
end

"""
    MinMaxScaling(data; a=0, b=1)

Returns a feature scaler for `data` with minimum `a` and `maximum `b`.
"""
function MinMaxScaling(data; a=0, b=1)
    data_min, data_max = extrema(data)
    return MinMaxScaling{typeof(data_min)}(a, b, data_min, data_max)
end

scale(x, s::MinMaxScaling) = s.a + (x - s.data_min) * (s.b - s.a) / (s.data_max - s.data_min)
unscale(y, s::MinMaxScaling) = s.data_min .+ (y .- s.a) * (s.data_max - s.data_min) / (s.b - s.a)

#####
##### Convenience functions
#####

(s::AbstractFeatureScaling)(x) = scale(x, s)
Base.inv(s::AbstractFeatureScaling) = y -> unscale(y, s)

struct DiffusivityScaling{T} <: AbstractFeatureScaling
    ν₀ :: T
    κ₀ :: T
    ν₁ :: T
    κ₁ :: T
end

function DiffusivityScaling(ν₀=1e-5, κ₀=1e-5, ν₁=0.1, κ₁=0.1)
    return DiffusivityScaling(ν₀, κ₀, ν₁, κ₁)
end

function scale(x, s::DiffusivityScaling)
    ν, κ = x
    ν₀, κ₀, ν₁, κ₁ = s.ν₀, s.κ₀, s.ν₁, s.κ₁
    return ν₀ + ν * ν₁, κ₀ + κ * κ₁
end

function unscale(y, s::DiffusivityScaling)
    ν, κ = y
    ν₀, κ₀, ν₁, κ₁ = s.ν₀, s.κ₀, s.ν₁, s.κ₁
    return (ν - ν₀) / ν₁, (κ - κ₀) / κ₁
end

(s::DiffusivityScaling)(x) = scale(x, s)
Base.inv(s::DiffusivityScaling) = y -> unscale(y, s)


