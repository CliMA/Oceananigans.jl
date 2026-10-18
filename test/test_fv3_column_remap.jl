using Test
using Random

isdefined(@__MODULE__, :remap_column!) || include(joinpath(@__DIR__, "..", "src", "VerticalRemapping", "fv3_column_remap.jl"))

function random_interfaces(rng, Nz; ptop = 100.0, psurface = 1.0e5, thin = false)
    δ = rand(rng, Nz) .+ 0.2
    thin && (δ[rand(rng, 2:Nz-1)] = 1e-6)
    p = ptop .+ (psurface - ptop) .* cumsum([0; δ]) ./ sum(δ)
    p[end] = psurface
    return p
end

content(q, p) = sum(q[k] * (p[k+1] - p[k]) for k in eachindex(q))

profiles = [FV3ScalarProfile(; kord = 8, iv = 1, minimum_value = 184.0),
            FV3ScalarProfile(; kord = 8, iv = 0, minimum_value = 0.0),
            FV3CubicSplineProfile(; kord = 8, iv = -1),
            FV3ScalarProfile(; kord = 10, iv = 0, minimum_value = 0.0),
            FV3CubicSplineProfile(; kord = 9, iv = 1),
            FV3CubicSplineProfile(; kord = 17, iv = 1)]

@testset "FV3 column remap: manufactured columns" begin
    rng = Xoshiro(20261018)
    Nz = 32
    w = FV3RemapWorkspace(Float64, Nz)
    q₂ = zeros(Nz)

    for profile in profiles, trial in 1:20
        p₁ = random_interfaces(rng, Nz; thin = isodd(trial))
        p₂ = random_interfaces(rng, Nz)

        @testset "constant preservation $(typeof(profile)) kord=$(profile.kord) iv=$(profile.iv)" begin
            remap_column!(q₂, p₂, fill(3.25, Nz), p₁, 3.25, profile, w)
            @test maximum(abs, q₂ .- 3.25) <= 64 * eps(3.25)
        end

        q₁ = profile.iv == 0 ? rand(rng, Nz) .* (rand(rng, Nz) .> 0.3) : 280 .+ 20 .* randn(rng, Nz)
        remap_column!(q₂, p₂, q₁, p₁, NaN, profile, w)
        @test all(isfinite, q₂)
        scale = sum(abs(q₁[k]) * (p₁[k+1] - p₁[k]) for k in 1:Nz)
        @test abs(content(q₂, p₂) - content(q₁, p₁)) <= 1e-12 * scale
        profile.iv == 0 && @test minimum(q₂) >= 0

        remap_column!(q₂, p₁, q₁, p₁, NaN, profile, w)
        @test maximum(abs, q₂ .- q₁) <= 1e-10 * maximum(abs, q₁)
    end
end

@testset "FV3 column remap: map1_q2 normalization uses the supplied thickness" begin
    rng = Xoshiro(7)
    Nz = 16
    w = FV3RemapWorkspace(Float64, Nz)
    p₁ = random_interfaces(rng, Nz); p₂ = random_interfaces(rng, Nz)
    q₁ = rand(rng, Nz)
    profile = FV3ScalarProfile(; kord = 8, iv = 0, minimum_value = 0.0)
    a = remap_column!(zeros(Nz), p₂, q₁, p₁, NaN, profile, w)
    b = remap_column!(zeros(Nz), p₂, q₁, p₁, NaN, profile, w; target_thickness = 2 .* diff(p₂))
    fractional = [k for k in 1:Nz if !any(l -> p₁[l] <= p₂[k] && p₂[k+1] <= p₁[l+1], 1:Nz)]
    @test all(b[k] ≈ a[k] / 2 for k in fractional)
end

@testset "fortran_min/fortran_max tie semantics" begin
    @test signbit(fortran_max(-0.0, 0.0))
    @test !signbit(fortran_max(0.0, -0.0))
    @test signbit(fortran_min(-0.0, 0.0))
    @test !signbit(fortran_min(0.0, -0.0))
end
