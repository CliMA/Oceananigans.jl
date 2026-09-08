include("dependencies_for_runtests.jl")

using Oceananigans.Advection: smoothness_indicator
using Random

# Smoothness indicator coefficients tabulated in the literature (Balsara & Shu, 2000, for orders 7 and 9), in
# the value form β = ∑ᵢ ψᵢ (Cᵢᵢ ψᵢ + ∑ⱼ₌ᵢ₊₁ Cᵢⱼ ψⱼ) over the stencil values ψ ordered from left to right.
# The tables are exact: they are integers divided by `reference_smoothness_denominator[buffer]`.
const reference_smoothness_denominator = Dict(2 => 1, 3 => 1, 4 => 1000, 5 => 100000, 6 => 10000000)

const reference_smoothness_coefficients = Dict(
    (2, 0) => (1, -2, 1),
    (2, 1) => (1, -2, 1),
    (3, 0) => (10, -31, 11, 25, -19, 4),
    (3, 1) => (4, -13, 5, 13, -13, 4),
    (3, 2) => (4, -19, 11, 25, -31, 10),
    (4, 0) => (2107, -9402, 7042, -1854, 11003, -17246, 4642, 7043, -3882, 547),
    (4, 1) => (547, -2522, 1922, -494, 3443, -5966, 1602, 2843, -1642, 267),
    (4, 2) => (267, -1642, 1602, -494, 2843, -5966, 1922, 3443, -2522, 547),
    (4, 3) => (547, -3882, 4642, -1854, 7043, -17246, 7042, 11003, -9402, 2107),
    (5, 0) => (107918, -649501, 758823, -411487, 86329, 1020563, -2462076, 1358458, -288007, 1521393, -1704396, 364863, 482963, -208501, 22658),
    (5, 1) => (22658, -140251, 165153, -88297, 18079, 242723, -611976, 337018, -70237, 406293, -464976, 99213, 138563, -60871, 6908),
    (5, 2) => (6908, -51001, 67923, -38947, 8209, 104963, -299076, 179098, -38947, 231153, -299076, 67923, 104963, -51001, 6908),
    (5, 3) => (6908, -60871, 99213, -70237, 18079, 138563, -464976, 337018, -88297, 406293, -611976, 165153, 242723, -140251, 22658),
    (5, 4) => (22658, -208501, 364863, -288007, 86329, 482963, -1704396, 1358458, -411487, 1521393, -2462076, 758823, 1020563, -649501, 107918),
    (6, 0) => (6150211, -47460464, 76206736, -63394124, 27060170, -4712740, 94851237, -311771244, 262901672, -113206788, 19834350, 260445372, -444003904, 192596472, -33918804, 190757572, -166461044, 29442256, 36480687, -12950184, 1152561),
    (6, 1) => (1152561, -9117992, 14742480, -12183636, 5134574, -880548, 19365967, -65224244, 55053752, -23510468, 4067018, 56662212, -97838784, 42405032, -7408908, 43093692, -37913324, 6694608, 8449957, -3015728, 271779),
    (6, 2) => (271779, -2380800, 4086352, -3462252, 1458762, -245620, 5653317, -20427884, 17905032, -7727988, 1325006, 19510972, -35817664, 15929912, -2792660, 17195652, -15880404, 2863984, 3824847, -1429976, 139633),
    (6, 3) => (139633, -1429976, 2863984, -2792660, 1325006, -245620, 3824847, -15880404, 15929912, -7727988, 1458762, 17195652, -35817664, 17905032, -3462252, 19510972, -20427884, 4086352, 5653317, -2380800, 271779),
    (6, 4) => (271779, -3015728, 6694608, -7408908, 4067018, -880548, 8449957, -37913324, 42405032, -23510468, 5134574, 43093692, -97838784, 55053752, -12183636, 56662212, -65224244, 14742480, 19365967, -9117992, 1152561),
    (6, 5) => (1152561, -12950184, 29442256, -33918804, 19834350, -4712740, 36480687, -166461044, 192596472, -113206788, 27060170, 190757572, -444003904, 262901672, -63394124, 260445372, -311771244, 76206736, 94851237, -47460464, 6150211),
)

# The reference smoothness indicator of the `buffer` stencil values `ψ`, evaluated exactly (in BigFloat from
# the rational coefficients)
function reference_smoothness_indicator(ψ, buffer, stencil)
    C = reference_smoothness_coefficients[(buffer, stencil)] .// reference_smoothness_denominator[buffer]
    β = zero(BigFloat)
    c = 1
    for i in 1:buffer
        β += ψ[i] * sum(BigFloat(C[c + j - i]) * ψ[j] for j in i:buffer)
        c += buffer - i + 1
    end
    return β
end

@testset "WENO smoothness indicators match the reference tables" begin
    Random.seed!(1234)

    for buffer in 2:6
        order = 2buffer - 1
        @info "Testing WENO$order smoothness indicators against the reference tables..."

        for stencil in 0:buffer-1, FT in (Float64, Float32)
            scheme = WENO(FT; order)
            # The exact coefficients are rounded to FT and the sum of squares has no cancellation,
            # so the result is accurate to a few ulps (the tolerance leaves room for the rounding of δ)
            rtol = FT == Float64 ? 1e-13 : 1e-4

            for _ in 1:20
                # A stencil with a large mean and O(1) variations, in BigFloat for the reference
                ψ = ntuple(_ -> 300 + randn(BigFloat), buffer)
                δ = ntuple(i -> FT(ψ[i+1] - ψ[i]), buffer - 1)

                β_ref = reference_smoothness_indicator(ψ, buffer, stencil)
                β     = smoothness_indicator(δ, scheme, Val(stencil))

                @test β ≈ β_ref rtol=rtol
            end
        end
    end
end
