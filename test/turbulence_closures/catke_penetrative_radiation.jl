include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

const TKEClosures = Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities

using .TKEClosures: convective_buoyancy_production, compensation_depth, convective_layer_depth, effective_buoyancy_flux

# Beer's law in two bands: a surface radiative buoyancy flux, a fraction ϵ of which is absorbed over 1 / κ₁ and the rest over 1 / κ₂
struct TwoBandRadiation{FT}
    surface_buoyancy_flux :: FT
    first_band_fraction :: FT
    first_absorption_coefficient :: FT
    second_absorption_coefficient :: FT
end

SingleBandRadiation(Jʳ, κ) = TwoBandRadiation(Jʳ, 1.0, κ, κ)

TKEClosures.surface_radiative_buoyancy_flux(i, j, grid, R::TwoBandRadiation, buoyancy, fields) = R.surface_buoyancy_flux

function TKEClosures.transmitted_fraction(R::TwoBandRadiation, i, j, grid, d)
    ϵ, κ₁, κ₂ = R.first_band_fraction, R.first_absorption_coefficient, R.second_absorption_coefficient
    return ϵ * exp(-κ₁ * d) + (1 - ϵ) * exp(-κ₂ * d)
end

function TKEClosures.transmitted_thickness(R::TwoBandRadiation, i, j, grid, h)
    ϵ, κ₁, κ₂ = R.first_band_fraction, R.first_absorption_coefficient, R.second_absorption_coefficient
    return - ϵ * expm1(-κ₁ * h) / κ₁ - (1 - ϵ) * expm1(-κ₂ * h) / κ₂
end

# Root of w★³ = W(h) + h Jᵇᵋ on the rising branch of W, by bisection to round-off
function bisected_convective_layer_depth(grid, radiation, w★³, Jᵇ, Jʳ, hᶜ, Jᵇᵋ)
    h⁻, h⁺ = 0.0, hᶜ
    for _ in 1:100
        h = (h⁻ + h⁺) / 2
        shallow = convective_buoyancy_production(1, 1, grid, radiation, h, Jᵇ, Jʳ) + h * Jᵇᵋ < w★³
        h⁻, h⁺ = shallow ? (h, h⁺) : (h⁻, h)
    end
    return (h⁻ + h⁺) / 2
end

function cooled_column(arch, penetrative_radiation; Jᵇᴮᶜ = 2e-8, steps = 144)
    grid = RectilinearGrid(arch, size = 32, z = (-128, 0), topology = (Flat, Flat, Bounded))
    closure = CATKEVerticalDiffusivity(; penetrative_radiation)
    boundary_conditions = (b = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇᴮᶜ)),
                           u = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4)))

    model = HydrostaticFreeSurfaceModel(grid; closure, boundary_conditions,
                                        buoyancy = BuoyancyTracer(), tracers = :b,
                                        coriolis = FPlane(f = 1e-4))

    set!(model, b = z -> 1e-5 * z)

    for _ in 1:steps
        time_step!(model, 10minutes)
    end

    return model
end

@testset "CATKE with penetrating radiation" begin
    @info "Testing CATKE with penetrating radiation..."

    # Non-radiative surface cooling Jᵇᴮᶜ under a stronger sun Jʳ: the column warms on net, Jᵇ < 0
    Jᵇᴮᶜ, Jʳ = 2e-8, 5e-8
    Jᵇ = Jᵇᴮᶜ - Jʳ
    Jᵇᵋ = 1e-11
    grid = RectilinearGrid(size = 100, z = (-500, 0), topology = (Flat, Flat, Bounded))
    radiation = SingleBandRadiation(Jʳ, 1 / 10)

    @testset "Radiation absorbed at the surface or passing through" begin
        opaque = SingleBandRadiation(Jʳ, 1e6)
        transparent = SingleBandRadiation(Jʳ, 1e-8)
        @test effective_buoyancy_flux(1, 1, grid, opaque, 30.0, Jᵇ, Jʳ) ≈ Jᵇ rtol = 1e-6
        @test effective_buoyancy_flux(1, 1, grid, transparent, 30.0, Jᵇ, Jʳ) ≈ Jᵇᴮᶜ rtol = 1e-6

        # Cooled on net, the convective layer depth reduces to the stock one with the corresponding flux
        cooling, sun = 5e-8, 2e-8
        stock_with_total_flux = convective_layer_depth(1, 1, grid, nothing, 1e-6, cooling - sun, 0.0, Inf, Jᵇᵋ)
        stock_without_radiation = convective_layer_depth(1, 1, grid, nothing, 1e-6, cooling, 0.0, Inf, Jᵇᵋ)
        @test convective_layer_depth(1, 1, grid, SingleBandRadiation(sun, 1e6), 1e-6, cooling - sun, sun, Inf, Jᵇᵋ) ≈ stock_with_total_flux rtol = 1e-6
        @test convective_layer_depth(1, 1, grid, SingleBandRadiation(sun, 1e-8), 1e-6, cooling - sun, sun, Inf, Jᵇᵋ) ≈ stock_without_radiation rtol = 1e-6
    end

    @testset "Compensation depth" begin
        hᶜ = compensation_depth(1, 1, grid, radiation, Jᵇ, Jʳ)
        W(h) = convective_buoyancy_production(1, 1, grid, radiation, h, Jᵇ, Jʳ)

        @test hᶜ ≈ 10 * log(Jʳ / (Jʳ - Jᵇᴮᶜ)) rtol = 1e-6
        @test W(hᶜ) > W(hᶜ - 1) && W(hᶜ) > W(hᶜ + 1)
        @test compensation_depth(1, 1, grid, radiation, 1e-8, Jʳ) == Inf
        @test compensation_depth(1, 1, grid, nothing, Jᵇ, Jʳ) == Inf
    end

    @testset "Convective layer depth" begin
        # A single band, and two bands with the red one absorbed in the top meter, in an afternoon column
        for (R, cooling) in ((radiation, Jᵇᴮᶜ), (TwoBandRadiation(1.2e-7, 0.58, 1 / 0.35, 1 / 23), 7.2e-8))
            sun = R.surface_buoyancy_flux
            net = cooling - sun
            hᶜ = compensation_depth(1, 1, grid, R, net, sun)
            Wᶜ = convective_buoyancy_production(1, 1, grid, R, hᶜ, net, sun) + hᶜ * Jᵇᵋ

            # The root is found up to the peak of W, and clamped at the compensation depth above it
            for fraction in (0.1, 0.5, 0.9, 0.9999)
                w★³ = fraction * Wᶜ
                h = bisected_convective_layer_depth(grid, R, w★³, net, sun, hᶜ, Jᵇᵋ)
                @test convective_layer_depth(1, 1, grid, R, w★³, net, sun, hᶜ, Jᵇᵋ) ≈ h rtol = 2e-3
            end

            @test convective_layer_depth(1, 1, grid, R, 2Wᶜ, net, sun, hᶜ, Jᵇᵋ) == hᶜ
        end
    end

    for arch in archs
        @testset "Time stepping with penetrating radiation [$(typeof(arch))]" begin
            # Without radiation the new path reduces to the stock closure
            stock = cooled_column(arch, nothing)
            dark = cooled_column(arch, SingleBandRadiation(0.0, 1 / 10))
            @test Array(interior(dark.tracers.b)) ≈ Array(interior(stock.tracers.b)) rtol = 1e-10
            @test Array(interior(dark.tracers.e)) ≈ Array(interior(stock.tracers.e)) rtol = 1e-10

            sunny = cooled_column(arch, radiation)
            closure_fields = sunny.closure_fields
            @test all(isfinite, Array(interior(sunny.tracers.e)))
            @test Array(interior(closure_fields.Jʳ))[1] ≈ Jʳ
            @test Array(interior(closure_fields.Jᵇ))[1] ≈ Jᵇ
            @test Array(interior(closure_fields.hᶜ))[1] ≈ 10 * log(Jʳ / (Jʳ - Jᵇᴮᶜ)) rtol = 1e-5
        end
    end
end
