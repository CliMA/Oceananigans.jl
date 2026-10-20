include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

const TKEClosures = Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities

using .TKEClosures: convective_buoyancy_production, convective_layer_depth, effective_buoyancy_flux

# Beer's law in two bands: a surface radiative buoyancy flux (≤ 0, positive upward), a fraction ϵ of which is absorbed over
# 1 / κ₁ and the rest over 1 / κ₂
struct TwoBandRadiation{FT}
    surface_buoyancy_flux :: FT
    first_band_fraction :: FT
    first_absorption_coefficient :: FT
    second_absorption_coefficient :: FT
end

SingleBandRadiation(Jʳ, κ) = TwoBandRadiation(Jʳ, 1.0, κ, κ)

Oceananigans.TurbulenceClosures.penetrative_radiation(R::TwoBandRadiation) = R
TKEClosures.surface_radiative_buoyancy_flux(i, j, grid, R::TwoBandRadiation, buoyancy, fields) = R.surface_buoyancy_flux

function TKEClosures.transmitted_fraction(R::TwoBandRadiation, i, j, grid, d)
    ϵ, κ₁, κ₂ = R.first_band_fraction, R.first_absorption_coefficient, R.second_absorption_coefficient
    return ϵ * exp(-κ₁ * d) + (1 - ϵ) * exp(-κ₂ * d)
end

function TKEClosures.transmitted_fraction_derivative(R::TwoBandRadiation, i, j, grid, d)
    ϵ, κ₁, κ₂ = R.first_band_fraction, R.first_absorption_coefficient, R.second_absorption_coefficient
    return - ϵ * κ₁ * exp(-κ₁ * d) - (1 - ϵ) * κ₂ * exp(-κ₂ * d)
end

function TKEClosures.transmitted_thickness(R::TwoBandRadiation, i, j, grid, h)
    ϵ, κ₁, κ₂ = R.first_band_fraction, R.first_absorption_coefficient, R.second_absorption_coefficient
    return - ϵ * expm1(-κ₁ * h) / κ₁ - (1 - ϵ) * expm1(-κ₂ * h) / κ₂
end

# The radiation heats the column as the divergence of the radiative flux through each cell
@inline function radiative_flux(i, j, k, grid, R)
    d = - znode(i, j, k, grid, Center(), Center(), Face())
    return R.surface_buoyancy_flux * TKEClosures.transmitted_fraction(R, i, j, grid, d)
end

@inline (R::TwoBandRadiation)(i, j, k, grid, clock, fields) = - ∂zᶜᶜᶜ(i, j, k, grid, radiative_flux, R)

# Root of w★³ = W(h) + h Jᵇᵋ, by bisection to round-off
function bisected_convective_layer_depth(grid, radiation, w★³, Jᵇ, Jʳ, Jᵇᵋ)
    h⁻, h⁺ = 0.0, 1e6
    for _ in 1:200
        h = (h⁻ + h⁺) / 2
        shallow = convective_buoyancy_production(1, 1, grid, radiation, h, Jᵇ, Jʳ) + h * Jᵇᵋ < w★³
        h⁻, h⁺ = shallow ? (h, h⁺) : (h⁻, h)
    end
    return (h⁻ + h⁺) / 2
end

column_grid(arch) = RectilinearGrid(arch, size = 32, z = (-128, 0), topology = (Flat, Flat, Bounded))

function sunny_column(grid, radiation; Jᵇᴮᶜ, steps = 144, closure = CATKEVerticalDiffusivity())

    forcing = isnothing(radiation) ? NamedTuple() : (; b = radiation)
    boundary_conditions = (b = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇᴮᶜ)),
                           u = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4)))

    model = HydrostaticFreeSurfaceModel(grid; closure, boundary_conditions, forcing,
                                        buoyancy = BuoyancyTracer(), tracers = :b,
                                        coriolis = FPlane(f = 1e-4))

    set!(model, b = z -> 1e-5 * z)

    for _ in 1:steps
        time_step!(model, 10minutes)
    end

    return model
end

@testset "CATKE with penetrating radiation" begin
    # Fluxes positive upward: surface cooling Jᵇᴮᶜ > 0 stronger than the sun Jʳ < 0, so the column loses buoyancy on net, Jᵇ > 0
    Jᵇᴮᶜ, Jʳ = 5e-8, -2e-8
    Jᵇ = Jᵇᴮᶜ + Jʳ
    Jᵇᵋ = 1e-11
    grid = RectilinearGrid(size = 100, z = (-500, 0), topology = (Flat, Flat, Bounded))

    @testset "Radiation absorbed at the surface or passing through" begin
        opaque = SingleBandRadiation(Jʳ, 1e6)
        transparent = SingleBandRadiation(Jʳ, 1e-8)
        @test effective_buoyancy_flux(1, 1, grid, opaque, 30.0, Jᵇ, Jʳ) ≈ Jᵇ rtol = 1e-6
        @test effective_buoyancy_flux(1, 1, grid, transparent, 30.0, Jᵇ, Jʳ) ≈ Jᵇᴮᶜ rtol = 1e-6

        # The convective layer depth reduces to the stock one with the total or with the non-radiative flux
        stock_with_total_flux = convective_layer_depth(1, 1, grid, nothing, 1e-6, Jᵇ, 0.0, Jᵇᵋ)
        stock_without_radiation = convective_layer_depth(1, 1, grid, nothing, 1e-6, Jᵇᴮᶜ, 0.0, Jᵇᵋ)
        @test convective_layer_depth(1, 1, grid, opaque, 1e-6, Jᵇ, Jʳ, Jᵇᵋ) ≈ stock_with_total_flux rtol = 1e-6
        @test convective_layer_depth(1, 1, grid, transparent, 1e-6, Jᵇ, Jʳ, Jᵇᵋ) ≈ stock_without_radiation rtol = 1e-6
    end

    @testset "Convective layer depth" begin
        # A single band, and two bands with the red one absorbed in the top meter under a sun that nearly offsets the cooling
        for (R, cooling) in ((SingleBandRadiation(Jʳ, 1 / 10), Jᵇᴮᶜ), (TwoBandRadiation(-1.41e-7, 0.58, 1 / 0.35, 1 / 23), 1.44e-7))
            sun = R.surface_buoyancy_flux
            net = cooling + sun

            for depth in (0.1, 2, 10, 50, 200, 1000)
                w★³ = convective_buoyancy_production(1, 1, grid, R, depth, net, sun) + depth * Jᵇᵋ
                h = bisected_convective_layer_depth(grid, R, w★³, net, sun, Jᵇᵋ)
                @test convective_layer_depth(1, 1, grid, R, w★³, net, sun, Jᵇᵋ) ≈ h rtol = 2e-3
            end
        end
    end

    @testset "CATKE takes the radiation from the model forcing" begin
        radiation = SingleBandRadiation(Jʳ, 1 / 10)
        closure = (CATKEVerticalDiffusivity(), ScalarDiffusivity(ν = 1e-4))
        forcing = (; b = (radiation, Forcing((z, t) -> 0)))
        model = HydrostaticFreeSurfaceModel(grid; closure, forcing, buoyancy = BuoyancyTracer(), tracers = :b)
        @test first(model.closure).penetrative_radiation === radiation
    end

    @testset "Ensemble members reproduce their single columns" begin
        radiation = SingleBandRadiation(Jʳ, 1 / 10)

        # The members differ in the dissipation length, which sets the averaging time of the surface buoyancy flux
        dissipation(x) = TKEClosures.CATKEEquation(CʰⁱD = 0.579x, CˡᵒD = 1.604x, CᵘⁿD = 0.923x, CᶜD = 3.254x)
        closures = [CATKEVerticalDiffusivity(turbulent_kinetic_energy_equation = dissipation(x))
                    for x in (1.0, 0.5, 2.0), _ in 1:1]

        ensemble_grid = RectilinearGrid(size = ColumnEnsembleSize(Nz = 32, ensemble = size(closures)), z = (-128, 0),
                                        topology = (Flat, Flat, Bounded))

        ensemble = sunny_column(ensemble_grid, radiation; Jᵇᴮᶜ, closure = closures, steps = 72)

        for m in 1:size(closures, 1)
            column = sunny_column(column_grid(CPU()), radiation; Jᵇᴮᶜ, closure = closures[m, 1], steps = 72)
            @test interior(ensemble.tracers.e, m, 1, :) == interior(column.tracers.e, 1, 1, :)
        end
    end

    for arch in archs
        @testset "Time stepping with penetrating radiation [$(typeof(arch))]" begin
            # Without radiation the new path reduces to the stock closure
            stock = sunny_column(column_grid(arch), nothing; Jᵇᴮᶜ)
            dark = sunny_column(column_grid(arch), SingleBandRadiation(0.0, 1 / 10); Jᵇᴮᶜ)
            @test Array(interior(dark.tracers.b)) ≈ Array(interior(stock.tracers.b)) rtol = 1e-10
            @test Array(interior(dark.tracers.e)) ≈ Array(interior(stock.tracers.e)) rtol = 1e-10

            # Cooled on net, and warmed on net by a sun stronger than the cooling
            for (sun, cooling) in ((Jʳ, Jᵇᴮᶜ), (-Jᵇᴮᶜ, -Jʳ))
                sunny = sunny_column(column_grid(arch), SingleBandRadiation(sun, 1 / 10); Jᵇᴮᶜ = cooling)
                @test all(isfinite, Array(interior(sunny.tracers.e)))
                @test Array(interior(sunny.closure_fields.Jʳ))[1] ≈ sun
                @test Array(interior(sunny.closure_fields.Jᵇ))[1] ≈ cooling + sun
            end
        end
    end
end
