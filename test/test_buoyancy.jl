using Oceananigans.Models: TracerFields

using Oceananigans.Buoyancy:
    required_tracers, ρ′, ∂x_b, ∂y_b,
    thermal_expansionᶜᶜᶜ, thermal_expansionᶠᶜᶜ, thermal_expansionᶜᶠᶜ, thermal_expansionᶜᶜᶠ,
    haline_contractionᶜᶜᶜ, haline_contractionᶠᶜᶜ, haline_contractionᶜᶠᶜ, haline_contractionᶜᶜᶠ

function instantiate_linear_equation_of_state(FT, α, β)
    eos = LinearEquationOfState(FT, α=α, β=β)
    return eos.α == FT(α) && eos.β == FT(β)
end

function instantiate_roquet_equations_of_state(FT, flavor; coeffs=nothing)
    eos = (coeffs == nothing ? RoquetIdealizedNonlinearEquationOfState(FT, flavor) :
                               RoquetIdealizedNonlinearEquationOfState(FT, flavor, polynomial_coeffs=coeffs))
    return typeof(eos.polynomial_coeffs.R₁₀₀) == FT
end

function instantiate_seawater_buoyancy(FT, EquationOfState; kwargs...)
    buoyancy = SeawaterBuoyancy(FT, equation_of_state=EquationOfState(FT); kwargs...)
    return typeof(buoyancy.gravitational_acceleration) == FT
end

function density_perturbation_works(arch, FT, eos)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), length=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid, (:T, :S)))
    density_anomaly = ρ′(2, 2, 2, grid, eos, C.T, C.S)
    return true
end

function ∂x_b_works(arch, FT, buoyancy)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), length=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid, required_tracers(buoyancy)))
    dbdx = ∂x_b(2, 2, 2, grid, buoyancy, C)
    return true
end

function ∂y_b_works(arch, FT, buoyancy)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), length=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid, required_tracers(buoyancy)))
    dbdy = ∂y_b(2, 2, 2, grid, buoyancy, C)
    return true
end

function ∂z_b_works(arch, FT, buoyancy)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), length=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid, required_tracers(buoyancy)))
    dbdz = ∂z_b(2, 2, 2, grid, buoyancy, C)
    return true
end

function thermal_expansion_works(arch, FT, eos)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), length=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid, (:T, :S)))
    α = thermal_expansionᶜᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    α = thermal_expansionᶠᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    α = thermal_expansionᶜᶠᶜ(2, 2, 2, grid, eos, C.T, C.S)
    α = thermal_expansionᶜᶜᶠ(2, 2, 2, grid, eos, C.T, C.S)
    return true
end

function haline_contraction_works(arch, FT, eos)
    grid = RegularCartesianGrid(FT, size=(3, 3, 3), length=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid, (:T, :S)))
    β = haline_contractionᶜᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    β = haline_contractionᶠᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    β = haline_contractionᶜᶠᶜ(2, 2, 2, grid, eos, C.T, C.S)
    β = haline_contractionᶜᶜᶠ(2, 2, 2, grid, eos, C.T, C.S)
    return true
end

EquationsOfState = (LinearEquationOfState, RoquetIdealizedNonlinearEquationOfState)
buoyancy_kwargs = (Dict(), Dict(:constant_salinity=>35.0), Dict(:constant_temperature=>20.0))

@testset "Buoyancy" begin
    @info "Testing buoyancy..."

    @testset "Equations of State" begin
        for FT in float_types
            @test instantiate_linear_equation_of_state(FT, 0.1, 0.3)

            testcoeffs = (R₀₁₀ = π, R₁₀₀ = ℯ, R₀₂₀ = 2π, R₀₁₁ = 2ℯ, R₂₀₀ = 3π, R₁₀₁ = 3ℯ, R₁₁₀ = 4π)
            for flavor in (:linear, :cabbeling, :cabbeling_thermobaricity, :freezing, :second_order)
                @test instantiate_roquet_equations_of_state(FT, flavor)
                @test instantiate_roquet_equations_of_state(FT, flavor, coeffs=testcoeffs)
            end

            for EOS in EquationsOfState
                for kwargs in buoyancy_kwargs
                    @test instantiate_seawater_buoyancy(FT, EOS; kwargs...)
                end
            end

            for arch in archs
                @test density_perturbation_works(arch, FT, RoquetIdealizedNonlinearEquationOfState())
            end

            for arch in archs
                for buoyancy in (BuoyancyTracer(), nothing, SeawaterBuoyancy(FT),
                                 SeawaterBuoyancy(FT, equation_of_state=RoquetIdealizedNonlinearEquationOfState(FT)))

                    @test ∂x_b_works(arch, FT, buoyancy)
                    @test ∂y_b_works(arch, FT, buoyancy)
                    @test ∂z_b_works(arch, FT, buoyancy)
                end
            end

            for arch in archs
                for EOS in EquationsOfState
                    @test thermal_expansion_works(arch, FT, EOS())
                    @test haline_contraction_works(arch, FT, EOS())
                end
            end
        end
    end
end
