function instantiate_coriolis(T)
    coriolis = FPlane(T, f=π)
    return coriolis.f == T(π)
end

function instantiate_linear_equation_of_state(T, α, β)
    eos = LinearEquationOfState(T, α=α, β=β)
    return eos.α == T(α) && eos.β == T(β)
end

function instantiate_roquet_equations_of_state(T, flavor; coeffs=nothing)
    eos = (coeffs == nothing ? RoquetIdealizedNonlinearEquationOfState(T, flavor) :
                               RoquetIdealizedNonlinearEquationOfState(T, flavor, coeffs=coeffs))
    return typeof(eos.coeffs.R₁₀₀) == T
end

function instantiate_seawater_buoyancy(T, EquationOfState) 
    buoyancy = SeawaterBuoyancy(T, equation_of_state=EquationOfState(T))
    return typeof(buoyancy.gravitational_acceleration) == T
end

function density_perturbation_works(arch, T, eos)
    grid = RegularCartesianGrid(T, N=(3, 3, 3), L=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid))
    density_anomaly = ρ′(2, 2, 2, grid, eos, C)
    return true
end

function buoyancy_frequency_squared_works(arch, T, buoyancy)
    grid = RegularCartesianGrid(N=(3, 3, 3), L=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid))
    N² = buoyancy_frequency_squared(2, 2, 2, grid, buoyancy, C)
    return true
end

function thermal_expansion_works(arch, T, eos)
    grid = RegularCartesianGrid(T, N=(3, 3, 3), L=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid))
    α = thermal_expansion(2, 2, 2, grid, eos, C)
    return true
end

function haline_contraction_works(arch, T, eos)
    grid = RegularCartesianGrid(N=(3, 3, 3), L=(1, 1, 1))
    C = datatuple(TracerFields(arch, grid))
    β = haline_contraction(2, 2, 2, grid, eos, C)
    return true
end

@testset "Coriolis and Buoyancy" begin
    println("Testing Coriolis and buoyancy...")

    @testset "Coriolis" begin
        for T in float_types
            @test instantiate_coriolis(T)
        end
    end

    @testset "Equations of State" begin
        for T in float_types
            @test instantiate_linear_equation_of_state(T, 0.1, 0.3)

            testcoeffs = (R₀₁₀ = π, R₁₀₀ = ℯ, R₀₂₀ = 2π, R₀₁₁ = 2ℯ, R₂₀₀ = 3π, R₁₀₁ = 3ℯ, R₁₁₀ = 4π)
            for flavor in (:linear, :cabbeling, :cabbeling_thermobaricity, :freezing, :second_order)
                @test instantiate_roquet_equations_of_state(T, flavor)
                @test instantiate_roquet_equations_of_state(T, flavor, coeffs=testcoeffs)
            end

            for EOS in EquationsOfState
                @test instantiate_seawater_buoyancy(T, EOS)
            end

            for arch in archs
                @test density_perturbation_works(arch, T, RoquetIdealizedNonlinearEquationOfState())
            end

            for arch in archs
                for EOS in EquationsOfState
                    buoyancy = SeawaterBuoyancy(T, equation_of_state=EOS(T))
                    @test buoyancy_frequency_squared_works(arch, T, buoyancy)
                end

                for buoyancy in (BuoyancyTracer(), nothing)
                    @test buoyancy_frequency_squared_works(arch, T, buoyancy)
                end
            end

            for arch in archs
                for EOS in EquationsOfState 
                    @test thermal_expansion_works(arch, T, EOS())
                    @test haline_contraction_works(arch, T, EOS())
                end
            end
        end
    end
end
