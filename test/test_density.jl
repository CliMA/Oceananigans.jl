include("dependencies_for_runtests.jl")

using Oceananigans.Models
using Oceananigans.Models: model_geopotential_height
using SeawaterPolynomials: ρ, BoussinesqEquationOfState, SecondOrderSeawaterPolynomial
using Oceananigans.BuoyancyModels: Zᶜᶜᶜ

tracers = (:S, :T)
S_testval, T_testval = (34.7, 0.5)

function error_non_Boussinesq(arch, FT)

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy()
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    SeawaterDensity(model) # throws error

    return nothing
end

Roquet_eos = (RoquetEquationOfState(:Linear), RoquetEquationOfState(:Cabbeling),
              RoquetEquationOfState(:CabbelingThermobaricity), RoquetEquationOfState(:Freezing),
              RoquetEquationOfState(:SecondOrder), RoquetEquationOfState(:SimplestRealistic))

function Roquet_eos_works(arch, FT, eos::BoussinesqEquationOfState{<:SecondOrderSeawaterPolynomial})

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state = eos)
    model = NonhydrostaticModel(; grid, buoyancy, tracers)

    return SeawaterDensity(model) isa KernelFunctionOperation
end

function insitu_Roquet_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:SecondOrderSeawaterPolynomial})

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state = eos)
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    set!(model, S = S_testval, T = T_testval)
    d_field = compute!(Field(SeawaterDensity(model)))

    # Computation from SeawaterPolynomials to check against
    geopotential_height = model_geopotential_height(model)
    T_testval_vec = fill(T_testval, model.grid.Nz)
    S_testval_vec = fill(S_testval, model.grid.Nz)
    eos_vec = fill(model.buoyancy.model.equation_of_state, model.grid.Nz)
    SWP_ρ = similar(interior(d_field))
    SWP_ρ .= SeawaterPolynomials.ρ.(T_testval_vec, S_testval_vec, geopotential_height, eos_vec)

    equal_values = sum(CUDA.@allowscalar interior(d_field) .== SWP_ρ)

    return equal_values == model.grid.Nx * model.grid.Ny * model.grid.Nz
end

function potential_Roquet_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:SecondOrderSeawaterPolynomial})

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state = eos)
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    set!(model, S = S_testval, T = T_testval)
    d_field = compute!(Field(SeawaterDensity(model, geopotential_height = 0)))

    # Computation from SeawaterPolynomials to check against
    geopotential_height = zeros(size(model.grid))
    T_testval_vec = fill(T_testval, model.grid.Nz)
    S_testval_vec = fill(S_testval, model.grid.Nz)
    eos_vec = fill(model.buoyancy.model.equation_of_state, model.grid.Nz)
    SWP_ρ = similar(interior(d_field))
    SWP_ρ .= SeawaterPolynomials.ρ.(T_testval_vec, S_testval_vec, geopotential_height, eos_vec)

    equal_values = sum(CUDA.@allowscalar interior(d_field) .== SWP_ρ)

    return equal_values == model.grid.Nx * model.grid.Ny * model.grid.Nz
end

TEOS10_eos = TEOS10EquationOfState()

function TEOS10_eos_works(arch, FT, eos::BoussinesqEquationOfState{<:TEOS10SeawaterPolynomial})

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state = eos)
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    return SeawaterDensity(model) isa KernelFunctionOperation
end

function insitu_TEOS10_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:TEOS10SeawaterPolynomial})

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state = eos)
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    set!(model, S = S_testval, T = T_testval)
    d_field = compute!(Field(SeawaterDensity(model)))

    # Computation from SeawaterPolynomials to check against
    geopotential_height = model_geopotential_height(model)
    T_testval_vec = fill(T_testval, model.grid.Nz)
    S_testval_vec = fill(S_testval, model.grid.Nz)
    eos_vec = fill(model.buoyancy.model.equation_of_state, model.grid.Nz)
    SWP_ρ = similar(interior(d_field))
    SWP_ρ .= SeawaterPolynomials.ρ.(T_testval_vec, S_testval_vec, geopotential_height, eos_vec)

    equal_values = sum(CUDA.@allowscalar interior(d_field) .== SWP_ρ)

    return equal_values == model.grid.Nx * model.grid.Ny * model.grid.Nz
end

function potential_TEOS10_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:TEOS10SeawaterPolynomial})

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy(equation_of_state = eos)
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    set!(model, S = S_testval, T = T_testval)
    d_field = compute!(Field(SeawaterDensity(model, geopotential_height = 0)))

    # Computation from SeawaterPolynomials to check against
    geopotential_height = zeros(size(model.grid))
    T_testval_vec = fill(T_testval, model.grid.Nz)
    S_testval_vec = fill(S_testval, model.grid.Nz)
    eos_vec = fill(model.buoyancy.model.equation_of_state, model.grid.Nz)
    SWP_ρ = similar(interior(d_field))
    SWP_ρ .= SeawaterPolynomials.ρ.(T_testval_vec, S_testval_vec, geopotential_height, eos_vec)

    equal_values = sum(CUDA.@allowscalar interior(d_field) .== SWP_ρ)

    return equal_values == model.grid.Nx * model.grid.Ny * model.grid.Nz
end

@testset "Density models" begin

    @testset "Error for non-`BoussinesqEquationOfState`" begin
        for FT in float_types
            for arch in archs
                @test_throws ArgumentError error_non_Boussinesq(arch, FT)
            end
        end
    end

    @testset "SeawaterDensity `KernelFunctionOperation` instantiation" begin
        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test Roquet_eos_works(arch, FT, eos)
                end
                @test TEOS10_eos_works(arch, FT, TEOS10_eos)
            end
        end
    end

    @testset "In-situ density computation tests" begin
        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test insitu_Roquet_computation_test(arch, FT, eos)
                end
                @test insitu_TEOS10_computation_test(arch, FT, TEOS10_eos)
            end
        end
    end

    @testset "Potential density computation tests" begin
        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test potential_Roquet_computation_test(arch, FT, eos)
                end
                @test potential_TEOS10_computation_test(arch, FT, TEOS10_eos)
            end
        end
    end

end
