include("dependencies_for_runtests.jl")

using Oceananigans.Models
using Oceananigans.Models: model_geopotential_height
using SeawaterPolynomials: ρ, BoussinesqEquationOfState
using SeawaterPolynomials: SecondOrderSeawaterPolynomial, RoquetEquationOfState
using SeawaterPolynomials: TEOS10EquationOfState, TEOS10SeawaterPolynomial
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

"""
    function insitu_density_Roquet_computation_test(arch, FT, eos)
Use the `KernelFunctionOperation` returned from `SeawaterDensity` to compute a density `Field`
and compare the computed values to density values explicitly calculate using
`SeawaterPolynomials.ρ`. The funciton `sum`s over the `Array{Bool}` that is returned from the
equality comparison between the density `Field` (computed using `SeawaterDensity`) and the
values calculated using `SeawaterPolynomials.ρ`. If the `sum` is the same size as the product of
the dimensions of the grid then all values must be equal the returned value for the equality
comparison is 1 if `true`.

This same method is used for testing the output from `SeawaterDensity` in
`potential_density_Roquet_computation_test`, `insitu_density_TEOS10_computation_test` and
`potential_density_TEOS10_computation_test`.
"""
function insitu_density_Roquet_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:SecondOrderSeawaterPolynomial})

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

function potential_density_Roquet_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:SecondOrderSeawaterPolynomial})

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

function insitu_density_TEOS10_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:TEOS10SeawaterPolynomial})

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

function potential_density_TEOS10_computation_test(arch, FT, eos::BoussinesqEquationOfState{<:TEOS10SeawaterPolynomial})

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
                    @test insitu_density_Roquet_computation_test(arch, FT, eos)
                end
                @test insitu_density_TEOS10_computation_test(arch, FT, TEOS10_eos)
            end
        end
    end

    @testset "Potential density computation tests" begin
        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test potential_density_Roquet_computation_test(arch, FT, eos)
                end
                @test potential_density_TEOS10_computation_test(arch, FT, TEOS10_eos)
            end
        end
    end

end
