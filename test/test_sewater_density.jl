include("dependencies_for_runtests.jl")

using Oceananigans.Models
using Oceananigans.Models: model_temperature, model_salinity, model_geopotential_height
using Oceananigans.Models: ConstantTemperatureSB, ConstantSalinitySB
using SeawaterPolynomials: ρ, BoussinesqEquationOfState
using SeawaterPolynomials: SecondOrderSeawaterPolynomial, RoquetEquationOfState
using SeawaterPolynomials: TEOS10EquationOfState, TEOS10SeawaterPolynomial
using Oceananigans.BuoyancyModels: Zᶜᶜᶜ

tracers = (:S, :T)
ST_testvals = (S = 34.7, T = 0.5)
Roquet_eos = (RoquetEquationOfState(:Linear), RoquetEquationOfState(:Cabbeling),
              RoquetEquationOfState(:CabbelingThermobaricity), RoquetEquationOfState(:Freezing),
              RoquetEquationOfState(:SecondOrder), RoquetEquationOfState(:SimplestRealistic))

TEOS10_eos = TEOS10EquationOfState()

"Return and `Array` on `arch` that is `size(grid)` flled with `value`."
function grid_size_value(arch, grid, value)

    value_array = fill(value, size(grid))

    return arch_array(arch, value_array)

end

"Check the error thrown for non-`BoussinesqEquationOfState`."
function error_non_Boussinesq(arch, FT)

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy = SeawaterBuoyancy()
    model = NonhydrostaticModel(; grid, buoyancy, tracers)
    SeawaterDensity(model) # throws error

    return nothing
end

"""
    function eos_works(arch, FT, eos::BoussinesqEquationOfState)
Check if using a `BoussinesqEquationOfState` returns a `KernelFunctionOperation`.
"""
function eos_works(arch, FT, eos::BoussinesqEquationOfState;
                   constant_temperature = nothing, constant_salinity = nothing)

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))

    buoyancy =  if !isnothing(constant_temperature)
                    SeawaterBuoyancy(equation_of_state = eos; constant_temperature)
                elseif !isnothing(constant_salinity)
                    SeawaterBuoyancy(equation_of_state = eos; constant_salinity)
                else
                    SeawaterBuoyancy(equation_of_state = eos)
                end

    model = NonhydrostaticModel(; grid, buoyancy, tracers)

    return SeawaterDensity(model) isa KernelFunctionOperation
end

"""
    function insitu_density(arch, FT, eos)
Use the `KernelFunctionOperation` returned from `SeawaterDensity` to compute a density `Field`
and compare the computed values to density values explicitly calculate using
`SeawaterPolynomials.ρ`. Similar function is used to test the potential density computation.
"""
function insitu_density(arch, FT, eos::BoussinesqEquationOfState;
                        constant_temperature = nothing, constant_salinity = nothing)

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy =  if !isnothing(constant_temperature)
                    SeawaterBuoyancy(equation_of_state = eos; constant_temperature)
                elseif !isnothing(constant_salinity)
                    SeawaterBuoyancy(equation_of_state = eos; constant_salinity)
                else
                    SeawaterBuoyancy(equation_of_state = eos)
                end
    model = NonhydrostaticModel(; grid, buoyancy, tracers)

    if !isnothing(constant_temperature)
        set!(model, S = ST_testvals.S)
    elseif !isnothing(constant_salinity)
        set!(model, T = ST_testvals.T)
    else
        set!(model, S = ST_testvals.S, T = ST_testvals.T)
    end

    d_field = compute!(Field(SeawaterDensity(model)))

    # Computation from SeawaterPolynomials to check against
    geopotential_height = model_geopotential_height(model)
    T_vec = grid_size_value(arch, grid, ST_testvals.T)
    S_vec = grid_size_value(arch, grid, ST_testvals.S)
    eos_vec = grid_size_value(arch, grid, model.buoyancy.model.equation_of_state)
    SWP_ρ = similar(interior(d_field))
    SWP_ρ .= SeawaterPolynomials.ρ.(T_vec, S_vec, geopotential_height, eos_vec)

    return all(CUDA.@allowscalar interior(d_field) .== SWP_ρ)
end

function potential_density(arch, FT, eos::BoussinesqEquationOfState;
                           constant_temperature = nothing, constant_salinity = nothing)

    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    buoyancy =  if !isnothing(constant_temperature)
                    SeawaterBuoyancy(equation_of_state = eos; constant_temperature)
                elseif !isnothing(constant_salinity)
                    SeawaterBuoyancy(equation_of_state = eos; constant_salinity)
                else
                    SeawaterBuoyancy(equation_of_state = eos)
                end
    model = NonhydrostaticModel(; grid, buoyancy, tracers)

    if !isnothing(constant_temperature)
        set!(model, S = ST_testvals.S)
    elseif !isnothing(constant_salinity)
        set!(model, T = ST_testvals.T)
    else
        set!(model, S = ST_testvals.S, T = ST_testvals.T)
    end

    d_field = compute!(Field(SeawaterDensity(model, geopotential_height = 0)))

    # Computation from SeawaterPolynomials to check against
    geopotential_height = grid_size_value(arch, grid, 0)
    T_vec = grid_size_value(arch, grid, ST_testvals.T)
    S_vec = grid_size_value(arch, grid, ST_testvals.S)
    eos_vec = grid_size_value(arch, grid, model.buoyancy.model.equation_of_state)
    SWP_ρ = similar(interior(d_field))
    SWP_ρ .= SeawaterPolynomials.ρ.(T_vec, S_vec, geopotential_height, eos_vec)

    return all(CUDA.@allowscalar interior(d_field) .== SWP_ρ)
end

@testset "Density models" begin
    @info "Testing `SeawaterDensity`..."

    @testset "Error for non-`BoussinesqEquationOfState`" begin
        @info "Testing error is thrown... "
        for FT in float_types
            for arch in archs
                @test_throws ArgumentError error_non_Boussinesq(arch, FT)
            end
        end
    end

    @testset "SeawaterDensity `KernelFunctionOperation` instantiation" begin
        @info "Testing `KernelFunctionOperation` is returned..."

        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test eos_works(arch, FT, eos)
                    @test eos_works(arch, FT, eos, constant_temperature = ST_testvals.T)
                    @test eos_works(arch, FT, eos, constant_salinity = ST_testvals.S)
                end
                @test eos_works(arch, FT, TEOS10_eos)
                @test eos_works(arch, FT, TEOS10_eos, constant_temperature = ST_testvals.T)
                @test eos_works(arch, FT, TEOS10_eos, constant_salinity = ST_testvals.S)
            end
        end
    end

    @testset "In-situ density computation tests" begin
        @info "Testing in-situ density compuation..."

        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test insitu_density(arch, FT, eos)
                    @test insitu_density(arch, FT, eos, constant_temperature = ST_testvals.T)
                    @test insitu_density(arch, FT, eos, constant_salinity = ST_testvals.S)
                end
                @test insitu_density(arch, FT, TEOS10_eos)
                @test insitu_density(arch, FT, TEOS10_eos, constant_temperature = ST_testvals.T)
                @test insitu_density(arch, FT, TEOS10_eos, constant_salinity = ST_testvals.S)
            end
        end
    end

    @testset "Potential density computation tests" begin
        @info "Testing a potential density comnputation..."

        for FT in float_types
            for arch in archs
                for eos in Roquet_eos
                    @test potential_density(arch, FT, eos)
                    @test potential_density(arch, FT, eos, constant_temperature = ST_testvals.T)
                    @test potential_density(arch, FT, eos, constant_salinity = ST_testvals.S)
                end
                @test potential_density(arch, FT, TEOS10_eos)
                @test potential_density(arch, FT, TEOS10_eos, constant_temperature = ST_testvals.T)
                @test potential_density(arch, FT, TEOS10_eos, constant_salinity = ST_testvals.S)
            end
        end
    end

end
