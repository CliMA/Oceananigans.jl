<<<<<<< HEAD
using Test
using Printf
using Random
using Statistics
using LinearAlgebra
using Logging

using CUDA
using MPI
using JLD2
using FFTW
using OffsetArrays
using SeawaterPolynomials

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Advection
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Coriolis
using Oceananigans.BuoyancyModels
using Oceananigans.Forcings
using Oceananigans.Solvers
using Oceananigans.Models
using Oceananigans.Simulations
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.TurbulenceClosures
using Oceananigans.AbstractOperations
using Oceananigans.Distributed
using Oceananigans.Logger
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Architectures: device # to resolve conflict with CUDA.device

using Dates: DateTime, Nanosecond
using TimesDates: TimeDate
using Statistics: mean
using LinearAlgebra: norm
using NCDatasets: Dataset
using KernelAbstractions: @kernel, @index, Event

import Oceananigans.Fields: interior
import Oceananigans.Utils: launch!, datatuple

Logging.global_logger(OceananigansLogger())

#####
##### Testing parameters
#####

float_types = (Float32, Float64)

archs = CUDA.has_cuda() ? (GPU(),) : (CPU(),)

closures = (
    :IsotropicDiffusivity,
    :AnisotropicDiffusivity,
    :AnisotropicBiharmonicDiffusivity,
    :TwoDimensionalLeith,
    :SmagorinskyLilly,
    :AnisotropicMinimumDissipation,
    :HorizontallyCurvilinearAnisotropicDiffusivity,
    :HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity,
    :ConvectiveAdjustmentVerticalDiffusivity
)

#####
##### Run tests!
#####

CUDA.allowscalar(true)

include("data_dependencies.jl")

group = get(ENV, "TEST_GROUP", :all) |> Symbol

=======
using CUDA
using Test
using Printf
using Statistics

using KernelAbstractions: @kernel, @index, Event

using Oceananigans
>>>>>>> ss/latitude_longitude_grid
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper, update_state!

import Oceananigans.Fields: interior

test_architectures() = CUDA.has_cuda() ? tuple(GPU()) : tuple(CPU())

function summarize_regression_test(fields, correct_fields)
    for (field_name, φ, φ_c) in zip(keys(fields), fields, correct_fields)
        Δ = φ .- φ_c

        Δ_min      = minimum(Δ)
        Δ_max      = maximum(Δ)
        Δ_mean     = mean(Δ)
        Δ_abs_mean = mean(abs, Δ)
        Δ_std      = std(Δ)

        matching    = sum(φ .≈ φ_c)
        grid_points = length(φ_c)

        @info @sprintf("Δ%s: min=%+.6e, max=%+.6e, mean=%+.6e, absmean=%+.6e, std=%+.6e (%d/%d matching grid points)",
                       field_name, Δ_min, Δ_max, Δ_mean, Δ_abs_mean, Δ_std, matching, grid_points)
    end
end

#####
##### Useful kernels
#####

@kernel function ∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

@kernel function divergence!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_∇²!(∇²ϕ, ϕ, arch, grid)
    fill_halo_regions!(ϕ, arch)
    child_arch = child_architecture(arch)
    event = launch!(child_arch, grid, :xyz, ∇²!, ∇²ϕ, grid, ϕ, dependencies=Event(device(child_arch)))
    wait(device(child_arch), event)
    fill_halo_regions!(∇²ϕ, arch)
    return nothing
end

#####
##### Useful utilities
#####

const AB2Model = NonhydrostaticModel{<:QuasiAdamsBashforth2TimeStepper}
const RK3Model = NonhydrostaticModel{<:RungeKutta3TimeStepper}

# For time-stepping without a Simulation
function ab2_or_rk3_time_step!(model::AB2Model, Δt, n)
    n == 1 && update_state!(model)
    time_step!(model, Δt, euler=n==1)
    return nothing
end

function ab2_or_rk3_time_step!(model::RK3Model, Δt, n)
    n == 1 && update_state!(model)
    time_step!(model, Δt)
    return nothing
end

interior(a, grid) = view(a, grid.Hx+1:grid.Nx+grid.Hx,
                            grid.Hy+1:grid.Ny+grid.Hy,
                            grid.Hz+1:grid.Nz+grid.Hz)

datatuple(A) = NamedTuple{propertynames(A)}(Array(data(a)) for a in A)
datatuple(args, names) = NamedTuple{names}(a.data for a in args)

function get_model_field(field_name, model)
    if field_name ∈ (:u, :v, :w)
        return getfield(model.velocities, field_name)
    else
        return getfield(model.tracers, field_name)
    end
end

function get_output_tuple(output, iter, tuplename)
    file = jldopen(output.filepath, "r")
    output_tuple = file["timeseries/$tuplename/$iter"]
    close(file)
    return output_tuple
end

function run_script(replace_strings, script_name, script_filepath, module_suffix="")
    file_content = read(script_filepath, String)
    test_script_filepath = script_name * "_test.jl"

    for strs in replace_strings
        new_file_content = replace(file_content, strs[1] => strs[2])
        if new_file_content == file_content
            error("$(strs[1]) => $(strs[2]) replacement not found in $script_filepath. " *
                  "Make sure the script has not changed, otherwise the test might take a long time.")
            return false
        else
            file_content = new_file_content
        end
    end

    open(test_script_filepath, "w") do f
        # Wrap test script inside module to avoid polluting namespaces
        write(f, "module _Test_$script_name" * "_$module_suffix\n")
        write(f, file_content)
        write(f, "\nend # module")
    end

    try
        include(test_script_filepath)
    catch err
        @error sprint(showerror, err)

        # Print the content of the file to the test log, with line numbers, for debugging
        test_file_content = read(test_script_filepath, String)
        delineated_file_content = split(test_file_content, '\n')
        for (number, line) in enumerate(delineated_file_content)
            @printf("% 3d %s\n", number, line)
        end

        rm(test_script_filepath)
        return false
    end

    # Delete the test script (if it hasn't been deleted already)
    rm(test_script_filepath)

    return true
end

#####
##### Boundary condition utils
#####

discrete_func(i, j, grid, clock, model_fields) = - model_fields.u[i, j, grid.Nz]
parameterized_discrete_func(i, j, grid, clock, model_fields, p) = - p.μ * model_fields.u[i, j, grid.Nz]

parameterized_fun(ξ, η, t, p) = p.μ * cos(p.ω * t)
field_dependent_fun(ξ, η, t, u, v, w) = - w * sqrt(u^2 + v^2)
exploding_fun(ξ, η, t, T, S, p) = - p.μ * cosh(S - p.S0) * exp((T - p.T0) / p.λ)

# Many bc. Very many
                 integer_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, 1)
                   float_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, FT(π))
              irrational_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, π)
                   array_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, ArrayType(rand(FT, 1, 1)))
         simple_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, (ξ, η, t) -> exp(ξ) * cos(η) * sin(t))
  parameterized_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, parameterized_fun, parameters=(μ=0.1, ω=2π))
field_dependent_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, field_dependent_fun, field_dependencies=(:u, :v, :w))
       discrete_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, discrete_func, discrete_form=true)

       parameterized_discrete_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, parameterized_discrete_func, discrete_form=true, parameters=(μ=0.1,))
parameterized_field_dependent_function_bc(C, FT=Float64, ArrayType=Array) = BoundaryCondition(C, exploding_fun, field_dependencies=(:T, :S), parameters=(S0=35, T0=100, μ=2π, λ=FT(2)))
