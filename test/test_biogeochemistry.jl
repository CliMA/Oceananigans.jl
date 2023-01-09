include("test/dependencies_for_runtests.jl") # remove

using Oceananigans.Biogeochemistry: AbstractBiogeochemistry, AbstractContinuousFormBiogeochemistry
using Oceananigans.Fields: ConstantField
using Oceananigans.Utils: launch!
using KernelAbstractions, CUDA

import Oceananigans.Biogeochemistry:
       required_biogeochemical_tracers,
       required_biogeochemical_auxiliary_fields,
       biogeochemical_drift_velocity,
       biogeochemical_advection_scheme,
       biogeochemical_auxiliary_fieilds,
       update_biogeochemical_state!

import Adapt: adapt_structure

#####
##### Define the biogeochemical models
#####

# Discrete version
struct MinimalDiscreteBiogeochemistry{FT, PAR, S, A} <: AbstractBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT

    PAR_field :: PAR
    sinking_velocity :: S
    advection_scheme :: A
end

@inline function (bgc::MinimalDiscreteBiogeochemistry)(i, j, k, grid, ::Val{:P}, clock, fields)
    μ₀ = bgc.growth_rate
    m = bgc.mortality_rate

    z = znode(Center(), k, grid)
    P = @inbounds fields.P[i, j, k]
    PAR = @inbounds fields.PAR[i, j, k]
 
    return (μ₀ * (1 - PAR) - m) * P
end

@inline adapt_structure(to, mdb::MinimalDiscreteBiogeochemistry) = MinimalDiscreteBiogeochemistry(mdb.growth_rate, mdb.mortality_rate, adapt_structure(to, mdb.PAR_field), mdb.sinking_velocity, adapt_structure(to, mdb.advection_scheme))

# Continuous version
struct MinimalContinuousBiogeochemistry{FT, PAR, S, A} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT

    PAR_field :: PAR
    sinking_velocity :: S
    advection_scheme :: A
end

@inline function (bgc::MinimalContinuousBiogeochemistry)(::Val{:P}, x, y, z, t, P, PAR)
    μ₀ = bgc.growth_rate
    m = bgc.mortality_rate
 
    return (μ₀ * (1 - PAR) - m) * P
end

@inline adapt_structure(to, mcb::MinimalContinuousBiogeochemistry) = MinimalContinuousBiogeochemistry(mcb.growth_rate, mcb.mortality_rate, adapt_structure(to, mcb.PAR_field), mcb.sinking_velocity, adapt_structure(to, mcb.advection_scheme))

# Required method definitions

MinimalBiogeochemistry = Union{MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry}

@inline required_biogeochemical_tracers(::MinimalBiogeochemistry) = (:P, )

@inline required_biogeochemical_auxiliary_fields(::MinimalBiogeochemistry) = (:PAR, )

@inline biogeochemical_drift_velocity(bgc::MinimalBiogeochemistry, ::Val{:P}) = bgc.sinking_velocity

@inline biogeochemical_advection_scheme(bgc::MinimalBiogeochemistry, ::Val{:P}) = bgc.advection_scheme

@inline biogeochemical_auxiliary_fieilds(bgc::MinimalBiogeochemistry) = (PAR = bgc.PAR_field, )

# Update state test (won't actually change between calls but here to check it gets called)

@kernel function integrate_PAR!(PAR, grid)
    i, j, k = @index(Global, NTuple)

    z = znode(Center(), k, grid)

    @inbounds PAR[i, j, k] = exp(z / 5)
end

@inline function update_biogeochemical_state!(bgc::MinimalBiogeochemistry, model)
    event = launch!(architecture(model.grid), model.grid, :xyz, integrate_PAR!, bgc.PAR_field, model.grid)
    wait(event)
end

#####
##### Test a `bgc` model in a `model` with `arch`
#####

function test_biogeochemical_model(arch, bgc, model)
    grid = RectilinearGrid(arch; size = (2, 2, 2), extent = (2, 2, 2))

    PAR = CenterField(grid)
    u, v, w = ConstantField.((0.0, 0.0, -200/day))

    biogeochemistry = bgc(1/day, 0.3/day, PAR, (; u, v, w), CenteredSecondOrder())

    model_instance = model(; grid, biogeochemistry)

    set!(model_instance, P = 1.0)

    @test :P in keys(model_instance.tracers)

    time_step!(model_instance, 1.0)

    @test CUDA.@allowscalar any(biogeochemistry.PAR_field .!= 0.0) # update state did get called
    @test CUDA.@allowscalar any(model_instance.tracers.P .!= 1.0) # bgc forcing did something
end

#####
##### Run the tests
#####

@testset "Biogeochemistry" begin
    @info "Testing biogeochemistry setup..."
    for bgc in (MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry), model in (NonhydrostaticModel, HydrostaticFreeSurfaceModel), arch in (GPU(), )
        @testset "$bgc in $model on $arch" begin
            test_biogeochemical_model(arch, bgc, model)
        end
    end
end