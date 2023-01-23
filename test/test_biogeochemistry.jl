include("dependencies_for_runtests.jl")

using KernelAbstractions
using CUDA

using Oceananigans.Fields: ConstantField
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry, AbstractContinuousFormBiogeochemistry

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

# "Minimal" biogeochemistry model for tesing (discrete form) AbstractBiogeochemistry
struct MinimalDiscreteBiogeochemistry{FT, PAR, S, A} <: AbstractBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT
    PAR :: PAR
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

@inline function adapt_structure(to, mdb::MinimalDiscreteBiogeochemistry)
    return MinimalDiscreteBiogeochemistry(mdb.growth_rate,
                                          mdb.mortality_rate,
                                          adapt_structure(to, mdb.PAR),
                                          mdb.sinking_velocity,
                                          adapt_structure(to, mdb.advection_scheme))
end

# "Minimal" biogeochemistry model for tesing AbstractContinuousFormBiogeochemistry
struct MinimalContinuousBiogeochemistry{FT, PAR, S, A} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT
    PAR :: PAR
    sinking_velocity :: S
    advection_scheme :: A
end

@inline function (bgc::MinimalContinuousBiogeochemistry)(::Val{:P}, x, y, z, t, P, PAR)
    μ₀ = bgc.growth_rate
    m = bgc.mortality_rate
    return (μ₀ * (1 - PAR) - m) * P
end

@inline function adapt_structure(to, mcb::MinimalContinuousBiogeochemistry)
    return MinimalContinuousBiogeochemistry(mcb.growth_rate,
                                            mcb.mortality_rate,
                                            adapt_structure(to, mcb.PAR),
                                            mcb.sinking_velocity,
                                            adapt_structure(to, mcb.advection_scheme))
end

# Required method definitions

const MB = Union{MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry}

@inline          required_biogeochemical_tracers(::MB) = tuple(:P)
@inline required_biogeochemical_auxiliary_fields(::MB) = tuple(:PAR)
@inline      biogeochemical_auxiliary_fieilds(bgc::MB) = (; PAR = bgc.PAR)
@inline   biogeochemical_drift_velocity(bgc::MB, ::Val{:P}) = bgc.sinking_velocity
@inline biogeochemical_advection_scheme(bgc::MB, ::Val{:P}) = bgc.advection_scheme

# Update state test (won't actually change between calls but here to check it gets called)

@kernel function integrate_PAR!(PAR, grid)
    i, j, k = @index(Global, NTuple)
    z = znode(Center(), k, grid)
    @inbounds PAR[i, j, k] = exp(z / 5)
end

@inline function update_biogeochemical_state!(bgc::MB, model)
    event = launch!(architecture(model), model.grid, :xyz, integrate_PAR!, bgc.PAR, model.grid)
    wait(event)
    return nothing
end

#####
##### Test a `bgc` model in a `model` with `arch`
#####

function test_biogeochemistry!(arch, MinimalBiogeochemistryType, ModelType)
    grid = RectilinearGrid(arch; size = (2, 2, 2), extent = (2, 2, 2))

    PAR = CenterField(grid)
    u, v, w = ConstantField.((0, 0, -200/day))
    velocities = (; u, v, w)
    advection_scheme = CenteredSecondOrder()
    growth_rate = 1/day
    mortality_rate = 0.3/day
    biogeochemistry = MinimalBiogeochemistryType(growth_rate, 
                                                 mortality_rate, 
                                                 PAR, 
                                                 velocities, 
                                                 advection_scheme)

    model = ModelType(; grid, biogeochemistry)
    set!(model, P = 1)

    @test :P in keys(model.tracers)

    time_step!(model, 1)

    @test CUDA.@allowscalar any(biogeochemistry.PAR .!= 0) # update state did get called
    @test CUDA.@allowscalar any(model.tracers.P .!= 1) # bgc forcing did something

    return nothing
end

#####
##### Run the tests
#####

@testset "Biogeochemistry" begin
    @info "Testing biogeochemistry setup..."
    for bgc in (MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry)
        for model in (NonhydrostaticModel, HydrostaticFreeSurfaceModel)
            for arch in archs
                @info "Testing $bgc in $model on $arch..."
                test_biogeochemistry!(arch, bgc, model)
            end
        end
    end
end

