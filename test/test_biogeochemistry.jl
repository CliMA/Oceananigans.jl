include("dependencies_for_runtests.jl")

using KernelAbstractions
using CUDA

using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry, AbstractContinuousFormBiogeochemistry

import Oceananigans.Biogeochemistry:
       required_biogeochemical_tracers,
       required_biogeochemical_auxiliary_fields,
       biogeochemical_drift_velocity,
       biogeochemical_auxiliary_fields,
       update_biogeochemical_state!

import Adapt: adapt_structure

#####
##### Define the biogeochemical models
#####

# "Minimal" biogeochemistry model for tesing (discrete form) AbstractBiogeochemistry
struct MinimalDiscreteBiogeochemistry{FT, I, S} <: AbstractBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT
    photosynthetic_active_radiation :: I
    sinking_velocity :: S
end

@inline function (bgc::MinimalDiscreteBiogeochemistry)(i, j, k, grid, ::Val{:P}, clock, fields)
    μ₀ = bgc.growth_rate
    m = bgc.mortality_rate
    P = @inbounds fields.P[i, j, k]
    Iᴾᴬᴿ = @inbounds fields.Iᴾᴬᴿ[i, j, k]
    return P * (μ₀ * (1 - Iᴾᴬᴿ) - m)
end

@inline function adapt_structure(to, mdb::MinimalDiscreteBiogeochemistry)
    return MinimalDiscreteBiogeochemistry(mdb.growth_rate,
                                          mdb.mortality_rate,
                                          adapt_structure(to, mdb.photosynthetic_active_radiation),
                                          mdb.sinking_velocity)
end

# "Minimal" biogeochemistry model for tesing AbstractContinuousFormBiogeochemistry
struct MinimalContinuousBiogeochemistry{FT, I, S} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT
    photosynthetic_active_radiation :: I
    sinking_velocity :: S
end

@inline function (bgc::MinimalContinuousBiogeochemistry)(::Val{:P}, x, y, z, t, P, Iᴾᴬᴿ)
    μ₀ = bgc.growth_rate
    m = bgc.mortality_rate
    return (μ₀ * (1 - Iᴾᴬᴿ) - m) * P
end

@inline function adapt_structure(to, mcb::MinimalContinuousBiogeochemistry)
    return MinimalContinuousBiogeochemistry(mcb.growth_rate,
                                            mcb.mortality_rate,
                                            adapt_structure(to, mcb.photosynthetic_active_radiation),
                                            mcb.sinking_velocity)
end

# Required method definitions

const MB = Union{MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry}

@inline          required_biogeochemical_tracers(::MB) = tuple(:P)
@inline required_biogeochemical_auxiliary_fields(::MB) = tuple(:Iᴾᴬᴿ)
@inline      biogeochemical_auxiliary_fields(bgc::MB) = (; Iᴾᴬᴿ = bgc.photosynthetic_active_radiation)
@inline   biogeochemical_drift_velocity(bgc::MB, ::Val{:P}) = bgc.sinking_velocity

# Update state test (won't actually change between calls but here to check it gets called)

@kernel function integrate_photosynthetic_active_radiation!(Iᴾᴬᴿ, grid)
    i, j, k = @index(Global, NTuple)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds Iᴾᴬᴿ[i, j, k] = exp(z / 5)
end

@inline function update_biogeochemical_state!(bgc::MB, model)
    launch!(architecture(model), model.grid, :xyz, integrate_photosynthetic_active_radiation!,
                    bgc.photosynthetic_active_radiation, model.grid)
    return nothing
end

#####
##### Test a `bgc` model in a `model` with `arch`
#####

function test_biogeochemistry(grid, MinimalBiogeochemistryType, ModelType)
    Iᴾᴬᴿ = CenterField(grid)

    u = ZeroField()
    v = ZeroField()
    w = ConstantField(-200/day)
    drift_velocities = (; u, v, w)

    growth_rate = 1/day
    mortality_rate = 0.3/day

    biogeochemistry = MinimalBiogeochemistryType(growth_rate, 
                                                 mortality_rate, 
                                                 Iᴾᴬᴿ, 
                                                 drift_velocities)

    model = ModelType(; grid, biogeochemistry)
    set!(model, P = 1)

    @test :P in keys(model.tracers)

    time_step!(model, 1)

    @test CUDA.@allowscalar any(biogeochemistry.photosynthetic_active_radiation .!= 0) # update state did get called
    @test CUDA.@allowscalar any(model.tracers.P .!= 1) # bgc forcing did something

    return nothing
end

#####
##### Run the tests
#####

@testset "Biogeochemistry" begin
    @info "Testing biogeochemistry setup..."
    for bgc in (MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry), 
        model in (NonhydrostaticModel, HydrostaticFreeSurfaceModel),
        arch in archs,
        grid in (RectilinearGrid(arch; size = (2, 2, 2), extent = (2, 2, 2)), 
                 LatitudeLongitudeGrid(arch; size = (5, 5, 5), longitude = (-180, 180), latitude = (-85, 85), z = (-2, 0)),
                 conformal_cubed_sphere_panel(arch; size = (3, 3, 3), z = (-2, 0)))

        if !((model == NonhydrostaticModel) && ((grid isa LatitudeLongitudeGrid) | (grid isa OrthogonalSphericalShellGrid)))
            @info "Testing $bgc in $model on $grid..."
            test_biogeochemistry(grid, bgc, model)
        end
    end
end
