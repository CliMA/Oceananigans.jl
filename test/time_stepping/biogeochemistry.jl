include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using KernelAbstractions
using CUDA

using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OrthogonalSphericalShellGrids: ConformalCubedSpherePanelGrid
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry,
                                    AbstractContinuousFormBiogeochemistry,
                                    biogeochemical_transition,
                                    include_biogeochemistry_transitions

using Oceananigans.Models.NonhydrostaticModels: compute_interior_tendency_contributions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_tracer_tendencies!

import Oceananigans.Models.NonhydrostaticModels
import Oceananigans.Models.HydrostaticFreeSurfaceModels

import Oceananigans.Biogeochemistry:
       required_biogeochemical_tracers,
       required_biogeochemical_auxiliary_fields,
       biogeochemical_drift_velocity,
       biogeochemical_auxiliary_fields,
       update_biogeochemical_state!,
       separate_tracer_transitions

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
@inline       biogeochemical_auxiliary_fields(bgc::MB) = (; Iᴾᴬᴿ = bgc.photosynthetic_active_radiation)
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
##### Two-tracer BGC models. Only `:P` has its transition computed separately when `Split = true`,
##### so both the split and the inline paths are exercised in the same model.
#####

struct SeparableDiscreteBGC{Split, FT, I, S} <: AbstractBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT
    photosynthetic_active_radiation :: I
    sinking_velocity :: S
end

struct SeparableContinuousBGC{Split, FT, I, S} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    mortality_rate :: FT
    photosynthetic_active_radiation :: I
    sinking_velocity :: S
end

const SeparableBGC{Split} = Union{SeparableDiscreteBGC{Split}, SeparableContinuousBGC{Split}} where Split

@inline function (bgc::SeparableDiscreteBGC)(i, j, k, grid, ::Val{:P}, clock, fields)
    P = @inbounds fields.P[i, j, k]
    Z = @inbounds fields.Z[i, j, k]
    Iᴾᴬᴿ = @inbounds fields.Iᴾᴬᴿ[i, j, k]
    return P * (bgc.growth_rate * (1 - Iᴾᴬᴿ) - bgc.mortality_rate) + 0.1 * Z
end

@inline function (bgc::SeparableDiscreteBGC)(i, j, k, grid, ::Val{:Z}, clock, fields)
    P = @inbounds fields.P[i, j, k]
    Z = @inbounds fields.Z[i, j, k]
    return bgc.mortality_rate * P - 0.5 * bgc.growth_rate * Z
end

@inline (bgc::SeparableContinuousBGC)(::Val{:P}, x, y, z, t, P, Z, Iᴾᴬᴿ) =
    P * (bgc.growth_rate * (1 - Iᴾᴬᴿ) - bgc.mortality_rate) + 0.1 * Z

@inline (bgc::SeparableContinuousBGC)(::Val{:Z}, x, y, z, t, P, Z, Iᴾᴬᴿ) =
    bgc.mortality_rate * P - 0.5 * bgc.growth_rate * Z

adapt_structure(to, bgc::SeparableDiscreteBGC{S}) where S =
    SeparableDiscreteBGC{S}(bgc.growth_rate, bgc.mortality_rate,
                            adapt_structure(to, bgc.photosynthetic_active_radiation), bgc.sinking_velocity)

adapt_structure(to, bgc::SeparableContinuousBGC{S}) where S =
    SeparableContinuousBGC{S}(bgc.growth_rate, bgc.mortality_rate,
                              adapt_structure(to, bgc.photosynthetic_active_radiation), bgc.sinking_velocity)

SeparableDiscreteBGC{S}(args...) where S = SeparableDiscreteBGC{S, typeof(args[1]), typeof(args[3]), typeof(args[4])}(args...)
SeparableContinuousBGC{S}(args...) where S = SeparableContinuousBGC{S, typeof(args[1]), typeof(args[3]), typeof(args[4])}(args...)

required_biogeochemical_tracers(::SeparableBGC) = (:P, :Z)
required_biogeochemical_auxiliary_fields(::SeparableBGC) = (:Iᴾᴬᴿ,)
biogeochemical_auxiliary_fields(bgc::SeparableBGC) = (; Iᴾᴬᴿ = bgc.photosynthetic_active_radiation)
biogeochemical_drift_velocity(bgc::SeparableBGC, ::Val{:P}) = bgc.sinking_velocity
separate_tracer_transitions(::SeparableBGC{true}) = (:P,)

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

    if ModelType == HydrostaticFreeSurfaceModel && grid isa OrthogonalSphericalShellGrid
        model = ModelType(grid; biogeochemistry, momentum_advection = VectorInvariant())
    else
        model = ModelType(grid; biogeochemistry)
    end
    set!(model, P = 1)

    @test :P in keys(model.tracers)

    time_step!(model, 1)

    @test @allowscalar any(biogeochemistry.photosynthetic_active_radiation .!= 0) # update state did get called
    @test @allowscalar any(model.tracers.P .!= 1) # bgc forcing did something

    return nothing
end

#####
##### Building and comparing models
#####

function separable_bgc(BGCType, grid, split; drift = true)
    Iᴾᴬᴿ = CenterField(grid)
    set!(Iᴾᴬᴿ, (x, y, z) -> exp(z / 5) / 2)
    w = drift ? ConstantField(-200 / day) : ZeroField()
    sinking_velocity = (; u = ZeroField(), v = ZeroField(), w)
    return BGCType{split}(1 / day, 0.3 / day, Iᴾᴬᴿ, sinking_velocity)
end

function separable_model(ModelType, grid, BGCType, split, timestepper; drift = true)
    biogeochemistry = separable_bgc(BGCType, grid, split; drift)
    model = ModelType(grid; biogeochemistry, timestepper)
    kw = ModelType == HydrostaticFreeSurfaceModel && grid.z isa MutableVerticalDiscretization ? (; η = 0.1) : (;)
    set!(model; P = (x, y, z) -> 1 + sin(x) * exp(z / 3) / 4, Z = (x, y, z) -> 0.5 + cos(y) * exp(z / 4) / 8, kw...)
    return model
end

same(a, b) = isapprox(Array(interior(a)), Array(interior(b)); rtol = 1e-12)

# `build_grid` is called once per model: a z-star grid is mutated by its model, so it cannot be shared
function test_separate_transitions(ModelType, build_grid, BGCType, timestepper)
    reference = separable_model(ModelType, build_grid(), BGCType, false, timestepper)
    split     = separable_model(ModelType, build_grid(), BGCType, true,  timestepper)
    nodrift   = separable_model(ModelType, build_grid(), BGCType, true,  timestepper; drift = false)

    @test separate_tracer_transitions(reference.biogeochemistry) == ()
    @test separate_tracer_transitions(split.biogeochemistry) == (:P,)

    P₀ = Array(interior(split.tracers.P))

    for _ in 1:4
        time_step!(reference, 60)
        time_step!(split, 60)
        time_step!(nodrift, 60)
    end

    for name in (:P, :Z)
        @test same(reference.tracers[name], split.tracers[name])
        @test same(reference.timestepper.Gⁿ[name], split.timestepper.Gⁿ[name])
        hasproperty(reference.timestepper, :G⁻) &&
            @test same(reference.timestepper.G⁻[name], split.timestepper.G⁻[name])
    end

    # The tracer evolved and the drift velocity still advects
    @test !isapprox(Array(interior(split.tracers.P)), P₀; rtol = 1e-6)
    @test !same(split.tracers.P, nodrift.tracers.P)

    return nothing
end

compute_inline_tendencies!(model::NonhydrostaticModel) = compute_interior_tendency_contributions!(model, :xyz)
compute_inline_tendencies!(model::HydrostaticFreeSurfaceModel) = compute_hydrostatic_tracer_tendencies!(model, :xyz)

compute_separate_transitions!(model::NonhydrostaticModel) = NonhydrostaticModels.compute_biogeochemical_transitions!(model, :xyz)
compute_separate_transitions!(model::HydrostaticFreeSurfaceModel) = HydrostaticFreeSurfaceModels.compute_biogeochemical_transitions!(model, :xyz)

# Check that the tracer tendency kernel omits the transition of `:P` when it is split off,
# and that the separate kernel adds exactly the omitted transition (and nothing for `:Z`)
function test_transition_computed_separately(ModelType, grid, BGCType)
    reference = separable_model(ModelType, grid, BGCType, false, :QuasiAdamsBashforth2)
    split     = separable_model(ModelType, grid, BGCType, true,  :QuasiAdamsBashforth2)

    compute_inline_tendencies!(reference)
    compute_inline_tendencies!(split)

    Gʳ = reference.timestepper.Gⁿ
    Gˢ = split.timestepper.Gⁿ

    # Only the transition of `:P` is missing from the inline tendency
    @test same(Gʳ.Z, Gˢ.Z)
    @test !same(Gʳ.P, Gˢ.P)

    inline_GZ = Array(interior(Gˢ.Z))

    compute_separate_transitions!(reference)
    compute_separate_transitions!(split)

    # The separate kernel adds the transition of `:P` only, and nothing for the inline model
    @test same(Gʳ.P, Gˢ.P)
    @test Array(interior(Gˢ.Z)) == inline_GZ
    @test same(Gʳ.Z, Gˢ.Z)

    return nothing
end

#####
##### Run the tests
#####

@testset "Biogeochemistry" begin
    @info "Testing biogeochemistry setup..."
    for arch in archs
        grids = (RectilinearGrid(arch; size = (2, 2, 2), extent = (2, 2, 2)),
                 LatitudeLongitudeGrid(arch; size = (5, 5, 5), longitude = (-180, 180), latitude = (-85, 85), z = (-2, 0)),
                 ConformalCubedSpherePanelGrid(arch; size = (3, 3, 3), z = (-2, 0)))

        for bgc in (MinimalDiscreteBiogeochemistry, MinimalContinuousBiogeochemistry),
            model in (NonhydrostaticModel, HydrostaticFreeSurfaceModel),
            grid in grids

            if !((model == NonhydrostaticModel) && ((grid isa LatitudeLongitudeGrid) | (grid isa OrthogonalSphericalShellGrid)))
                @info "Testing $bgc in $model on $grid..."
                test_biogeochemistry(grid, bgc, model)
            end
        end
    end

    @testset "Separately computed biogeochemical transitions" begin
        @info "Testing separately computed biogeochemical transitions..."

        @testset "include_biogeochemistry_transitions" begin
            grid = RectilinearGrid(size = (2, 2, 2), extent = (1, 1, 1))
            for BGCType in (SeparableDiscreteBGC, SeparableContinuousBGC)
                inline = separable_bgc(BGCType, grid, false)
                split  = separable_bgc(BGCType, grid, true)

                @test @inferred(include_biogeochemistry_transitions(inline, Val(:P)))
                @test @inferred(include_biogeochemistry_transitions(inline, Val(:Z)))
                @test !@inferred(include_biogeochemistry_transitions(split, Val(:P)))
                @test @inferred(include_biogeochemistry_transitions(split, Val(:Z)))
                @test @inferred(include_biogeochemistry_transitions(nothing, Val(:P)))

                # The result is known at compile time, so the kernel argument `Val(...)` is concretely inferred
                @test @inferred((bgc -> Val(include_biogeochemistry_transitions(bgc, Val(:P))))(split)) === Val(false)
                @test @inferred((bgc -> Val(include_biogeochemistry_transitions(bgc, Val(:Z))))(split)) === Val(true)
            end
        end

        for arch in archs
            grid = RectilinearGrid(arch; size = (6, 6, 6), x = (0, 2π), y = (0, 2π), z = (-2, 0),
                                   topology = (Periodic, Periodic, Bounded))

            for ModelType in (NonhydrostaticModel, HydrostaticFreeSurfaceModel),
                BGCType in (SeparableDiscreteBGC, SeparableContinuousBGC)

                @testset "Transition is computed separately: $(nameof(ModelType)), $(nameof(BGCType)) [$(typeof(arch))]" begin
                    test_transition_computed_separately(ModelType, grid, BGCType)
                end
            end
        end

        for arch in archs
            underlying_grid(z = (-2, 0)) = RectilinearGrid(arch; size = (6, 6, 6), x = (0, 2π), y = (0, 2π), z,
                                                           topology = (Periodic, Periodic, Bounded))
            bottom(x, y) = -2 + 1.2 * exp(-((x - π)^2 + (y - π)^2))

            rectilinear     = () -> underlying_grid()
            immersed_map    = () -> ImmersedBoundaryGrid(underlying_grid(), GridFittedBottom(bottom); active_cells_map = true)
            immersed_no_map = () -> ImmersedBoundaryGrid(underlying_grid(), GridFittedBottom(bottom); active_cells_map = false)
            zstar           = () -> underlying_grid(MutableVerticalDiscretization((-2, 0)))

            cases = ((NonhydrostaticModel,         (:QuasiAdamsBashforth2, :RungeKutta3),      (rectilinear, immersed_map, immersed_no_map)),
                     (HydrostaticFreeSurfaceModel, (:QuasiAdamsBashforth2, :SplitRungeKutta3), (rectilinear, immersed_map, immersed_no_map, zstar)))

            for (ModelType, timesteppers, grids) in cases,
                BGCType in (SeparableDiscreteBGC, SeparableContinuousBGC),
                timestepper in timesteppers,
                build_grid in grids

                # Immersed and z-star grids are only tested with the first timestepper to limit compilation time
                build_grid === rectilinear || timestepper == first(timesteppers) || continue

                @testset "$(nameof(ModelType)), $(nameof(BGCType)), $timestepper, $(summary(build_grid())) [$(typeof(arch))]" begin
                    test_separate_transitions(ModelType, build_grid, BGCType, timestepper)
                end
            end
        end
    end
end
