include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Test
using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry,
                                    AbstractContinuousFormBiogeochemistry,
                                    TransitionFree,
                                    biogeochemical_transition,
                                    tendency_biogeochemistry

import Oceananigans.Biogeochemistry:
       required_biogeochemical_tracers,
       required_biogeochemical_auxiliary_fields,
       biogeochemical_drift_velocity,
       biogeochemical_auxiliary_fields,
       separate_transition_tracers

import Adapt: adapt_structure

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
separate_transition_tracers(::SeparableBGC{true}) = (:P,)

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

    @test separate_transition_tracers(reference.biogeochemistry) == ()
    @test separate_transition_tracers(split.biogeochemistry) == (:P,)

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

@testset "Separately computed biogeochemical transitions" begin
    @info "Testing separately computed biogeochemical transitions..."

    @testset "TransitionFree and tendency_biogeochemistry" begin
        grid = RectilinearGrid(size = (2, 2, 2), extent = (1, 1, 1))
        for BGCType in (SeparableDiscreteBGC, SeparableContinuousBGC)
            inline = separable_bgc(BGCType, grid, false)
            split  = separable_bgc(BGCType, grid, true)

            @test @inferred(tendency_biogeochemistry(inline, Val(:P))) === inline
            @test @inferred(tendency_biogeochemistry(split, Val(:Z))) === split
            @test @inferred(tendency_biogeochemistry(split, Val(:P))) isa TransitionFree
            @test @inferred(tendency_biogeochemistry(nothing, Val(:P))) === nothing

            tf = tendency_biogeochemistry(split, Val(:P))
            @test !(tf isa AbstractBiogeochemistry)
            @test tf.biogeochemistry === split
            @test biogeochemical_drift_velocity(tf, Val(:P)) === biogeochemical_drift_velocity(split, Val(:P))
            @test biogeochemical_auxiliary_fields(tf) === biogeochemical_auxiliary_fields(split)
            @test adapt_structure(nothing, tf) isa TransitionFree

            fields = (; P = CenterField(grid), Z = CenterField(grid), Iᴾᴬᴿ = CenterField(grid))
            @test biogeochemical_transition(1, 1, 1, grid, tf, Val(:P), Clock(grid), fields) == 0
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
