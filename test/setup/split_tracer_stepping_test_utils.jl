using Random
using Statistics: mean
using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Fields: ZeroField, interior
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: MutableVerticalDiscretization, znodes, topology
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers, biogeochemical_drift_velocity
import Adapt: adapt_structure

#####
##### Diagnostics shared by the split tracer stepping tests and validation scripts
#####

"""
    active_cells(grid)

Return a CPU `Bool` array that is `true` on the active (non-immersed) tracer cells of `grid`.
"""
function active_cells(grid)
    mask = CenterField(grid)
    set!(mask, 1)
    mask_immersed_field!(mask)
    return Array(interior(mask)) .== 1
end

include(joinpath(@__DIR__, "volume_integrals.jl"))

tracer_inventory(c) = volume_integral(c)

function maximum_uniform_deviation(c, value, active)
    data = Array(interior(c))
    return maximum(abs, data[active] .- value) / abs(value)
end

function relative_errors(c, reference, active)
    a = Array(interior(c))[active]
    b = Array(interior(reference))[active]
    L² = sqrt(sum(abs2, a .- b) / sum(abs2, b))
    L∞ = maximum(abs, a .- b) / maximum(abs, b)
    return L², L∞
end

"""
    long_step_courant_numbers(model, T; drift = nothing)

Return the horizontal and the vertical Courant numbers of the slow velocities of the most recent
long step of length `T`. `drift` adds a vertical drift velocity to the vertical Courant number.
"""
function long_step_courant_numbers(model, T; drift = nothing)
    ū, v̄, w̄ = model.tracer_time_step_splitting.velocities
    grid = model.grid
    horizontal = T / cell_advection_timescale(grid, (u=ū, v=v̄, w=ZeroField()))
    vertical_velocity = isnothing(drift) ? w̄ : compute!(Field(w̄ + drift))
    vertical = T / cell_advection_timescale(grid, (u=ZeroField(), v=ZeroField(), w=vertical_velocity))
    return horizontal, vertical
end

#####
##### A barotropically non-divergent overturning circulation
#####

"""
    overturning_velocity(grid)

Return a CPU array with the interior of an `XFaceField` holding an overturning velocity
`u = - δz ψ / Δz` in the x-z plane. The streamfunction `ψ` vanishes at the top, at the bottom
and on every face that touches an immersed cell, so that `∫ u dz = 0` in every column and
`u = 0` on immersed faces. With `northern_rows_at_rest > 0` the flow also vanishes in the northernmost rows
(for example, next to the fold of a `TripolarGrid`). The free surface stays at rest and the vertical velocity at the top
vanishes, which makes the tracer inventory exactly conserved also with a static vertical coordinate.
"""
function overturning_velocity(grid; northern_rows_at_rest = 0)
    Nx, Ny, Nz = size(grid)
    active = active_cells(grid)
    TX = topology(grid, 1)
    Nu = TX == Bounded ? Nx + 1 : Nx

    wet(i, j, k) = 1 ≤ k ≤ Nz && active[mod1(i, Nx), j, k] && (TX != Bounded || 1 ≤ i ≤ Nx)

    ψ = zeros(Nu, Ny, Nz + 1)
    for k in 1:Nz+1, j in 1:Ny, i in 1:Nu
        surrounded = wet(i-1, j, k-1) && wet(i, j, k-1) && wet(i-1, j, k) && wet(i, j, k) && j ≤ Ny - northern_rows_at_rest
        ψ[i, j, k] = surrounded ? sinpi(2 * (i - 1) / Nu) + 1 / 2 * cospi(3 * (k - 1) / Nz) * sinpi(j / Ny) : 0
    end

    underlying_grid = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Δz = vec(Array(zspacings(underlying_grid, Center(), Center(), Center())))
    u = zeros(Nu, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nu
        u[i, j, k] = - (ψ[i, j, k+1] - ψ[i, j, k]) / Δz[k]
    end

    return u
end

@inline function oscillating_overturning_forcing(i, j, k, grid, clock, model_fields, p)
    return @inbounds - p.amplitude * p.frequency * sin(p.frequency * clock.time) * p.structure[i, j, k]
end

"""
    overturning_model(grid; ratio = 1, speed = 0.1, period = 1hour, ...)

A model whose barotropically non-divergent overturning circulation oscillates in time
with `period`, and whose free surface stays at rest.
"""
function overturning_model(grid; ratio = 1,
                           slow_tracers = (:c, :constant, :smooth),
                           speed = 0.1,
                           period = 1hour,
                           tracer_advection = WENO(order=5),
                           timestepper = :SplitRungeKutta3,
                           free_surface = SplitExplicitFreeSurface(grid; substeps=8),
                           closure = nothing,
                           northern_rows_at_rest = 0)

    structure = XFaceField(grid)
    u = overturning_velocity(grid; northern_rows_at_rest)
    u = speed .* u ./ maximum(abs, u)
    set!(structure, u)
    fill_halo_regions!(structure)

    parameters = (; amplitude = convert(eltype(grid), 1/2), frequency = convert(eltype(grid), 2π / period), structure)
    u_forcing = Forcing(oscillating_overturning_forcing; discrete_form = true, parameters)

    splitting = isnothing(ratio) ? nothing : TracerTimeStepSplitting(tracers = slow_tracers, ratio = ratio)

    model = HydrostaticFreeSurfaceModel(grid; free_surface, timestepper, tracer_advection, closure,
                                        momentum_advection = nothing,
                                        tracers = (:fast, :c, :constant, :smooth),
                                        forcing = (; u = u_forcing),
                                        tracer_time_step_splitting = splitting)

    Random.seed!(1234)
    set!(model, u = structure, c = (x, y, z) -> rand(), constant = 1, fast = (x, y, z) -> rand(),
         smooth = smooth_tracer_initial_condition(grid))

    return model
end

"""
    baroclinic_adjustment_model(grid; ratio = 1, ...)

A z-star model with a buoyancy front whose adjustment moves the free surface.
"""
function baroclinic_adjustment_model(grid; ratio = 1,
                                     slow_tracers = (:c, :constant, :smooth),
                                     tracer_advection = WENO(order=5),
                                     vertical_coordinate = ZStarCoordinate(),
                                     free_surface = SplitExplicitFreeSurface(grid; substeps=8),
                                     closure = nothing,
                                     buoyancy_contrast = 0.05)

    splitting = isnothing(ratio) ? nothing : TracerTimeStepSplitting(tracers = slow_tracers, ratio = ratio)

    model = HydrostaticFreeSurfaceModel(grid; free_surface, tracer_advection, closure, vertical_coordinate,
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c, :constant, :smooth),
                                        tracer_time_step_splitting = splitting)

    x = first(nodes(grid, Center(), Center(), Center()))
    x₀ = (minimum(x) + maximum(x)) / 2
    bᵢ(x, y, z) = x < x₀ ? buoyancy_contrast : buoyancy_contrast / 5

    Random.seed!(1234)
    set!(model, b = bᵢ, c = (x, y, z) -> rand(), constant = 1, smooth = smooth_tracer_initial_condition(grid))

    return model
end

function smooth_tracer_initial_condition(grid)
    x, y, z = nodes(grid, Center(), Center(), Center())
    x₀, Lx = (minimum(x) + maximum(x)) / 2, maximum(x) - minimum(x)
    y₀, Ly = (minimum(y) + maximum(y)) / 2, maximum(y) - minimum(y)
    z₀, Lz = (minimum(z) + maximum(z)) / 2, maximum(z) - minimum(z)
    return (x, y, z) -> 1 + exp(- ((x - x₀) / (Lx / 4))^2 - ((y - y₀) / (Ly / 3))^2 - ((z - z₀) / (Lz / 3))^2)
end

#####
##### A minimal nutrient-phytoplankton-zooplankton-detritus model
#####

"""
    MinimalNPZD(; growth_rate, grazing_rate, mortality_rate, remineralization_rate,
                  half_saturation, light_scale, sinking_velocity)

A minimal nutrient-phytoplankton-zooplankton-detritus model whose sources sum to zero pointwise,
so that total nitrogen `N + P + Z + D` is conserved. Detritus sinks with `sinking_velocity`.
"""
struct MinimalNPZD{FT, W} <: AbstractContinuousFormBiogeochemistry
    growth_rate :: FT
    grazing_rate :: FT
    mortality_rate :: FT
    remineralization_rate :: FT
    half_saturation :: FT
    light_scale :: FT
    sinking_velocity :: W
end

function MinimalNPZD(grid; growth_rate = 1/day, grazing_rate = 1/2day, mortality_rate = 1/10day,
                     remineralization_rate = 1/5day, half_saturation = 1, light_scale = 10,
                     sinking_speed = 0)

    FT = eltype(grid)
    sinking_velocity = sinking_speed == 0 ? nothing : sinking_velocity_field(grid, sinking_speed)

    return MinimalNPZD(convert(FT, growth_rate), convert(FT, grazing_rate), convert(FT, mortality_rate),
                       convert(FT, remineralization_rate), convert(FT, half_saturation),
                       convert(FT, light_scale), sinking_velocity)
end

adapt_structure(to, bgc::MinimalNPZD) =
    MinimalNPZD(bgc.growth_rate, bgc.grazing_rate, bgc.mortality_rate, bgc.remineralization_rate,
                bgc.half_saturation, bgc.light_scale, adapt_structure(to, bgc.sinking_velocity))

"""
    sinking_velocity_field(grid, speed)

A downward drift velocity `(u, v, w)` with `w = - speed` on every interior face above an active cell
and `w = 0` on the bottom, the top and immersed faces, so the sinking flux leaves no closed domain.
"""
function sinking_velocity_field(grid, speed)
    w = ZFaceField(grid)
    active = active_cells(grid)
    Nx, Ny, Nz = size(grid)
    data = zeros(Nx, Ny, Nz + 1)
    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        data[i, j, k] = active[i, j, k-1] && active[i, j, k] ? - speed : 0
    end
    set!(w, data)
    fill_halo_regions!(w)
    return (u = ZeroField(), v = ZeroField(), w = w)
end

required_biogeochemical_tracers(::MinimalNPZD) = (:N, :P, :Z, :D)

biogeochemical_drift_velocity(bgc::MinimalNPZD, ::Val{:D}) = bgc.sinking_velocity
biogeochemical_drift_velocity(bgc::MinimalNPZD{<:Any, Nothing}, ::Val{:D}) = nothing

@inline npzd_growth(bgc, z, N, P) = bgc.growth_rate * exp(z / bgc.light_scale) * N / (N + bgc.half_saturation) * P
@inline npzd_grazing(bgc, P, Z) = bgc.grazing_rate * P * Z
@inline npzd_mortality(bgc, P) = bgc.mortality_rate * P
@inline npzd_remineralization(bgc, D) = bgc.remineralization_rate * D

@inline (bgc::MinimalNPZD)(::Val{:N}, x, y, z, t, N, P, Z, D) = - npzd_growth(bgc, z, N, P) + npzd_remineralization(bgc, D)
@inline (bgc::MinimalNPZD)(::Val{:P}, x, y, z, t, N, P, Z, D) =   npzd_growth(bgc, z, N, P) - npzd_grazing(bgc, P, Z) - npzd_mortality(bgc, P)
@inline (bgc::MinimalNPZD)(::Val{:Z}, x, y, z, t, N, P, Z, D) =   npzd_grazing(bgc, P, Z)
@inline (bgc::MinimalNPZD)(::Val{:D}, x, y, z, t, N, P, Z, D) =   npzd_mortality(bgc, P) - npzd_remineralization(bgc, D)

"""
    npzd_model(grid; ratio, biogeochemistry_substeps, ...)

A z-star baroclinic adjustment carrying the `MinimalNPZD` tracers as the slow group.
"""
function npzd_model(grid; ratio = 1,
                    biogeochemistry_substeps = nothing,
                    sinking_speed = 10 / day,
                    remineralization_rate = 1 / 5day,
                    growth_rate = 1 / day,
                    tracer_advection = WENO(order=5),
                    vertical_coordinate = ZStarCoordinate(),
                    closure = nothing,
                    buoyancy_contrast = 0.05)

    biogeochemistry = MinimalNPZD(grid; sinking_speed, remineralization_rate, growth_rate)

    splitting = isnothing(ratio) ? nothing :
                TracerTimeStepSplitting(tracers = (:N, :P, :Z, :D), ratio = ratio, biogeochemistry_substeps = biogeochemistry_substeps)

    model = HydrostaticFreeSurfaceModel(grid; biogeochemistry, tracer_advection, closure, vertical_coordinate,
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=8),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :N, :P, :Z, :D),
                                        tracer_time_step_splitting = splitting)

    x = first(nodes(grid, Center(), Center(), Center()))
    x₀ = (minimum(x) + maximum(x)) / 2
    bᵢ(x, y, z) = x < x₀ ? buoyancy_contrast : buoyancy_contrast / 5

    Random.seed!(1234)
    set!(model, b = bᵢ,
         N = (x, y, z) -> 4 + rand(),
         P = (x, y, z) -> 0.5 + 0.1 * rand(),
         Z = (x, y, z) -> 0.2 + 0.05 * rand(),
         D = (x, y, z) -> 0.3 + 0.1 * rand())

    return model
end

total_nitrogen(model) = sum(tracer_inventory(model.tracers[name]) for name in (:N, :P, :Z, :D))

"""
    random_flow_model(grid; ratio = 1, speed = 1, ...)

A model with random, horizontally non-divergent initial velocities of order `speed` derived from a random
streamfunction, and a buoyancy front, as in the tripolar z-star conservation tests.
"""
function random_flow_model(grid; ratio = 1,
                           slow_tracers = (:c, :constant, :smooth),
                           speed = 1,
                           tracer_advection = WENO(order=5),
                           vertical_coordinate = ZStarCoordinate(),
                           free_surface = SplitExplicitFreeSurface(grid; substeps=8))

    splitting = isnothing(ratio) ? nothing : TracerTimeStepSplitting(tracers = slow_tracers, ratio = ratio)

    model = HydrostaticFreeSurfaceModel(grid; free_surface, tracer_advection, vertical_coordinate,
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c, :constant, :smooth),
                                        tracer_time_step_splitting = splitting)

    Random.seed!(1234)
    ψ = Field{Center, Center, Center}(grid)
    Δ = (mean(xspacings(grid, Face(), Face(), Center())) + mean(yspacings(grid, Face(), Face(), Center()))) / 2
    set!(ψ, speed * Δ * rand(size(ψ)...))
    fill_halo_regions!(ψ)

    set!(model, u = ∂y(ψ), v = -∂x(ψ), b = (x, y, z) -> y < 0 ? 0.06 : 0.01,
         c = (x, y, z) -> rand(), constant = 1, smooth = smooth_tracer_initial_condition(grid))

    return model
end
