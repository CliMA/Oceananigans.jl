include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Biogeochemistry: AbstractBiogeochemistry
using Oceananigans.Fields: ZeroField

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers, biogeochemical_drift_velocity
import Adapt: adapt_structure

# A tracer that only sinks, with no sources
struct SinkingParticles{W} <: AbstractBiogeochemistry
    sinking_velocity :: W
end

adapt_structure(to, bgc::SinkingParticles) = SinkingParticles(adapt_structure(to, bgc.sinking_velocity))

required_biogeochemical_tracers(::SinkingParticles) = (:D,)
biogeochemical_drift_velocity(bgc::SinkingParticles, ::Val{:D}) = bgc.sinking_velocity

# `w = - speed` on interior faces and `w = 0` on the bottom and the top, so that the sinking flux stays in the domain
function sinking_velocity(grid, speed)
    w = ZFaceField(grid)
    Nz = size(grid, 3)
    set!(w, (x, y, z) -> - speed)
    interior(w, :, :, 1) .= 0
    interior(w, :, :, Nz + 1) .= 0
    fill_halo_regions!(w)
    return (u = ZeroField(), v = ZeroField(), w = w)
end

function center_of_mass(c)
    grid = c.grid
    z = reshape(znodes(grid, Center()), 1, 1, size(grid, 3))
    data = Array(interior(c))
    return sum(z .* data) / sum(data)
end

function sink_blob(arch, timestepper, Δt, steps; speed = 100 / day)
    grid = RectilinearGrid(arch; size = (4, 4, 40), extent = (1, 1, 400), halo = (4, 4, 4),
                           topology = (Periodic, Periodic, Bounded))

    advection = WENO(order = 5, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5))
    biogeochemistry = SinkingParticles(sinking_velocity(grid, speed))

    model = HydrostaticFreeSurfaceModel(grid; biogeochemistry, timestepper,
                                        tracer_advection = advection,
                                        momentum_advection = nothing,
                                        free_surface = nothing)

    set!(model, D = (x, y, z) -> exp(- (z + 100)^2 / 2 / 20^2))

    inventory₀ = sum(Array(interior(model.tracers.D)))
    z₀ = center_of_mass(model.tracers.D)

    for step in 1:steps
        time_step!(model, Δt)
    end

    inventory_drift = abs(sum(Array(interior(model.tracers.D))) - inventory₀) / inventory₀
    descent = z₀ - center_of_mass(model.tracers.D)
    expected_descent = speed * Δt * steps

    return (; inventory_drift, descent, expected_descent)
end

@testset "Sinking tracers with adaptive implicit vertical advection" begin
    for arch in archs, timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @info "  Testing a sinking tracer with AdaptiveVerticallyImplicitDiscretization and $timestepper [$(typeof(arch))]..."

        # Sinking Courant number 0.4: the explicit part carries all of the flux
        explicit = sink_blob(arch, timestepper, 0.04day, 30)

        # Sinking Courant number 4: most of the flux must go through the implicit part
        implicit = sink_blob(arch, timestepper, 0.4day, 3)

        @info "    descent at Courant 0.4: $(explicit.descent) m, at Courant 4: $(implicit.descent) m, expected $(implicit.expected_descent) m"

        @test explicit.inventory_drift < 1e-12
        @test implicit.inventory_drift < 1e-12
        @test isapprox(explicit.descent, explicit.expected_descent, rtol = 0.03)
        @test isapprox(implicit.descent, implicit.expected_descent, rtol = 0.05)
    end
end
