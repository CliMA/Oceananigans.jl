include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures: TriadIsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z,
                                       ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization,
                                       compute_diffusivities!

"""
Test that TriadIsopycnalSkewSymmetricDiffusivity can be constructed and time-stepped
with both explicit and vertically implicit time discretizations.
"""
function time_step_with_triad_isopycnal_diffusivity(arch, time_disc)
    grid = RectilinearGrid(arch, size=(4, 4, 8), extent=(100, 100, 100))

    closure = TriadIsopycnalSkewSymmetricDiffusivity(time_disc, Float64,
                                                      κ_skew = 100.0,
                                                      κ_symmetric = 100.0)

    # TriadIsopycnalSkewSymmetricDiffusivity only works with HydrostaticFreeSurfaceModel
    model = HydrostaticFreeSurfaceModel(; grid, closure,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c))

    # Set up a simple stratified initial condition
    bᵢ(x, y, z) = 1e-5 * z
    set!(model, b=bᵢ)

    # Attempt to time-step
    time_step!(model, 1)

    return true
end


@testset "TriadIsopycnalSkewSymmetricDiffusivity" begin
    @info "Testing TriadIsopycnalSkewSymmetricDiffusivity..."

    for arch in archs
        @testset "Time stepping with TriadIsopycnalSkewSymmetricDiffusivity [$arch]" begin
            for time_discretization in [ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization()]
                @info "  Testing time stepping with $(typeof(time_discretization)) on $arch..."
                @test time_step_with_triad_isopycnal_diffusivity(arch, time_discretization)
           end
        end
    end
end
