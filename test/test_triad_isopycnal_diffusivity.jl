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

"""
Test that flux calculations work correctly and receive the clock argument.
This specifically tests for the bug where the clock argument was missing in
the explicit_R₃₃_∂z_c function call.
"""
function test_triad_flux_calculations_with_clock(arch, time_disc)
    grid = RectilinearGrid(arch, size=(4, 4, 8), extent=(100, 100, 100))

    closure = TriadIsopycnalSkewSymmetricDiffusivity(time_disc, Float64,
                                                      κ_skew = 100.0,
                                                      κ_symmetric = 100.0)

    # Create necessary fields
    tracers = TracerFields((:b, :c), grid)
    clock = Clock(time=0.0)
    buoyancy = BuoyancyTracer()

    # Set up a stratified state with lateral gradients
    b = tracers.b
    c = tracers.c

    # Create a tilted isopycnal surface: b = α*x + β*z
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        x, y, z = nodes((Center, Center, Center), grid, i, j, k)
        b[i, j, k] = 1e-5 * x + 1e-4 * z
        c[i, j, k] = 1.0 + 0.1 * sin(2π * x / 100) * sin(2π * z / 100)
    end

    fill_halo_regions!(tracers)

    # Compute diffusivity fields if needed for vertically implicit
    K = DiffusivityFields(grid, (:b, :c), NamedTuple(), closure)

    if time_disc isa VerticallyImplicitTimeDiscretization
        # Create a mock model for compute_diffusivities!
        mock_model = (architecture = arch,
                     grid = grid,
                     clock = clock,
                     tracers = tracers,
                     buoyancy = buoyancy)

        compute_diffusivities!(K, closure, mock_model)
    end

    model_fields = datatuple(tracers)

    # Test that flux calculations work without error
    # This will fail if clock argument is missing
    @allowscalar begin
        i, j, k = 2, 2, 4

        # Test diffusive fluxes - these should not error
        # The key is that these functions internally call explicit_R₃₃_∂z_c
        # which now requires the clock argument
        flux_x = diffusive_flux_x(i, j, k, grid, closure, K, Val(1), c, clock, model_fields, buoyancy)
        flux_y = diffusive_flux_y(i, j, k, grid, closure, K, Val(1), c, clock, model_fields, buoyancy)
        flux_z = diffusive_flux_z(i, j, k, grid, closure, K, Val(1), c, clock, model_fields, buoyancy)

        # Fluxes should be real numbers (not NaN or Inf)
        @test isfinite(flux_x)
        @test isfinite(flux_y)
        @test isfinite(flux_z)
    end

    return true
end

"""
Test that time discretization affects flux computation correctly.
Explicit should compute R₃₃ contribution, implicit should precompute it.
"""
function test_triad_time_discretization_behavior(arch)
    grid = RectilinearGrid(arch, size=(4, 4, 8), extent=(100, 100, 100))

    # Test with both time discretizations
    for time_disc in (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
        closure = TriadIsopycnalSkewSymmetricDiffusivity(time_disc, Float64,
                                                          κ_skew = 100.0,
                                                          κ_symmetric = 100.0)

        K = DiffusivityFields(grid, (:b, :c), NamedTuple(), closure)

        if time_disc isa VerticallyImplicitTimeDiscretization
            # Should have ϵκR₃₃ field
            @test hasfield(typeof(K), :ϵκR₃₃)
            @test K.ϵκR₃₃ isa Field
        else
            # Should be nothing for explicit
            @test K === nothing
        end
    end

    return true
end

"""
Test that the closure works with time-varying diffusivities.
"""
function test_triad_with_time_varying_diffusivity(arch)
    grid = RectilinearGrid(arch, size=(4, 4, 8), extent=(100, 100, 100))

    # Time-varying diffusivity
    κ_time_varying(x, y, z, t) = 100.0 * (1.0 + 0.1 * sin(2π * t / 86400))

    closure = TriadIsopycnalSkewSymmetricDiffusivity(ExplicitTimeDiscretization(), Float64,
                                                      κ_skew = κ_time_varying,
                                                      κ_symmetric = κ_time_varying)

    model = HydrostaticFreeSurfaceModel(; grid, closure,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c))

    # Set up initial condition
    bᵢ(x, y, z) = 1e-5 * z
    set!(model, b=bᵢ)

    # Time-step at different times - the clock should be passed correctly
    for n in 1:3
        time_step!(model, 1000.0)  # 1000 second time steps
        @test model.clock.time ≈ n * 1000.0
    end

    return true
end

@testset "TriadIsopycnalSkewSymmetricDiffusivity" begin
    @info "Testing TriadIsopycnalSkewSymmetricDiffusivity..."

    for arch in archs
        @testset "Time stepping with TriadIsopycnalSkewSymmetricDiffusivity [$arch]" begin
            @info "  Testing time stepping with explicit time discretization on $arch..."
            @test time_step_with_triad_isopycnal_diffusivity(arch, ExplicitTimeDiscretization())

            @info "  Testing time stepping with vertically implicit time discretization on $arch..."
            @test time_step_with_triad_isopycnal_diffusivity(arch, VerticallyImplicitTimeDiscretization())
        end

        @testset "Flux calculations with clock argument [$arch]" begin
            @info "  Testing flux calculations receive clock (explicit) on $arch..."
            @test test_triad_flux_calculations_with_clock(arch, ExplicitTimeDiscretization())

            @info "  Testing flux calculations receive clock (implicit) on $arch..."
            @test test_triad_flux_calculations_with_clock(arch, VerticallyImplicitTimeDiscretization())
        end

        @testset "Time discretization behavior [$arch]" begin
            @info "  Testing time discretization affects diffusivity fields on $arch..."
            @test test_triad_time_discretization_behavior(arch)
        end

        @testset "Time-varying diffusivity [$arch]" begin
            @info "  Testing time-varying diffusivity with clock on $arch..."
            @test test_triad_with_time_varying_diffusivity(arch)
        end
    end
end
