include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, DiffusiveFormulation
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedTracer
using Oceananigans.OutputReaders: FieldTimeSeries, InMemory

@testset "Prescribed Tracers" begin
    @info "Testing prescribed tracers..."

    for arch in archs

        #####
        ##### Test 1: Basic construction and non-stepping with BuoyancyTracer
        #####

        @testset "Basic construction and non-stepping [$arch]" begin
            @info "  Testing basic PrescribedTracer construction [$arch]..."

            grid = RectilinearGrid(arch, size=(4, 4),
                                   x=(0, 1), z=(-1, 0),
                                   topology=(Bounded, Flat, Bounded))

            times = [0.0, 10.0]
            b_fts = FieldTimeSeries{Center, Center, Center}(grid, times)
            set!(b_fts, (x, z) -> x / 1000 + 1e-5 * z, 1)
            set!(b_fts, (x, z) -> x / 1000 + 1e-5 * z, 2)

            closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)

            model = HydrostaticFreeSurfaceModel(grid;
                buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (b = PrescribedTracer(b_fts),))

            # Time-step the model
            time_step!(model, 1.0)

            # Prescribed tracer b should be a TimeSeriesInterpolation, not a Field
            @test !(model.tracers.b isa Field)

            # No NaN values should be present
            @test isfinite(model.tracers.b[1, 1, 1])

            @info "    Basic construction test passed [$arch]"
        end

        #####
        ##### Test 2: Prescribed tracer with Function input
        #####

        @testset "PrescribedTracer with Function [$arch]" begin
            @info "  Testing PrescribedTracer with Function [$arch]..."

            grid = RectilinearGrid(arch, size=(4, 4),
                                   x=(0, 1), z=(-1, 0),
                                   topology=(Bounded, Flat, Bounded))

            # On Flat y-topology, FunctionField expects func(x, z, t)
            b_func(x, z, t) = x / 1000 + 1e-5 * z

            closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)

            model = HydrostaticFreeSurfaceModel(grid;
                buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (b = PrescribedTracer(b_func),))

            # Time-step the model
            for _ in 1:5
                time_step!(model, 1.0)
            end

            @test !(model.tracers.b isa Field)
            @test isfinite(model.tracers.b[1, 1, 1])

            @info "    PrescribedTracer with Function test passed [$arch]"
        end

        #####
        ##### Test 3: Prescribed tracer alongside prognostic tracers
        #####

        @testset "Prescribed and prognostic tracers [$arch]" begin
            @info "  Testing prescribed + prognostic tracers [$arch]..."

            grid = RectilinearGrid(arch, size=(4, 4),
                                   x=(0, 1), z=(-1, 0),
                                   topology=(Bounded, Flat, Bounded))

            times = [0.0, 10.0]
            b_fts = FieldTimeSeries{Center, Center, Center}(grid, times)
            set!(b_fts, (x, z) -> x / 1000 + 1e-5 * z, 1)
            set!(b_fts, (x, z) -> x / 1000 + 1e-5 * z, 2)

            closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)

            # Pass prescribed b and prognostic c as a CenterField
            c_field = CenterField(grid)
            model = HydrostaticFreeSurfaceModel(grid;
                buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (b = PrescribedTracer(b_fts), c = c_field))

            set!(model, c = 1.0)
            time_step!(model, 1.0)

            @test !(model.tracers.b isa Field)
            @test model.tracers.c isa Field
            @test !any(isnan, model.tracers.c)

            @info "    Prescribed + prognostic tracers test passed [$arch]"
        end

        #####
        ##### Test 4: Replicate GM test with prescribed buoyancy tracer
        #####
        ##### Run the same ISSD test as test_gm_infinite_slope.jl but with
        ##### prescribed `b` from the first run.
        #####

        @testset "Replicate GM test with prescribed tracer [$arch]" begin
            @info "  Testing GM replication with prescribed tracer [$arch]..."

            nx = 8
            nz = 8

            grid = RectilinearGrid(arch,
                                   size = (nx, nz),
                                   x = (0, 1),
                                   z = (-1, 0),
                                   topology = (Bounded, Flat, Bounded))

            closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)

            # --- Run 1: Standard prognostic simulation ---
            model1 = HydrostaticFreeSurfaceModel(grid;
                buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (:b, :c))

            set!(model1, b = (x, z) -> x / 10000, c = 1.0)

            Nsteps = 5
            Δt = 1.0

            # Save b field at each time step
            saved_times = Float64[0.0]
            b_snapshots = [deepcopy(interior(model1.tracers.b))]

            for n in 1:Nsteps
                time_step!(model1, Δt)
                push!(saved_times, model1.clock.time)
                push!(b_snapshots, deepcopy(interior(model1.tracers.b)))
            end

            # Also save the closure field ϵ_R₃₃ from the last step of run 1
            ϵ_R₃₃_run1 = deepcopy(interior(model1.closure_fields.ϵ_R₃₃))

            # --- Build FieldTimeSeries from saved data ---
            b_fts = FieldTimeSeries{Center, Center, Center}(grid, saved_times)
            for (n, snapshot) in enumerate(b_snapshots)
                interior(b_fts[n]) .= snapshot
                fill_halo_regions!(b_fts[n])
            end

            # --- Run 2: Same setup but with prescribed b ---
            c_field = CenterField(grid)
            model2 = HydrostaticFreeSurfaceModel(grid;
                buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (b = PrescribedTracer(b_fts), c = c_field))

            set!(model2, c = 1.0)

            for n in 1:Nsteps
                time_step!(model2, Δt)
            end

            # The closure field ϵ_R₃₃ should match between the two runs
            # (both use the same b field at the final time step)
            ϵ_R₃₃_run2 = interior(model2.closure_fields.ϵ_R₃₃)
            @test ϵ_R₃₃_run1 ≈ ϵ_R₃₃_run2

            # The prognostic tracer c in run 2 should not contain NaN
            @test !any(isnan, model2.tracers.c)

            @info "    GM replication test passed [$arch]"
        end

        #####
        ##### Test 5: Analytical isopycnal slope check with prescribed b
        #####
        ##### For a linear buoyancy field b(x, z) = M² x + N² z,
        ##### the isopycnal slope is Sx = -∂x_b / ∂z_b = -M²/N².
        ##### The rotation tensor component R₃₃ = Sx² = (M²/N²)².
        ##### With tapering factor ϵ ≤ 1, the stored ϵ_R₃₃ should equal ϵ * (M²/N²)².
        #####

        @testset "Analytical isopycnal slope with prescribed tracer [$arch]" begin
            @info "  Testing analytical isopycnal slope [$arch]..."

            nx = 4
            nz = 4

            grid = RectilinearGrid(arch,
                                   size = (nx, nz),
                                   x = (0, 1e5),
                                   z = (-1e3, 0),
                                   topology = (Bounded, Flat, Bounded))

            M² = 1e-7  # horizontal buoyancy gradient
            N² = 1e-5  # vertical buoyancy gradient (stratification)

            # On Flat y-topology, FunctionField expects func(x, z, t)
            b_func(x, z, t) = M² * x + N² * z

            closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)

            model = HydrostaticFreeSurfaceModel(grid;
                buoyancy = BuoyancyTracer(),
                closure = closure,
                tracers = (b = PrescribedTracer(b_func),))

            # Trigger state update which computes closure fields
            time_step!(model, 1.0)

            # Expected isopycnal slope
            Sx = M² / N²  # = 0.01
            expected_R₃₃ = Sx^2

            # The tapering factor ϵ = min(1, max_slope² / slope²)
            # With Sx = 0.01 and max_slope = 0.1 (default FluxTapering):
            # ϵ = min(1, 0.01 / 0.0001) = min(1, 100) = 1
            # So ϵ_R₃₃ should be ≈ Sx² = 1e-4

            ϵ_R₃₃ = interior(model.closure_fields.ϵ_R₃₃)

            # Check interior points (avoiding boundaries where stencils may differ)
            for i in 2:nx-1, k in 2:nz
                @test ϵ_R₃₃[i, 1, k] ≈ expected_R₃₃ atol=1e-10
            end

            @info "    Analytical isopycnal slope test passed [$arch]"
        end

    end
end
