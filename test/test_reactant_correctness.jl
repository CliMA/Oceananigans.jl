include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using Random
using CUDA
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid, OrthogonalSphericalShellGrid
using SeawaterPolynomials: TEOS10EquationOfState

# Helper to generate all combinations
all_combos(xs...) = vec(collect(Iterators.product(xs...)))

@testset "Reactant correctness" begin
    @info "Testing Reactant correctness (comparing vanilla Oceananigans vs ReactantState)..."

    # Get vanilla architecture from TEST_ARCHITECTURE env var (set by reactant_test_utils.jl)
    vanilla_arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
    reactant_arch = ReactantState()

    # Field locations to test
    all_locations = all_combos((Center, Face), (Center, Face), (Center, Face))

    # Topologies to test for RectilinearGrid
    # Note: We exclude (Periodic, Periodic, Periodic) because it triggers a segfault in
    # Reactant's MLIR pattern rewriting when using raise=true (RecognizeRotate pass bug)
    all_topologies = filter(topo -> topo != (Periodic, Periodic, Periodic),
                            all_combos((Periodic, Bounded), (Periodic, Bounded), (Periodic, Bounded)))

    # JIT raise modes: raise=false is default, raise=true is needed for autodiff
    # raise_first controls whether the first pass uses raise mode
    raise_modes = (false, true)
    raise_first_modes = (false, true)

    #####
    ##### Halo filling tests
    #####

    @testset "fill_halo_regions! correctness" begin

        # Test RectilinearGrid with all topologies
        for topo in all_topologies
            @testset "RectilinearGrid topology=$topo" begin
                @info "Testing fill_halo_regions! correctness on RectilinearGrid with topology=$topo..."
                kw = (size=(3, 4, 2), halo=(1, 1, 1), extent=(1, 1, 1), topology=topo)
                vanilla_grid = RectilinearGrid(vanilla_arch; kw...)
                reactant_grid = RectilinearGrid(reactant_arch; kw...)

                for loc in all_locations, raise in raise_modes, raise_first in raise_first_modes
                    LX, LY, LZ = loc
                    @testset "loc=$loc raise=$raise raise_first=$raise_first" begin
                        vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                        reactant_field = Field{LX, LY, LZ}(reactant_grid)

                        Random.seed!(12345)
                        data = randn(size(vanilla_field)...)
                        set!(vanilla_field, data)
                        set!(reactant_field, data)

                        fill_halo_regions!(vanilla_field)
                        @jit raise=raise raise_first=raise_first fill_halo_regions!(reactant_field)

                        @test compare_parent("halo", vanilla_field, reactant_field)
                    end
                end
            end
        end

        # Test LatitudeLongitudeGrid with LX ∈ (Periodic, Bounded)
        for LX in (Periodic, Bounded)
            topo = (LX, Bounded, Bounded)
            @testset "LatitudeLongitudeGrid topology=$topo" begin
                @info "Testing fill_halo_regions! correctness on LatitudeLongitudeGrid with topology=$topo..."
                kw = (size=(4, 4, 2), halo=(1, 1, 1), longitude=(0, 10), latitude=(0, 10), z=(0, 1), topology=topo)
                vanilla_grid = LatitudeLongitudeGrid(vanilla_arch; kw...)
                reactant_grid = LatitudeLongitudeGrid(reactant_arch; kw...)

                for loc in all_locations, raise in raise_modes, raise_first in raise_first_modes
                    LXf, LYf, LZf = loc
                    @testset "loc=$loc raise=$raise raise_first=$raise_first" begin
                        vanilla_field = Field{LXf, LYf, LZf}(vanilla_grid)
                        reactant_field = Field{LXf, LYf, LZf}(reactant_grid)

                        Random.seed!(12345)
                        data = randn(size(vanilla_field)...)
                        set!(vanilla_field, data)
                        set!(reactant_field, data)

                        fill_halo_regions!(vanilla_field)
                        @jit raise=raise raise_first=raise_first fill_halo_regions!(reactant_field)

                        @test compare_parent("halo", vanilla_field, reactant_field)
                    end
                end
            end
        end

        # Test TripolarGrid (fixed topology)
        @testset "TripolarGrid" begin
            @info "Testing fill_halo_regions! correctness on TripolarGrid..."
            kw = (size=(8, 10, 2), halo=(1, 1, 1), z=(0, 1))
            vanilla_grid = TripolarGrid(vanilla_arch; kw...)
            reactant_grid = TripolarGrid(reactant_arch; kw...)

            for loc in all_locations, raise in raise_modes, raise_first in raise_first_modes
                LX, LY, LZ = loc
                @testset "loc=$loc raise=$raise raise_first=$raise_first" begin
                    vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
                    reactant_field = Field{LX, LY, LZ}(reactant_grid)

                    Random.seed!(12345)
                    data = randn(size(vanilla_field)...)
                    set!(vanilla_field, data)
                    set!(reactant_field, data)

                    fill_halo_regions!(vanilla_field)
                    @jit raise=raise raise_first=raise_first fill_halo_regions!(reactant_field)

                    @test compare_parent("halo", vanilla_field, reactant_field)
                end
            end
        end
    end

    #####
    ##### Tendency computation tests
    #####

    @testset "compute_simple_Gu! correctness" begin
        advection_schemes = (nothing, Centered(), WENO())
        
        # Helper to get advection name for testset
        adv_name(::Nothing) = "nothing"
        adv_name(a) = string(nameof(typeof(a)))

        # Topologies to test
        gu_topologies = (
            (Periodic, Periodic, Bounded),
            (Periodic, Bounded, Bounded),
            (Bounded, Periodic, Bounded),
            (Bounded, Bounded, Bounded),
        )

        # RectilinearGrid tests
        @testset "RectilinearGrid" begin
            for topo in gu_topologies
                @testset "topology=$topo" begin
                    kw = (size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1), topology=topo)
                    vanilla_grid = RectilinearGrid(vanilla_arch; kw...)
                    reactant_grid = RectilinearGrid(reactant_arch; kw...)

                    # Set up velocity fields
                    vanilla_velocities = (u=XFaceField(vanilla_grid), v=YFaceField(vanilla_grid), w=ZFaceField(vanilla_grid))
                    reactant_velocities = (u=XFaceField(reactant_grid), v=YFaceField(reactant_grid), w=ZFaceField(reactant_grid))

                    Random.seed!(54321)
                    for (vf, rf) in zip(vanilla_velocities, reactant_velocities)
                        data = randn(size(vf)...)
                        set!(vf, data)
                        set!(rf, data)
                        fill_halo_regions!(vf)
                        @jit fill_halo_regions!(rf)
                    end

                    vanilla_Gu = XFaceField(vanilla_grid)
                    reactant_Gu = XFaceField(reactant_grid)

                    # Test with coriolis=nothing
                    @testset "coriolis=nothing" begin
                        @info "Testing compute_simple_Gu! on RectilinearGrid($topo) with coriolis=nothing..."
                        for advection in advection_schemes, raise in raise_modes, raise_first in raise_first_modes
                            @testset "advection=$(adv_name(advection)) raise=$raise raise_first=$raise_first" begin
                                fill!(vanilla_Gu, 0)
                                fill!(reactant_Gu, 0)
                                compute_simple_Gu!(vanilla_Gu, advection, nothing, vanilla_velocities)
                                @jit raise=raise raise_first=raise_first compute_simple_Gu!(reactant_Gu, advection, nothing, reactant_velocities)
                                @test compare_interior("Gu", vanilla_Gu, reactant_Gu)
                            end
                        end
                    end

                    # Test with FPlane Coriolis
                    @testset "coriolis=FPlane" begin
                        @info "Testing compute_simple_Gu! on RectilinearGrid($topo) with FPlane Coriolis..."
                        coriolis = FPlane(f=1e-4)
                        for advection in advection_schemes, raise in raise_modes, raise_first in raise_first_modes
                            @testset "advection=$(adv_name(advection)) raise=$raise raise_first=$raise_first" begin
                                fill!(vanilla_Gu, 0)
                                fill!(reactant_Gu, 0)
                                compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                                @jit raise=raise raise_first=raise_first compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
                                @test compare_interior("Gu", vanilla_Gu, reactant_Gu)
                            end
                        end
                    end
                end
            end
        end

        # LatitudeLongitudeGrid tests
        @testset "LatitudeLongitudeGrid" begin
            kw = (size=(4, 4, 4), halo=(3, 3, 3), longitude=(0, 10), latitude=(0, 10), z=(0, 1))
            vanilla_grid = LatitudeLongitudeGrid(vanilla_arch; kw...)
            reactant_grid = LatitudeLongitudeGrid(reactant_arch; kw...)

            # Set up velocity fields
            vanilla_velocities = (u=XFaceField(vanilla_grid), v=YFaceField(vanilla_grid), w=ZFaceField(vanilla_grid))
            reactant_velocities = (u=XFaceField(reactant_grid), v=YFaceField(reactant_grid), w=ZFaceField(reactant_grid))

            Random.seed!(54321)
            for (vf, rf) in zip(vanilla_velocities, reactant_velocities)
                data = randn(size(vf)...)
                set!(vf, data)
                set!(rf, data)
                fill_halo_regions!(vf)
                @jit fill_halo_regions!(rf)
            end

            vanilla_Gu = XFaceField(vanilla_grid)
            reactant_Gu = XFaceField(reactant_grid)

            # Test with coriolis=nothing (advection-only)
            # Note: WENO on LatitudeLongitudeGrid fails to compile with raise=true
            @testset "coriolis=nothing" begin
                @info "Testing compute_simple_Gu! on LatitudeLongitudeGrid with coriolis=nothing..."
                for advection in advection_schemes, raise in raise_modes, raise_first in raise_first_modes
                    @testset "advection=$(adv_name(advection)) raise=$raise raise_first=$raise_first" begin
                        fill!(vanilla_Gu, 0)
                        fill!(reactant_Gu, 0)
                        compute_simple_Gu!(vanilla_Gu, advection, nothing, vanilla_velocities)
                        @jit raise=raise raise_first=raise_first compute_simple_Gu!(reactant_Gu, advection, nothing, reactant_velocities)
                        @test compare_interior("Gu", vanilla_Gu, reactant_Gu)
                    end
                end
            end

            # Test with HydrostaticSphericalCoriolis
            # Note: This has known numerical differences with raise=true (δ ≈ 8.6e-06)
            @testset "coriolis=HydrostaticSphericalCoriolis" begin
                @info "Testing compute_simple_Gu! on LatitudeLongitudeGrid with HydrostaticSphericalCoriolis..."
                coriolis = HydrostaticSphericalCoriolis()
                for advection in advection_schemes, raise in raise_modes, raise_first in raise_first_modes
                    @testset "advection=$(adv_name(advection)) raise=$raise raise_first=$raise_first" begin
                        fill!(vanilla_Gu, 0)
                        fill!(reactant_Gu, 0)
                        compute_simple_Gu!(vanilla_Gu, advection, coriolis, vanilla_velocities)
                        @jit raise=raise raise_first=raise_first compute_simple_Gu!(reactant_Gu, advection, coriolis, reactant_velocities)
                        @test compare_interior("Gu", vanilla_Gu, reactant_Gu)
                    end
                end
            end
        end
    end

    #####
    ##### Time-stepping tests with HydrostaticFreeSurfaceModel
    #####

    @testset "HydrostaticFreeSurfaceModel time-stepping" begin

        # Topologies to test
        hfsm_topologies = (
            (Periodic, Periodic, Bounded),
            (Periodic, Bounded, Bounded),
            (Bounded, Periodic, Bounded),
            (Bounded, Bounded, Bounded),
        )

        # Grid setup
        # Note: WENOVectorInvariant requires halo >= (6, 6, 3)
        Nx, Ny, Nz = 8, 8, 4
        Lx, Ly, Lz = 1e5, 1e5, 100  # 100km x 100km x 100m domain

        # Common model parameters
        coriolis = FPlane(f=1e-4)
        advection = WENO()
        momentum_advection = WENOVectorInvariant()
        equation_of_state = TEOS10EquationOfState()
        buoyancy = SeawaterBuoyancy(; equation_of_state)
        tracers = (:T, :S)

        # Closures to test
        # Note: CATKEVerticalDiffusivity takes too long to compile with Reactant (>10 min)
        closures = (nothing,)

        closure_name(::Nothing) = "nothing"
        closure_name(c) = string(nameof(typeof(c)))

        for topo in hfsm_topologies
            @testset "topology=$topo" begin
                kw = (size=(Nx, Ny, Nz), halo=(6, 6, 3), x=(0, Lx), y=(0, Ly), z=(-Lz, 0), topology=topo)
                vanilla_grid = RectilinearGrid(vanilla_arch; kw...)
                reactant_grid = RectilinearGrid(reactant_arch; kw...)

                # Use SplitExplicitFreeSurface (FFT-based ImplicitFreeSurface doesn't work with Reactant)
                free_surface = SplitExplicitFreeSurface(vanilla_grid; substeps = 10)

                for closure in closures
                    @testset "closure=$(closure_name(closure))" begin
                        @info "Testing HydrostaticFreeSurfaceModel($topo) with closure=$(closure_name(closure))..."

                        model_kw = (; coriolis, buoyancy, free_surface, tracers, tracer_advection=advection, momentum_advection, closure)
                        vanilla_model = HydrostaticFreeSurfaceModel(vanilla_grid; model_kw...)
                        reactant_model = HydrostaticFreeSurfaceModel(reactant_grid; model_kw...)

                        # Set initial conditions
                        # Note: Use pre-computed arrays, NOT functions with randn(),
                        # because randn() would be called at different times for each model
                        Random.seed!(98765)
                        
                        # Small velocity perturbations
                        u_init = 0.1 * randn(Nx, Ny, Nz)
                        v_init = 0.1 * randn(Nx, Ny, Nz)
                        
                        # Realistic T/S with vertical gradient and small perturbations
                        T_init = [20.0 + 5.0 * (k - 0.5) / Nz + 0.01 * randn() for i=1:Nx, j=1:Ny, k=1:Nz]
                        S_init = [35.0 + 0.01 * randn() for i=1:Nx, j=1:Ny, k=1:Nz]

                        set!(vanilla_model, u=u_init, v=v_init, T=T_init, S=S_init)
                        set!(reactant_model, u=u_init, v=v_init, T=T_init, S=S_init)

                        # Small time-step for stability
                        Δt = 60.0  # 1 minute
                        stop_iteration = 3

                        # Run vanilla simulation
                        vanilla_simulation = Simulation(vanilla_model; Δt, stop_iteration, verbose=false)
                        run!(vanilla_simulation)

                        # Compile and run Reactant simulation
                        @jit Oceananigans.TimeSteppers.update_state!(reactant_model)

                        r_first_time_step! = @compile sync=true Oceananigans.TimeSteppers.first_time_step!(reactant_model, Δt)
                        r_time_step! = @compile sync=true Oceananigans.TimeSteppers.time_step!(reactant_model, Δt)

                        reactant_simulation = Simulation(reactant_model; Δt, stop_iteration, verbose=false)
                        r_run!(reactant_simulation, r_time_step!, r_first_time_step!)

                        # Compare results
                        @testset "velocity fields" begin
                            @test compare_interior("u", vanilla_model.velocities.u, reactant_model.velocities.u)
                            @test compare_interior("v", vanilla_model.velocities.v, reactant_model.velocities.v)
                        end

                        @testset "tracer fields" begin
                            @test compare_interior("T", vanilla_model.tracers.T, reactant_model.tracers.T)
                            @test compare_interior("S", vanilla_model.tracers.S, reactant_model.tracers.S)
                        end

                        @testset "simulation state" begin
                            @test iteration(vanilla_simulation) == iteration(reactant_simulation)
                            @test time(vanilla_simulation) ≈ time(reactant_simulation)
                        end
                    end
                end
            end
        end
    end
end
