include("reactant_test_utils.jl")
using Reactant: @trace

#####
##### Reactant HydrostaticFreeSurfaceModel tests (non-FFT free surface)
#####
# Tests construction and compiled time-stepping of HydrostaticFreeSurfaceModel
# with ExplicitFreeSurface (no FFT) on various topologies including Bounded.
# Uses raise=true and raise_first=true to surface any MLIR compilation errors.

@testset "Reactant HydrostaticFreeSurfaceModel (non-FFT)" begin
    @info "Performing Reactant HydrostaticFreeSurfaceModel (non-FFT) tests..."
    reactant_arch = ReactantState()

    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for i in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    @testset "3D (Periodic, Periodic, Bounded) — ExplicitFreeSurface" begin
        @info "  Testing HFSM 3D (Periodic, Periodic, Bounded)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Bounded))
        model = HydrostaticFreeSurfaceModel(grid;
                    free_surface = ExplicitFreeSurface(),
                    timestepper = :QuasiAdamsBashforth2,
                    buoyancy = nothing,
                    tracers = ())

        @testset "Construction" begin
            @test model isa HydrostaticFreeSurfaceModel
            @test model.grid.architecture isa ReactantState
            @test model.free_surface isa ExplicitFreeSurface
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end

    @testset "3D (Bounded, Bounded, Bounded) — ExplicitFreeSurface" begin
        @info "  Testing HFSM 3D (Bounded, Bounded, Bounded)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Bounded, Bounded, Bounded))
        model = HydrostaticFreeSurfaceModel(grid;
                    free_surface = ExplicitFreeSurface(),
                    timestepper = :QuasiAdamsBashforth2,
                    buoyancy = nothing,
                    tracers = ())

        @testset "Construction" begin
            @test model isa HydrostaticFreeSurfaceModel
            @test model.grid.architecture isa ReactantState
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end
end
