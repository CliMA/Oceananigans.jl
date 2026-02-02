#=
FFT Compilation Tests (no differentiation)
Tests model construction with FFT-based solvers across topology combinations.
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Test

Reactant.set_default_backend("cpu")

@testset "FFT Compilation - Model Construction" begin

    @testset "NonhydrostaticModel" begin
        # Periodic in all dimensions (FFT only)
        @testset "Periodic, Periodic, Periodic" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4, 4), extent=(100, 100, 100),
                halo=(3, 3, 3), topology=(Periodic, Periodic, Periodic))
            model = NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
            @test model !== nothing
            @test model.pressure_solver !== nothing
        end

        # Bounded z requires DCT - should error
        @testset "Periodic, Periodic, Bounded (DCT - unsupported)" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4, 4), extent=(100, 100, 100),
                halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))
            @test_throws ErrorException NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
        end

        # Bounded x requires DCT - should error
        @testset "Bounded, Periodic, Periodic (DCT - unsupported)" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4, 4), extent=(100, 100, 100),
                halo=(3, 3, 3), topology=(Bounded, Periodic, Periodic))
            @test_throws ErrorException NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
        end

        # Flat dimension (no transform needed)
        @testset "Periodic, Periodic, Flat" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4), extent=(100, 100),
                halo=(3, 3), topology=(Periodic, Periodic, Flat))
            model = NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
            @test model !== nothing
            @test model.pressure_solver !== nothing
        end

        @testset "Periodic, Flat, Periodic" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4), extent=(100, 100),
                halo=(3, 3), topology=(Periodic, Flat, Periodic))
            model = NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
            @test model !== nothing
            @test model.pressure_solver !== nothing
        end
    end

    @testset "HydrostaticFreeSurfaceModel" begin
        # ImplicitFreeSurface uses FFT
        @testset "ImplicitFreeSurface - Periodic, Periodic, Bounded (DCT - unsupported)" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4, 4), extent=(100, 100, 100),
                halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))
            @test_throws ErrorException HydrostaticFreeSurfaceModel(grid;
                free_surface=ImplicitFreeSurface(),
                tracers=:T, buoyancy=nothing, closure=nothing)
        end

        # ExplicitFreeSurface - no FFT solver
        @testset "ExplicitFreeSurface - no FFT (baseline)" begin
            grid = RectilinearGrid(ReactantState(),
                size=(4, 4, 4), extent=(100, 100, 100),
                halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))
            model = HydrostaticFreeSurfaceModel(grid;
                free_surface=ExplicitFreeSurface(),
                tracers=:T, buoyancy=nothing, closure=nothing)
            @test model !== nothing
        end
    end
end
