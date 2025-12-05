using Oceananigans: Oceananigans
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using Test: @testset, @test

@testset "Aqua" begin
    @info "testing quality assurance via Aqua"
    Aqua.test_all(Oceananigans; ambiguities=false)
end

@testset "ExplicitImports" begin

    modules = (Oceananigans.Utils, Oceananigans.OrthogonalSphericalShellGrids, Oceananigans.Diagnostics, Oceananigans.AbstractOperations, Oceananigans.Models.HydrostaticFreeSurfaceModels, Oceananigans.TimeSteppers, Oceananigans.ImmersedBoundaries)

    @testset "Explicit Imports [$(mod)]" for mod in modules
        @info "Testing no implicit imports for module $(mod)"
        @test ExplicitImports.check_no_implicit_imports(mod) === nothing
    end

    @testset "Import via Owner" begin
        @info "Testing no imports via owner"
        @test ExplicitImports.check_all_explicit_imports_via_owners(Oceananigans) === nothing
    end

    @testset "Stale Explicit Imports" begin
        @info "Testing no stale implicit imports"
        @test ExplicitImports.check_no_stale_explicit_imports(Oceananigans) === nothing
    end

    @testset "Qualified Accesses" begin
        @info "Testing no qualified access via owners"
        @test ExplicitImports.check_all_qualified_accesses_via_owners(Oceananigans) === nothing
    end

    @testset "Self Qualified Accesses" begin
        @info "Testing no self qualified accesses"
        @test ExplicitImports.check_no_self_qualified_accesses(Oceananigans) === nothing
    end
end
