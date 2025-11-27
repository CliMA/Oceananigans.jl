using Oceananigans: Oceananigans
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using Test: @testset, @test

@testset "Aqua" begin
    Aqua.test_all(Oceananigans; ambiguities=false, undefined_exports=false)
end

@testset "ExplicitImports" begin

    modules = (Oceananigans.Utils,)

    @testset "Explicit Imports [$(mod)]" for mod in modules
        @test ExplicitImports.check_no_implicit_imports(mod) === nothing
    end

    @testset "Import via Owner [$(mod)]" for mod in modules
        @test ExplicitImports.check_all_explicit_imports_via_owners(mod) === nothing
    end

    @testset "Stale Explicit Imports [$(mod)]" for mod in modules
        @test ExplicitImports.check_no_stale_explicit_imports(mod) === nothing
    end

    @testset "Qualified Accesses" begin
        @test ExplicitImports.check_all_qualified_accesses_via_owners(Oceananigans) === nothing
    end

    @testset "Self Qualified Accesses" begin
        @test ExplicitImports.check_no_self_qualified_accesses(Oceananigans) === nothing
    end
end
