include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

#####
##### Run pressure solver tests 2
#####

@testset "Poisson solvers 2" begin
    @info "Testing Poisson solvers (vertically stretched grid)..."

    # Vertically stretched topologies to test.
    vs_topos = [
        (Periodic, Periodic, Bounded),
        (Periodic, Bounded,  Bounded),
        (Bounded,  Periodic, Bounded),
        (Bounded,  Bounded,  Bounded),
        (Flat,     Bounded,  Bounded),
        (Flat,     Periodic, Bounded),
        (Bounded,  Flat,     Bounded),
        (Periodic, Flat,     Bounded)
    ]

    for arch in archs, topo in vs_topos
        @testset "Vertically stretched Poisson solver [FACR, $(typeof(arch)), $topo]" begin
            @info "  Testing vertically stretched Poisson solver [FACR, $(typeof(arch)), $topo]..."

            @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 8, 1:8)
            @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 7, 1:7)
            @test vertically_stretched_poisson_solver_correct_answer(Float32, arch, topo, 8, 8, 1:8)

            zF_even = [1, 2, 4, 7, 11, 16, 22, 29, 37]      # Nz = 8
            zF_odd  = [1, 2, 4, 7, 11, 16, 22, 29, 37, 51]  # Nz = 9

            for zF in [zF_even, zF_odd]
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8,  8, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 16, 8, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 16, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 11, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 5,  8, zF)
                @test vertically_stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 13, zF)
            end
        end
    end
end
