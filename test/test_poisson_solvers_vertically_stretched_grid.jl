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
        @testset "Irregular-grid Poisson solver [FACR, $(typeof(arch)), $topo]" begin

            faces_even = [1, 2, 4, 7, 11, 16, 22, 29, 37]      # Nz = 8
            faces_odd  = [1, 2, 4, 7, 11, 16, 22, 29, 37, 51]  # Nz = 9
            for irregular_axis in (1, 2, 3)
                if topo[irregular_axis] == Bounded
                    @info "  Testing stretched Poisson solver [FACR, $(typeof(arch)), $topo, $irregular_axis]..."
                    @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 8, 1:8; irregular_axis)
                    @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 7, 1:7; irregular_axis)
                    @test stretched_poisson_solver_correct_answer(Float32, arch, topo, 8, 8, 1:8; irregular_axis)

                    for faces in [faces_even, faces_odd]
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8,  8, faces; irregular_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 16, 8, faces; irregular_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 16, faces; irregular_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 11, faces; irregular_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 5,  8, faces; irregular_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 13, faces; irregular_axis)
                    end
                end
            end
        end
    end
end
