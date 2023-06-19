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
        (Periodic, Bounded,  Periodic),
        (Bounded,  Periodic, Periodic),
        (Bounded,  Bounded,  Periodic),
        (Flat,     Bounded,  Bounded),
        (Flat,     Periodic, Bounded),
        (Bounded,  Flat,     Bounded),
        (Periodic, Flat,     Bounded)
    ]

    for arch in archs, topo in vs_topos
        @testset "Irregular-grid Poisson solver [FACR, $(typeof(arch)), $topo]" begin

            faces_even = [1, 2, 4, 7, 11, 16, 22, 29, 37]      # Nz = 8
            faces_odd  = [1, 2, 4, 7, 11, 16, 22, 29, 37, 51]  # Nz = 9
            for stretched_axis in (1, 2, 3,)
                if topo[stretched_axis] == Bounded
                    @info "  Testing stretched Poisson solver [FACR, $(typeof(arch)), $topo, stretched axis = $stretched_axis]..."
                    @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 4, 5, 1:4; stretched_axis)
                    @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 8, 1:8; stretched_axis)
                    @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 7, 1:7; stretched_axis)
                    @test stretched_poisson_solver_correct_answer(Float32, arch, topo, 8, 8, 1:8; stretched_axis)

                    for faces in [faces_even, faces_odd]
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8,  8, faces; stretched_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 16, 8, faces; stretched_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 16, faces; stretched_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 8, 11, faces; stretched_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 5,  8, faces; stretched_axis)
                        @test stretched_poisson_solver_correct_answer(Float64, arch, topo, 7, 13, faces; stretched_axis)
                    end
                end
            end
        end
    end
end
