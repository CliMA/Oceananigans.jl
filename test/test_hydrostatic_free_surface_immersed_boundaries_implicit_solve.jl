include("dependencies_for_runtests.jl")

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Architectures: arch_array
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_vertically_integrated_volume_flux!, 
                                                        compute_implicit_free_surface_right_hand_side!,
                                                        implicit_free_surface_step!,
                                                        pressure_correct_velocities!,
                                                        finalize_solver!
using Oceananigans.Solvers: initialize_AMGX, finalize_AMGX

@testset "Immersed boundaries test divergent flow solve with hydrostatic free surface models" begin
    for arch in archs
        initialize_AMGX(arch)
        A = typeof(arch)
        @info "Testing immersed boundaries divergent flow solve [$A]"

        Nx = 11 
        Ny = 11 
        Nz = 1

        underlying_grid = RectilinearGrid(arch,
                                          size = (Nx, Ny, Nz),
                                          extent = (Nx, Ny, 1),
                                          halo = (3, 3, 3),
                                          topology = (Periodic, Periodic, Bounded))

        imm1=Int(floor((Nx+1)/2)  )
        imp1=Int(floor((Nx+1)/2)+1)
        jmm1=Int(floor((Ny+1)/2)  )
        jmp1=Int(floor((Ny+1)/2)+1)

        bottom = [-1. for j=1:Ny, i=1:Nx]
        bottom[imm1-1:imp1+1, jmm1-1:jmp1+1] .= 0

        B = arch_array(arch, bottom)
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(B))

        free_surfaces = [ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, gravitational_acceleration=1.0),
                         ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, gravitational_acceleration=1.0),
                         ImplicitFreeSurface(solver_method=:Multigrid, gravitational_acceleration=1.0),
                         ImplicitFreeSurface(gravitational_acceleration=1.0)]

        sol = ()
        f = ()

        for free_surface in free_surfaces

            model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                                buoyancy = nothing,
                                                tracers = nothing,
                                                closure = nothing)

            # Now create a divergent flow field and solve for pressure correction
            u, v, w = model.velocities
            u[imm1, jmm1, 1:Nz] .=  1
            u[imp1, jmm1, 1:Nz] .= -1
            v[imm1, jmm1, 1:Nz] .=  1
            v[imm1, jmp1, 1:Nz] .= -1
            
            implicit_free_surface_step!(model.free_surface, model, 1.0, 1.5)

            sol = (sol..., model.free_surface.η)
            f  = (f..., model.free_surface)

            finalize_solver!(model.free_surface.implicit_step_solver)
        end

        @test all(interior(sol[1]) .≈ interior(sol[2]) .≈ interior(sol[3]))
        finalize_AMGX(arch)
    end
end