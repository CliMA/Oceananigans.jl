using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Architectures: arch_array
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_vertically_integrated_volume_flux!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_implicit_free_surface_right_hand_side!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: pressure_correct_velocities!

@testset "Immersed boundaries test divergent flow solve with hydrostatic free surface models" begin
    @info "Testing immersed boundaries divergent flow solve"

    for arch in archs
        Nx = 11 
        Ny = 11 
        Nz = 1

        underlying_grid = RectilinearGrid(arch,
                                          size = (Nx, Ny, Nz),
                                          extent = (Nx, Ny, 1),
                                          halo = (3, 3, 3),
                                          topology = (Periodic, Periodic, Bounded))

        imm1=Int( floor((Nx+1)/2)   )
        imp1=Int( floor((Nx+1)/2)+1 )
        jmm1=Int( floor((Ny+1)/2)   )
        jmp1=Int( floor((Ny+1)/2)+1 )

        bottom = [-1. for i=1:Nx, j=1:Ny ]
        bottom[imm1-1:imp1+1, jmm1-1:jmp1+1] .= 0

        B = arch_array(arch, bottom)
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(B))

        free_surfaces = [ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, gravitational_acceleration=1.0),
                         ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, gravitational_acceleration=1.0), 
                         ImplicitFreeSurface(gravitational_acceleration=1.0)]

        sol = ()
        f = ()

        for free_surface in free_surfaces

            model = HydrostaticFreeSurfaceModel(grid = grid,
                                                free_surface = free_surface,
                                                buoyancy = nothing,
                                                tracers = nothing,
                                                closure = nothing)

            # Now create a divergent flow field and solve for 
            # pressure correction
            u, v, w     = model.velocities
            u[imm1, jmm1, 1:Nz ] .=  1
            u[imp1, jmm1, 1:Nz ] .= -1
            v[imm1, jmm1, 1:Nz ] .=  1
            v[imm1, jmp1, 1:Nz ] .= -1

              η = model.free_surface.η
              g = model.free_surface.gravitational_acceleration
            ∫ᶻQ = model.free_surface.barotropic_volume_flux
            rhs = model.free_surface.implicit_step_solver.right_hand_side
            solver = model.free_surface.implicit_step_solver;
                Δt = 1.
            compute_vertically_integrated_volume_flux!(∫ᶻQ, model)
            rhs_event = compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, ∫ᶻQ, η)
            wait(device(arch), rhs_event)

            solve!(η, solver, rhs, g, Δt)

            fill_halo_regions!(η)

            #=
            println("model.free_surface.gravitational_acceleration = ",model.free_surface.gravitational_acceleration)
            println("∫ᶻQ.u")
            show(stdout,"text/plain", ∫ᶻQ.u.data[1:Nx, 1:Ny, 1])
            println("")

            println("η")
            show(stdout,"text/plain", η.data[1:Nx, 1:Ny, 1])
            println("")

            pressure_correct_velocities!(model, Δt)
            fill_halo_regions!(u)

            println("u")
            show(stdout,"text/plain", u.data[1:Nx, 1:Ny, 1])
            println("")
            =#

            fs = model.free_surface
            vertically_integrated_lateral_areas = fs.implicit_step_solver.vertically_integrated_lateral_areas
            ∫Axᶠᶜᶜ = vertically_integrated_lateral_areas.xᶠᶜᶜ
            ∫Ayᶜᶠᶜ = vertically_integrated_lateral_areas.yᶜᶠᶜ

            sol = (sol..., model.free_surface.η)
            f  = (f..., model.free_surface)
        end

        @test all(interior(sol[1]) .≈ interior(sol[2]) .≈ interior(sol[3]))
    end
end
