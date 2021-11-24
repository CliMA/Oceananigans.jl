
using Oceananigans
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_vertically_integrated_volume_flux!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_implicit_free_surface_right_hand_side!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: pressure_correct_velocities!
using Oceananigans.Solvers: solve!
using BenchmarkTools

N = 512

arch = GPU()

Nx = N
Ny = N 
Nz =  1

grid = RectilinearGrid(architecture = arch,
                               size = (Nx, Ny, Nz),
                             extent = (Nx, Ny, 1),
                           topology = (Periodic, Periodic, Bounded))


free_surface = ImplicitFreeSurface(gravitational_acceleration=1.0)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = arch,
                                    free_surface = free_surface,
                                    tracer_advection = WENO5(),
                                    buoyancy = nothing,
                                    tracers = nothing,
                                    closure = nothing);

# Now create a divergent flow field and solve for 
# pressure correction
u, v, w     = model.velocities
parent(u) .=  1.
parent(v) .=  1.

        η = model.free_surface.η
        g = model.free_surface.gravitational_acceleration
    ∫ᶻQ = model.free_surface.barotropic_volume_flux
    rhs = model.free_surface.implicit_step_solver.right_hand_side
solver = model.free_surface.implicit_step_solver;
        Δt = 1.


@benchmark begin 
    compute_vertically_integrated_volume_flux!(∫ᶻQ, model)
    
    rhs_event = compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, ∫ᶻQ, η)
    wait(device(arch), rhs_event)
    
    solve!(η, solver, rhs, g, Δt)

    fill_halo_regions!(η, arch)

    println(typeof(model))

    pressure_correct_velocities!(model, Δt)

    fill_halo_regions!(u, arch)
end
