using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization, z_viscosity
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom

Nx, Nz = 64, 32

underlying_grid = RectilinearGrid(arch, size = (Nx, Nz), extent = (10, 5), topology = (Periodic, Flat, Bounded))

Δz = underlying_grid.Δzᵃᵃᶜ
Δx = underlying_grid.Δxᶠᵃᵃ
Lz = underlying_grid.Lz
nsteps = 100
diff   = zeros(nsteps)

@inline wedge(x, y) = @. max(0, min( 1/2.5 * x - 1, -2/5 * x + 3))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(wedge))

explicit_closure = IsotropicDiffusivity(ν = 1.0)
implicit_closure = IsotropicDiffusivity(ν = 1.0, time_discretization = VerticallyImplicitTimeDiscretization())

explicit_model = HydrostaticFreeSurfaceModel(grid = grid,
                                            closure = explicit_closure,
                                        free_surface = ImplicitFreeSurface())

implicit_model = HydrostaticFreeSurfaceModel(grid = grid,
                                            closure = implicit_closure,
                                        free_surface = ImplicitFreeSurface())

# initial divergence-free velocity
initial_velocity(x, y, z) = z > - Lz / 2 ? 1 : 0

ue, ve, we = explicit_model.velocities
ui, vi, wi = implicit_model.velocities

set!(ue, initial_velocity)
set!(ui, initial_velocity)

# CFL condition (advective and diffusion) = 0.01
Δt = min( 1.0/Δx, Δz^2 / explicit_closure.ν) / 100

for step in 1:nsteps
    time_step!(explicit_model, Δt)
    time_step!(implicit_model, Δt)

    diff[step] = norm(ue .- ui) / norm(ue)
end

u_explicit = interior(explicit_model.velocities.u)[:, 1, :]
u_implicit = interior(implicit_model.velocities.u)[:, 1, :]
η_explicit = interior(explicit_model.free_surface.η)[:, 1, 1]
η_implicit = interior(implicit_model.free_surface.η)[:, 1, 1]
