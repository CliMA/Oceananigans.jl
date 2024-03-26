using Oceananigans
using Oceananigans.Advection: DefaultStencil, OnlySelfUpwinding 
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: configured_kernel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: 
                immersed_boundary_condition, 
                top_tracer_boundary_conditions,
                compute_hydrostatic_free_surface_Gu!,
                compute_hydrostatic_free_surface_Gv!,
                compute_hydrostatic_free_surface_Gc!

using Oceananigans.ImmersedBoundaries: active_interior_map
    
using CUDA
CUDA.device!(1)
a = CPU()

grid = LatitudeLongitudeGrid(a, size = (20, 20, 20), latitude = (-1, 1), longitude = (-1, 1), z = (0, 1), halo = (7, 7, 7))
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y)->x); active_cells_map = true)

momentum_advection = VectorInvariant(vorticity_scheme  = WENO(; order = 9),  
                                     divergence_scheme = WENO(),
                                     vertical_scheme   = Centered()) #, vorticity_stencil = DefaultStencil())

tracer_advection = Oceananigans.Advection.TracerAdvection(WENO(; order = 7), WENO(; order = 7), Centered())

# momentum_advection = nothing

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection) #, tracer_advection)

arch = architecture(grid)
velocities = model.velocities

u_immersed_bc = immersed_boundary_condition(velocities.u)
v_immersed_bc = immersed_boundary_condition(velocities.v)

start_momentum_kernel_args = (model.advection.momentum,
                              model.coriolis,
                              model.closure)

end_momentum_kernel_args = (velocities,
                            model.free_surface,
                            model.tracers,
                            model.buoyancy,
                            model.diffusivity_fields,
                            model.pressure.pHY′,
                            model.auxiliary_fields,
                            model.forcing,
                            model.clock)

u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args...)
v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args...)

c_immersed_bc  = immersed_boundary_condition(model.tracers.T)
top_tracer_bcs = top_tracer_boundary_conditions(grid, model.tracers)

T_kernel_args = tuple(Val(1),
                      Val(:T),
                      model.advection.T,
                      nothing,
                      c_immersed_bc,
                      model.buoyancy,
                      model.biogeochemistry,
                      model.velocities,
                      model.free_surface,
                      model.tracers,
                      top_tracer_bcs,
                      model.diffusivity_fields,
                      model.auxiliary_fields,
                      model.forcing.T,
                      model.clock)

active_cells_map = active_interior_map(model.grid)

loop_u! = configured_kernel(arch, grid, :xyz,
                            compute_hydrostatic_free_surface_Gu!;
                            active_cells_map)

loop_v! = configured_kernel(arch, grid, :xyz,
                            compute_hydrostatic_free_surface_Gv!;
                            active_cells_map)

loop_T! = configured_kernel(arch, grid, :xyz,
                            compute_hydrostatic_free_surface_Gc!;
                            active_cells_map)

@show "Kernels are configured"

kernel_u = Oceananigans.Utils.retrieve_kernel(loop_u!, model.timestepper.Gⁿ.u, grid, active_cells_map, u_kernel_args)
kernel_v = Oceananigans.Utils.retrieve_kernel(loop_v!, model.timestepper.Gⁿ.v, grid, active_cells_map, v_kernel_args)
kernel_T = Oceananigans.Utils.retrieve_kernel(loop_T!, model.timestepper.Gⁿ.T, grid, active_cells_map, T_kernel_args)
 
@show CUDA.registers(kernel_u) CUDA.registers(kernel_v) CUDA.registers(kernel_T)

nothing
