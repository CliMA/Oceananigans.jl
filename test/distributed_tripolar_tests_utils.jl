using JLD2
using MPI
using Oceananigans.DistributedComputations: reconstruct_global_field, reconstruct_global_grid
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids: analytical_immersed_tripolar_grid

import Oceananigans.BoundaryConditions: _fill_north_halo!
using Oceananigans.BoundaryConditions: ZBC, CCLocation, FCLocation

include("dependencies_for_runtests.jl")

# The serial version of the TripolarGrid substitutes the second half of the last row of the grid
# This is not done in the distributed version, so we need to undo this substitution if we want to
# compare the results. Otherwise very tiny differences caused by finite precision compuations
# will appear in the last row of the grid.

# Mask the singularity of the grid in a region of `radius` degrees around the singularities
function analytical_immersed_tripolar_grid(underlying_grid::TripolarGrid; radius = 5) # degrees
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    φm = underlying_grid.conformal_mapping.southernmost_latitude

    Lz = underlying_grid.Lz

    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) | (φ < φm) ? 0 : - Lz

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

    return grid
end

# tracers or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::ZBC, ::CCLocation, args...) = my_fold_north_center_center!(i, k, grid, bc.condition, c)
@inline _fill_north_halo!(i, k, grid, u, bc::ZBC, ::FCLocation, args...) = my_fold_north_face_center!(i, k, grid, bc.condition, u)

@inline function my_fold_north_face_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! elemesnt Nx + 1 does not exist!
    sign  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end

@inline function my_fold_north_center_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
    end

    return nothing
end

# Run the distributed grid simulation and save down reconstructed results
function run_distributed_tripolar_grid(arch, filename)
    distributed_grid = TripolarGrid(arch; size = (40, 40, 1), z = (-1000, 0), halo = (5, 5, 5))
    distributed_grid = analytical_immersed_tripolar_grid(distributed_grid)
    simulation       = run_tripolar_simulation(distributed_grid)

    η = reconstruct_global_field(simulation.model.free_surface.η)
    u  = reconstruct_global_field(simulation.model.velocities.u)
    v  = reconstruct_global_field(simulation.model.velocities.v)
    c  = reconstruct_global_field(simulation.model.tracers.c)

    if arch.local_rank == 0
        jldsave(filename; u = Array(interior(u, :, :, 1)),
                          v = Array(interior(v, :, :, 1)), 
                          c = Array(interior(c, :, :, 1)),
                          η = Array(interior(η, :, :, 1))) 
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()

    return nothing
end

# Just a random simulation on a tripolar grid
function run_tripolar_simulation(grid)

    model = HydrostaticFreeSurfaceModel(; grid = grid,
                                          free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                          tracers = :c,
                                          buoyancy = nothing, 
                                          tracer_advection = WENO(), 
                                          momentum_advection = WENOVectorInvariant(order=3),
                                          coriolis = HydrostaticSphericalCoriolis())

    # Setup the model with a gaussian sea surface height
    # near the physical north poles and one near the equator
    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)

    set!(model, c = ηᵢ, η = ηᵢ)

    simulation = Simulation(model, Δt = 5minutes, stop_iteration = 100)
    
    run!(simulation)

    return simulation
end