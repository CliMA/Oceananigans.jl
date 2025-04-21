using JLD2
using MPI
using Oceananigans.DistributedComputations: reconstruct_global_field, reconstruct_global_grid
using Oceananigans.Units
using Reactant
using Oceananigans.TimeSteppers: first_time_step!

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
    model            = run_distributed_simulation(distributed_grid)

    η = reconstruct_global_field(model.free_surface.η)
    u = reconstruct_global_field(model.velocities.u)
    v = reconstruct_global_field(model.velocities.v)
    c = reconstruct_global_field(model.tracers.c)

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

# Run the distributed grid simulation and save down reconstructed results
function run_distributed_latitude_longitude_grid(arch, filename)
    Random.seed!(1234)
    bottom_height = - rand(40, 40, 1) .* 500 .- 500

    flat_distributed_grid = LatitudeLongitudeGrid(arch,
        size = (40, 40),
        longitude = (0, 360),
        latitude = (-90, 90),
        topology = (Periodic, Bounded, Flat))

    @test isnothing(flat_distributed_grid.z)

    distributed_grid = LatitudeLongitudeGrid(arch;
                                             size = (40, 40, 10),
                                             longitude = (0, 360),
                                             latitude = (-10, 10),
                                             z = (-1000, 0),
                                             halo = (5, 5, 5))

    distributed_grid = ImmersedBoundaryGrid(distributed_grid, GridFittedBottom(bottom_height))
    model = run_distributed_simulation(distributed_grid)

    η = reconstruct_global_field(model.free_surface.η)
    u = reconstruct_global_field(model.velocities.u)
    v = reconstruct_global_field(model.velocities.v)
    c = reconstruct_global_field(model.tracers.c)

    if arch.local_rank == 0
        jldsave(filename; u = Array(interior(u, :, :, 10)),
                          v = Array(interior(v, :, :, 10)),
                          c = Array(interior(c, :, :, 10)),
                          η = Array(interior(η, :, :, 1)))
    end

    return nothing
end

# Just a random simulation on a tripolar grid
function run_distributed_simulation(grid)

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
    set!(model, c=ηᵢ, η=ηᵢ)

    Δt = 5minutes
    arch = architecture(grid)
    if arch isa ReactantState || arch isa Distributed{<:ReactantState}
        @info "Compiling first_time_step..."
        r_first_time_step! = @compile sync=true raise=true first_time_step!(model, Δt)

        @info "Compiling time_step..."
        r_time_step! = @compile sync=true raise=true time_step!(model, Δt)
    else
        r_first_time_step! = first_time_step!
        r_time_step! = time_step!
    end

    @info "Running first time step..."
    r_first_time_step!(model, Δt)
    @info "Running time steps..."
    for N in 2:100
        r_time_step!(model, Δt)
    end

    return model
end

