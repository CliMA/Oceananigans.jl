using JLD2
using MPI
using Oceananigans.DistributedComputations: reconstruct_global_field, reconstruct_global_grid
using Oceananigans.Units
using Reactant
using Oceananigans.TimeSteppers: first_time_step!

include("dependencies_for_runtests.jl")

# Mask the singularity of the grid in a region of `radius` degrees around the singularities
function analytical_immersed_tripolar_grid(underlying_grid::TripolarGrid; radius = 5) # degrees
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    φm = underlying_grid.conformal_mapping.southernmost_latitude

    Lz = underlying_grid.Lz

    # We need a bottom height field that ``masks'' the singularities.
    # Use φm + radius (not φm) to ensure the south boundary is immersed for FPivot grids,
    # where southernmost_latitude is at the cell face, leaving j=1 centers slightly north of φm.
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 360) < radius) & (abs(φp - φ) < radius)) | (φ < φm + radius) ? 0 : - Lz

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

    return grid
end

# Run the distributed grid simulation and save down reconstructed results
function run_distributed_tripolar_grid(arch, filename; fold_topology = RightCenterFolded, Nx = 80, Ny = 80)
    distributed_grid = TripolarGrid(arch; size = (Nx, Ny, 1), z = (-1000, 0), halo = (5, 5, 5), fold_topology)
    distributed_grid = analytical_immersed_tripolar_grid(distributed_grid)

    model = setup_simulation(distributed_grid)
    run_simulation!(model)

    η = reconstruct_global_field(model.free_surface.displacement)
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

    η = reconstruct_global_field(model.free_surface.displacement)
    u = reconstruct_global_field(model.velocities.u)
    v = reconstruct_global_field(model.velocities.v)
    c = reconstruct_global_field(model.tracers.c)

    if arch.local_rank == 0
        jldopen(filename, "w") do file
            file["u"] = u.data
            file["v"] = v.data
            file["c"] = c.data
            file["η"] = η.data
        end
    end

    return nothing
end

# Create model and set initial conditions (shared by serial and distributed)
function setup_simulation(grid)
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                        tracers = :c,
                                        tracer_advection = WENO(),
                                        momentum_advection = WENOVectorInvariant(order=3),
                                        coriolis = HydrostaticSphericalCoriolis())

    # Setup the model with a gaussian sea surface height
    # near the physical north poles and one near the equator
    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)
    set!(model; c=ηᵢ, η=ηᵢ)

    return model
end

# Time-step a model (shared by serial and distributed)
function run_simulation!(model; Δt = 5minutes, Nt = 100)
    arch = architecture(model.grid)
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
    for N in 2:Nt
        r_time_step!(model, Δt)
    end

    return model
end

# Convenience wrapper: setup + run (backward compat for LatLon tests etc.)
function run_distributed_simulation(grid)
    model = setup_simulation(grid)
    return run_simulation!(model)
end
