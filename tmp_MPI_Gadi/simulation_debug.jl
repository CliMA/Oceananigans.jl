# Simulation debug: runs serial + distributed models side by side,
# compares fields at steps 0, 1, 10, 100.
# Usage: mpiexec -n N julia --check-bounds=yes simulation_debug.jl <upivot|fpivot> <Rx> <Ry>

using MPI
MPI.Init()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: RightFaceFolded, RightCenterFolded, topology
using Oceananigans.DistributedComputations: reconstruct_global_field
using Oceananigans.TimeSteppers: first_time_step!

fold_arg = ARGS[1]
Rx = parse(Int, ARGS[2])
Ry = parse(Int, ARGS[3])

if fold_arg == "upivot"
    fold_topology = RightCenterFolded
    Ny = 80
elseif fold_arg == "fpivot"
    fold_topology = RightFaceFolded
    Ny = 81
else
    error("Unknown fold topology: $fold_arg. Use 'upivot' or 'fpivot'.")
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Mask the singularity of the grid
function analytical_immersed_tripolar_grid(underlying_grid; radius = 5)
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    φm = underlying_grid.conformal_mapping.southernmost_latitude
    Lz = underlying_grid.Lz

    # Use φm + radius (not φm) to ensure the south boundary is immersed for FPivot grids,
    # where southernmost_latitude is at the cell face, leaving j=1 centers slightly north of φm.
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 360) < radius) & (abs(φp - φ) < radius)) | (φ < φm + radius) ? 0 : - Lz

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
end

function compare_one(name, is, id, Ny_grid)
    diff = abs.(is .- id)
    maxdiff = maximum(diff)
    match = all(is .≈ id)

    if !match
        idx = argmax(diff)
        j_max = idx[2]
        near_fold = j_max > Ny_grid - 10

        # Count mismatches per j-row
        mismatches = .!(is .≈ id)
        nmismatch = sum(mismatches)
        mismatch_by_j = [sum(mismatches[:, j, :]) for j in axes(is, 2)]
        j_with_mm = findall(mismatch_by_j .> 0)

        println("  $name: MISMATCH  maxdiff=$maxdiff  at $(Tuple(idx))  near_fold=$near_fold  mismatches=$nmismatch/$(length(is))")
        println("         serial=$(is[idx])  distributed=$(id[idx])")
        if !isempty(j_with_mm)
            j_min, j_max_mm = extrema(j_with_mm)
            println("         mismatches in j=$j_min..$j_max_mm (Ny=$Ny_grid)")
            # Print per-j counts for rows near the fold (top 20 rows)
            fold_start = max(1, Ny_grid - 20)
            for j in fold_start:length(mismatch_by_j)
                if mismatch_by_j[j] > 0
                    println("           j=$j: $(mismatch_by_j[j]) mismatches")
                end
            end
        end
    else
        println("  $name: MATCH  (maxdiff=$maxdiff)")
    end
end

function compare_fields(label, serial_model, dist_model, rank, Ny_grid)
    fields_s = (u = serial_model.velocities.u,
                v = serial_model.velocities.v,
                c = serial_model.tracers.c,
                η = serial_model.free_surface.displacement)

    fields_d = (u = dist_model.velocities.u,
                v = dist_model.velocities.v,
                c = dist_model.tracers.c,
                η = dist_model.free_surface.displacement)

    for name in (:u, :v, :c, :η)
        fd_global = reconstruct_global_field(fields_d[name])
        if rank == 0
            compare_one(name, interior(fields_s[name]), interior(fd_global), Ny_grid)
        end
    end

    # Also compare barotropic velocities
    baro_s = serial_model.free_surface.barotropic_velocities
    baro_d = dist_model.free_surface.barotropic_velocities
    for (name, fs, fd) in [(:U, baro_s.U, baro_d.U), (:V, baro_s.V, baro_d.V)]
        fd_global = reconstruct_global_field(fd)
        if rank == 0
            is = interior(fs)
            id = interior(fd_global)
            # Sizes may differ if halos extended differently; truncate to min
            nx = min(size(is,1), size(id,1))
            ny = min(size(is,2), size(id,2))
            nz = min(size(is,3), size(id,3))
            compare_one(name, is[1:nx, 1:ny, 1:nz], id[1:nx, 1:ny, 1:nz], Ny_grid)
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)
end

# Setup
arch = Distributed(CPU(); partition = Partition(Rx, Ry))

dist_grid = TripolarGrid(arch; size=(80, Ny, 1), z=(-1000, 0), halo=(5, 5, 5), fold_topology)
dist_grid = analytical_immersed_tripolar_grid(dist_grid)

serial_grid = TripolarGrid(; size=(80, Ny, 1), z=(-1000, 0), halo=(5, 5, 5), fold_topology)
serial_grid = analytical_immersed_tripolar_grid(serial_grid)

if rank == 0
    println("=== Simulation Debug: $fold_arg Partition($Rx,$Ry) Ny=$Ny ===")
end

local_g = dist_grid.underlying_grid
println("[Rank $rank] local Ny=$(local_g.Ny), Hy=$(local_g.Hy), topo_y=$(topology(local_g, 2))")
MPI.Barrier(MPI.COMM_WORLD)

ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)

dist_model = HydrostaticFreeSurfaceModel(dist_grid;
    free_surface = SplitExplicitFreeSurface(dist_grid; substeps=20),
    tracers = :c, tracer_advection = WENO(),
    momentum_advection = WENOVectorInvariant(; order=3),
    coriolis = HydrostaticSphericalCoriolis())

serial_model = HydrostaticFreeSurfaceModel(serial_grid;
    free_surface = SplitExplicitFreeSurface(serial_grid; substeps=20),
    tracers = :c, tracer_advection = WENO(),
    momentum_advection = WENOVectorInvariant(; order=3),
    coriolis = HydrostaticSphericalCoriolis())

set!(dist_model; c=ηᵢ, η=ηᵢ)
set!(serial_model; c=ηᵢ, η=ηᵢ)

# Step 0
if rank == 0; println("\n=== Step 0 (after set!) ==="); end
MPI.Barrier(MPI.COMM_WORLD)
compare_fields("step0", serial_model, dist_model, rank, Ny)

# Step 1
Δt = 5minutes
first_time_step!(dist_model, Δt)
first_time_step!(serial_model, Δt)
if rank == 0; println("\n=== Step 1 ==="); end
MPI.Barrier(MPI.COMM_WORLD)
compare_fields("step1", serial_model, dist_model, rank, Ny)

# Steps 2-10
for n in 2:10
    time_step!(dist_model, Δt)
    time_step!(serial_model, Δt)
end
if rank == 0; println("\n=== Step 10 ==="); end
MPI.Barrier(MPI.COMM_WORLD)
compare_fields("step10", serial_model, dist_model, rank, Ny)

# Steps 11-100
for n in 11:100
    time_step!(dist_model, Δt)
    time_step!(serial_model, Δt)
end
if rank == 0; println("\n=== Step 100 ==="); end
MPI.Barrier(MPI.COMM_WORLD)
compare_fields("step100", serial_model, dist_model, rank, Ny)

if rank == 0; println("\n=== Done ==="); end
MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
