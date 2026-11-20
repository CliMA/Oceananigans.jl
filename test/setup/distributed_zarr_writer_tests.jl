#####
##### MPI worker script — exercised by `test/distributed/zarr_writer.jl` via mpiexec.
#####
##### Args: --partition x|y|xy  --output <path.zarr>  [--grid rectilinear|tripolar]  [--fold center|face]
#####

using MPI
MPI.Init(threadlevel=:multiple)

using Oceananigans
using Zarr
using Oceananigans.DistributedComputations: Distributed, Partition

partition = "x"
out_path  = "dist_zarr_out.zarr"
grid_type = "rectilinear"
fold      = "center"
let args = ARGS
    for (i, a) in enumerate(args)
        if a == "--partition" && i < length(args)
            global partition = args[i+1]
        elseif a == "--output" && i < length(args)
            global out_path = args[i+1]
        elseif a == "--grid" && i < length(args)
            global grid_type = args[i+1]
        elseif a == "--fold" && i < length(args)
            global fold = args[i+1]
        end
    end
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if partition == "x"
    arch = Distributed(CPU(); partition=Partition(nranks, 1, 1))
elseif partition == "y"
    arch = Distributed(CPU(); partition=Partition(1, nranks, 1))
elseif partition == "xy"
    @assert nranks == 4
    arch = Distributed(CPU(); partition=Partition(2, 2, 1))
else
    error("Unknown partition: $partition")
end

tripolar_kw = (size = (16, 12, 4), z = (-1, 0), fold_topology = fold == "face" ? FaceFolded : CenterFolded)

grid = grid_type == "tripolar" ? TripolarGrid(arch; tripolar_kw...) :
                                 RectilinearGrid(arch,
                                                 size     = (8, 8, 4),
                                                 extent   = (1, 1, 1),
                                                 topology = (Periodic, Periodic, Periodic))

model = grid_type == "tripolar" ? HydrostaticFreeSurfaceModel(grid; tracers=:c) : NonhydrostaticModel(grid; tracers=:c)
# Seed every grid point with x-coordinate to make per-rank slabs distinguishable.
set!(model, u=(x, y, z) -> 10*x + y,
            c=(x, y, z) -> 100*x + y + z)

# NOTE: avoid the Simulation/run! harness for now — its update_state! interacts with
# distributed halos in a way that is independent of the writer. The writer's MPI
# behavior is the thing under test, so we drive initialize! + write_output! directly.
writer = ZarrWriter(model, (u=model.velocities.u, c=model.tracers.c);
                    filename = out_path,
                    dir = ".",
                    schedule = IterationInterval(1),
                    overwrite_files = true,
                    with_halos = false)

Oceananigans.initialize!(writer, model)
# Mimic 3 write steps with incrementing model time.
for n in 0:2
    model.clock.iteration = n
    model.clock.time = Float64(n)
    Oceananigans.write_output!(writer, model)
end

MPI.Barrier(comm)

# Root rank verifies global content using the serial reader.
if rank == 0
    g = Zarr.zopen(out_path)
    times = g["time"][:]
    @assert length(times) == 3 "expected 3 timesteps, got $(length(times))"

    u_arr = g["u"]
    c_arr = g["c"]

    if grid_type == "tripolar"
        # Every rank's slab must land where the serial grid puts it, including the fold rows
        serial = HydrostaticFreeSurfaceModel(TripolarGrid(CPU(); tripolar_kw...); tracers=:c)
        set!(serial, u=(x, y, z) -> 10*x + y, c=(x, y, z) -> 100*x + y + z)
        for (name, arr, field) in (("u", u_arr, serial.velocities.u), ("c", c_arr, serial.tracers.c))
            @assert size(arr) == (size(field)..., 3) "$name global shape wrong: $(size(arr)) vs $(size(field))"
            @assert arr[:, :, :, 3] == Float32.(interior(field)) "$name differs from the serial field"
        end
    else
        @assert size(u_arr) == (8, 8, 4, 3) "u global shape wrong: $(size(u_arr))"
        @assert size(c_arr) == (8, 8, 4, 3) "c global shape wrong: $(size(c_arr))"
    end

    # Spot-check expected pattern from the seed function. u at (Face, Center, Center)
    # so x-nodes are at face positions.
    u0 = u_arr[:, :, :, 1]
    @assert any(u0 .> 0) "u all zero"

    # rank_topology recorded
    rt = g.attrs["rank_topology"]
    @assert length(rt) == 3 "rank_topology not 3-vector"

    println("DISTRIBUTED_ZARR_OK rank_topology=$(rt) partition=$partition nranks=$nranks grid=$grid_type fold=$fold")
end

MPI.Finalize()
