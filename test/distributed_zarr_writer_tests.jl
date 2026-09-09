#####
##### MPI worker script — exercised by `test_distributed_zarr_writer.jl` via mpiexec.
#####
##### Args: --partition x|y|xy  --output <path.zarr>
#####

using MPI
MPI.Init()

using Oceananigans
using Zarr
using Oceananigans.DistributedComputations: Distributed, Partition

partition = "x"
out_path  = "dist_zarr_out.zarr"
let args = ARGS
    for (i, a) in enumerate(args)
        if a == "--partition" && i < length(args)
            global partition = args[i+1]
        elseif a == "--output" && i < length(args)
            global out_path = args[i+1]
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

grid = RectilinearGrid(arch,
                       size     = (8, 8, 4),
                       extent   = (1, 1, 1),
                       topology = (Periodic, Periodic, Periodic))

model = NonhydrostaticModel(grid; tracers=:c)
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
                    overwrite_existing = true,
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
    @assert size(u_arr) == (8, 8, 4, 3) "u global shape wrong: $(size(u_arr))"
    @assert size(c_arr) == (8, 8, 4, 3) "c global shape wrong: $(size(c_arr))"

    # Spot-check expected pattern from the seed function. u at (Face, Center, Center)
    # so x-nodes are at face positions.
    u0 = u_arr[:, :, :, 1]
    @assert any(u0 .> 0) "u all zero"

    # rank_topology recorded
    rt = g.attrs["rank_topology"]
    @assert length(rt) == 3 "rank_topology not 3-vector"

    println("DISTRIBUTED_ZARR_OK rank_topology=$(rt) partition=$partition nranks=$nranks")
end

MPI.Finalize()
