# `(; uuid, pci_address, gpu_aware_mpi, memory_pool)` of this rank's GPU (`gpu_aware_mpi = nothing` when unknown), or `nothing`
gpu_configuration(child_architecture) = nothing

# Linux CPU list, e.g. "0-3,8,10-11"
function parse_cpu_list(list)
    cpus = Int[]
    for range in split(strip(list), ',')
        bounds = parse.(Int, split(range, '-'))
        append!(cpus, first(bounds):last(bounds))
    end
    return cpus
end

function warn_on_ranks(condition, communicator, message)
    affected_ranks = MPI.Allreduce(Int(condition), +, communicator)
    @root communicator if affected_ranks > 0
        @warn "$message (on $affected_ranks of $(MPI.Comm_size(communicator)) MPI ranks)" _id=message maxlog=1
    end
    return nothing
end

function validate_distributed_configuration(arch)
    child_architecture = arch.child_architecture
    communicator = mpi_communicator(arch.communicator)
    cpus = Sys.islinux() ? parse_cpu_list(match(r"Cpus_allowed_list:\s*(\S+)", read("/proc/self/status", String))[1]) : nothing

    warn_on_ranks(has_spare_thread() && !use_progress_worker(), communicator,
                  "MPI was initialized without MPI_THREAD_MULTIPLE, so halo messages progress only when the main thread calls MPI")

    warn_on_ranks(!isnothing(cpus) && length(cpus) < Threads.nthreads(:default), communicator,
                  "MPI ranks have fewer CPU cores than Julia threads; give each rank at least one core per thread, e.g. with Slurm's --cpus-per-task")

    gpu = gpu_configuration(child_architecture)
    isnothing(gpu) && return nothing

    node_communicator = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(communicator))
    node_ranks = MPI.Comm_size(node_communicator)
    shared_gpu = count(==(gpu.uuid), MPI.Allgather(gpu.uuid, node_communicator)) > 1
    gpu_cpu_list = "/sys/bus/pci/devices/$(gpu.pci_address)/local_cpulist"
    distant_gpu = !isnothing(cpus) && isfile(gpu_cpu_list) && isdisjoint(cpus, parse_cpu_list(read(gpu_cpu_list, String)))
    mpi_moves_gpu_buffers = arch.communicator isa MPI.Comm

    warn_on_ranks(mpi_moves_gpu_buffers && gpu.gpu_aware_mpi === false, communicator,
                  "MPI is not GPU-aware (on Cray MPICH, set MPICH_GPU_SUPPORT_ENABLED=1), but Oceananigans passes GPU buffers to MPI; use a GPU-aware MPI or NCCLDistributed")

    warn_on_ranks(shared_gpu, communicator, "Several MPI ranks share one GPU; launch at most one rank per GPU")

    warn_on_ranks(!shared_gpu && node_ranks > 1 && ndevices(child_architecture) == 1, communicator,
                  "Each MPI rank sees only its own GPU (e.g. with Slurm's --gpus-per-task), which can prevent direct GPU-to-GPU copies between ranks; use --gpus-per-node instead")

    warn_on_ranks(mpi_moves_gpu_buffers && gpu.memory_pool && node_ranks > 1, communicator,
                  "CUDA.jl's memory pool prevents CUDA IPC, so MPI copies GPU buffers between ranks on the same node through the host; set JULIA_CUDA_MEMORY_POOL=none")

    warn_on_ranks(distant_gpu, communicator,
                  "MPI ranks run on CPU cores that are not attached to their GPU; bind each rank to its GPU's cores, e.g. with Slurm's --cpu-bind, or choose GPUs with the `devices` keyword")

    return nothing
end
