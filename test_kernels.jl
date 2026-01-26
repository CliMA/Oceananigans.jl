import KernelAbstractions as KA
using CUDA
using Oceananigans
using CUDA: CUDABackend

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function my_cuda_kernel(obj::KA.Kernel{CUDABackend}, args...; ndrange=nothing, workgroupsize=nothing)
    backend = KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KA.workgroupsize(obj) <: KA.StaticSize
        maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = @cuda launch=false always_inline=backend.always_inline maxthreads=maxthreads obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        config = CUDA.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        if backend.prefer_blocks
            # Prefer blocks over threads
            threads = min(prod(ndrange), config.threads)
            # XXX: Some kernels performs much better with all blocks active
            cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
            threads = cld(prod(ndrange), cu_blocks)
        else
            threads = config.threads
        end

        workgroupsize = threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    if blocks == 0
        return nothing
    end

    return kernel
end

using Random
Random.seed!(1234)

grid = TripolarGrid(GPU(), size = (200, 200, 100), z = (-1000, 0), halo=(6,6,6))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -500 * rand() - 500); active_cells_map = true)

free_surface = SplitExplicitFreeSurface(grid; substeps = 20)
model = HydrostaticFreeSurfaceModel(grid; free_surface, tracers = :c, momentum_advection = nothing, tracer_advection = WENO(order=7))
set!(model, u = (x, y, z) -> rand(), v = (x, y, z) -> rand(), c = (x, y, z) -> rand())

arch = model.architecture
grid = model.grid

tracer_index = 1
tracer_name = :c

@inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
@inbounds c_advection   = model.advection[tracer_name]
@inbounds c_forcing     = model.forcing[tracer_name]
@inbounds c_immersed_bc = nothing

active_cells_map = Oceananigans.Grids.get_active_cells_map(model.grid, Val(:interior))

args = tuple(Val(tracer_index),
             Val(tracer_name),
             c_advection,
             model.closure,
             c_immersed_bc,
             model.buoyancy,
             model.biogeochemistry,
             model.transport_velocities,
             model.free_surface,
             model.tracers,
             model.closure_fields,
             model.auxiliary_fields,
             model.clock,
             c_forcing)

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_Gc!

Gc = model.timestepper.Gⁿ.c
kernel, _ = Oceananigans.Utils.configure_kernel(arch, grid, :xyz, compute_hydrostatic_free_surface_Gc!; active_cells_map)
cuda_kernel = my_cuda_kernel(kernel, Gc, grid, args)
CUDA.registers(cuda_kernel) # 123 for mine # What will it be for AI?? 120 for the new one