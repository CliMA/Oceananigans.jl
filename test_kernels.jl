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

grid = TripolarGrid(GPU(), size = (200, 200, 100), z = (-1000, 0), halo=(7, 7, 7))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -500 * rand() - 500); active_cells_map = true)

free_surface = SplitExplicitFreeSurface(grid; substeps = 20)
model = HydrostaticFreeSurfaceModel(grid; free_surface, tracers = :c, momentum_advection = WENOVectorInvariant(), tracer_advection = WENO(order=7))
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

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_Gc!, 
                                                        compute_hydrostatic_free_surface_Gu!, 
                                                        compute_hydrostatic_free_surface_Gv!

Gc = model.timestepper.Gⁿ.c
kernel, _ = Oceananigans.Utils.configure_kernel(arch, grid, :xyz, compute_hydrostatic_free_surface_Gc!; active_cells_map)
cuda_kernel = my_cuda_kernel(kernel, Gc, grid, args)
@info "Tracer registers $(CUDA.registers(cuda_kernel))" # 123 for mine # What will it be for AI?? 120 for the new one

u_immersed_bc = nothing
v_immersed_bc = nothing

u_forcing = model.forcing.u
v_forcing = model.forcing.v

start_momentum_kernel_args = (model.advection.momentum,
                              model.coriolis,
                              model.closure)

end_momentum_kernel_args = (model.velocities,
                            model.free_surface,
                            model.tracers,
                            model.buoyancy,
                            model.closure_fields,
                            model.pressure.pHY′,
                            model.auxiliary_fields,
                            model.vertical_coordinate,
                            model.clock)

u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args..., u_forcing)
v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args..., v_forcing)


u_kernel, _ = Oceananigans.Utils.configure_kernel(arch, grid, :xyz, compute_hydrostatic_free_surface_Gu!; active_cells_map)
v_kernel, _ = Oceananigans.Utils.configure_kernel(arch, grid, :xyz, compute_hydrostatic_free_surface_Gv!; active_cells_map)

Gu = model.timestepper.Gⁿ.u
Gv = model.timestepper.Gⁿ.v

cuda_u_kernel = my_cuda_kernel(u_kernel, Gu, grid, u_kernel_args)
cuda_v_kernel = my_cuda_kernel(v_kernel, Gv, grid, v_kernel_args)

@info "UVel registers $(CUDA.registers(cuda_u_kernel))"
@info "VVel registers $(CUDA.registers(cuda_v_kernel))"

# Detailed kernel analysis
for (name, kernel) in [("Tracer", cuda_kernel), ("UVel", cuda_u_kernel), ("VVel", cuda_v_kernel)]
    regs = CUDA.registers(kernel)
    maxthreads = CUDA.maxthreads(kernel)
    local_mem = CUDA.memory(kernel).local
    shared_mem = CUDA.memory(kernel).shared
    @info "$name kernel:" registers=regs maxthreads=maxthreads local_mem_bytes=local_mem shared_mem_bytes=shared_mem

    # Compute occupancy
    dev = CUDA.device()
    mp_count = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    max_regs = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    max_threads_mp = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    warp_size = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)
    threads_per_block = min(maxthreads, 256)
    warps_per_block = cld(threads_per_block, warp_size)
    regs_per_warp = regs * warp_size
    max_warps_by_regs = div(max_regs, regs_per_warp)
    max_warps_total = div(max_threads_mp, warp_size)
    active_warps = min(max_warps_by_regs, max_warps_total)
    occupancy = active_warps / max_warps_total
    @info "  Occupancy estimate: $(round(occupancy * 100, digits=1))% ($(active_warps)/$(max_warps_total) warps)"
    @info "  Register-limited warps/SM: $max_warps_by_regs"
end