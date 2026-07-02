# CUDA-side per-launch host-allocation fixes for Oceananigans.
# See notes/gpu-cuda-launch-allocation-investigation.md for the full derivation.
#
# These are method overrides into CUDA.jl's `CUDAKernels`/`CUDACore` modules, applied with `Core.eval`
# at load time (from `__init__` via `apply_allocation_fixes!`) — evaluating into a closed module at
# top level breaks incremental precompilation. They belong upstream in CUDA.jl (the mechanisms are
# backend-agnostic Julia codegen properties); they live here until then.

import Adapt

const _CUDAKernels = parentmodule(CUDABackend)   # CUDACore.CUDAKernels
const _CUDACore    = parentmodule(_CUDAKernels)  # CUDACore
const _KernelAdaptor = _CUDACore.KernelAdaptor

# (0) Flat, non-recursive device conversion of NamedTuple/Tuple kernel arguments.
#     Base's recursive tuple `map` inside `adapt(::KernelAdaptor, ::NamedTuple)` loses inference at
#     width ≥ 4 (the hand-unrolled path ends at 3), falling back to the generic NamedTuple constructor —
#     a heap rebuild of every wide container argument on every kernel launch (~75 KB/step on the CATKE
#     benchmark; `Base.promote_op` stays concrete, so only the value path degrades). A flat `@generated`
#     splat keeps each element conversion independently inferred and the whole conversion on the stack.
@generated function flat_adapt_namedtuple(to::_KernelAdaptor, nt::NamedTuple{names}) where {names}
    exprs = [:(Adapt.adapt(to, getfield(nt, $i))) for i in 1:length(names)]
    return :(NamedTuple{names}(($(exprs...),)))
end

@generated function flat_adapt_tuple(to::_KernelAdaptor, t::Tuple)
    exprs = [:(Adapt.adapt(to, getfield(t, $i))) for i in 1:fieldcount(t)]
    return :(($(exprs...),))
end

Adapt.adapt_structure(to::_KernelAdaptor, nt::NamedTuple) = flat_adapt_namedtuple(to, nt)
Adapt.adapt_structure(to::_KernelAdaptor, t::Tuple) = flat_adapt_tuple(to, t)

# (1) Kernel-launch functor.
#   a. Split the heavy trailing-`Vararg` functor into a tiny `@inline` dispatcher that bundles `args` into a
#      `Tuple` and a fixed-arity `cuda_run` worker. A non-inlined trailing-`Vararg` callee is invoked through
#      the jlcall ABI, which heap-boxes every non-isbits argument (each `Field`) on every launch; a fixed-arity
#      `Tuple` argument passes by pointer instead (mirrors KernelAbstractions' CPU `__run`).
#   b. Build the launch type signature from the argument *types* by inference, and call the (already cached)
#      `cufunction` directly, instead of `@cuda launch=false` — which materializes every converted argument
#      just to read its type, allocating for big nested args (NamedTuples/Tuples of fields) on every launch.
const _cuda_kernels_fix = quote
    @inline (obj::KA.Kernel{CUDABackend})(args...; ndrange=nothing, workgroupsize=nothing) =
        cuda_run(obj, ndrange, workgroupsize, args)

    const _oceananigans_compiler_configs = Dict{Tuple{CuDevice, Bool, Union{Int, Nothing}}, Any}()

    @noinline function _oceananigans_config(dev, always_inline::Bool, maxthreads::Union{Int, Nothing})
        key = (dev, always_inline, maxthreads)
        cfg = get(_oceananigans_compiler_configs, key, nothing)
        if cfg === nothing
            cfg = CUDACore.compiler_config(dev; always_inline, maxthreads)
            _oceananigans_compiler_configs[key] = cfg
        end
        return cfg
    end

    # Positional clone of `CUDACore.cufunction`: the kwargs path pays `pairs` iteration and
    # `hash(kwargs)` inside `compiler_config` on every (cached) launch.
    function _oceananigans_cufunction(f::F, tt::TT, always_inline::Bool, maxthreads::Union{Int, Nothing}) where {F, TT}
        cuda = CUDACore.active_state()
        Base.@lock CUDACore.cufunction_lock begin
            cache = CUDACore.compiler_cache(cuda.context)
            source = CUDACore.methodinstance(F, tt)
            config = _oceananigans_config(cuda.device, always_inline, maxthreads)::CUDACore.CUDACompilerConfig
            fun = CUDACore.GPUCompiler.cached_compilation(cache, source, config, CUDACore.compile, CUDACore.link)
            key = (objectid(source), hash(fun), f)
            kernel = get(CUDACore._kernel_instances, key, nothing)
            if kernel === nothing
                state = CUDACore.KernelState(CUDACore.create_exceptions!(fun.mod), UInt32(0))
                kernel = CUDACore.HostKernel{F, tt}(f, fun, state)
                CUDACore._kernel_instances[key] = kernel
            end
            return kernel::CUDACore.HostKernel{F, tt}
        end
    end

    function cuda_run(obj::KA.Kernel{CUDABackend}, ndrange, workgroupsize, args::Tuple)
        backend = KA.backend(obj)

        ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)

        maxthreads = KA.workgroupsize(obj) <: KA.StaticSize ? prod(KA.get(KA.workgroupsize(obj))) : nothing

        kernel_f = cudaconvert(obj.f)
        tt = Tuple{Base.promote_op(cudaconvert, typeof(ctx)),
                   map(a -> Base.promote_op(cudaconvert, typeof(a)), args)...}
        kernel = _oceananigans_cufunction(kernel_f, tt, backend.always_inline, maxthreads)

        if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
            config = CUDACore.launch_configuration(kernel.fun; max_threads=prod(ndrange))
            threads = if backend.prefer_blocks
                cu_blocks = max(cld(prod(ndrange), min(prod(ndrange), config.threads)), config.blocks)
                cld(prod(ndrange), cu_blocks)
            else
                config.threads
            end
            workgroupsize = threads_to_workgroupsize(threads, ndrange)
            iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
            ctx = KA.mkcontext(obj, ndrange, iterspace)
        end

        blocks  = length(KA.blocks(iterspace))
        threads = length(KA.workitems(iterspace))

        blocks == 0 && return nothing

        kernel(ctx, args...; threads, blocks)

        return nothing
    end
end

# (2) `prepare_cuda_state` runs before every driver call (e.g. the graph-capture check on every
# `pointer(::CuArray)`) and allocated a fresh `Ref{CUcontext}` each time; reuse a per-thread Ref instead.
# (The `activate` below is CUDACore context activation; this is not Pkg.activate.)
# Threads can be spawned after load (e.g. CUDA's synchronization worker), so the cache is sized
# generously and threads beyond it fall back to a fresh Ref — never index out of bounds.
const _cuda_core_fix = quote
    const _oceananigans_ctx_refs = Base.RefValue{CUcontext}[Ref{CUcontext}() for _ in 1:(Threads.maxthreadid() + 32)]

    function prepare_cuda_state()
        state = task_local_state!()
        tid = Threads.threadid()
        ctx = tid <= length(_oceananigans_ctx_refs) ? (@inbounds _oceananigans_ctx_refs[tid]) : Ref{CUcontext}()
        cuCtxGetCurrent(ctx)
        if ctx[] != state.context.handle
            activate(state.context)
        end
        return
    end
end

# (3) The driver `launch` allocates a fresh `attrs = CUlaunchAttribute[]` and `config = Ref(CUlaunchConfig)`
# on every kernel launch (they are handed to `cuLaunchKernelEx`, so escape analysis cannot stack them).
# Reuse a shared empty attrs vector (attributes are only needed for cooperative/cluster launches) and a
# per-thread config Ref. (~96 B x #launches per step.)
const _cuda_launch_fix = quote
    const _oceananigans_empty_attrs = CUlaunchAttribute[]
    const _oceananigans_config_refs = [Ref{CUlaunchConfig}() for _ in 1:(Threads.maxthreadid() + 32)]

    function launch(f::CuFunction, args::Vararg{Any,N}; blocks::CuDim=1, threads::CuDim=1,
                    clustersize::CuDim=1, cooperative::Bool=false, shmem::Integer=0,
                    stream::CuStream=stream()) where {N}
        blockdim = CuDim3(blocks)
        threaddim = CuDim3(threads)
        clusterdim = CuDim3(clustersize)

        simple = !cooperative && clusterdim.x == 1 && clusterdim.y == 1 && clusterdim.z == 1
        attrs = if simple
            _oceananigans_empty_attrs
        else
            a = CUlaunchAttribute[]
            if cooperative
                resize!(a, length(a)+1)
                attr = pointer(a, length(a))
                attr.id = CUDACore.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
                attr.value.cooperative = 1
            end
            if clusterdim.x != 1 || clusterdim.y != 1 || clusterdim.z != 1
                resize!(a, length(a)+1)
                attr = pointer(a, length(a))
                attr.id = CUDACore.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                attr.value.clusterDim.x = clusterdim.x
                attr.value.clusterDim.y = clusterdim.y
                attr.value.clusterDim.z = clusterdim.z
            end
            a
        end

        GC.@preserve attrs stream begin
            tid = Threads.threadid()
            config = tid <= length(_oceananigans_config_refs) ? (@inbounds _oceananigans_config_refs[tid]) :
                                                                Ref{CUlaunchConfig}()
            config[] = CUlaunchConfig(blockdim.x, blockdim.y, blockdim.z,
                                      threaddim.x, threaddim.y, threaddim.z,
                                      shmem, stream.handle, pointer(attrs), length(attrs))
            try
                pack_arguments(args...) do kernelParams
                    cuLaunchKernelEx(config, f, kernelParams, C_NULL)
                end
            catch err
                diagnose_launch_failure(f, config, err; blockdim, threaddim, clusterdim, shmem)
            end
        end
    end
end

function apply_allocation_fixes!()
    Core.eval(_CUDAKernels, _cuda_kernels_fix)
    Core.eval(_CUDACore, _cuda_core_fix)
    Core.eval(_CUDACore, _cuda_launch_fix)
    return nothing
end
