# CUDA-side per-launch host-allocation fixes for Oceananigans.
# See notes/gpu-cuda-launch-allocation-investigation.md for the full derivation.
#
# Most fixes are ordinary method definitions in this extension (precompile-friendly): the kernel-launch
# functor is overloaded on signatures *strictly more specific* than CUDA.jl's own
# `(obj::Kernel{CUDABackend})(args...)` — one method per workgroup-size type — so nothing is overwritten.
# Only the two driver tweaks at the bottom replace same-signature CUDA.jl methods and must be applied with
# `Core.eval` at load time, guarded against downstream precompilation.
#
# CUDA.jl ≥ 6.1 splits the implementation into the `CUDACore` subpackage while older versions keep it in
# `CUDA` itself, so the implementation modules are resolved through `parentmodule` and never referred to
# by name. Overrides that rely on version-specific internals are feature-gated and degrade to public-API
# fallbacks when the internals are absent.

import Adapt
import KernelAbstractions
const _KA = KernelAbstractions

const _CUDAKernels = parentmodule(CUDABackend)   # CUDACore.CUDAKernels (≥ 6.1) or CUDA.CUDAKernels
const _CUDACore    = parentmodule(_CUDAKernels)  # CUDACore (≥ 6.1) or CUDA
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

# (1) Positional path to the (cached) compiled kernel: the kwargs `cufunction` pays `pairs` iteration and
# `hash(kwargs)` inside `compiler_config` on every launch. Uses CUDA.jl internals, so it is feature-gated
# at extension-precompile time (the extension precompiles against the resolved CUDA.jl version) and
# replaced by the public kwargs `cufunction` when the internals are missing.
const _cufunction_internals = (:active_state, :cufunction_lock, :compiler_cache, :methodinstance,
                               :compiler_config, :CUDACompilerConfig, :GPUCompiler, :compile, :link,
                               :_kernel_instances, :KernelState, :create_exceptions!, :HostKernel,
                               :CuDevice)

if all(name -> isdefined(_CUDACore, name), _cufunction_internals)
    const _compiler_configs = Dict{Tuple{_CUDACore.CuDevice, Bool, Union{Int, Nothing}}, Any}()

    @noinline function cached_compiler_config(dev, always_inline::Bool, maxthreads::Union{Int, Nothing})
        key = (dev, always_inline, maxthreads)
        cfg = get(_compiler_configs, key, nothing)
        if cfg === nothing
            cfg = _CUDACore.compiler_config(dev; always_inline, maxthreads)
            _compiler_configs[key] = cfg
        end
        return cfg
    end

    function cached_cufunction(f::F, tt::TT, always_inline::Bool, maxthreads::Union{Int, Nothing}) where {F, TT}
        cuda = _CUDACore.active_state()
        Base.@lock _CUDACore.cufunction_lock begin
            cache = _CUDACore.compiler_cache(cuda.context)
            source = _CUDACore.methodinstance(F, tt)
            config = cached_compiler_config(cuda.device, always_inline, maxthreads)::_CUDACore.CUDACompilerConfig
            fun = _CUDACore.GPUCompiler.cached_compilation(cache, source, config, _CUDACore.compile, _CUDACore.link)
            key = (objectid(source), hash(fun), f)
            kernel = get(_CUDACore._kernel_instances, key, nothing)
            if kernel === nothing
                state = _CUDACore.KernelState(_CUDACore.create_exceptions!(fun.mod), UInt32(0))
                kernel = _CUDACore.HostKernel{F, tt}(f, fun, state)
                _CUDACore._kernel_instances[key] = kernel
            end
            return kernel::_CUDACore.HostKernel{F, tt}
        end
    end
else
    @inline cached_cufunction(f, tt, always_inline, maxthreads) =
        _CUDACore.cufunction(f, tt; always_inline, maxthreads)
end

# (2) Kernel-launch functor.
#   a. Split the heavy trailing-`Vararg` functor into a tiny `@inline` dispatcher that bundles `args` into a
#      `Tuple` and a fixed-arity `cuda_run` worker. A non-inlined trailing-`Vararg` callee is invoked through
#      the jlcall ABI, which heap-boxes every non-isbits argument (each `Field`) on every launch; a fixed-arity
#      `Tuple` argument passes by pointer instead (mirrors KernelAbstractions' CPU `__run`).
#   b. Build the launch type signature from the argument *types* by inference, and call the (already cached)
#      `cufunction` directly, instead of `@cuda launch=false` — which materializes every converted argument
#      just to read its type, allocating for big nested args (NamedTuples/Tuples of fields) on every launch.
#
# One dispatcher per workgroup-size type: each signature is strictly more specific than CUDA.jl's
# `(obj::Kernel{CUDABackend})(args...)`, so these are new methods (legal in precompilation), yet together
# they cover every kernel CUDA.jl constructs.
@inline (obj::_KA.Kernel{CUDABackend, <:_KA.NDIteration.StaticSize})(args...; ndrange=nothing, workgroupsize=nothing) =
    cuda_run(obj, ndrange, workgroupsize, args)

@inline (obj::_KA.Kernel{CUDABackend, <:_KA.NDIteration.DynamicSize})(args...; ndrange=nothing, workgroupsize=nothing) =
    cuda_run(obj, ndrange, workgroupsize, args)

# `promote_op` is not inferable for every adapt rule (e.g. closure reconstruction of `Returns` forcings
# gives `Any`, tripping GPUCompiler's `isdispatchtuple` assertion); recover the concrete type from the
# converted value in exactly those cases.
@inline function device_argument_type(a)
    a isa Type && return Core.Typeof(a)
    T = Base.promote_op(_CUDACore.cudaconvert, typeof(a))
    return isconcretetype(T) ? T : Core.Typeof(_CUDACore.cudaconvert(a))
end

function cuda_run(obj::_KA.Kernel{CUDABackend}, ndrange, workgroupsize, args::Tuple)
    backend = _KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = _KA.launch_config(obj, ndrange, workgroupsize)
    ctx = _KA.mkcontext(obj, ndrange, iterspace)

    maxthreads = _KA.workgroupsize(obj) <: _KA.NDIteration.StaticSize ? prod(_KA.get(_KA.workgroupsize(obj))) : nothing

    kernel_f = _CUDACore.cudaconvert(obj.f)
    tt = Tuple{device_argument_type(ctx), map(device_argument_type, args)...}
    kernel = cached_cufunction(kernel_f, tt, backend.always_inline, maxthreads)

    if _KA.workgroupsize(obj) <: _KA.NDIteration.DynamicSize && workgroupsize === nothing
        config = _CUDACore.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        threads = if backend.prefer_blocks
            cu_blocks = max(cld(prod(ndrange), min(prod(ndrange), config.threads)), config.blocks)
            cld(prod(ndrange), cu_blocks)
        else
            config.threads
        end
        workgroupsize = _CUDAKernels.threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = _KA.partition(obj, ndrange, workgroupsize)
        ctx = _KA.mkcontext(obj, ndrange, iterspace)
    end

    blocks  = length(_KA.blocks(iterspace))
    threads = length(_KA.workitems(iterspace))

    blocks == 0 && return nothing

    kernel(ctx, args...; threads, blocks)

    return nothing
end

# (3) `prepare_cuda_state` runs before every driver call (e.g. the graph-capture check on every
# `pointer(::CuArray)`) and allocated a fresh `Ref{CUcontext}` each time; reuse a per-thread Ref instead.
# (The `activate` below is CUDA context activation; this is not Pkg.activate.)
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

# (4) The driver `launch` allocates a fresh `attrs = CUlaunchAttribute[]` and `config = Ref(CUlaunchConfig)`
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
                attr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE
                attr.value.cooperative = 1
            end
            if clusterdim.x != 1 || clusterdim.y != 1 || clusterdim.z != 1
                resize!(a, length(a)+1)
                attr = pointer(a, length(a))
                attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
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

const _launch_fix_internals = (:CUlaunchAttribute, :CUlaunchConfig, :cuLaunchKernelEx, :pack_arguments,
                               :diagnose_launch_failure, :CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                               :CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)

function apply_allocation_fixes!()
    # These two replace same-signature CUDA.jl methods, which is legal only at runtime load: evaluating
    # into another module while a downstream package is being precompiled would not survive into its
    # image and errors. Skip there; `__init__` runs again in the actual session and applies them.
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing

    if isdefined(_CUDACore, :task_local_state!) && isdefined(_CUDACore, :prepare_cuda_state)
        Core.eval(_CUDACore, _cuda_core_fix)
    end

    if all(name -> isdefined(_CUDACore, name), _launch_fix_internals)
        Core.eval(_CUDACore, _cuda_launch_fix)
    end

    return nothing
end
