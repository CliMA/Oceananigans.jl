module OceananigansCUDAExt

using InteractiveUtils: versioninfo
using CUDA: CUDA, CuArray, CuContext, CuDevice, CuDeviceArray, CuPtr, context,
    context!, cu, CUDA.CUDABackend
# TODO: when we'll support CUDA.jl v6 only, add `cuSPARSE`/`CUDACore`/`cuFFT` to
# list of triggers of this extensions, and simplify the conditions below.
if isdefined(CUDA, :cuSPARSE)
    using CUDA.cuSPARSE: CuSparseMatrixCSC
else
    using CUDA.CUSPARSE: CuSparseMatrixCSC
end
if isdefined(CUDA, :cuFFT)
    using CUDA.cuFFT: cuFFT
else
    const cuFFT = CUDA.CUFFT
end
using GPUArraysCore: allowscalar
using GPUArrays: unsafe_free!
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata

import Oceananigans.Architectures as AC
import Oceananigans.BoundaryConditions as BC
import Oceananigans.DistributedComputations as DC
import Oceananigans.Fields as FD
import Oceananigans.MultiRegion as MR
import Oceananigans.Utils: apply_regionally!
import Oceananigans.Grids as GD
import Oceananigans.Solvers as SO
import Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces as SE
import Oceananigans.Utils as UT
import SparseArrays: SparseMatrixCSC
import KernelAbstractions: __iterspace, __dynamic_checkbounds, __validindex
import Oceananigans.DistributedComputations: Distributed

const GPUVar = Union{CuArray, CuContext, CuPtr, Ptr}

function __init__()
    if CUDA.functional()
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end

        allowscalar(false)
    end
end

const CUDAGPU = AC.GPU{<:CUDABackend}
CUDAGPU() = AC.GPU(CUDABackend(always_inline=true))

# Keep default CUDA backend
function AC.GPU()
    if CUDA.has_cuda_gpu()
        return CUDAGPU()
    else
        msg = """We cannot make a GPU with the CUDA backend:
                 a CUDA GPU was not found!"""
        throw(ArgumentError(msg))
    end
end

function UT.versioninfo_with_gpu(::CUDAGPU)
    s = sprint(versioninfo)
    gpu_name = CUDA.CuDevice(0) |> CUDA.name
    return "CUDA GPU: $gpu_name"
end

Base.summary(::CUDAGPU) = "CUDAGPU"
AC.device!(::CUDAGPU, i) = CUDA.device!(i)

AC.architecture(::CuArray) = CUDAGPU()
AC.architecture(::Type{CuArray}) = CUDAGPU()
AC.architecture(::CuDeviceArray) = CUDAGPU()
AC.architecture(::Type{CuDeviceArray}) = CUDAGPU()
AC.architecture(::CuSparseMatrixCSC) = CUDAGPU()
AC.array_type(::AC.GPU{CUDABackend}) = CuArray

AC.on_architecture(::CUDAGPU, a::Number) = a
AC.on_architecture(::AC.CPU, a::CuArray) = Array(a)
AC.on_architecture(::CUDAGPU, a::Array) = CuArray(a)
AC.on_architecture(::CUDAGPU, a::CuArray) = a
AC.on_architecture(::CUDAGPU, a::BitArray) = CuArray(a)
AC.on_architecture(::CUDAGPU, a::StepRangeLen) = a
AC.on_architecture(arch::Distributed, a::CuArray) = AC.on_architecture(AC.child_architecture(arch), a)

@inline AC.sparse_matrix_constructors(::AC.GPU{CUDABackend}, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))
@inline AC.sparse_matrix_constructors(::AC.CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline AC.sparse_matrix_constructors(::AC.GPU{CUDABackend}, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)

@inline AC.sparse_matrix(::AC.GPU{CUDABackend}, constr::Tuple) = CuSparseMatrixCSC(constr...)

@inline AC.on_architecture(::AC.CPU, A::CuSparseMatrixCSC)              = SparseMatrixCSC(AC.sparse_matrix_constructors(AC.CPU(), A)...)
@inline AC.on_architecture(::AC.GPU{CUDABackend}, A::SparseMatrixCSC)   = CuSparseMatrixCSC(AC.sparse_matrix_constructors(AC.GPU(), A)...)
@inline AC.on_architecture(::AC.GPU{CUDABackend}, A::CuSparseMatrixCSC) = A

# cu alters the type of `a`, so we convert it back to the correct type
AC.unified_array(::CUDAGPU, a::AbstractArray) = map(eltype(a), cu(a; unified = true))

## GPU to GPU copy of contiguous data
@inline function AC.device_copy_to!(dst::CuArray, src::CuArray; async::Bool = false)
    n = length(src)
    context!(context(src)) do
        GC.@preserve src dst begin
            unsafe_copyto!(pointer(dst, 1), pointer(src, 1), n; async)
        end
    end
    return dst
end

@inline AC.unsafe_free!(a::CuArray) = unsafe_free!(a)

@inline AC.convert_to_device(::CUDAGPU, args) = CUDA.cudaconvert(args)
@inline AC.convert_to_device(::CUDAGPU, args::Tuple) = map(CUDA.cudaconvert, args)

BC.validate_boundary_condition_architecture(::CuArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::CuArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `CuArray` on CPU architectures!"))

function SO.plan_forward_transform(A::CuArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return cuFFT.plan_fft!(A, dims)
end

FD.set!(v::FD.Field, a::CuArray) = FD.set_to_array!(v, a)
FD.set!(v::DC.DistributedField, a::CuArray) = FD.set_to_array!(v, a)
FD.set!(v::MR.MultiRegionField, a::CuArray) = apply_regionally!(FD.set!, v, a)

function SO.plan_backward_transform(A::CuArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return cuFFT.plan_ifft!(A, dims)
end

# CUDA version, the indices are passed implicitly
# You must not use KA here as this code is executed in another scope
CUDA.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        index = @inbounds linear_expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
        return index ≤ __linear_ndrange(ctx)
    else
        return true
    end
end

@inline UT.sync_device!(::CuDevice)      = CUDA.synchronize()
@inline UT.sync_device!(::CUDAGPU)       = CUDA.synchronize()
@inline UT.sync_device!(::CUDABackend)   = CUDA.synchronize()

# Use faster versions of `newton_div` on Nvidia GPUs
CUDA.@device_override UT.newton_div(::Type{UT.BackendOptimizedDivision}, a, b) = a * fast_inv_cuda(b)

function fast_inv_cuda(a::Float64)
    # Get the approximate reciprocal
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp-approx-ftz-f64
    # This instruction chops off last 32bits of mantissa and computes inverse
    # while treating all subnormal numbers as 0.0
    # If reciprocal would be subnormal, underflows to 0.0
    # 32 least significant bits of the result are filled with 0s
    inv_a = ccall("llvm.nvvm.rcp.approx.ftz.d", llvmcall, Float64, (Float64,), a)

    # Approximate the missing 32bits of mantissa with a single cubic iteration
    e = fma(inv_a, -a, 1.0)
    e = fma(e, e, e)
    inv_a = fma(e, inv_a, inv_a)
    return inv_a
end

function fast_inv_cuda(a::Float32)
    # This instruction just computes reciprocal flushing subnormals to 0.0
    # Hence for subnormal inputs it returns Inf
    # For large number whose reciprocal is subnormal it underflows to 0.0
    inv_a = ccall("llvm.nvvm.rcp.approx.ftz.f", llvmcall, Float32, (Float32,), a)
    return inv_a
end

#####
##### Barotropic substepping as a replayed CUDA graph
#####

struct BarotropicGraph{S, W}
    owner :: WeakRef
    weights :: W
    executable :: CUDA.CuGraphExec
    step_pointer :: CuPtr{Nothing}
    host_step :: Vector{S}
    device_buffers :: Tuple{CuArray, CuArray, CuArray}
end

const barotropic_graphs = Dict{Tuple{CuContext, CuPtr{Nothing}}, BarotropicGraph}()
const barotropic_graphs_lock = ReentrantLock()

# Δτᴮ is the third argument of both kernels and the clock the eighth argument of the free-surface kernel
function SE.substep_barotropic_mode!(::CUDAGPU, free_surface, barotropic_velocity_kernel!, free_surface_kernel!,
                                     converted_U_args, converted_η_args, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps

    step_values = (; Δτ = converted_U_args[3], clock = converted_η_args[8])
    owner = parent(free_surface.displacement.data)
    key = (context(), convert(CuPtr{Nothing}, pointer(owner)))
    graph = Base.@lock barotropic_graphs_lock get(barotropic_graphs, key, nothing)

    captured_weights = (weights, transport_weights)
    CapturedGraph = BarotropicGraph{typeof(step_values), typeof(captured_weights)}

    if graph isa CapturedGraph && graph.owner.value === owner && graph.weights === captured_weights
        replay_barotropic_graph!(graph, step_values)
    else
        graph = capture_barotropic_graph(owner, step_values, barotropic_velocity_kernel!, free_surface_kernel!,
                                         converted_U_args, converted_η_args, weights, transport_weights, Val(Nsubsteps))
        Base.@lock barotropic_graphs_lock begin
            filter!(entry -> !isnothing(last(entry).owner.value), barotropic_graphs)
            barotropic_graphs[key] = graph
        end
    end

    return nothing
end

function replay_barotropic_graph!(graph, step_values)
    copy_step_values!(graph.step_pointer, graph.host_step, step_values)
    CUDA.launch(graph.executable)
    return nothing
end

# `host_step` stays pageable: the asynchronous copy stages it before returning, so it can be overwritten on the next step
function copy_step_values!(step_pointer, host_step::Vector{S}, step_values::S) where S
    @inbounds host_step[1] = step_values
    GC.@preserve host_step unsafe_copyto!(convert(CuPtr{S}, step_pointer), pointer(host_step), 1; async=true)
    return nothing
end

function capture_barotropic_graph(owner, step_values, barotropic_velocity_kernel!, free_surface_kernel!,
                                  converted_U_args, converted_η_args, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps

    step = CuArray{typeof(step_values)}(undef, 1)
    host_step = [step_values]
    averaging_weights_array = CuArray(collect(weights))
    transport_weights_array = CuArray(collect(transport_weights))

    Δτ = SE.StepValue{:Δτ}(CUDA.cudaconvert(step))
    clock = SE.StepValue{:clock}(CUDA.cudaconvert(step))
    velocity_arguments = Base.setindex(converted_U_args, Δτ, 3)
    free_surface_arguments = Base.setindex(Base.setindex(converted_η_args, Δτ, 3), clock, 8)
    device_averaging_weights = CUDA.cudaconvert(averaging_weights_array)
    device_transport_weights = CUDA.cudaconvert(transport_weights_array)

    substeps! = () -> for substep in 1:Nsubsteps
        barotropic_velocity_kernel!(SE.SubstepWeight(substep, device_transport_weights), velocity_arguments...)
        free_surface_kernel!(SE.SubstepWeight(substep, device_averaging_weights), free_surface_arguments...)
    end

    step_pointer = convert(CuPtr{Nothing}, pointer(step))
    copy_step_values!(step_pointer, host_step, step_values)

    # the uncaptured run advances this step and compiles the kernels before capture
    substeps!()
    executable = CUDA.instantiate(CUDA.capture(substeps!))

    return BarotropicGraph(WeakRef(owner), (weights, transport_weights), executable, step_pointer, host_step,
                           (step, averaging_weights_array, transport_weights_array))
end

end # module OceananigansCUDAExt
