module Architecture
    using CUDA
    using Oceananigans

    import Oceananigans.Architectures: 
        CPU, GPU, available,
        device, architecture, array_type, arch_array, unified_array,
        device_copy_to!, unsafe_free!

    device(::GPU) = CUDA.CUDABackend(; always_inline=true)
    architecture(::CuArray) = GPU()
    available(::GPU) = CUDA.has_cuda()
    array_type(::GPU) = CuArray

    arch_array(::GPU, a::Array)   = CuArray(a)
    arch_array(::CPU, a::CuArray) = Array(a)
    arch_array(::GPU, a::CuArray) = a
    arch_array(::GPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
    arch_array(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
    arch_array(::GPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)

    unified_array(::GPU, a) = a

    function unified_array(::GPU, arr::AbstractArray) 
        buf = Mem.alloc(Mem.Unified, sizeof(arr))
        vec = unsafe_wrap(CuArray{eltype(arr),length(size(arr))}, convert(CuPtr{eltype(arr)}, buf), size(arr))
        finalizer(vec) do _
            Mem.free(buf)
        end
        copyto!(vec, arr)
        return vec
    end

    ## Only for contiguous data!! (i.e. only if the offset for pointer(dst::CuArrat, offset::Int) is 1)
    @inline function device_copy_to!(dst::CuArray, src::CuArray; async::Bool = false) 
        n = length(src)
        context!(context(src)) do
            GC.@preserve src dst begin
                unsafe_copyto!(pointer(dst, 1), pointer(src, 1), n; async)
            end
        end
        return dst
    end

    @inline unsafe_free!(a::CuArray) = CUDA.unsafe_free!(a)
end
