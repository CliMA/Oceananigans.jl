import Oceananigans.Grids: new_data

@inline infer_architecture(::Nothing)   = CPU()
@inline infer_architecture(dev::Tuple)  = length(dev) == 1 ? GPU() : MultiGPU()
@inline infer_architecture(dev::Number) = dev == 1 ? GPU() : MultiGPU()

@inline assoc_device(a)                     = CPU()
@inline assoc_device(cu::CuArray)           = CUDA.device(cu)
@inline assoc_device(cu::CuContext)         = CUDA.device(cu)
@inline assoc_device(cu::Union{CuPtr, Ptr}) = CUDA.device(cu)

@inline switch_device!(::CPU)    = nothing
@inline switch_device!(dev::Int) = CUDA.device!(dev)
@inline switch_device!(dev::CuDevice) = CUDA.device!(dev)
@inline switch_device!(mrg::AbstractMultiGrid, i)  = switch_device!(assoc_device(mrg, i))
@inline switch_device!(mrf::AbstractMultiField, i) = switch_device!(assoc_device(mrf, i))

@inline switch_device!(dev::Tuple, i) = switch_device!(dev[i])

@inline regions(mrg::AbstractMultiGrid)  = regions(mrg.partition)
@inline regions(arr::Tuple)              = 1:length(arr)
@inline regions(par::AbstractPartition)  = 1:length(par)

@inline underlying_arch(::MultiGPU) = GPU()
@inline underlying_arch(::GPU) = GPU()
@inline underlying_arch(::CPU) = CPU()

const TupleOrGridOrField = Union{AbstractMultiGrid, AbstractMultiField, Tuple}

function MultiRegionObject(mrg::TupleOrGridOrField, constructor, args, iter_args, kwargs, iter_kwargs)
    regional_obj = []
    for i in regions(mrg)
        switch_device!(mrg, i)
        if isnothing(args) & isnothing(kwargs)
            push!(regional_obj, constructor())
        elseif isnothing(kwargs)
            push!(regional_obj, constructor(iterate_args(args, iter_args, i)...))
        elseif isnothing(kwargs)
            push!(regional_obj, constructor(; iterate_args(kwargs, iter_args, i)...))
        else
            push!(regional_obj, constructor(iterate_args(args, iter_args, i)...; iterate_args(kwargs, iter_kwargs, i)...))
        end
    end
    return MultiRegionObject(Tuple(regional_obj))
end

function multi_region_function!(mrg::TupleOrGridOrField, func!, args, iter_args, kwargs, iter_kwargs)
    for i in regions(mrg)
        switch_device!(mrg, i)
        if isnothing(args) & isnothing(kwargs)
            func!()
        elseif isnothing(kwargs)
            func!(iterate_args(args, iter_args, i)...)
        elseif isnothing(kwargs)
            func!(; iterate_args(kwargs, iter_args, i)...)
        else
            func!(iterate_args(args, iter_args, i)...; iterate_args(kwargs, iter_kwargs, i)...)
        end
    end
end

function validate_devices(partition, devices)
    @assert length(unique(devices)) <= length(CUDA.devices())
    @assert maximum(devices) <= length(CUDA.devices())
    @assert length(devices) <= length(partition)
    return devices
end

function validate_devices(partition, devices::Number)
    @assert devices <= length(CUDA.devices())
    @assert devices <= length(partition)
    return devices
end

validate_devices(p, ::Nothing) = nothing

assign_devices(p, ::Nothing) = Tuple(CPU() for i in 1:length(p))

function assign_devices(p::AbstractPartition, dev::Number) 
    part     = length(p)
    repeat   = part ÷ dev
    leftover = mod(part, dev)
    devices  = []

    for i in 1:dev
        CUDA.device!(i-1)
        for j in 1:repeat
            push!(devices, CUDA.device())
        end
        if i <= leftover
            push!(devices, CUDA.device())
        end
    end
    return Tuple(devices)
end

function assign_devices(p::AbstractPartition, dev::Tuple) 
    part     = length(p)
    repeat   = part ÷ length(dev)
    leftover = mod(part, length(dev))
    devices  = []

    for i in 1:length(dev)
        CUDA.device!(dev[i])
        for j in 1:repeat
            push!(devices, CUDA.device())
        end
        if i <= leftover
            push!(devices, CUDA.device())
        end
    end
    return Tuple(devices)
end

iterate_args(::Nothing, args...)               = (nothing, nothing)
iterate_args(args::Tuple, iterate, idx)        = Tuple(to_iterate(args[i], iterate[i], idx) for i in 1:length(args))
iterate_args(kwargs::NamedTuple, iterate, idx) = NamedTuple{keys(kwargs)}(to_iterate(kwargs[i], iterate[i], idx) for i in 1:length(kwargs))

to_iterate(arg, iter, idx) = iter == 1 ? arg[idx] : arg