
@inline infer_architecture(::Nothing)   = CPU()
@inline infer_architecture(dev::Tuple)  = length(dev) == 1 ? GPU() : MultiGPU()
@inline infer_architecture(dev::Number) = dev == 1 ? GPU() : MultiGPU()

@inline switch_device!(dev::CPU) = nothing
@inline switch_device!(dev::Int) = CUDA.device!(dev)
@inline switch_device!(dev::CuDevice) = CUDA.device!(dev)

@inline assoc_device(a) = CPU()
@inline assoc_device(cu::CuArray)               = CUDA.device(cu)
@inline assoc_device(cu::CuContext)             = CUDA.device(cu)
@inline assoc_device(cu::Union{CuPtr, Ptr})     = CUDA.device(cu)

function validate_devices(partition, devices)
    @assert length(devices) <= length(CUDA.devices())
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
