getdevice(a, i)                     = CPU()
getdevice(cu::CuArray, i)           = CUDA.device(cu)
getdevice(cu::CuContext, i)         = CUDA.device(cu)
getdevice(cu::Union{CuPtr, Ptr}, i) = CUDA.device(cu)
getdevice(cu::OffsetArray, i)       = getdevice(cu.parent)

getdevice(a)                     = CPU()
getdevice(cu::CuArray)           = CUDA.device(cu)
getdevice(cu::CuContext)         = CUDA.device(cu)
getdevice(cu::Union{CuPtr, Ptr}) = CUDA.device(cu)
getdevice(cu::OffsetArray)       = getdevice(cu.parent)

switch_device!(::CPU)    = nothing
switch_device!(dev::Int) = CUDA.device!(dev)
switch_device!(dev::CuDevice) = CUDA.device!(dev)
switch_device!(dev::Tuple, i) = switch_device!(dev[i])

struct Reference{R}
    ref :: R
end

struct Iterate{I}
    iter :: I
end

getregion(mo, i) = mo
getregion(ref::Reference, i) = ref.ref
getregion(iter::Iterate, i)  = iter.iter[i]

isregional(a) = false

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
