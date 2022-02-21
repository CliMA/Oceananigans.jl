infer_architecture(::Nothing)   = CPU()
infer_architecture(dev::Tuple)  = length(dev) == 1 ? GPU() : GPU()
infer_architecture(dev::Number) = dev == 1 ? GPU() : GPU()

getdevice(a, i)                     = CPU()
getdevice(cu::OffsetArray, i)       = CUDA.device(cu.parent)
getdevice(cu::CuArray, i)           = CUDA.device(cu)
getdevice(cu::CuContext, i)         = CUDA.device(cu)
getdevice(cu::Union{CuPtr, Ptr}, i) = CUDA.device(cu)

getdevice(a)                     = CPU()
getdevice(cu::OffsetArray)       = CUDA.device(cu.parent)
getdevice(cu::CuArray)           = CUDA.device(cu)
getdevice(cu::CuContext)         = CUDA.device(cu)
getdevice(cu::Union{CuPtr, Ptr}) = CUDA.device(cu)

switch_device!(::CPU)    = nothing
switch_device!(dev::Int) = CUDA.device!(dev)
switch_device!(dev::CuDevice) = CUDA.device!(dev)
switch_device!(dev::Tuple, i) = switch_device!(dev[i])

underlying_arch(::MultiGPU) = GPU()
underlying_arch(::GPU) = GPU()
underlying_arch(::CPU) = CPU()

getregion(mo, i) = mo
isregional(mo) = hasproperty(mo, :devices)

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
