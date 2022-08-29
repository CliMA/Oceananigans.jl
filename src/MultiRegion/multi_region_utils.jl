
validate_devices(partition, ::CPU, devices) = nothing
validate_devices(p, ::CPU, ::Nothing) = nothing

# If no device is specified on the GPU, use only the default device
validate_devices(p, ::CUDAGPU, ::Nothing) = 1
validate_devices(p, ::ROCMGPU, ::Nothing) = 1

function validate_devices(partition, ::CUDAGPU, devices)
    @assert length(unique(devices)) <= length(CUDA.devices())
    @assert maximum(devices) <= length(CUDA.devices())
    @assert length(devices) <= length(partition)
    return devices
end

function validate_devices(partition, ::ROCMGPU, devices)
    @assert length(unique(devices)) <= length(AMDGPU.get_agents(:gpu))
    @assert maximum(devices) <= length(AMDGPU.get_agents(:gpu))
    @assert length(devices) <= length(partition)
    return devices
end

function validate_devices(partition, ::CUDAGPU, devices::Number)
    @assert devices <= length(CUDA.devices())
    @assert devices <= length(partition)
    return devices
end

function validate_devices(partition, ::ROCMGPU, devices::Number)
    @assert devices <= length(AMDGPU.get_agents(:gpu))
    @assert devices <= length(partition)
    return devices
end

assign_devices(p, ::Nothing) = Tuple(CPU() for i in 1:length(p))

function assign_devices(p::AbstractPartition, dev::Number) 
    part     = length(p)
    repeat   = part ÷ dev
    leftover = mod(part, dev)
    devices  = []

    PROVIDER = AMDGPU.has_rocm_gpu() ? AMDGPU : CUDA

    for i in 1:dev
        PROVIDER.device!(i-1)
        for j in 1:repeat
            push!(devices, PROVIDER.device())
        end
        if i <= leftover
            push!(devices, PROVIDER.device())
        end
    end
    return Tuple(devices)
end

function assign_devices(p::AbstractPartition, dev::Tuple) 
    part     = length(p)
    repeat   = part ÷ length(dev)
    leftover = mod(part, length(dev))
    devices  = []

    PROVIDER = AMDGPU.has_rocm_gpu() ? AMDGPU : CUDA

    for i in 1:length(dev)
        PROVIDER.device!(dev[i])
        for j in 1:repeat
            push!(devices, PROVIDER.device())
        end
        if i <= leftover
            push!(devices, PROVIDER.device())
        end
    end
    return Tuple(devices)
end
