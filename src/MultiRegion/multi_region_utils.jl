using Oceananigans.Architectures: device, device!
using Oceananigans.Fields: Fields, flatten_tuple

Fields.flatten_tuple(mro::MultiRegionObject) = flatten_tuple(mro.regional_objects)

validate_devices(partition, ::CPU, devices) = nothing
validate_devices(p, ::CPU, ::Nothing) = nothing

# If no device is specified on the GPU, use only the default device
validate_devices(p, ::GPU, ::Nothing) = 1

function validate_devices(partition, arch::GPU, devices)
    @assert length(unique(devices)) ≤ length(devices)
    @assert maximum(devices) ≤ length(devices)
    @assert length(devices) ≤ length(partition)
    return devices
end

function validate_devices(partition, ::GPU, devices::Number)
    @assert devices ≤ length(devices)
    @assert devices ≤ length(partition)
    return devices
end

assign_devices(arch::AbstractArchitecture, p::AbstractPartition, ::Nothing) = Tuple(arch for i in 1:length(p))

function assign_devices(arch::AbstractArchitecture, p::AbstractPartition, dev::Number)
    part     = length(p)
    repeat   = part ÷ dev
    leftover = mod(part, dev)
    devices  = []

    for i in 1:dev
        device!(arch, i-1)
        for _ in 1:repeat
            push!(devices, device(arch))
        end
        if i ≤ leftover
            push!(devices, device(arch))
        end
    end
    return Tuple(devices)
end

function assign_devices(arch::AbstractArchitecture, p::AbstractPartition, dev::Tuple)
    part     = length(p)
    repeat   = part ÷ length(dev)
    leftover = mod(part, length(dev))
    devices  = []

    for i in 1:length(dev)
        device!(arch, dev[i])
        for _ in 1:repeat
            push!(devices, device(arch))
        end
        if i ≤ leftover
            push!(devices, device(arch))
        end
    end
    return Tuple(devices)
end
