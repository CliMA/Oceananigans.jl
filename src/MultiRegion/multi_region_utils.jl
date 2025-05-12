import Oceananigans.Fields: flatten_tuple

flatten_tuple(mro::MultiRegionObject) = flatten_tuple(mro.regional_objects)

validate_devices(partition, ::CPU, devices) = nothing
validate_devices(p, ::CPU, ::Nothing) = nothing

# If no device is specified on the GPU, use only the default device
validate_devices(p, ::GPU, ::Nothing) = 1

function validate_devices(partition, ::GPU, devices)
    @assert length(unique(devices)) ≤ length(KA.devices())
    @assert maximum(devices) ≤ length(KA.devices())
    @assert length(devices) ≤ length(partition)
    return devices
end

function validate_devices(partition, ::GPU, devices::Number)
    @assert devices ≤ length(KA.devices())
    @assert devices ≤ length(partition)
    return devices
end

assign_devices(p, ::Nothing) = Tuple(CPU() for i in 1:length(p))

function assign_devices(p::AbstractPartition, dev::Number)
    part     = length(p)
    repeat   = part ÷ dev
    leftover = mod(part, dev)
    devices  = []

    for i in 1:dev
        KA.device!(i-1)
        for _ in 1:repeat
            push!(devices, KA.device())
        end
        if i ≤ leftover
            push!(devices, KA.device())
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
        KA.device!(dev[i])
        for _ in 1:repeat
            push!(devices, KA.device())
        end
        if i ≤ leftover
            push!(devices, KA.device())
        end
    end
    return Tuple(devices)
end

maybe_enable_peer_access!(devices) = nothing

# # Enable peer access by copying fake CuArrays between all devices
function maybe_enable_peer_access!(devices::NTuple{N, D}) where {N, D}

    fake_arrays = []
    for dev in devices
        switch_device!(dev)
        backend = KA.get_backend(dev)
        push!(fake_arrays, adapt(backend, zeros(2, 2, 2)))
    end

    sync_all_devices!(devices)

    for (idx_dst, dev_dst) in enumerate(devices)
        for (idx_src, dev_src) in enumerate(devices)
            if idx_dst != idx_src
                switch_device!(dev_src)
                src = fake_arrays[idx_src]
                switch_device!(dev_dst)
                dst = fake_arrays[idx_dst]
                copyto!(dst, src)
            end
        end
    end

    sync_all_devices!(devices)
    return nothing
end
