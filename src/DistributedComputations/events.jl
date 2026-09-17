record_event(arch) = sync_device!(arch)

record_event(arch::Distributed) = record_event(arch.child_architecture)

sync_event(event) = nothing
