record_event(arch) = nothing

record_event(arch::Distributed) = record_event(arch.child_architecture)

sync_event(event) = nothing
