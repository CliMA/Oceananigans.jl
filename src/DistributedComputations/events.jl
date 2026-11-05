# Each communication state owns one event, re-recorded after every pack of its send buffers
new_event(arch) = nothing
new_event(arch::Distributed) = new_event(arch.child_architecture)

record_event!(event, arch) = sync_device!(arch)
record_event!(event, arch::Distributed) = record_event!(event, arch.child_architecture)

event_done(event) = true
bind_thread_to_device!(event) = nothing

function wait_for_event(event)
    while !event_done(event)
        Threads.nthreads() > 1 && yield()
    end
    return nothing
end
