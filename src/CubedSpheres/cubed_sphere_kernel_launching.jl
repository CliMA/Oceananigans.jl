using KernelAbstractions: MultiEvent
using Oceananigans.Architectures: device

import Oceananigans.Utils: launch!

function launch!(arch, grid::ConformalCubedSphereGrid, args...; kwargs...)
    events = [launch!(arch, grid_face, args...; kwargs...) for grid_face in grid.faces]

    # We should return the events but let's just wait here because errors.
    events = filter(e -> e isa Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
