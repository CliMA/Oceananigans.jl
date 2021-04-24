using KernelAbstractions: Event, MultiEvent

using Oceananigans.Architectures: device
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, PrescribedVelocityFields
import Oceananigans.Utils: launch!

function launch!(arch, grid::MultiRegionGrid, dims, kernel!, args...; kwargs...)

    events = []

    for (region_index, regional_grid) in enumerate(grid.regions)
        region_args = Tuple(get_region(arg, region_index) for arg in args)
        event = launch!(arch, regional_grid, dims, kernel!, region_args...; kwargs...)
        push!(events, event)
    end

    events = filter(e -> e isa Event, events)

    return MultiEvent(Tuple(events))
end
