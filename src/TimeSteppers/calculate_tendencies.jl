"""
calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model, fill_halo_events)

    arch = model.architecture

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain
    interior_events = calculate_hydrostatic_tendency_contributions!(model, :interior; dependencies = device_event(arch))

    dependencies = fill_halo_events[end]

    boundary_events = []
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :west;   dependencies))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :east;   dependencies))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :south;  dependencies))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :north;  dependencies))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :bottom; dependencies))
    push!(boundary_events, calculate_hydrostatic_tendency_contributions!(model, :top;    dependencies))

    wait(device(arch), MultiEvent(tuple(fill_halo_events..., interior_events..., boundary_events...)))

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model)

    return nothing
end

@inline tendency_kernel_size(N, H, ::Val{:interior}) = N .- H
@inline tendency_kernel_size(N, H, ::Val{:west})   = (H[1], N[2], N[3])
@inline tendency_kernel_size(N, H, ::Val{:east})   = (H[1], N[2], N[3])
@inline tendency_kernel_size(N, H, ::Val{:south})  = (N[1], H[2], N[3])
@inline tendency_kernel_size(N, H, ::Val{:north})  = (N[1], H[2], N[3])
@inline tendency_kernel_size(N, H, ::Val{:bottom}) = (N[1], N[2], H[3])
@inline tendency_kernel_size(N, H, ::Val{:top})    = (N[1], N[2], H[3])

@inline tendency_kernel_offset(N, H, ::Val{:interior}) = H
@inline tendency_kernel_offset(N, H, ::Val{:west})   = (0, 0, 0)
@inline tendency_kernel_offset(N, H, ::Val{:east})   = (N[1] - H[1], 0, 0)
@inline tendency_kernel_offset(N, H, ::Val{:south})  = (0, 0, 0)
@inline tendency_kernel_offset(N, H, ::Val{:north})  = (0, N[2] - H[2], 0)
@inline tendency_kernel_offset(N, H, ::Val{:bottom}) = (0, 0, 0)
@inline tendency_kernel_offset(N, H, ::Val{:top})    = (0, 0, N[3] - H[3])
