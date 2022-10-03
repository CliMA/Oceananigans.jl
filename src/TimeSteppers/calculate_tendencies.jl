using Oceananigans.Grids: size, halo_size

"""
calculate_tendencies!(model::NonhydrostaticModel)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(model, fill_halo_events)

    arch = model.architecture

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain

    N = size(model.grid)
    H = halo_size(model.grid)

    interior_events = calculate_tendency_contributions!(model, :interior; dependencies = device_event(arch))
    
    wait(device(arch), MultiEvent(Tuple(fill_halo_events)))

    if validate_kernel_size(N, H) # Split communication and computation only for 3D simulations
        boundary_events = []
        dependencies    = fill_halo_events[end]

        boundary_events = []
        for region in [:west, :east :south, :north, :bottom, :top]
            push!(boundary_events, calculate_tendency_contributions!(model, region;   dependencies)...)
        end
    else
        boundary_events = [NoneEvent()]
    end

    wait(device(arch), MultiEvent(tuple(interior_events..., boundary_events...)))

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    calculate_boundary_tendency_contributions!(model)

    return nothing
end

@inline validate_kernel_size(N, H) = all(N .- 2 .* H .> 0)

@inline function tendency_kernel_size(grid, ::Val{:interior}) 
    N = size(grid)
    H = halo_size(grid)
    return ifelse(validate_kernel_size(N, H), N .- 2 .* H, N)
end

## The edges and corners are calculated in the x direction

@inline tendency_kernel_size(grid, ::Val{:west})   = (halo_size(grid, 1), size(grid, 2), size(grid, 3))
@inline tendency_kernel_size(grid, ::Val{:south})  = (size(grid, 1) - 2*halo_size(grid, 1), halo_size(grid, 2), size(grid, 3) - 2*halo_size(grid, 3))
@inline tendency_kernel_size(grid, ::Val{:bottom}) = (size(grid, 1) - 2*halo_size(grid, 1), size(grid, 2) - 2*halo_size(grid, 2), halo_size(grid, 3))

@inline tendency_kernel_size(grid, ::Val{:east})   = tendency_kernel_size(grid, Val(:west))
@inline tendency_kernel_size(grid, ::Val{:north})  = tendency_kernel_size(grid, Val(:south))
@inline tendency_kernel_size(grid, ::Val{:top})    = tendency_kernel_size(grid, Val(:bottom))

@inline function tendency_kernel_offset(grid, ::Val{:interior}) 
    N = size(grid)
    H = halo_size(grid)
    return ifelse(validate_kernel_size(N, H), H, (0, 0, 0))
end

@inline tendency_kernel_offset(grid, ::Val{:west})   = (0, 0, 0)
@inline tendency_kernel_offset(grid, ::Val{:east})   = (size(grid, 1) - halo_size(grid, 1), 0, 0)
@inline tendency_kernel_offset(grid, ::Val{:south})  = (halo_size(grid, 1), 0, halo_size(grid, 3))
@inline tendency_kernel_offset(grid, ::Val{:north})  = (halo_size(grid, 1), size(grid, 2) - halo_size(grid, 2), halo_size(grid, 3))
@inline tendency_kernel_offset(grid, ::Val{:bottom}) = (halo_size(grid, 1), halo_size(grid, 2), 0)
@inline tendency_kernel_offset(grid, ::Val{:top})    = (halo_size(grid, 1), halo_size(grid, 2), size(grid, 3) - halo_size(grid, 3))
