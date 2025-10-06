using Oceananigans.BoundaryConditions: permute_boundary_conditions,
                                       fill_halo_event!, get_boundary_kernels,
                                       DistributedCommunication

using Oceananigans.DistributedComputations: cooperative_waitall!,
                                            recv_from_buffers!,
                                            fill_corners!,
                                            loc_id,
                                            DCBCT

using Oceananigans.Fields: location

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!

@inline instantiate(T::DataType) = T()
@inline instantiate(T) = T

const DistributedZipper = BoundaryCondition{<:DistributedCommunication, <:ZipperHaloCommunicationRanks}

switch_north_halos!(c, north_bc, grid, loc) = nothing

function switch_north_halos!(c, north_bc::DistributedZipper, grid, loc)
    sign  = north_bc.condition.sign
    hz = halo_size(grid)
    sz = size(grid)

    _switch_north_halos!(parent(c), loc, sign, sz, hz)

    return nothing
end

@inline reversed_halos(::Tuple{<:Any, <:Center, <:Any}, Ny, Hy) = Ny+2Hy-1:-1:Ny+Hy+1
@inline reversed_halos(::Tuple{<:Any, <:Face,   <:Any}, Ny, Hy) = Ny+2Hy:-1:Ny+Hy+2

@inline adjust_x_face!(c, loc, north_halos, Px) = nothing
@inline adjust_x_face!(c, ::Tuple{<:Face, <:Any, <:Any}, north_halos, Px) = view(c, 2:Px, north_halos, :) .= view(c, 1:Px-1, north_halos, :)

# We throw away the first point!
@inline function _switch_north_halos!(c, loc, sign, (Nx, Ny, Nz), (Hx, Hy, Hz))

    # Domain indices common for all locations
    north_halos = Ny+Hy+1:Ny+2Hy-1
    west_corner = 1:Hx
    east_corner = Nx+Hx+1:Nx+2Hx
    interior    = Hx+1:Nx+Hx

    # Location - dependent halo indices
    reversed_north_halos = reversed_halos(loc, Ny, Hy)

    view(c, west_corner, north_halos, :) .= sign .* reverse(view(c, west_corner, reversed_north_halos, :), dims = 1)
    view(c, east_corner, north_halos, :) .= sign .* reverse(view(c, east_corner, reversed_north_halos, :), dims = 1)
    view(c, interior,    north_halos, :) .= sign .* reverse(view(c, interior,    reversed_north_halos, :), dims = 1)

    # throw out first point for the x - face locations
    adjust_x_face!(c, loc, north_halos, size(c, 1))

    return nothing
end

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DistributedTripolarGridOfSomeKind, buffers, args...;
                            only_local_halos=false, fill_open_bcs=true, kwargs...)

    north_bc = bcs.north

    arch = architecture(grid)
    kernels!, ordered_bcs = get_boundary_kernels(bcs, c, grid, loc, indices)

    number_of_tasks = length(kernels!)
    outstanding_requests = length(arch.mpi_requests)

    for task = 1:number_of_tasks
        @inbounds fill_halo_event!(c, kernels![task], ordered_bcs[task], loc, grid, buffers, args...; kwargs...)
    end

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; only_local_halos, kwargs...)

    # We increment the request counter only if we have actually initiated the MPI communication.
    # This is the case only if at least one of the boundary conditions is a distributed communication
    # boundary condition (DCBCT) _and_ the `only_local_halos` keyword argument is false.
    if length(arch.mpi_requests) > outstanding_requests
        arch.mpi_tag[] += 1
    end

    switch_north_halos!(c, north_bc, grid, loc)

    return nothing
end

function synchronize_communication!(field::Field{<:Any, <:Any, <:Any, <:Any, <:DistributedTripolarGridOfSomeKind})
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests)
        cooperative_waitall!(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[] = 0

        # Reset MPI requests
        empty!(arch.mpi_requests)
    end

    recv_from_buffers!(field.data, field.communication_buffers, field.grid)

    north_bc = field.boundary_conditions.north
    instantiated_location = map(instantiate, location(field))

    switch_north_halos!(field, north_bc, field.grid, instantiated_location)

    return nothing
end
