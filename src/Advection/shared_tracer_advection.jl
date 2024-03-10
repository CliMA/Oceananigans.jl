using KernelAbstractions: @index, @kernel

function tracer_advection(Gⁿ, tracers, grid, tracer_advection, velocities; parameters = :xyz)

    return nothing
end

@inline retrieve_x_advection(tracer_advection) = tracer_advection
@inline retrieve_y_advection(tracer_advection) = tracer_advection
@inline retrieve_z_advection(tracer_advection) = tracer_advection

@inline retrieve_x_advection(tracer_advection::TracerAdvection) = tracer_advection.x
@inline retrieve_y_advection(tracer_advection::TracerAdvection) = tracer_advection.y
@inline retrieve_z_advection(tracer_advection::TracerAdvection) = tracer_advection.z

@kernel function tracer_advection_x(Gⁿ, tracers, grid, tracer_advection, U, ::Val{Ntracers}) where Ntracers
    global_id = @index(Global, Linear)
    block_id  = 
    thread_id = @index(Local)

    k_id = 

    tracer_flux = @localmem FT (, )

    @unroll for t in 1:Ntracers
        tracer_name = @inbounds tracer_names[t]
        tracer    = @inbounds tracers[t]
        advection = @inbounds tracer_advection[tracer_name]
        tendency  = @inbounds Gⁿ[tracer_name]
        advection = retrieve_x_advection(advection)

        # calculate fluxes into the shared memory
        tracer_flux[] = _advective_tracer_flux_x(i, j, k, grid, advection, U, c)

        @synchronize
        # put div in memory
        tendency[i, j, k] = (tracer_flux[t] - tracer_flux[t-1]) / Vᶜᶜᶜ(i, j, k, grid)
    end
end

@kernel function tracer_advection_y(Gⁿ, tracers, grid, tracer_advection, V)
    i, j, k = @index(Global, NTuple)
   
    # Allocate shared memory of the dimensions of the grid


    for (c, advection) in zip(tracers, tracer_advection)
        # calculate fluxes into the shared memory

        __syncthreads()
        # put div in memory

        # exit
    end
end

@kernel function tracer_advection_z(Gⁿ, tracers, grid::ActiveCellsIBG, tracer_advection, W)
    i, j, k = @index(Global, NTuple)

    # Allocate shared memory of the dimensions of the grid

    for (c, advection) in zip(tracers, tracer_advection)
        # calculate fluxes into the shared memory

        __syncthreads()
        # put div in memory

        # exit
    end
end