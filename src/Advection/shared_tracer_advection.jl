using KernelAbstractions: @index, @kernel

function tracer_advection(Gⁿ, tracers, grid, tracer_advection, velocities; parameters = :xyz)

    return nothing
end

@kernel function tracer_advection_x(Gⁿ, tracers, grid, tracer_advection, U)
    i, j, k = @index(Global, NTuple)
   
    glo_id = @index(Global)
    loc_id = @index(Local)

    tracer_flux = @localmem FT (N, grp_size)

    @unroll for tracer_name in tracer_names
        advection = retrieve_x_advection(tracer_advection, tracer_name)

        # calculate fluxes into the shared memory
        tracer_flux[] = _advective_tracer_flux_x(i, j, k, grid, advection, U, c)

        @synchronize
        # put div in memory
        Gⁿ[i, j, k] = (tracer_flux[t] - tracer_flux[t-1]) / Vᶜᶜᶜ(i, j, k, grid)
        # exit
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