

function tracer_advection(Gⁿ, tracers, grid, tracer_advection, velocities; parameters = :xyz)

    return nothing
end

@kernel function tracer_advection_x(Gⁿ, tracers, grid, tracer_advection, U)
    i, j, k = @index(Global, NTuple)
   
    # Allocate shared memory of the dimensions of the grid

    for (c, advection) in zip(tracers, tracer_advection)
        # calculate fluxes into the shared memory

        __syncthreads()
        # put div in memory

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