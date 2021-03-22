using Oceananigans 
using Oceananigans.Models
using Profile
using Oceananigans.Grids: halo_size

Nx = 8

topologies = (  (Periodic, Periodic, Bounded), (Periodic, Periodic, Flat),
                (Periodic, Flat,     Flat),    (Flat,     Flat,     Flat))
sizes = ( (Nx,Nx,1), (Nx, Nx), (Nx), () )
extents = ( (1, 1, 1), (1, 1), (1), ()) 

for (iter, topo) in enumerate(topologies)

    grid = RegularRectilinearGrid(size=sizes[iter], extent=extents[iter], topology=topo)

    # Warning: model.grid will have correct halos but grid will not necessary
    # Do we show a warning or something else?

    model = ShallowWaterModel(architecture=CPU(), grid=grid, advection=WENO5(), gravitational_acceleration=1)

    set!(model,h=1)

    simulation = Simulation(model, Δt = 1e-3, stop_time = 1e-2)

    #run!(simulation)
    
    print("Finished iter = ", iter, "\n\n")
end

