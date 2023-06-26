using Oceananigans
using Oceananigans.MultiRegion: getregion
using Oceananigans.Utils: Iterate,
                          get_lat_lon_nodes_and_vertices,
                          get_cartesian_nodes_and_vertices
using Oceananigans.BoundaryConditions: fill_halo_regions!


function multi_region_tracer_advection!(Nx, Ny, Nt, tracer_field)

    grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz = 1), z = (-1, 0), radius=1, horizontal_direction_halo = 1, 
                                    partition = CubedSpherePartition(; R = 1))
    
    u0 = Oceananigans.Fields.ZeroField()
    v0 = Oceananigans.Fields.ZeroField()
    
    velocities = PrescribedVelocityFields(; u = u0, v = v0)

    model = HydrostaticFreeSurfaceModel(; grid, velocities, tracers = :θ, buoyancy = nothing)

    θᵢ(x, y, z) = exp(-x^2/100 - y^2/100)
    set!(model, θ = θᵢ)

    fill_halo_regions!(model.tracers.θ)
    
    Δt = 1.0
    T = Nt * Δt
    
    simulation = Simulation(model, Δt=Δt, stop_time=T)
    
    # Figure out OutputWriters for a CubedSphere (`reconstruct_global_grid` is not defined for a CubedSphere neither 
    # does it make sense to define it for a CubedSphere). Then we can use the following lines of code:
    # 
    # simulation.output_writers[:fields] = JLD2OutputWriter(model, model.tracers, schedule = TimeInterval(0.02),
    #                                                       filename = "tracer_advection_over_bump")
    #                                                                             
    # simulation.output_writers[:fields] = Checkpointer(model, schedule = TimeInterval(1.0), 
    #                                                   prefix = "multi_region_tracer_advection")

    function save_tracer(sim)
        push!(tracer_field, sim.model.tracers.θ)
    end

    simulation.callbacks[:save_tracer] = Callback(save_tracer, TimeInterval(1.0))

    run!(simulation)
    
    return nothing
end 


Nx = 10
Ny = 10
Nt = 10

tracer_fields = Field[]
multi_region_tracer_advection!(Nx, Ny, Nt, tracer_fields)
tracer = tracer_fields[1]

@info "Making an animation from the saved data..."

n = Observable(1)

θGrid = @lift begin
    interior(tracer_fields[$n], :, :, 1)
end

fig = Figure(resolution = (850, 750))
title_θ = @lift "Tracer Concentration after " *string(round(times[$n], digits = 1))*" seconds"

@apply_regionally heatsphere!(ax, θGrid)

ax_θ = Axis3(fig[1,1]; xlabel = "Longitude (m)", ylabel = "Meridional Distance (m)", xlabelsize = 22.5, ylabelsize = 22.5, 
             xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1.0, 
             title = title_θ, titlesize = 27.5, titlegap = 15, titlefont = :bold)

@apply_regionally heatsphere!(ax_θ, model.tracers.θ)

Colorbar(fig[1,2], hm_θ; label = "Tracer Concentration", labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

frames = 1:length(times)

CairoMakie.record(fig, filename * "_" * immersed_boundary_type * "_" * plot_type * ".mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end