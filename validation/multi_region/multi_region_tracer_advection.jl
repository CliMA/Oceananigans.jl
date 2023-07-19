using Oceananigans
using Oceananigans.MultiRegion: getregion
using Oceananigans.Utils: Iterate,
                          get_lat_lon_nodes_and_vertices,
                          get_cartesian_nodes_and_vertices
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, OneField
using Oceananigans.Grids: λnode, φnode

using GeoMakie, GLMakie
GLMakie.activate!()

include("multi_region_cubed_sphere.jl")

u_by_region(region,grid) = region == 1 || region == 2 ? OneField() : ZeroField()
v_by_region(region,grid) = region == 4 || region == 5 ? OneField() : ZeroField()

function multi_region_tracer_advection!(Nx, Ny, Nt, tracer_fields)

    grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz = 1), z = (-1, 0), radius=1, horizontal_direction_halo = 1, 
                                    partition = CubedSpherePartition(; R = 1))

    @apply_regionally u0 = u_by_region(Iterate(1:6), grid)
    @apply_regionally v0 = v_by_region(Iterate(1:6), grid)
    
    velocities = PrescribedVelocityFields(; u = u0, v = v0)

    model = HydrostaticFreeSurfaceModel(; grid, velocities, tracers = :θ, buoyancy = nothing)
    
    facing_panel_index = 5
    θ₀ = 1
    x₀ = λnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Face(), Center())
    y₀ = φnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Center(), Face())
    R₀ = 10.0
    θᵢ(x, y, z) = 0.0
    
    initial_condition = :uniform_patch # Choose initial_condition to be :uniform_patch or :Gaussian.
    
    if initial_condition == :uniform_patch
        θᵢ(x, y, z) = abs(x - x₀) < R₀ && abs(y - y₀) < R₀ ? θ₀ : 0.0
    elseif initial_condition == :Gaussian
        θᵢ(x, y, z) = θ₀*exp(-((x - x₀)^2 + (y - y₀)^2)/(R₀^2))
    end
    
    set!(model, θ = θᵢ)
    fill_halo_regions!(model.tracers.θ)
    
    Δt = 0.01
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
        push!(tracer_fields, deepcopy(sim.model.tracers.θ))
    end

    simulation.callbacks[:save_tracer] = Callback(save_tracer, TimeInterval(10Δt))

    run!(simulation)
    
    return nothing
    
end 

Nx = 50
Ny = 50
Nt = 1000

tracer_fields = Field[]
multi_region_tracer_advection!(Nx, Ny, Nt, tracer_fields)

@info "Making an animation from the saved data..."

fig = Figure(resolution = (850, 750))
title = "Tracer Concentration"

ax = Axis3(fig[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = title, titlesize = 27.5, titlegap = 15, 
           titlefont = :bold)

heatsphere!(ax, tracer_fields[1]; colorrange = (0, 1))

frames = 1:length(tracer_fields)

GLMakie.record(fig, "multi_region_tracer_advection.mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    heatsphere!(ax, tracer_fields[i]; colorrange = (0, 1))
end