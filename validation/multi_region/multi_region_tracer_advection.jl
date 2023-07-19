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

function multi_region_tracer_advection!(Nx, Ny, Nt, tracer_fields)

    grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz = 1), z = (-1, 0), radius=1, horizontal_direction_halo = 1, 
                                    partition = CubedSpherePartition(; R = 1))
    
    facing_panel_index = 5 
    # The tracer is initially placed on the equator at the center of panel 5, which which is oriented towards the 
    # viewer in a heatsphere plot. This optimal positioning allows the viewer to effectively track the tracer's initial 
    # movement as it starts advecting along the equator.
    
    prescribed_velocity_type = :solid_body_rotation 
    # Choose prescribed_velocity_type to be :zonal or :solid_body_rotation.
    
    if prescribed_velocity_type == :zonal
        u_by_region(region,grid) = region == 1 || region == 2 ? OneField() : ZeroField()
        v_by_region(region,grid) = region == 4 || region == 5 ? OneField() : ZeroField()
        @apply_regionally u0 = u_by_region(Iterate(1:6), grid)
        @apply_regionally v0 = v_by_region(Iterate(1:6), grid)
        velocities = PrescribedVelocityFields(; u = u0, v = u0)
    elseif prescribed_velocity_type == :solid_body_rotation
        u_solid_body_rotation(λ,φ,z) = cosd(φ)
        u0 = XFaceField(grid) 
        set!(u0, u_solid_body_rotation)
        velocities = PrescribedVelocityFields(; v = u0) 
        # Note that v = u0 (and not u = u0) above. This is because the local meridional velocity in panel 5, on which 
        # the tracer is initialized, is oriented along the global zonal direction.
    end

    model = HydrostaticFreeSurfaceModel(; grid, velocities, tracers = :θ, buoyancy = nothing)
    
    θ₀ = 1
    x₀ = λnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Face(), Center())
    y₀ = φnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Center(), Face())
    R₀ = 10.0
    
    initial_condition = :Gaussian # Choose initial_condition to be :uniform_patch or :Gaussian.
    
    if initial_condition == :uniform_patch
        θᵢ(x, y, z) = abs(x - x₀) < R₀ && abs(y - y₀) < R₀ ? θ₀ : 0.0
    elseif initial_condition == :Gaussian
        θᵢ(x, y, z) = θ₀*exp(-((x - x₀)^2 + (y - y₀)^2)/(R₀^2))
    end
    
    set!(model, θ = θᵢ)
    fill_halo_regions!(model.tracers.θ)
    
    Δt = 0.005
    T = Nt * Δt
    
    simulation = Simulation(model, Δt=Δt, stop_time=T)

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

plot_type = :heatsphere # Choose plot_type to be :heatsphere or :heatlatlon.

if plot_type == :heatsphere
    ax = Axis3(fig[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = title, titlesize = 27.5, titlegap = 15, 
               titlefont = :bold, aspect = (1,1,1))
    heatsphere!(ax, tracer_fields[1]; colorrange = (-1, 1))
end
if plot_type == :heatlatlon
    ax = Axis(fig[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = title, titlesize = 27.5, titlegap = 15, 
              titlefont = :bold)
    heatlatlon!(ax, tracer_fields[1]; colorrange = (-1, 1))
end

frames = 1:length(tracer_fields)

GLMakie.record(fig, "multi_region_tracer_advection.mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    if plot_type == :heatsphere
        heatsphere!(ax, tracer_fields[i]; colorrange = (-1, 1), colormap = :balance)
    elseif plot_type == :heatlatlon
        heatlatlon!(ax, tracer_fields[i]; colorrange = (-1, 1), colormap = :balance)
    end
end