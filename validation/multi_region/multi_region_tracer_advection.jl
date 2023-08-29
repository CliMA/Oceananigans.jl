using Oceananigans
using Oceananigans.Grids: φnode, λnode
using Oceananigans.MultiRegion: getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_velocity_halos!
using GeoMakie, GLMakie
GLMakie.activate!()

include("multi_region_cubed_sphere.jl")

Nx, Ny, Nz, Nt = 50, 50, 1, 150

cubed_sphere_radius = 1
grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz = 1), z = (-1, 0), radius=cubed_sphere_radius, 
                                horizontal_direction_halo = 1, partition = CubedSpherePartition(; R = 1))

time_period = 25
u_advection = (2π * cubed_sphere_radius) / time_period
                                
U(λ, φ, z) = u_advection * cosd(λ) * sind(φ)
V(λ, φ, z) = - u_advection * sind(λ)
    
u = XFaceField(grid) 
v = YFaceField(grid) 

set_velocities_panelwise = true

if set_velocities_panelwise

    set!(getregion(u, 1), U)
    set!(getregion(u, 2), U)
    set!(getregion(u, 3), U)
    set!(getregion(u, 4), V)
    set!(getregion(u, 5), V)
    set!(getregion(u, 6), V)

    set!(getregion(v, 1), V)
    set!(getregion(v, 2), V)
    set!(getregion(v, 3), V)
    set!(getregion(v, 4), U)
    set!(getregion(v, 5), U)
    set!(getregion(v, 6), U)
    
else

    set!(u, U)
    set!(v, V)
    
end

for _ in 1:2
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_velocity_halos!((; u = u, v = v, w = nothing), grid)
end

velocities = PrescribedVelocityFields(; u = u, v = v)

model = HydrostaticFreeSurfaceModel(; grid, velocities, tracers = :θ, buoyancy = nothing)

facing_panel_index = 1
θ₀ = 1
λ₀ = λnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Face(), Center())
φ₀ = φnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Center(), Face())
R₀ = 10

initial_condition = :Gaussian # Choose initial_condition to be :uniform_patch or :Gaussian.

θᵢ(λ, φ, z) = θ₀*exp(-((λ - λ₀)^2 + (φ - φ₀)^2)/(R₀^2))

set!(model, θ = θᵢ)

fill_halo_regions!(model.tracers.θ)

Δt = 0.025
T = Nt * Δt

simulation = Simulation(model, Δt=Δt, stop_time=T)

tracer_fields = Field[]

function save_tracer(sim)
    push!(tracer_fields, deepcopy(sim.model.tracers.θ))
end

simulation.callbacks[:save_tracer] = Callback(save_tracer, TimeInterval(Δt))

run!(simulation)

@info "Making an animation from the saved data..."

plot_type = :heatlatlon # Choose plot_type to be :heatsphere or :heatlatlon.

fig = Figure(resolution = (850, 750))
title = "Tracer Concentration"

if plot_type == :heatsphere
    ax = Axis3(fig[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = title, titlesize = 27.5, titlegap = 15, 
               titlefont = :bold, aspect = (1,1,1))
    heatsphere!(ax, tracer_fields[1]; colorrange = (-1, 1))
elseif plot_type == :heatlatlon
    ax = Axis(fig[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = title, titlesize = 27.5, titlegap = 15, 
              titlefont = :bold)
    heatlatlon!(ax, tracer_fields[1]; colorrange = (-1, 1))
end

frames = 1:length(tracer_fields)

GLMakie.record(fig, "multi_region_tracer_advection.mp4", frames, framerate = 1) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    if plot_type == :heatsphere
        heatsphere!(ax, tracer_fields[i]; colorrange = (-1, 1), colormap = :balance)
    elseif plot_type == :heatlatlon
        heatlatlon!(ax, tracer_fields[i]; colorrange = (-1, 1), colormap = :balance)
    end
end