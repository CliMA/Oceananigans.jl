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

Nx = 20
Ny = 20
Nt = 2250
cubed_sphere_radius = 1

grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz = 1), z = (-1, 0), radius=CubedSphereRadius, 
                                horizontal_direction_halo = 1, partition = CubedSpherePartition(; R = 1))

facing_panel_index = 5 
# The tracer is initially placed on the equator at the center of panel 5, which which is oriented towards the 
# viewer in a heatsphere plot. This optimal positioning allows the viewer to effectively track the tracer's initial 
# movement as it starts advecting along the equator.

prescribed_velocity_type = :solid_body_rotation 
# Choose prescribed_velocity_type to be :zonal or :solid_body_rotation.

if prescribed_velocity_type == :zonal

    u_by_region(region,grid) = region == 1 || region == 2 ? OneField() : ZeroField()
    v_by_region(region,grid) = region == 4 || region == 5 ? OneField() : ZeroField()
    
elseif prescribed_velocity_type == :solid_body_rotation
    
    
    # First implementation of solid body rotation:
    
    # Ψ(λ, φ, z) = R * u_advection * cosd(λ) * cosd(φ)
    # U(λ, φ, z) = - 1/R          * ∂Ψ/∂φ =   u_advection * cosd(λ) * sind(φ)
    # V(λ, φ, z) = + 1/(R cos(φ)) * ∂Ψ/∂λ = - u_advection * sind(λ)

    time_period = 10
    u_advection = (2π * CubedSphereRadius) / time_period

    U(λ, φ, z) =   u_advection * cosd(λ) * sind(φ)
    V(λ, φ, z) = - u_advection * sind(λ)
    
    u₀ = XFaceField(grid) 
    set!(u₀, U)
    
    v₀ = YFaceField(grid) 
    set!(v₀, V)
    
    for _ in 1:2
        fill_halo_regions!(u₀)
        fill_halo_regions!(v₀)
        @apply_regionally replace_horizontal_velocity_halos!((; u = u₀, v = v₀, w = nothing), grid)
    end

    
    # α = 90
    
    # Ψ(λ, φ, z) = - CubedSphereRadius * u_advection * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))
    
    # Ψ₀ = CubedSphereRadius * u_advection
    
    # Ψᶠᶠᶜ = Field{Face, Face, Center}(grid)
    # uᶠᶜᶜ = Field{Face, Center, Center}(grid)
    # vᶜᶠᶜ = Field{Center, Face, Center}(grid)
    
    # for region in 1:6
    
    #     for i in 1:Nx+1, j in 1:Ny+1
    #         λᶠᶠᵃ = λnode(i, j, getregion(grid, region), Face(), Face())
    #         φᶠᶠᵃ = φnode(i, j, getregion(grid, region), Face(), Face())
    #         getregion(Ψᶠᶠᶜ, region).data[i, j, 1] = Ψ(λᶠᶠᵃ, φᶠᶠᵃ, 0)
    #     end
        
    # end

    # for region in 1:6
    
    #     for i in 1:Nx+1, j in 1:Ny
    #         getregion(uᶠᶜᶜ, region).data[i, j, 1] = (
    #         (getregion(Ψᶠᶠᶜ, region).data[i, j, 1] 
    #          - getregion(Ψᶠᶠᶜ, region).data[i, j+1, 1]) / getregion(grid, region).Δyᶠᶜᵃ[i, j])
    #     end
        
    #     for i in 1:Nx, j in 1:Ny+1
    #         getregion(vᶜᶠᶜ, region).data[i, j, 1] = (
    #         (getregion(Ψᶠᶠᶜ, region).data[i+1, j, 1] 
    #          - getregion(Ψᶠᶠᶜ, region).data[i, j, 1]) / getregion(grid, region).Δxᶜᶠᵃ[i, j])
    #     end
        
    # end
    
    # u_by_region(region,grid) = getregion(uᶠᶜᶜ, region)
    # v_by_region(region,grid) = getregion(vᶜᶠᶜ, region)
end

# @apply_regionally u₀ = u_by_region(Iterate(1:6), grid)
# @apply_regionally v₀ = v_by_region(Iterate(1:6), grid)

Ψ₀ = CubedSphereRadius * u_advection

velocities = PrescribedVelocityFields(; u = u₀, v = v₀)

@info "Plotting prescribed velocities..."

plot_type = :heatlatlon # Choose plot_type to be :heatsphere or :heatlatlon.

fig_u = Figure(resolution = (850, 750))
fig_v = Figure(resolution = (850, 750))

if plot_type == :heatsphere
    ax_u = Axis3(fig_u[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = "Prescribed Zonal Velocity", 
    titlesize = 27.5, titlegap = 15, titlefont = :bold, aspect = (1,1,1))
    ax_v = Axis3(fig_v[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = "Prescribed Meridional Velocity", 
    titlesize = 27.5, titlegap = 15, titlefont = :bold, aspect = (1,1,1))
    heat_u = heatsphere!(ax_u, u₀; colorrange = (-Ψ₀, Ψ₀))
    heat_v = heatsphere!(ax_v, v₀; colorrange = (-Ψ₀, Ψ₀))
elseif plot_type == :heatlatlon
    ax_u = Axis(fig_u[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = "Prescribed Zonal Velocity", 
                titlesize = 27.5, titlegap = 15, titlefont = :bold)
    ax_v = Axis(fig_v[1,1]; xticklabelsize = 17.5, yticklabelsize = 17.5, title = "Prescribed Meridional Velocity", 
                titlesize = 27.5, titlegap = 15, titlefont = :bold)
    heat_u = heatlatlon!(ax_u, u₀; colorrange = (-Ψ₀, Ψ₀))
    heat_v = heatlatlon!(ax_v, v₀; colorrange = (-Ψ₀, Ψ₀))
end

save("prescribed_zonal_velocity.png", fig_u)
save("prescribed_meridional_velocity.png", fig_v)

model = HydrostaticFreeSurfaceModel(; grid, velocities, tracers = :θ, buoyancy = nothing)

θ₀ = 1
x₀ = λnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Face(), Center())
y₀ = φnode(Nx÷2+1, Ny÷2+1, getregion(grid, facing_panel_index), Center(), Face())
R₀ = 10

initial_condition = :Gaussian # Choose initial_condition to be :uniform_patch or :Gaussian.

θᵢ(x, y, z) = if initial_condition == :uniform_patch
    abs(x - x₀) < R₀ && abs(y - y₀) < R₀ ? θ₀ : 0.0
elseif initial_condition == :Gaussian
    θ₀*exp(-((x - x₀)^2 + (y - y₀)^2)/(R₀^2))
end

θᵢ_field = CenterField(grid)
getregion(θᵢ_field, 1).data[Nx-5:Nx, Int.(round(Ny/4)):Int.(round(3Ny/4)), 1] .= 1

set!(model, θ = θᵢ_field)
fill_halo_regions!(model.tracers.θ)

Δt = 0.005
T = Nt * Δt

simulation = Simulation(model, Δt=Δt, stop_time=T)

tracer_fields = Field[]

function save_tracer(sim)
    push!(tracer_fields, deepcopy(sim.model.tracers.θ))
end

simulation.callbacks[:save_tracer] = Callback(save_tracer, TimeInterval(10Δt))

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

GLMakie.record(fig, "multi_region_tracer_advection.mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    if plot_type == :heatsphere
        heatsphere!(ax, tracer_fields[i]; colorrange = (-1, 1), colormap = :balance)
    elseif plot_type == :heatlatlon
        heatlatlon!(ax, tracer_fields[i]; colorrange = (-1, 1), colormap = :balance)
    end
end
