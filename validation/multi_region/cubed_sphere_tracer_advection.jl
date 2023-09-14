using Oceananigans

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.MultiRegion: getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_velocity_halos!
using Oceananigans.Operators: Δyᶜᶠᶜ, Δxᶠᶜᶜ, ∂yᶠᶜᶜ, ∂xᶜᶠᶜ

using GLMakie
GLMakie.activate!()

Nλ = 180
Nφ = 180
Nz = 1
radius = R = 1
U = 1

grid = ConformalCubedSphereGrid(; panel_size = (Nλ, Nφ, Nz),
                                  z = (-1, 0),
                                  radius, 
                                  horizontal_direction_halo = 4,
                                  partition = CubedSpherePartition(; R = 1))

u = XFaceField(grid)
v = YFaceField(grid)

# Solid body rotation
φʳ = π/2      # Latitude pierced by the axis of rotation
α  = π/2 - φʳ # Angle between axis of rotation and north pole
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

Hx, Hy, Hz = halo_size(grid)

#ψ = Field{Face, Face, Center}(grid)
ψ = Field{Center, Center, Center}(grid)

# TODO: put this in set! eg
#
# function set!(ψ::CubedSphereField, f::Function)
# 
# return nothing
#
for region in 1:6

    region_grid = getregion(grid, region)

    i₁ = -Hx + 1
    i₂ = region_grid.Nx + Hx
    j₁ = -Hy + 1
    j₂ = region_grid.Ny + Hy
    k₁ = -Hz + 1
    k₂ = region_grid.Nz + Hz

    for k in k₁:k₂, j=j₁:j₂, i=i₁:i₂
        λ = λnode(i, j, k, region_grid, Face(), Face(), Center())
        φ = φnode(i, j, k, region_grid, Face(), Face(), Center())
        getregion(ψ, region)[i, j, k] = ψᵣ(λ, φ, 0)
    end
end

u = Field{Face,   Center, Center}(grid)
v = Field{Center, Face,   Center}(grid)

for region in 1:6

    region_grid = getregion(grid, region)

    i₁ = -Hx + 2
    i₂ = region_grid.Nx + Hx - 1  
    j₁ = -Hy + 2
    j₂ = region_grid.Ny + Hy - 1
    k₁ = -Hz + 2
    k₂ = region_grid.Nz + Hz - 1

    region_ψ = getregion(ψ, region)
    region_u = getregion(u, region)
    region_v = getregion(v, region)

    for k in k₁:k₂, j=j₁:j₂, i=i₁:i₂
        # region_u[i, j, k] = - (region_ψ[i, j+1, k] - region_ψ[i, j, k]) / Δyᶜᶠᶜ(i, j, 1, region_grid)
        # region_v[i, j, k] =   (region_ψ[i+1, j, k] - region_ψ[i, j, k]) / Δxᶠᶜᶜ(i, j, 1, region_grid)
        
        region_u[i, j, k] = - ∂yᶠᶜᶜ(i, j, 1, region_grid, region_ψ)
        region_v[i, j, k] = + ∂xᶜᶠᶜ(i, j, 1, region_grid, region_ψ)
    end
end

#=
for _ in 1:2
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_velocity_halos!((; u = u, v = v, w = nothing), grid)
end
=#

velocities = PrescribedVelocityFields(; u, v)

model = HydrostaticFreeSurfaceModel(; grid,
                                      velocities,
                                      momentum_advection = nothing,
                                      tracer_advection = WENO(),
                                      tracers = :θ,
                                      buoyancy = nothing)

# Initial condition for tracer: 4 Gaussians

i₁ = 3Nλ÷4+1
j₁ = Nφ÷4+1
panel = 1
λ₁ = λnode(i₁, j₁, getregion(grid, panel), Center(), Center())
φ₁ = φnode(i₁, j₁, getregion(grid, panel), Center(), Center())

i₂ = Nλ÷4+1
j₂ = 3Nφ÷4+1
panel = 4
λ₂ = λnode(i₂, j₂, getregion(grid, panel), Center(), Center())
φ₂ = φnode(i₂, j₂, getregion(grid, panel), Center(), Center())

i₃ = 3Nλ÷4+1
j₃ = 3Nφ÷4+1
panel = 3
λ₃ = λnode(i₃, j₃, getregion(grid, panel), Center(), Center())
φ₃ = φnode(i₃, j₃, getregion(grid, panel), Center(), Center())

i₄ = 3Nλ÷4+1
j₄ = 3Nφ÷4+1
panel = 6
λ₄ = λnode(i₄, j₄, getregion(grid, panel), Center(), Center())
φ₄ = φnode(i₄, j₄, getregion(grid, panel), Center(), Center())

δR = 2
θ₀ = 1

θᵢ(λ, φ, z) =  θ₀ * exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2δR^2) +
               θ₀ * exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2δR^2) +
               θ₀ * exp(-((λ - λ₃)^2 + (φ - φ₃)^2) / 2δR^2) + 
               θ₀ * exp(-((λ - λ₄)^2 + (φ - φ₄)^2) / 2δR^2)

set!(model, θ = θᵢ)

θ = model.tracers.θ
fill_halo_regions!(θ)

Δt = 0.0015
stop_iteration = 8000

simulation = Simulation(model; Δt, stop_iteration)

# Print a progress message
using Printf

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))


tracer_fields = Field[]

function save_tracer(sim)
    push!(tracer_fields, deepcopy(sim.model.tracers.θ))
end

simulation.callbacks[:save_tracer] = Callback(save_tracer, IterationInterval(20))

run!(simulation)


@info "Making an animation from the saved data..."

n = Observable(1)

Θₙ = []
for region in 1:6
    push!(Θₙ, @lift parent(getregion(tracer_fields[$n], region)[:, :, grid.Nz]))
end

function where_to_plot(region)
    region == 1 && return (3, 1)
    region == 2 && return (3, 2)
    region == 3 && return (2, 2)
    region == 4 && return (2, 3)
    region == 5 && return (1, 3)
    region == 6 && return (1, 4)
end

function heatlatlon!(ax::Axis, field, k=1; kwargs...)

    LX, LY, LZ = location(field)

    grid = field.grid
    _, (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(grid, LX(), LY(), LZ())

    quad_points = vcat([Point2.(λvertices[:, i, j], φvertices[:, i, j]) 
                        for i in axes(λvertices, 2), j in axes(λvertices, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1  j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)

    colors_per_point = vcat(fill.(vec(interior(field, :, :, k)), 4)...)

    mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = false, kwargs...)

    xlims!(ax, (-180, 180))
    ylims!(ax, (-90, 90))

    return ax
end

heatlatlon!(ax::Axis, field::CubedSphereField, k=1; kwargs...) = apply_regionally!(heatlatlon!, ax, field, k; kwargs...)
heatlatlon!(ax::Axis, field::Observable{<:CubedSphereField}, k=1; kwargs...) = apply_regionally!(heatlatlon!, ax, field.val, k; kwargs...)

#=
using GeoMakie
using Oceananigans.Utils: Iterate, get_lat_lon_nodes_and_vertices, get_cartesian_nodes_and_vertices, apply_regionally!

n = Observable(1)

Θₙ = @lift tracer_fields[$n]

fig = Figure(resolution = (1600, 1200), fontsize=30)
ax = GeoAxis(fig[1, 1], coastlines = true, lonlims = automatic)
heatlatlon!(ax, Θₙ, colorrange=(0, 0.5θ₀))

fig

frames = 1:length(tracer_fields)

GLMakie.record(fig, "multi_region_tracer_advection_latlon.mp4", frames, framerate = 48) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    Θₙ[] = tracer_fields[i]
    heatlatlon!(ax, Θₙ, colorrange=(0, 0.5θ₀))
end
=#
