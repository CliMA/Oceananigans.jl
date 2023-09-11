using Oceananigans

using Oceananigans.Grids: φnode, λnode
using Oceananigans.MultiRegion: getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_velocity_halos!

using GLMakie
GLMakie.activate!()

include("multi_region_cubed_sphere.jl")

Nx, Ny, Nz = 64, 64, 1

radius = 1
grid = ConformalCubedSphereGrid(; panel_size=(Nx, Ny, Nz),
                                  z = (-1, 0),
                                  radius, 
                                  horizontal_direction_halo = 4,
                                  partition = CubedSpherePartition(; R = 1))
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

u = XFaceField(grid)
v = YFaceField(grid)

R = getregion(grid, 1).radius # radius of the sphere (m)

α  = 0
u₀ = 1

ψ(λ, φ, z) = - R * u₀ * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

Ψ = Field{Face, Face, Center}(grid)
set!(Ψ, ψ)

#=
# Fill halo nodes manually.

for region in 1:6
    for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
        if (i >= 1-Hx & i <= 0) | (i >= Nx+1 & i <= Nx+Hx) | (j >= 1-Hy & j <= 0) | (j >= Ny+1 & j <= Ny+Hy)
            λAtNode = λnodes(getregion(grid, region), Face(), Face(), Center(); with_halos=true)[i, j, 1]
            φAtNode = φnodes(getregion(grid, region), Face(), Face(), Center(); with_halos=true)[i, j, 1]
            getregion(Ψ, region).data[i, j, 1] = ψ(λAtNode, φAtNode, -0.5)
        end
    end
end
=#

#=
Fill halo nodes using the fill_halo_regions! function.

fill_halo_regions!(Ψ)
=#

# Fill interior and halo nodes manually.

for region in 1:6
    for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
        λAtNode = λnodes(getregion(grid, region), Face(), Face(), Center(); with_halos=true)[i, j, 1]
        φAtNode = φnodes(getregion(grid, region), Face(), Face(), Center(); with_halos=true)[i, j, 1]
        getregion(Ψ, region).data[i, j, 1] = ψ(λAtNode, φAtNode, -0.5)
    end
end

u = compute!(Field(-∂y(Ψ)))
v = compute!(Field(+∂x(Ψ)))

for _ in 1:2
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    @apply_regionally replace_horizontal_velocity_halos!((; u = u, v = v, w = nothing), grid)
end

fig = Figure(resolution = (2200, 1200), fontsize=25)

# Plot streamfunction and velocities with halos.

for region in 1:6
    ax = Axis(fig[1, region], title="Ψ panel $region")
    heatmap!(ax, parent(getregion(Ψ, region)[:, :, 1]), colorrange = (-R * u₀, R * u₀), colormap = :balance)
    ax = Axis(fig[2, region], title="u panel $region")
    heatmap!(ax, parent(getregion(u, region)[:, :, 1]), colorrange = (-R * u₀, R * u₀), colormap = :balance)
    ax = Axis(fig[3, region], title="v panel $region")
    heatmap!(ax, parent(getregion(v, region)[:, :, 1]), colorrange = (-R * u₀, R * u₀), colormap = :balance)
end

#=
Plot streamfunction and velocities without halos.

for region in 1:6
    ax = Axis(fig[1, region], title="Ψ panel $region")
    heatmap!(ax, interior(getregion(Ψ, region), :, :, 1), colorrange = (-R * u₀, R * u₀), colormap = :balance)
    ax = Axis(fig[2, region], title="u panel $region")
    heatmap!(ax, interior(getregion(u, region), :, :, 1), colorrange = (-R * u₀, R * u₀), colormap = :balance)
    ax = Axis(fig[3, region], title="v panel $region")
    heatmap!(ax, interior(getregion(v, region), :, :, 1), colorrange = (-R * u₀, R * u₀), colormap = :balance)
end
=#

save("streamfunction_u_v.png", fig)

fig

#=
# An alternative way to compute velocities from streamfunction -- just to confirm sanity is in the room

U = Field{Face, Center, Center}(grid)
V = Field{Center, Face, Center}(grid)

for region in 1:6
    region_grid = getregion(grid, region)
    for k in 1:region_grid.Nz, j in 1:region_grid.Ny, i in 1:region_grid.Nx
        getregion(U, region)[i, j, k] = (getregion(Ψ, region)[i, j, k] - getregion(Ψ, region)[i, j+1, k]) / getregion(grid, region).Δyᶠᶜᵃ[i, j]
        getregion(V, region)[i, j, k] = (getregion(Ψ, region)[i+1, j, k] - getregion(Ψ, region)[i, j, k]) / getregion(grid, region).Δxᶜᶠᵃ[i, j]
    end
end

fig = Figure(resolution = (2200, 1200))

for region in 1:6
    ax = Axis(fig[1, region], title="Ψ panel $region")
    heatmap!(ax, interior(getregion(Ψ, region), :, :, 1), colorrange = (-R * u₀, R * u₀), colormap = :balance)
    ax = Axis(fig[2, region], title="U panel $region")
    heatmap!(ax, interior(getregion(U, region), :, :, 1), colorrange = (-R * u₀, R * u₀), colormap = :balance)
    ax = Axis(fig[3, region], title="V panel $region")
    heatmap!(ax, interior(getregion(V, region), :, :, 1), colorrange = (-R * u₀, R * u₀), colormap = :balance)
end

save("streamfunction_finite-differencing_u_v.png", fig)

fig
=#

velocities = PrescribedVelocityFields(; u , v)

model = HydrostaticFreeSurfaceModel(; grid,
                                      velocities,
                                      momentum_advection = VectorInvariant(vorticity_scheme = WENO()),
                                      tracer_advection = WENO(),
                                      tracers = :θ,
                                      buoyancy = nothing)

# initial condition for tracer
panel = 1

λ₀ = λnode(3Nx÷4+1, 3Ny÷4+1, getregion(grid, panel), Face(), Center())
φ₀ = φnode(3Nx÷4+1, 3Ny÷4+1, getregion(grid, panel), Center(), Face())
δR = 2

@show λ₀, φ₀

θ₀ = 1
θᵢ(λ, φ, z) = θ₀ * exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2δR^2)

set!(model, θ = θᵢ)

θ = model.tracers.θ

fill_halo_regions!(θ)

#=
# plot initial conditions
fig = Figure(resolution = (2200, 400))

for region in 1:6
    ax = Axis(fig[1, region], title="panel $region")
    heatmap!(ax, interior(getregion(θ, region), :, :, grid.Nz), colorrange=(0, θ₀))
end

fig
=#

Δt = 0.005
stop_iteration = 2500

simulation = Simulation(model; Δt, stop_iteration)

# Print a progress message
using Printf

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(1))

tracer_fields = Field[]

function save_tracer(sim)
    push!(tracer_fields, deepcopy(sim.model.tracers.θ))
end

simulation.callbacks[:save_tracer] = Callback(save_tracer, TimeInterval(Δt))

run!(simulation)

@info "Making an animation from the saved data..."

n = Observable(1)

Θₙ = []
for region in 1:6
    push!(Θₙ, @lift interior(getregion(tracer_fields[$n], region), :, :, grid.Nz))
end

fig = Figure(resolution = (2200, 400))

for region in 1:6
    ax = Axis(fig[1, region], title="panel $region")
    heatmap!(ax, Θₙ[region], colorrange=(-θ₀, θ₀), colormap=:balance)
end

fig

frames = 1:length(tracer_fields)

GLMakie.record(fig, "multi_region_tracer_advection.mp4", frames, framerate = 18) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    n[] = i
end