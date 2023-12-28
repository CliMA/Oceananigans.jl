using Oceananigans
using Oceananigans.Units
using StructArrays
using JLD2
using FileIO
using Printf
using Random
using Statistics
using Dates
using CUDA: CuArray
using Oceananigans.Models.LagrangianParticleTracking: ParticleVelocities, ParticleDiscreteForcing
using Oceananigans.Fields: VelocityFields
using Oceananigans.Architectures: device, architecture
using CairoMakie
using KernelAbstractions

Random.seed!(123)

grid = RectilinearGrid(Oceananigans.CPU(), Float64,
                       size = (4, 4, 4),
                       halo = (5, 5, 5),
                       x = (0, 1),
                       y = (0, 1),
                       z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

# b_initial_noisy(x, y, z) = 1e-3 * rand() * exp(z / 8)
b_initial_noisy(x, y, z) = 1e-3 * rand()

#%%
struct LagrangianPOC{T, V, A, R}
    x :: T
    y :: T
    z :: T
    u :: V
    v :: V
    w :: V
    u_particle :: V
    v_particle :: V
    w_particle :: V
    age :: A
    radius :: R
end

n_particles = 3

# x₀ = CuArray(zeros(n_particles))
# y₀ = CuArray(rand(n_particles))
# z₀ = CuArray(-0.1 * rand(n_particles))

# u₀ = CuArray(zeros(n_particles))
# v₀ = CuArray(zeros(n_particles))
# w₀ = CuArray(-0.1 * rand(n_particles))

# u₀_particle = deepcopy(u₀)
# v₀_particle = deepcopy(v₀)
# w₀_particle = deepcopy(w₀)

# age = CuArray(zeros(n_particles))
# radius = CuArray(ones(n_particles))

x₀ = zeros(n_particles)
y₀ = rand(n_particles)
z₀ = -0.1 * rand(n_particles)

u₀ = zeros(n_particles)
v₀ = zeros(n_particles)
w₀ = -1e-5 * rand(n_particles)

u₀_particle = deepcopy(u₀)
v₀_particle = deepcopy(v₀)
w₀_particle = deepcopy(w₀)

age = zeros(n_particles)
radius = ones(n_particles)

particles = StructArray{LagrangianPOC}((x₀, y₀, z₀, u₀, v₀, w₀, u₀_particle, v₀_particle, w₀_particle, age, radius))

@inline function w_sinking(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    w₀ = particles[p].w_particle
    w = w₀ + 1 / (2 * 24 * 60^2) * (w_fluid - w₀) * Δt

    # particles[p].w = w
    return w
end
w_forcing  = ParticleDiscreteForcing(w_sinking)
sinking = ParticleVelocities(w=w_forcing)

@kernel function update_particle_velocities!(particles, advective_velocity::ParticleVelocities, grid, clock, Δt, model_fields)
    p = @index(Global)
    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]

        u_fluid = particles.u[p]
        v_fluid = particles.v[p]
        w_fluid = particles.w[p]

        particles.u_particle[p] = advective_velocity.u(x, y, z, u_fluid, particles, p, grid, clock, Δt, model_fields)
        particles.v_particle[p] = advective_velocity.v(x, y, z, v_fluid, particles, p, grid, clock, Δt, model_fields)
        particles.w_particle[p] = advective_velocity.w(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    end
end

function update_lagrangian_particle_velocities!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    workgroup = min(length(particles), 256)
    worksize = length(particles)
    model_fields = merge(model.velocities, model.tracers, model.auxiliary_fields)

    # # Update particle "properties"
    # for (name, field) in pairs(particles.tracked_fields)
    #     compute!(field)
    #     particle_property = getproperty(particles.properties, name)
    #     ℓx, ℓy, ℓz = map(instantiate, location(field))

    #     update_field_property_kernel! = update_property!(device(arch), workgroup, worksize)

    #     update_field_property_kernel!(particle_property, particles.properties, model.grid,
    #                                   datatuple(field), ℓx, ℓy, ℓz)
    # end

    update_particle_velocities_kernel! = update_particle_velocities!(device(arch), workgroup, worksize)
    update_particle_velocities_kernel!(particles.properties, particles.advective_velocity, model.grid, model.clock, Δt, model_fields)

    return nothing
end
velocities = VelocityFields(grid)

# lagrangian_particles = LagrangianParticles(particles, dynamics=add_age, advective_forcing=sinking)
lagrangian_particles = LagrangianParticles(particles, advective_velocity=sinking, tracked_fields=velocities, dynamics=update_lagrangian_particle_velocities!)
# lagrangian_particles = LagrangianParticles(particles)

#%%
model = NonhydrostaticModel(; 
            grid = grid,
            velocities = velocities,
            closure = ScalarDiffusivity(ν=1e-5, κ=1e-5),
            coriolis = FPlane(f=1e-4),
            buoyancy = BuoyancyTracer(),
            tracers = (:b),
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            particles = lagrangian_particles
            )

set!(model, b=b_initial_noisy)
# set!(model, b=b_initial_noisy, c0=1e6)

b = model.tracers.b
u, v, w = model.velocities

simulation = Simulation(model, Δt=0.1seconds, stop_time=2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=60seconds, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, (<x>, <y>, <z>) (%6.3e, %6.3e, %6.3e), next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            mean(lagrangian_particles.properties.x),
            mean(lagrangian_particles.properties.y),
            mean(lagrangian_particles.properties.z),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    # file["metadata/parameters/coriolis_parameter"] = f
    # file["metadata/parameters/momentum_flux"] = Qᵁ
    # file["metadata/parameters/buoyancy_flux"] = Qᴮ
    # file["metadata/parameters/carbon_flux"] = Qᶜ
    # file["metadata/parameters/dbdz"] = dbdz
    # file["metadata/parameters/b_surface"] = b_surface
    return nothing
end

particle_outputs = (; model.particles)

simulation.output_writers[:particles] = JLD2OutputWriter(model, particle_outputs,
                                                          filename = "./particles.jld2",
                                                          schedule = TimeInterval(60seconds),
                                                          with_halos = true,
                                                          overwrite_existing = true,
                                                          init = init_save_some_metadata!)

run!(simulation)

#%%
times, particle_data = jldopen("./particles.jld2", "r") do file
    iters = keys(file["timeseries/t"])
    times = [file["timeseries/t/$(iter)"] for iter in iters]
    particle_timeseries = [file["timeseries/particles/$(iter)"] for iter in iters]
    return times, particle_timeseries
end

#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="z")
for i in 1:n_particles
    lines!(ax, times, [data.z[i] for data in particle_data])
end
display(fig)
#%%
#=
# FILE_DIR = "./LES/QU_0_QB_1.0e-6_QC_-1000.0_dbdz_0.0002_Nages_20_Lxz_64.0_128.0_halfc0_2_w_-0.002962962962962963_test7"

b_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "b", backend=OnDisk())
b_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "b", backend=OnDisk())
b_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "b", backend=OnDisk())

bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")
# particle_data = FieldDataset("$(FILE_DIR)/particles.jld2")


blim = (find_min(b_xy_data, b_yz_data, b_xz_data), find_max(b_xy_data, b_yz_data, b_xz_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))
# cbarlim = (find_min(csbar_data...), find_max(csbar_data...))
cbarlim = (find_min(csbar_data...), 100)

# c_xticks = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

Nt = length(bbar_data.times)

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.xᶜᵃᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

#%%
fig = Figure(resolution=(2000, 2000))

axb = Axis3(fig[1:2, 1:2], title="b", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

axbbar = Axis(fig[3, 1], title="<b>", xlabel="<b>", ylabel="z")
axcbar = Axis(fig[3, 2], title="<c>", xlabel="<c>", ylabel="z")

xs_xy = xC
ys_xy = yC
zs_xy = [zC[Nz] for x in xs_xy, y in ys_xy]

ys_yz = yC
xs_yz = range(xC[1], stop=xC[1], length=length(zC))
zs_yz = zeros(length(xs_yz), length(ys_yz))
for j in axes(zs_yz, 2)
  zs_yz[:, j] .= zC
end

xs_xz = xC
ys_xz = range(yC[1], stop=yC[1], length=length(zC))
zs_xz = zeros(length(xs_xz), length(ys_xz))
for i in axes(zs_xz, 1)
  zs_xz[i, :] .= zC
end

colormap = Reverse(:RdBu_10)
b_color_range = blim

n = Observable(1)

parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

bₙ_xy = @lift interior(b_xy_data[$n], :, :, 1)
bₙ_yz = @lift transpose(interior(b_yz_data[$n], 1, :, :))
bₙ_xz = @lift interior(b_xz_data[$n], :, 1, :)

bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)
csbarₙ = [@lift interior(data[$n], 1, 1, :) for data in csbar_data]

# cmin = @lift find_min([interior(data[$n], 1, 1, :) for data in csbar_data]..., -1e-5)
# cmax = @lift find_max([interior(data[$n], 1, 1, :) for data in csbar_data]..., 1e-5)

# cbarlim = @lift (find_min([interior(data[$n], 1, 1, :) for data in csbar_data]..., -1e-5), find_max([interior(data[$n], 1, 1, :) for data in csbar_data]..., 1e-5))

b_xy_surface = surface!(axb, xs_xy, ys_xy, zs_xy, color=bₙ_xy, colormap=colormap, colorrange = b_color_range)
b_yz_surface = surface!(axb, xs_yz, ys_yz, zs_yz, color=bₙ_yz, colormap=colormap, colorrange = b_color_range)
b_xz_surface = surface!(axb, xs_xz, ys_xz, zs_xz, color=bₙ_xz, colormap=colormap, colorrange = b_color_range)

lines!(axbbar, bbarₙ, zC)

for (i, data) in enumerate(csbarₙ)
    lines!(axcbar, data, zC, label="c$(i-1)")
end

Legend(fig[4, :], axcbar, tellwidth=false, orientation=:horizontal, nbanks=2)

xlims!(axbbar, bbarlim)
xlims!(axcbar, cbarlim)
# xlims!(axcbar, (cmin[], cmax[]))
# xlims!(axcbar, (-0.1, 0.1))

trim!(fig.layout)
display(fig)

record(fig, "$(FILE_DIR)/video.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"

#%%
#%%
# fig = Figure(resolution=(1800, 1500))

# axubar = Axis(fig[1, 1], title="<u>", xlabel="<u>", ylabel="z")
# axvbar = Axis(fig[1, 2], title="<v>", xlabel="<v>", ylabel="z")
# axbbar = Axis(fig[1, 3], title="<b>", xlabel="<b>", ylabel="z")
# axcbar = Axis(fig[2, 1], title="<c>", xlabel="<c>", ylabel="z")

# ubarlim = (minimum(ubar_data), maximum(ubar_data))
# vbarlim = (minimum(vbar_data), maximum(vbar_data))
# bbarlim = (minimum(bbar_data), maximum(bbar_data))
# cbarlim = (find_min(csbar_data...), find_max(csbar_data...))

# n = Observable(1)

# parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
#     return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
# end 

# time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
# title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

# ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
# vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
# bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)

# csbarₙ = [@lift interior(data[$n], 1, 1, :) for data in csbar_data]

# # cmin = @lift find_min([interior(data[$n], 1, 1, :) for data in csbar_data]..., -1e-5)
# # cmax = @lift find_max([interior(data[$n], 1, 1, :) for data in csbar_data]..., 1e-5)

# # cbarlim = @lift (find_min([interior(data[$n], 1, 1, :) for data in csbar_data]..., -1e-5), find_max([interior(data[$n], 1, 1, :) for data in csbar_data]..., 1e-5))

# lines!(axubar, ubarₙ, zC)
# lines!(axvbar, vbarₙ, zC)
# lines!(axbbar, bbarₙ, zC)

# for (i, data) in enumerate(csbarₙ)
#     lines!(axcbar, data, zC, label="c$(i-1)")
# end

# Legend(fig[2, 2], axcbar, tellwidth=false)

# xlims!(axubar, ubarlim)
# xlims!(axvbar, vbarlim)
# xlims!(axbbar, bbarlim)
# xlims!(axcbar, cbarlim)
# # xlims!(axcbar, (cmin[], cmax[]))
# # xlims!(axcbar, (-0.1, 0.1))

# trim!(fig.layout)

# record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
#     n[] = nn
# end

# @info "Animation completed"

# #%%
=#