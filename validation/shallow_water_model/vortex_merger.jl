using Printf
using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, shallow_water_velocities
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Statistics: mean
using JLD2
using CairoMakie

arch = GPU()

for Nh in [100, 200, 400, 800, 1600], stencil in [VorticityStencil, VelocityStencil]

    grid = RectilinearGrid(arch, size = (Nh, Nh), x = (0, 1), y = (0, 1), halo = (4, 4), topology = (Bounded, Bounded, Flat))

    g = 1.0
    f = 5.0
    H  = 1.0
    R = (g*H)^0.5 / f

    model = ShallowWaterModel(grid = grid,
                            gravitational_acceleration = g,
                            coriolis = FPlane(f = f),
                            mass_advection = WENO(),
                            momentum_advection = VectorInvariant(scheme=WENO(), stencil=stencil()),
                            formulation = VectorInvariantFormulation())

    # Model initialization
    h₀ = 0.2
    σ  = 0.07
    d  = 1.4σ

    gaussian(x, y, σ) = exp(-(x^2 + y^2)/(2*σ^2))
    hᵢ(x, y, z) = H + h₀ * (gaussian(x - d - 0.5, y - 0.5, σ) + gaussian(x - 0.5 + d, y - 0.5, σ))

    set!(model, h = hᵢ)
    set!(model, uh = - ∂y(model.solution.h) / f)
    set!(model, vh =   ∂x(model.solution.h) / f)
    @info "Model initialized"

    #####
    ##### Simulation setup
    #####

    g = model.gravitational_acceleration
    gravity_wave_speed = sqrt(g * maximum(model.solution.h)) # hydrostatic (shallow water) gravity wave speed

    # Time-scale for gravity wave propagation across the smallest grid cell
    wave_propagation_time_scale = grid.Δxᶜᵃᵃ / gravity_wave_speed

    Δt = wave_propagation_time_scale * 0.03

    simulation = Simulation(model, Δt = Δt, stop_time = 10)
    start_time = [time_ns()]

    function progress(sim)
        wall_time = (time_ns() - start_time[1]) * 1e-9

        u = sim.model.solution[1]

        @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, wall time: %s",
                        prettytime(sim.model.clock.time),
                        sim.model.clock.iteration, maximum(abs, u),
                        prettytime(wall_time))

        start_time[1] = time_ns()

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    h    = model.solution.h
    u, v = shallow_water_velocities(model)

    Eᵦ = g * H * H

    ζ = Field(∂x(v) - ∂y(u))
    compute!(ζ)

    KE = Field(h * (u^2 + v^2))
    PE = Field(g * h^2)
    compute!(KE)
    compute!(PE)

    PV = Field((ζ + f) / h)
    compute!(PV)

    save_interval = 0.1

    simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, h, ζ, KE, PE, PV),
                                                                schedule = TimeInterval(save_interval),
                                                                filename = "vortex_merger_$(Nh)_WENO",
                                                                overwrite_existing = true)

    run!(simulation)

    file = jldopen("vortex_merger_$(Nh)_WENO.jld2")
    iterations = parse.(Int, keys(file["timeseries/t"]))

    ke2 = zeros(length(iterations))
    pe2 = zeros(length(iterations))
    z2  = zeros(length(iterations))
    en  = zeros(length(iterations))
    ett = zeros(length(iterations))
    ztt = zeros(length(iterations))

    for (idx, iter) in enumerate(iterations)
        ke2[idx] = mean(file["timeseries/KE/" * string(iter)][:, :, 1])
        pe2[idx] = mean(file["timeseries/PE/" * string(iter)][:, :, 1])
        en[idx]  = mean(ke2[idx] .+ pe2[idx])
        z2[idx]  = mean(file["timeseries/ζ/" * string(iter)][:, :, 1] .^ 2)
        ett[idx] = (en[1] - en[idx]) / (en[1] - Eᵦ)
        ztt[idx] = (z2[1] - z2[idx]) / z2[1]
    end

    jldsave("energy_vorticity_$(Nh)_WENO.jld2", energy = ett, vorticity = ztt)

    file = jldopen("vortex_merger_$(Nh)_WENO.jld2")

    iter = Observable(0)
    iterations = parse.(Int, keys(file["timeseries/t"]))

    ζ′ = @lift(file["timeseries/ζ/" * string($iter)][:, 1:end-1, 1])
    h′ = @lift(file["timeseries/h/" * string($iter)][:, 1:end-1, 1])

    PV = @lift(file["timeseries/PV/" * string($iter)][:, :, 1]')

    xζ, yζ, z = nodes((Face, Face, Center), grid)
    xh, yh, z = nodes((Center, Center, Center), grid)

    title = @lift(@sprintf("Vorticity in Shallow Water Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
    fig = CairoMakie.Figure(resolution = (1000, 600))

    ax = CairoMakie.Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title)
    heatmap_plot = CairoMakie.heatmap!(ax, yζ, xζ, PV, colormap=Reverse(:balance), colorrange = (-9.0, 9.0))
    CairoMakie.Colorbar(fig[1,2], heatmap_plot, width=25)

    CairoMakie.record(fig, "vortex_merger.mp4", iterations[1:end], framerate=2) do i
        @info "Plotting iteration $i of $(iterations[end])..."
        iter[] = i
    end

    iter = iterations[end]
    save("vortex_merger_$(Nh)_WENO.png", fig)

end

#=
ettv1 = []
ettv2 = []
eetv3 = []
for Nh in [100, 200, 400, 800, 1600]
    filev2 = jldopen("energy_vorticity_$(Nh)_VelocityStencil.jld2")
    push!(ettv2, filev2["energy"])
    filev1 = jldopen("energy_vorticity_$(Nh)_VorticityStencil.jld2")
    push!(ettv1, filev1["energy"])
    if Nh < 200
        filev3 = jldopen("energy_vorticity_$(Nh)_WENO.jld2")
        push!(ettv3, filev3["energy"])
    end
end

ε = 1e-20
fig = Figure()
ax = Axis(fig[1, 1], yscale = log10)
lines!(ax, abs.(ettv1[1] .+ ε))
lines!(ax, abs.(ettv1[2] .+ ε))
lines!(ax, abs.(ettv1[3] .+ ε))
lines!(ax, abs.(ettv1[4] .+ ε))
lines!(ax, abs.(ettv1[5] .+ ε))
lines!(ax, abs.(ettv2[1] .+ ε), linestyle = :dash)
lines!(ax, abs.(ettv2[2] .+ ε), linestyle = :dash)
lines!(ax, abs.(ettv2[3] .+ ε), linestyle = :dash)
lines!(ax, abs.(ettv2[4] .+ ε), linestyle = :dash)
lines!(ax, abs.(ettv2[5] .+ ε), linestyle = :dash)
GLMakie.ylims!(ax, (1e-6, 1))

lines(ettv2[1],  color = :red)
lines!(ettv2[2], color = :green)
lines(ettv2[3], color = :blue)
lines!(ettv2[4], linestyle = :dash)
lines!(ettv2[5], linestyle = :dot)


for (idx, stencil) in enumerate((:VorticityStencil,)), Nh in [100, 200, 800, 1600]

    grid = RectilinearGrid(size = (Nh, Nh), x = (0, 1), y = (0, 1), halo = (4, 4), topology = (Bounded, Bounded, Flat))

    @show idx, Nh
    pv = Symbol(:pv, idx, :_, Nh)
    x = Symbol(:x, :_, Nh)
    y = Symbol(:y, :_, Nh)

    @eval $x, $y, _ = nodes((Face, Face, Center), $grid)

    file = jldopen("vortex_merger_$(Nh)_$(string(stencil)).jld2")

    key = parse(Int, keys(file["timeseries/PV"])[end])
    @eval $pv = file["timeseries/PV/" * string(key)][:, :, 1]
end

fig = Figure()
ga = fig[1, 1] = GridLayout()

axtl = GLMakie.Axis(ga[1, 1]) #, title = advection_scheme * " resolution = $(N1)²")
axtr = GLMakie.Axis(ga[1, 2]) #, title = advection_scheme * " resolution = $(N2)²")
axbl = GLMakie.Axis(ga[2, 1]) #, title = advection_scheme * " resolution = $(N3)²")
axbr = GLMakie.Axis(ga[2, 2]) #, title = advection_scheme * " resolution = $(N4)²")

heatmap!(axbl, y_200[1:100],     reverse(x_200[101:200]) , Array(pv1_200[101:200, 1:100]'),   colormap=Reverse(:balance), colorrange = (-9.0, 9.0))
heatmap!(axbr, y_100[51:100],    reverse(x_100[51:100]),   Array(pv1_100[51:100, 51:100]'),   colormap=Reverse(:balance), colorrange = (-9.0, 9.0))
heatmap!(axtl, y_800[1:400],     reverse(x_800[1:400]),    Array(pv1_800[1:400, 1:400]'),     colormap=Reverse(:balance), colorrange = (-9.0, 9.0))
heatmap!(axtr, y_1600[801:1600], reverse(x_1600[1:800]),   Array(pv1_1600[1:800, 801:1600]'), colormap=Reverse(:balance), colorrange = (-9.0, 9.0))
hidespines!(axtl)
hidespines!(axtr)
hidespines!(axbl)
hidespines!(axbr)
GLMakie.hidedecorations!(axtl)
GLMakie.hidedecorations!(axtr)
GLMakie.hidedecorations!(axbl)
GLMakie.hidedecorations!(axbr)

colgap!(ga, 0)
rowgap!(ga, 0)
=#