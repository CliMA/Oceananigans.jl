using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.OutputWriters
using CairoMakie
using Revise
using Statistics
using NCDatasets

# ------------------------
# 1. Construct your grid
# ------------------------

Nx, Ny = 64, 64

grid = EquatorialLatitudeLongitudeGrid(
    size = (Nx, Ny, 1),
    longitude = (-60, 60),
    latitude  = (-60, 60),
    z = (-1000, 0)
)

model = NonhydrostaticModel(grid, 
                            tracers = :c,
                            advection = WENO(order=3))

model.velocities.u .= 1.0
model.velocities.v .= 0.0

u, v = model.velocities

δ = ∂x(u) + ∂y(v)
ω = ∂x(v) - ∂y(u)
compute!(δ)
compute!(ω)

println("divergence extrema: = ", extrema(interior(δ)))
println("vorticity extrema:  = ", extrema(interior(ω)))

set!(model, u = 2, c = (x,y,z) -> exp(-(x^2 + y^2) / (2e2)))

simulation = Simulation(model, Δt = 1hours , stop_time=50*days)

function print_tracer_stats(sim)
    c = interior(sim.model.tracers.c)[:, :, 1]
    println("t = ", sim.model.clock.time,
            " | min(c) = ", minimum(c),
            " | max(c) = ", maximum(c),
            " | mean(c) = ", mean(c))
end

simulation.callbacks[:progress] =
    Callback(print_tracer_stats, IterationInterval(50))


saved_c = Vector{Array{Float64,2}}()
saved_t = Float64[]

simulation.callbacks[:save_c] = Callback(sim -> begin
    c = Array(interior(sim.model.tracers.c)[:, :, 1])
    push!(saved_c, c)
    push!(saved_t, sim.model.clock.time)
end, TimeInterval(1days))

run!(simulation)

print(maximum(interior(model.tracers.c)), " ")
print(minimum(interior(model.tracers.c)))

λ = collect(grid.λᶜᵃᵃ)
φ = collect(grid.φᵃᶜᵃ)

c = collect(interior(model.tracers.c))[:, :, 1]

fig = Figure()
ax = Axis(fig[1,1], xlabel="φ", ylabel="λ")

hm = heatmap!(ax, φ, λ, c')   # transpose for correct orientation
Colorbar(fig[1,2], hm)

display(fig)


λ = collect(grid.λᶜᵃᵃ)
φ = collect(grid.φᵃᶜᵃ)

using NCDatasets
using CairoMakie

# ------------------------
# 1. Load NetCDF file
# ------------------------

ds = NCDataset("tracer.nc")

c_data = ds["c"]          # (φ, λ, time)
time   = ds["time"]

Ny, Nx, Nt = size(c_data)

println("Loaded data size:", size(c_data))

# ------------------------
# 2. Build coordinate axes
# ------------------------
# (Use grid values if still in memory — otherwise reconstruct)

# If grid exists:
λ = collect(grid.λᶜᵃᵃ)
φ = collect(grid.φᵃᶜᵃ)

# If grid is NOT available, use simple indices:
# λ = 1:Nx
# φ = 1:Ny

# ------------------------
# 3. Plot a single frame
# ------------------------

n = 1   # first time step

c = c_data[:, :, n]

fig = Figure()
ax = Axis(fig[1,1],
    xlabel = "φ (degrees)",
    ylabel = "λ (degrees)",
    title = "Tracer field (t = $(round(time[n]/86400, digits=2)) days)"
)

hm = heatmap!(ax, φ, λ, c'; colorrange=(minimum(c), maximum(c)))
Colorbar(fig[1,2], hm)

display(fig)


# ------------------------
# 4. Animate all frames
# ------------------------

fig = Figure()
ax = Axis(fig[1,1],
    xlabel="φ (degrees)",
    ylabel="λ (degrees)"
)

c0 = c_data[:, :, 1]

# FIXED color range so animation is meaningful
crange = (minimum(c_data), maximum(c_data))

hm = heatmap!(ax, φ, λ, c0'; colorrange=crange)
Colorbar(fig[1,2], hm)

record(fig, "tracer_animation.mp4", 1:Nt) do n
    c = c_data[:, :, n]

    hm[3] = c'   # update heatmap

    ax.title = "t = $(round(time[n]/86400, digits=2)) days"
end

close(ds)

println("Animation saved as tracer_animation.mp4")



#=
println("Grid constructed successfully.")

Δx = collect(grid.Δxᶜᶜᵃ)
Δy = collect(grid.Δyᶠᶜᵃ)
Az = collect(grid.Azᶜᶜᵃ)

Δx_mat = repeat(Δx', size(Δy,1), 1)

ratio = Az ./ (Δx_mat .* Δy)
println(extrema(ratio[isfinite.(ratio)]))

# ------------------------
# 2. Extract nodes
# ------------------------

λC, φC, _ = nodes(grid, Center(), Center(), Center())

println("\nφ (latitude) range:")
println(extrema(φC))

# ------------------------
# 3. Access metrics
# ------------------------

Δx = grid.Δxᶜᶜᵃ
Δy = grid.Δyᶠᶜᵃ
Az = grid.Azᶜᶜᵃ

println("\nMetric ranges:")
println("Δx: ", extrema(Δx))
println("Δy: ", extrema(Δy))
println("Az: ", extrema(Az))

# ------------------------
# 4. Sanity checks
# ------------------------

println("\nSanity checks:")

# Δx should not vary in i-direction
println("\nCheck Δx variation along i (should be small):")
maximum(abs, Δx .- mean(Δx))

# Δy should vary with latitude (cos φ)
println("\nCheck Δy min/max (should vary):")
println(extrema(Δy))

# Area consistency (avoid division by zero)
println("\nCheck Az ≈ Δx * Δy:")

Δx_arr = parent(grid.Δxᶜᶜᵃ)
Δy_arr = isa(grid.Δyᶠᶜᵃ, Number) ? fill(grid.Δyᶠᶜᵃ, size(Δx_arr)) : parent(grid.Δyᶠᶜᵃ)
Az_arr = parent(grid.Azᶜᶜᵃ)

println("Δx range: ", extrema(Δx_arr))
println("Δy range: ", extrema(Δy_arr))
println("Az range: ", extrema(Az_arr))

# Simple check (no slicing!)
println("Check Δx variation:")
println(maximum(abs, Δx_arr .- mean(Δx_arr; dims=1)))

println("Check Δy variation:")
println(maximum(abs, Δy_arr .- mean(Δy_arr)))

Δx_vec = collect(grid.Δxᶜᶜᵃ)          # 1D
Δy_mat = collect(grid.Δyᶠᶜᵃ)          # 2D
Az_mat = collect(grid.Azᶜᶜᵃ)

# Broadcast Δx into 2D
Δx_mat = repeat(Δx_vec', size(Δy_mat, 1), 1)

ratio = Az_mat ./ (Δx_mat .* Δy_mat)

println("ratio range: ", extrema(ratio[isfinite.(ratio)]))

# ------------------------
# 5. Plot diagnostics
# ------------------------

fig = Figure(size = (1200, 400))

φ = collect(grid.φᵃᶜᵃ)
λ = collect(grid.λᶜᵃᵃ)

# Δx (1D)
ax1 = Axis(fig[1,1], title="Δx vs φ",
           xlabel="φ (degrees)", ylabel="Δx (m)")

lines!(ax1, φ, collect(grid.Δxᶜᶜᵃ))
scatter!(ax1, φ, collect(grid.Δxᶜᶜᵃ))

# Δy (2D)
ax2 = Axis(fig[1,2], title="Δy",
           xlabel="φ (degrees)", ylabel="λ (degrees)")

hm2 = heatmap!(ax2, φ, λ, collect(grid.Δyᶠᶜᵃ)')
Colorbar(fig[1,3], hm2)

# Az (2D)
ax3 = Axis(fig[1,4], title="Az",
           xlabel="φ (degrees)", ylabel="λ (degrees)")

hm3 = heatmap!(ax3, φ, λ, collect(grid.Azᶜᶜᵃ)')
Colorbar(fig[1,5], hm3)

display(fig)

# ------------------------
# 6. Optional: geometry in 3D
# ------------------------

λ = collect(grid.λᶜᵃᵃ)
φ = collect(grid.φᵃᶜᵃ)

Λ = reshape(λ, :, 1) .* ones(1, length(φ))
Φ = ones(length(λ), 1) .* reshape(φ, 1, :)

R = grid.radius

X = R .* cosd.(Φ) .* cosd.(Λ)
Y = R .* cosd.(Φ) .* sind.(Λ)
Z = R .* sind.(Φ)

fig2 = Figure(size = (500, 500))
ax = Axis3(fig2[1,1])

surface!(ax, X, Y, Z)

display(fig2)

println("\nDone.")
=#
