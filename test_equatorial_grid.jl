using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Operators
using CairoMakie
using Statistics

Nx, Ny = 64, 64

grid = EquatorialLatitudeLongitudeGrid(
    size = (Nx, Ny, 1),
    longitude = (-60, 60),
    latitude  = (-60, 60),
    z = (-1000, 0)
)

model = NonhydrostaticModel(
    grid,
    tracers = :c,
    advection = WENO(order=3)
)

# Constant flow
model.velocities.u .= 1.0
model.velocities.v .= 0.0

u, v = model.velocities

δ = ∂x(u) + ∂y(v)
ω = ∂x(v) - ∂y(u)

compute!(δ); compute!(ω)

println("div extremum  : ", extrema(interior(δ)))
println("vorticity ext : ", extrema(interior(ω)))

set!(model, u = 2, c = (x,y,z) -> begin
    φ = y / grid.radius
    λ = x / (grid.radius * cos(φ))
    exp(-(φ^2 + λ^2) / 0.1^2)
end)

simulation = Simulation(model, Δt = 1hours, stop_time = 50days)

saved_c = Matrix{Float64}[]
saved_t = Float64[]

simulation.callbacks[:save] = Callback(sim -> begin
    push!(saved_c, Array(interior(sim.model.tracers.c)[:, :, 1]))
    push!(saved_t, sim.model.clock.time)
end, TimeInterval(1days))

run!(simulation)

println("Final min/max: ",
    minimum(interior(model.tracers.c)), " ",
    maximum(interior(model.tracers.c)))

λ = collect(grid.λᶜᵃᵃ)[1:Nx]
φ = collect(grid.φᵃᶜᵃ)[1:Ny]

function plot_tracer(c, λ, φ; title="")

    cmin, cmax = extrema(c)

    if cmax - cmin < 1e-12
        cmax = cmin + 1e-6   # avoid Makie crash
    end

    fig = Figure()
    ax = Axis(fig[1,1],
              xlabel="φ (deg)",
              ylabel="λ (deg)",
              title=title)

    hm = heatmap!(ax, φ, λ, c'; colorrange=(cmin, cmax))
    Colorbar(fig[1,2], hm)

    display(fig)
end

c_final = saved_c[end]
plot_tracer(c_final, λ, φ; title="Final tracer")

Nt = length(saved_c)

# Global color scale (important!)
global_min = minimum(map(minimum, saved_c))
global_max = maximum(map(maximum, saved_c))

fig = Figure()
ax = Axis(fig[1,1], xlabel="φ", ylabel="λ")

hm = heatmap!(ax, φ, λ, saved_c[1]';
              colorrange=(global_min, global_max))

Colorbar(fig[1,2], hm)

record(fig, "tracer_animation.mp4", 1:Nt) do n
    hm[3] = saved_c[n]'
    ax.title = "t = $(round(saved_t[n]/86400, digits=1)) days"
end


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
