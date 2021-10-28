using Printf
using GLMakie
using Oceananigans
using Oceananigans.Units

# 2.8125 degree resolution
Nx = 128
Ny = 60
Nz = 18
reference_density = 1035

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

east_west_stress_path = "off_TAUXvar1.bin"
north_south_stress_path = "off_TAUY.bin"
sea_surface_temperature_path="sst25_128x60x12.bin"

thirty_days = 30days
Nmonths = 12
bytes = sizeof(Float32) * Nx * Ny

τˣ = - reshape(bswap.(reinterpret(Float32, read(east_west_stress_path, Nmonths * bytes))), (Nx, Ny, Nmonths)) ./ reference_density
τʸ = - reshape(bswap.(reinterpret(Float32, read(north_south_stress_path, Nmonths * bytes))), (Nx, Ny, Nmonths)) ./ reference_density
T★ = reshape(bswap.(reinterpret(Float32, read(sea_surface_temperature_path, Nmonths * bytes))), (Nx, Ny, Nmonths))

#####
##### Utils
#####

@inline current_time_index(time, interval=thirty_days, length=Nmonths) = mod(trunc(Int, time / interval),     length) + 1
@inline next_time_index(time, interval=thirty_days, length=Nmonths)    = mod(trunc(Int, time / interval) + 1, length) + 1

@inline cyclic_interpolate(u₁, u₂, time, interval=thirty_days) = u₁ + mod(time / interval, 1) * (u₂ - u₁)

@inline function cyclic_interpolate(τ::AbstractArray, time, interval=thirty_days, length=Nmonths)
    n₁ = current_time_index(time, interval, length)
    n₂ = next_time_index(time, interval, length)
    return cyclic_interpolate.(view(τ, :, :, n₁), view(τ, :, :, n₂), time, interval)
end

times = 0:1days:24*30days
discrete_times = 0:30days:11*30days

#=
i = 80
j = 12 # Southern ocean?
Nt = length(times)
τ_ij = zeros(Nt)

for (m, t) in enumerate(times)
    n₁ = current_time_index(t)
    n₂ = next_time_index(t)
    τ_ij[m] = cyclic_interpolate(τˣ[i, j, n₁], τˣ[i, j, n₂], t)
end

fig = Figure(resolution = (1200, 600))

ax = Axis(fig[1, 1], xlabel="Time (days)", ylabel="Target SST at (i, j) = ($i, $j)")
scatter!(ax, discrete_times ./ day, τˣ[i, j, :], markersize=20, marker=:utriangle, color=:red)
lines!(ax, times ./ day, τ_ij)

display(fig)
=#

#####
##### Analyze forcing
#####

fig = Figure(resolution = (1200, 800))

t = Node(0.0)

T★_continuous(t) = cyclic_interpolate(T★, t)
τˣ_continuous(t) = cyclic_interpolate(τˣ, t)
τʸ_continuous(t) = cyclic_interpolate(τʸ, t)

T★_t = @lift T★_continuous($t)
τˣ_t = @lift τˣ_continuous($t)
τʸ_t = @lift τʸ_continuous($t)

max_τ = maximum(abs, τˣ)
max_T = maximum(T★)
min_T = minimum(T★)

fig = Figure(resolution = (1200, 600))

ax_τˣ = Axis(fig[1, 1], title="East-west wind stress (m² s⁻²)")
hm_τˣ = heatmap!(ax_τˣ, τˣ_t, colorrange=(-max_τ, max_τ), colormap=:balance)
cb_τˣ = Colorbar(fig[1, 2], hm_τˣ)

ax_τʸ = Axis(fig[2, 1], title="North-south wind stress (m² s⁻²)")
hm_τʸ = heatmap!(ax_τʸ, τʸ_t, colorrange=(-max_τ, max_τ), colormap=:balance)
cb_τʸ = Colorbar(fig[2, 2], hm_τʸ)

ax_T = Axis(fig[3, 1], title="Sea surface temperature (ᵒC)")
hm_T = heatmap!(ax_T, T★_t, colorrange=(min_T, max_T), colormap=:thermal)
cb_T = Colorbar(fig[3, 2], hm_T)

title_str = @lift "Earth day = " * prettytime($t)
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, "annual_cycle_forcing.mp4", times, framerate=8) do tn
    t[] = tn
end
