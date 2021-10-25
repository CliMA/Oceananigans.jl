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

Nmonths = 12
bytes = sizeof(Float32) * Nx * Ny

τˣ = - reshape(bswap.(reinterpret(Float32, read(east_west_stress_path, Nmonths * bytes))), (Nx, Ny, Nmonths)) ./ reference_density
τʸ = - reshape(bswap.(reinterpret(Float32, read(north_south_stress_path, Nmonths * bytes))), (Nx, Ny, Nmonths)) ./ reference_density
T★ = reshape(bswap.(reinterpret(Float32, read(sea_surface_temperature_path, Nmonths * bytes))), (Nx, Ny, Nmonths))

#=
max_τ = maximum(abs, τˣ)
min_T = minimum(T★)
max_T = maximum(T★)

fig = Figure(resolution = (1200, 600))

for i = 1:6
    τˣi = τˣ[:, :, i]
    τʸi = τʸ[:, :, i]
    T★i = T★[:, :, i]
    
    ax_τˣ = Axis(fig[i, 1], title="Month $i τˣ (m² s⁻²)")
    hm_τˣ = heatmap!(ax_τˣ, τˣi, colorrange=(-max_τ, max_τ), colormap=:balance)
    cb_τˣ = Colorbar(fig[i, 2], hm_τˣ)
    
    ax_τʸ = Axis(fig[i, 3], title="Month $i τʸ (m² s⁻²)")
    hm_τʸ = heatmap!(ax_τʸ, τʸi, colorrange=(-max_τ, max_τ), colormap=:balance)
    cb_τʸ = Colorbar(fig[i, 4], hm_τʸ)
    
    ax_T = Axis(fig[i, 5], title="Month $i T★ (ᵒC)")
    hm_T = heatmap!(ax_T, T★i, colorrange=(min_T, max_T), colormap=:thermal)
    cb_T = Colorbar(fig[i, 6], hm_T)
end

display(fig)
=#

#####
##### Analyze forcing
#####

const thirty_days = 30days

@inline current_time_index(time) = mod(trunc(Int, time / thirty_days), 12) + 1
@inline next_time_index(time) = mod(trunc(Int, time / thirty_days) + 1, 12) + 1

@inline thirty_day_interpolate(u₁, u₂, time) = u₁ + mod1(time / thirty_days, 1.0) * (u₂ - u₁)

@inline function thirty_day_interpolate(τ::AbstractArray, time)
    n₁ = current_time_index(time)
    n₂ = next_time_index(time)
    return thirty_day_interpolate.(view(τ, :, :, n₁), view(τ, :, :, n₂), time)
end

times = 0.0:5days:12*30days

time = Node(0.0)

T★_continuous(t) = thirty_day_interpolate(T★, t)
τˣ_continuous(t) = thirty_day_interpolate(τˣ, t)
τʸ_continuous(t) = thirty_day_interpolate(τʸ, t)

T★_t = @lift T★_continuous($time)
τˣ_t = @lift τˣ_continuous($time)
τʸ_t = @lift τʸ_continuous($time)

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

title_str = @lift "Earth day = " * prettytime($time)
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, "annual_cycle_forcing.mp4", times, framerate=8) do tn
    time[] = tn
end

display(fig)
