using JLD2
using Printf
using GLMakie

Nx = 128
Ny = 60

bathymetry_path = "bathy_128x60var4.bin"
east_west_stress_path = "off_TAUXvar1.bin"
north_south_stress_path = "off_TAUY.bin"
sea_surface_temperature_path="sst25_128x60x12.bin"

Nbytes = sizeof(Float32) * Nx * Ny
bathymetry = reshape(bswap.(reinterpret(Float32, read(bathymetry_path, Nbytes))), (Nx, Ny))

τˣ = reshape(bswap.(reinterpret(Float32, read(east_west_stress_path,  12Nbytes))), (Nx, Ny, 12))
τʸ = reshape(bswap.(reinterpret(Float32, read(north_south_stress_path, 12Nbytes))), (Nx, Ny, 12))
target_sea_surface_temperature = reshape(bswap.(reinterpret(Float32, read(sea_surface_temperature_path, 12Nbytes))), (Nx, Ny, 12))

τˣ = τˣ[:, :, 1]
τʸ = τʸ[:, :, 1]
target_sea_surface_temperature = target_sea_surface_temperature[:, :, 1] 

# bathymetry = Array{Float64, 2}(bathymetry)
# τˣ = Array{Float64, 2}(τˣ)
# τʸ = Array{Float64, 2}(τʸ)
# target_sea_surface_temperature = Array{Float64, 2}(target_sea_surface_temperature)

bathymetry_file = jldopen("earth_bathymetry_128_60.jld2", "a+")
bathymetry_file["bathymetry"] = bathymetry
close(bathymetry_file)

println("Bathymetry: min= ", minimum(bathymetry)," , max= ", maximum(bathymetry))
println("τˣ: min= ", minimum(τˣ)," , max= ", maximum(τˣ))
println("τʸ: min= ", minimum(τʸ)," , max= ", maximum(τʸ))
println("SST★: min= ", minimum(target_sea_surface_temperature)," , max= ",maximum(target_sea_surface_temperature))

fig = Figure(resolution = (1920, 1080))
ax = fig[1, 1] = LScene(fig, title = "Bathymetry")
heatmap!(ax, bathymetry)

ax = fig[2, 1] = LScene(fig, title = "Bathymetry")
heatmap!(ax, target_sea_surface_temperature)

ax = fig[1, 2] = LScene(fig, title = "East-west wind stress")
heatmap!(ax, τˣ)

ax = fig[2, 2] = LScene(fig, title = "North-south wind stress")
heatmap!(ax, τʸ)

display(fig)

