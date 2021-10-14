using Printf
using GLMakie

Nx = 128
Ny = 60

bathymetry_path = "bathy_128x60var4.bin"
east_west_stress_path = "off_TAUXvar1.bin"
north_south_stress_path = "off_TAUY.bin"
sea_surface_temperature_path="sst25_128x60x12.bin"

bathymetry = reshape(bswap.(reinterpret(Float32, read(bathymetry_path,sizeof(Float32)*Nx*Ny))), (Nx, Ny))
τˣ = reshape(bswap.(reinterpret(Float32, read(east_west_stress_path,sizeof(Float32)*Nx*Ny))), (Nx, Ny))
τʸ = reshape(bswap.(reinterpret(Float32, read(north_south_stress_path,sizeof(Float32)*Nx*Ny))), (Nx, Ny))
target_sea_surface_temperature = reshape(bswap.(reinterpret(Float32, read(sea_surface_temperature_path, sizeof(Float32)*Nx*Ny))), (Nx, Ny))

bathymetry = Array{Float64, 2}(bathymetry)
τˣ = Array{Float64, 2}(τˣ)
τʸ = Array{Float64, 2}(τʸ)
target_sea_surface_temperature = Array{Float64, 2}(target_sea_surface_temperature)

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

