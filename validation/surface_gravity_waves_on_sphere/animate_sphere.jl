using Statistics
using JLD2
using Printf
using GLMakie

using Oceananigans.Utils: prettytime

file = jldopen("full_cubed_sphere_gravity_waves.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

# Azimuthal spherical coordinates:
# Convert to λ ∈ [0°, 360°]
# Convert to φ ∈ [0°, 180°] (0° at north pole)
geographic2x(λ, φ, r=1) = r * cosd(λ + 180) * sind(90 - φ)
geographic2y(λ, φ, r=1) = r * sind(λ + 180) * sind(90 - φ)
geographic2z(λ, φ, r=1) = r * cosd(90 - φ)

Nf = file["grid/faces"] |> keys |> length
Nx = file["grid/faces/1/Nx"]
Ny = file["grid/faces/1/Ny"]
Nz = file["grid/faces/1/Nz"]

#####
##### Plot η
#####

size_2d = (Nx, Ny * Nf)

flatten_cubed_sphere(field, size) = reshape(cat(field..., dims=4), size)

λ = flatten_cubed_sphere((file["grid/faces/$n/λᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)
φ = flatten_cubed_sphere((file["grid/faces/$n/φᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)

x = geographic2x.(λ, φ)
y = geographic2y.(λ, φ)
z = geographic2z.(λ, φ)

iter = Node(0)

plot_title = @lift @sprintf("Surface gravity waves on a cubed sphere: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

η = @lift flatten_cubed_sphere(file["timeseries/η/" * string($iter)], size_2d)

fig = Figure(resolution = (1920, 1080))
ax = fig[1, 1] = LScene(fig) # make plot area wider
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
sf = surface!(ax, x, y, z, color=η, colormap=:balance, colorrange=(-0.01, 0.01))

rotate_cam!(ax.scene, (3π/4, 0, 0))
zoom!(ax.scene, (0, 0, 0), 8, false)

cb1 = fig[1, 2] = Colorbar(fig, sf, label="sea surface height η′ (m)", width=30)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

record(fig, "surface_gravity_waves_on_a_cubed_sphere.mp4", iterations, framerate=15) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end

#####
##### Plot u
#####

# size_2d = (Nx+1, Ny * Nf)

# λ = flatten_cubed_sphere((file["grid/faces/$n/λᶠᶠᵃ"][2:34, 2:33] for n in 1:6), size_2d)
# φ = flatten_cubed_sphere((file["grid/faces/$n/φᶠᶠᵃ"][2:34, 2:33] for n in 1:6), size_2d)

# x = geographic2x.(λ, φ)
# y = geographic2y.(λ, φ)
# z = geographic2z.(λ, φ)

# iter = Node(0)

# plot_title = @lift @sprintf("Surface gravity waves on a cubed sphere: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

# u = @lift flatten_cubed_sphere(file["timeseries/u/" * string($iter)])

# fig = Figure(resolution = (1920, 1080))
# ax = fig[1, 1] = LScene(fig) # make plot area wider
# wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
# sf = surface!(ax, x, y, z, color=u, colormap=:balance, colorrange=(-1e-4, 1e-4))

# rotate_cam!(ax.scene, (3π/4, 0, 0))
# zoom!(ax.scene, (0, 0, 0), 8, false)

# cb1 = fig[1, 2] = Colorbar(fig, sf, label="u-velocity (m/s)", width=30)

# supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

# record(fig, "surface_gravity_waves_on_a_cubed_sphere_u.mp4", iterations[1:20], framerate=15) do i
#     @info "Animating iteration $i/$(iterations[end])..."
#     iter[] = i
# end

#####
##### Plot v
#####

# size_2d = (Nx, (Ny+1) * Nf)

# λ = flatten_cubed_sphere((file["grid/faces/$n/λᶠᶠᵃ"][2:33, 2:34] for n in 1:6), size_2d)
# φ = flatten_cubed_sphere((file["grid/faces/$n/φᶠᶠᵃ"][2:33, 2:34] for n in 1:6), size_2d)

# x = geographic2x.(λ, φ)
# y = geographic2y.(λ, φ)
# z = geographic2z.(λ, φ)

# iter = Node(0)

# plot_title = @lift @sprintf("Surface gravity waves on a cubed sphere: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

# v = @lift flatten_cubed_sphere(file["timeseries/v/" * string($iter)])

# fig = Figure(resolution = (1920, 1080))
# ax = fig[1, 1] = LScene(fig) # make plot area wider
# wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
# sf = surface!(ax, x, y, z, color=v, colormap=:balance, colorrange=(-1e-4, 1e-4))

# rotate_cam!(ax.scene, (3π/4, 0, 0))
# zoom!(ax.scene, (0, 0, 0), 8, false)

# cb1 = fig[1, 2] = Colorbar(fig, sf, label="v-velocity (m/s)", width=30)

# supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

# record(fig, "surface_gravity_waves_on_a_cubed_sphere_v.mp4", iterations[1:20], framerate=15) do i
#     @info "Animating iteration $i/$(iterations[end])..."
#     iter[] = i
# end

close(file)
