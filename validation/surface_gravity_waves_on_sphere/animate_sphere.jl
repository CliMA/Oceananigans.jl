using Statistics
using JLD2
using Printf
using GLMakie

file = jldopen("full_cubed_sphere_gravity_waves.jld2")

# Azimuthal spherical coordinates:
# Convert to λ ∈ [0°, 360°]
# Convert to φ ∈ [0°, 180°] (0° at north pole)
geographic2x(λ, φ, r=1) = r * cosd(λ + 180) * sind(90 - φ)
geographic2y(λ, φ, r=1) = r * sind(λ + 180) * sind(90 - φ)
geographic2z(λ, φ, r=1) = r * cosd(90 - φ)

# Need 2D arrays for `surface!`

# Nf = file["grid/faces"] |> keys |> length
# Nx = file["grid/faces/1/Nx"]
# Ny = file["grid/faces/1/Ny"]
# Nz = file["grid/faces/1/Nz"]

Nx, Ny, Nz, Nf = size(grid)
size_2d = (Nx, Ny * Nf)

field = model.free_surface.η

λ = reshape(λnodes(field), size_2d)
φ = reshape(φnodes(field), size_2d)

x = geographic2x.(λ, φ)
y = geographic2y.(λ, φ)
z = geographic2z.(λ, φ)

field_vals = cat(file["timeseries/η/72"]..., dims=4)
field_val = reshape(field_vals, size_2d)

fig = Figure(resolution = (1920, 1080))
ax = fig[1, 1] = LScene(fig) # make plot area wider
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
surface!(ax, x, y, z, color=field_val, colormap=:balance, colorrange=(-0.01, 0.01))

close(file)
