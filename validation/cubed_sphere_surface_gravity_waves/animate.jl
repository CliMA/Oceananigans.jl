using Statistics
using JLD2
using Printf
using GLMakie

# using Oceananigans.Grids
# using Oceananigans.Utils: prettytime, hours, day, days, years

function geographic2cartesian(λ, φ, r=1)
    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    φ_azimuthal = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

    x = @. r * cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. r * sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. r * cosd(φ_azimuthal)

    return x, y, z
end

ds_cs = jldopen("cubed_sphere_face_waves.jld2")

λᶜᶜᵃ = ds_cs["grid/λᶜᶜᵃ"][2:33, 2:33]
φᶜᶜᵃ = ds_cs["grid/φᶜᶜᵃ"][2:33, 2:33]

xc, yc, zc = geographic2cartesian(λᶜᶜᵃ, φᶜᶜᵃ)

iterations = parse.(Int, keys(ds_cs["timeseries/t"]))

iter = Node(0)

plot_title = @lift @sprintf("η′ on a cubed sphere face: time = %s", prettytime(ds_cs["timeseries/t/" * string($iter)]))

η = @lift ds_cs["timeseries/η/" * string($iter)][:, :, 1]

fig = Figure(resolution = (1920, 1080))

ax = fig[1, 1] = LScene(fig) # make plot area wider
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
sf = surface!(ax, xc, yc, zc, color=η, colormap=:balance, colorrange=(-0.01, 0.01))
rotate_cam!(ax.scene, (3π/4, π/6, 0))
zoom!(ax.scene, (0, 0, 0), 5, false)
# fig[2, 2 + 3*(n-1)] = Label(fig, statenames[n], textsize = 50) # put names in center

cb1 = fig[1, 2] = Colorbar(fig, sf, label="η′", width=30)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

record(fig, "cubed_sphere_waves.mp4", iterations, framerate=60) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end

close(ds_cs)
