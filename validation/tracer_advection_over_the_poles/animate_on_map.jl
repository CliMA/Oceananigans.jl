using Printf
using Glob
using JLD2
using PyCall

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

## Extract data

file = jldopen("tracer_advection_over_the_poles.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

Nf = file["grid/faces"] |> keys |> length
Nx = file["grid/faces/1/Nx"]
Ny = file["grid/faces/1/Ny"]
Nz = file["grid/faces/1/Nz"]

size_2d = (Nx, Ny * Nf)

flatten_cubed_sphere(field, size) = reshape(cat(field..., dims=4), size)

λ = flatten_cubed_sphere((file["grid/faces/$n/λᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)
φ = flatten_cubed_sphere((file["grid/faces/$n/φᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)

## Plot!

h_end = flatten_cubed_sphere(file["timeseries/h/$(iterations[end])"], size_2d)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.scatter(λ, φ, c=h_end, transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance, s=25, vmin=0, vmax=1000)
plt.savefig("h_end.png", dpi=100)
plt.close(fig)
@info "Saved: h_end.png"

# projection = ccrs.Orthographic(central_longitude=270)
projection = ccrs.Robinson()
transform = ccrs.PlateCarree()

for (n, i) in enumerate(iterations)
    @info "Plotting iteration $i/$(iterations[end])..."
    h = flatten_cubed_sphere(file["timeseries/h/$i"], size_2d)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.scatter(λ, φ, c=h, transform=transform, cmap=cmocean.cm.balance, s=25, vmin=0, vmax=1000)
    # ax.pcolormesh(λ, φ, η, transform=transform, cmap="seismic", vmin=-0.01, vmax=0.01)
    # ax.contourf(λ, φ, η, transform=transform, cmap="seismic")

    # ax.legend(loc="lower right")
    # ax.coastlines(resolution="50m")
    # ax.set_global()

    # plt.show()
    filename = @sprintf("tracer_avection_over_the_poles_%04d.png", n)
    plt.savefig(filename, dpi=200)
    plt.close(fig)
end

close(file)

run(`ffmpeg -y -i tracer_avection_over_the_poles_%04d.png -c:v libx264 -vf fps=10 -pix_fmt yuv420p out.mp4`)

[rm(f) for f in glob("*.png")];
