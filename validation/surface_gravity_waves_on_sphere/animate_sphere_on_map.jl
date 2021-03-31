using Printf
using JLD2
using PyCall

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

## Extract data

file = jldopen("full_cubed_sphere_gravity_waves.jld2")

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

# projection = ccrs.Orthographic(central_longitude=270)
projection = ccrs.Robinson()
transform = ccrs.PlateCarree()

for (n, i) in enumerate(iterations)
    @info "Plotting iteration $i/$(iterations[end])..."
    η = flatten_cubed_sphere(file["timeseries/η/$i"], size_2d)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.scatter(λ, φ, c=η, transform=transform, cmap=cmocean.cm.balance, s=25, vmin=-0.01, vmax=0.01)
    # ax.pcolormesh(λ, φ, η, transform=transform, cmap="seismic", vmin=-0.01, vmax=0.01)
    # ax.contourf(λ, φ, η, transform=transform, cmap="seismic")

    # ax.legend(loc="lower right")
    # ax.coastlines(resolution="50m")
    # ax.set_global()

    # plt.show()
    filename = @sprintf("surface_gravity_waves_on_a_cubed_sphere_η_%04d.png", n)
    plt.savefig(filename, dpi=200)
    plt.close(fig)
end
