using Printf
using Glob
using JLD2
using PyCall
using Oceananigans.Utils: prettytime

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

function animate_surface_gravity_waves_on_cubed_sphere(face_number)

    projection = ccrs.Robinson()
    transform = ccrs.PlateCarree()

    ## Extract data

    file = jldopen("surface_gravity_waves_on_cubed_sphere_face$face_number.jld2")

    iterations = parse.(Int, keys(file["timeseries/t"]))

    Nf = file["grid/faces"] |> keys |> length
    Nx = file["grid/faces/1/Nx"]
    Ny = file["grid/faces/1/Ny"]
    Nz = file["grid/faces/1/Nz"]

    size_2d = (Nx, Ny * Nf)
    size_u = (Nx+1, Ny * Nf)
    size_v = (Nx, (Ny+1) * Nf)

    flatten_cubed_sphere(field, size) = reshape(cat(field..., dims=4), size)

    λ = flatten_cubed_sphere((file["grid/faces/$n/λᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)
    φ = flatten_cubed_sphere((file["grid/faces/$n/φᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)

    ## Plot!

    for (n, i) in enumerate(iterations)
        @info "Plotting face $face_number iteration $i/$(iterations[end]) (frame $n/$(length(iterations)))..."
        η = flatten_cubed_sphere(file["timeseries/η/$i"], size_2d)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        ax.scatter(λ, φ, c=η, transform=transform, cmap=cmocean.cm.balance, s=25, vmin=-0.01, vmax=0.01)

        filename = @sprintf("surface_gravity_waves_on_a_cubed_sphere_face%d_η_%04d.png", face_number, n)
        plt.savefig(filename, dpi=200)
        plt.close(fig)
    end

    close(file)

    run(`ffmpeg -y -i surface_gravity_waves_on_a_cubed_sphere_face$(face_number)_η_%04d.png -c:v libx264 -vf fps=10 -pix_fmt yuv420p surface_gravity_waves_face$(face_number).mp4`)

    [rm(f) for f in glob("surface_gravity_waves_on_a_cubed_sphere_face$(face_number)_η_*.png")]

    return nothing
end
