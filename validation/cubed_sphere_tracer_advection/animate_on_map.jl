using Printf
using Glob
using JLD2
using PyCall
using Oceananigans.Utils: prettytime

plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

function animate_tracer_advection(face_number)

    projection = ccrs.Robinson()
    transform = ccrs.PlateCarree()

    ## Extract data

    file = jldopen("tracer_advection_over_the_poles_face$face_number.jld2")

    iterations = parse.(Int, keys(file["timeseries/t"]))

    Nf = file["grid/faces"] |> keys |> length
    Nx = file["grid/faces/1/Nx"]
    Ny = file["grid/faces/1/Ny"]
    Nz = file["grid/faces/1/Nz"]

    size_2d = (Nx, Ny * Nf)
    size_u = (Nx+1, Ny * Nf)
    size_v = (Nx, (Ny+1) * Nf)

    flatten_cubed_sphere(field, size) = reshape(cat(field..., dims=4), size)

    # FIXME: These are not correct. Grid does not actually have ᶠᶜᵃ and ᶜᶠᵃ right now.
    λu = flatten_cubed_sphere((file["grid/faces/$n/λᶠᶠᵃ"][2:34, 2:33] for n in 1:6), size_u)
    φu = flatten_cubed_sphere((file["grid/faces/$n/φᶠᶠᵃ"][2:34, 2:33] for n in 1:6), size_u)

    λv = flatten_cubed_sphere((file["grid/faces/$n/λᶠᶠᵃ"][2:33, 2:34] for n in 1:6), size_v)
    φv = flatten_cubed_sphere((file["grid/faces/$n/φᶠᶠᵃ"][2:33, 2:34] for n in 1:6), size_v)

    ## Plot u and v

    u = flatten_cubed_sphere(file["timeseries/u/$(iterations[end])"], size_u)
    v = flatten_cubed_sphere(file["timeseries/v/$(iterations[end])"], size_v)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    sc = ax.scatter(λu, φu, c=u, transform=transform, cmap=cmocean.cm.balance, s=25, vmin=-40, vmax=40)
    fig.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title("Tracer advection from face $face_number: u-velocity")
    filename = "tracer_advection_u_velocity_face$face_number.png"
    plt.savefig(filename, dpi=100)
    @info "Saved: $filename"
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    sc = ax.scatter(λv, φv, c=v, transform=transform, cmap=cmocean.cm.balance, s=25, vmin=-40, vmax=40)
    fig.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_title("Tracer advection from face $face_number: v-velocity")
    filename = "tracer_advection_v_velocity_face$face_number.png"
    plt.savefig(filename, dpi=100)
    @info "Saved: $filename"
    plt.close(fig)

    ## Makie movie of h

    λᶜᶜᵃ = flatten_cubed_sphere((file["grid/faces/$n/λᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)
    φᶜᶜᵃ = flatten_cubed_sphere((file["grid/faces/$n/φᶜᶜᵃ"][2:33, 2:33] for n in 1:6), size_2d)

    for (n, i) in enumerate(iterations)
        @info "Plotting face $face_number iteration $i/$(iterations[end]) (frame $n/$(length(iterations)))..."
        h = flatten_cubed_sphere(file["timeseries/h/$i"], size_2d)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        sc = ax.scatter(λᶜᶜᵃ, φᶜᶜᵃ, c=h, transform=transform, cmap=cmocean.cm.balance, s=25, vmin=0, vmax=1000)
        fig.colorbar(sc, ax=ax, shrink=0.6)

        t = prettytime(file["timeseries/t/$i"])
        ax.set_title("Tracer advection from face $face_number: tracer at t = $t")

        filename = @sprintf("tracer_avection_over_the_poles_face%d_%04d.png", face_number, n)
        plt.savefig(filename, dpi=100)
        plt.close(fig)
    end

    close(file)

    run(`ffmpeg -y -i tracer_avection_over_the_poles_face$(face_number)_%04d.png -c:v libx264 -vf fps=10 -pix_fmt yuv420p tracer_advection_face$(face_number).mp4`)

    [rm(f) for f in glob("tracer_avection_over_the_poles_face$(face_number)_*.png")]

    return nothing
end
