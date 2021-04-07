using Printf
using Glob
using JLD2
using PyCall

using Oceananigans.Utils: prettytime

np = pyimport("numpy")
ma = pyimport("numpy.ma")
plt = pyimport("matplotlib.pyplot")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

function plot_cubed_sphere_tracer_field!(fig, ax, var, grid; transform, cmap, vmin, vmax)

    Nf = grid["faces"] |> keys |> length
    Nx = grid["faces/1/Nx"]
    Ny = grid["faces/1/Ny"]
    Hx = grid["faces/1/Hx"]
    Hy = grid["faces/1/Hy"]

    for face in 1:Nf
        λᶠᶠᵃ = grid["faces/$face/λᶠᶠᵃ"][1+Hx:Nx+2Hx, 1+Hy:Ny+2Hy]
        φᶠᶠᵃ = grid["faces/$face/φᶠᶠᵃ"][1+Hx:Nx+2Hx, 1+Hy:Ny+2Hy]

        var_face = var[face][:, :, 1]

        # Remove very specific problematic grid cells near λ = 180° on face 6 that mess up the plot.
        # May be related to https://github.com/SciTools/cartopy/issues/1151
        if face == 6
            for i in 1:Nx+1, j in 1:Ny+1
                if isapprox(λᶠᶠᵃ[i, j], -180, atol=15)
                    var_face[min(i, Nx), min(j, Ny)] = NaN
                end
            end
        end

        var_face_masked = ma.masked_where(np.isnan(var_face), var_face)

        pc = ax.pcolormesh(λᶠᶠᵃ, φᶠᶠᵃ, var_face_masked; transform, cmap, vmin, vmax)

        face == Nf && fig.colorbar(pc, ax=ax, shrink=0.6)

        ax.set_global()
    end

    return ax
end

function animate_tracer_advection(; face_number, α, projection=ccrs.Robinson())

    ## Extract data

    file = jldopen("tracer_advection_over_the_poles_face$(face_number)_alpha$α.jld2")

    iterations = parse.(Int, keys(file["timeseries/t"]))

    ## Makie movie of tracer field h

    for (n, i) in enumerate(iterations)
        @info "Plotting face $face_number iteration $i/$(iterations[end]) (frame $n/$(length(iterations)))..."

        h = file["timeseries/h/$i"]

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        plot_cubed_sphere_tracer_field!(fig, ax, h, file["grid"], transform=ccrs.PlateCarree(), cmap=cmocean.cm.dense, vmin=0, vmax=1000)

        t = prettytime(file["timeseries/t/$i"])
        ax.set_title("Tracer advection from face $face_number at α = $(α)°: tracer at t = $t")

        filename = @sprintf("cubed_sphere_tracer_advection_face%d_alpha%d_%04d.png", face_number, α, n)
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    end

    close(file)

    filename_pattern = "cubed_sphere_tracer_advection_face$(face_number)_alpha$(α)_%04d.png"
    output_filename  = "cubed_sphere_tracer_advection_face$(face_number)_alpha$(α).mp4"

    # Need extra crop video filter in case we end up with odd number of pixels in width or height.
    # See: https://stackoverflow.com/a/29582287
    run(`ffmpeg -y -i $filename_pattern -c:v libx264 -vf "fps=15, crop=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p $output_filename`)

    [rm(f) for f in glob("tracer_avection_over_the_poles_face$(face_number)_*.png")]

    return nothing
end
