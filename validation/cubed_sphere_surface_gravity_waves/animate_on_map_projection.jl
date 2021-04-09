using Printf
using Glob
using JLD2
using PyCall

using Oceananigans.Utils: prettytime

np = pyimport("numpy")
ma = pyimport("numpy.ma")
plt = pyimport("matplotlib.pyplot")
mticker = pyimport("matplotlib.ticker")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

function plot_cubed_sphere_tracer_field!(fig, ax, var, grid; add_colorbar, transform, cmap, vmin, vmax)

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
        # Not needed for Orthographic or NearsidePerspective projection so let's use those.
        # if face == 6
        #     for i in 1:Nx+1, j in 1:Ny+1
        #         if isapprox(λᶠᶠᵃ[i, j], -180, atol=15)
        #             var_face[min(i, Nx), min(j, Ny)] = NaN
        #         end
        #     end
        # end

        var_face_masked = ma.masked_where(np.isnan(var_face), var_face)

        im = ax.pcolormesh(λᶠᶠᵃ, φᶠᶠᵃ, var_face_masked; transform, cmap, vmin, vmax)

        # Add colorbar below all the subplots.
        if add_colorbar && face == Nf
            ax_cbar = fig.add_axes([0.25, 0.1, 0.5, 0.02])
            fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        end

        ax.set_global()
    end

    return ax
end

function animate_surface_gravity_waves(; face_number, projections=[ccrs.Robinson()])

    ## Extract data

    file = jldopen("cubed_sphere_surface_gravity_waves_face$face_number.jld2")

    iterations = parse.(Int, keys(file["timeseries/t"]))

    ## Makie movie of tracer field h

    for (n, i) in enumerate(iterations)
        @info "Plotting face $face_number iteration $i/$(iterations[end]) (frame $n/$(length(iterations)))..."

        η = file["timeseries/η/$i"]

        t = prettytime(file["timeseries/t/$i"])
        plot_title = "Surface gravity waves from face $face_number: η(λ, φ) at t = $t"

        fig = plt.figure(figsize=(16, 9))
        n_subplots = length(projections)
        subplot_kwargs = (transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance, vmin=-0.01, vmax=0.01)

        for (n, projection) in enumerate(projections)
            ax = fig.add_subplot(1, n_subplots, n, projection=projection)
            plot_cubed_sphere_tracer_field!(fig, ax, η, file["grid"]; add_colorbar = (n == n_subplots), subplot_kwargs...)
            n_subplots == 1 && ax.set_title(plot_title)

            gl = ax.gridlines(color="gray", alpha=0.5, linestyle="--")
            gl.xlocator = mticker.FixedLocator(-180:30:180)
            gl.ylocator = mticker.FixedLocator(-80:20:80)
        end

        n_subplots > 1 && fig.suptitle(plot_title, y=0.85)

        filename = @sprintf("cubed_sphere_surface_gravity_waves_face%d_%04d.png", face_number, n)
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    end

    close(file)

    filename_pattern = "cubed_sphere_surface_gravity_waves_face$(face_number)_%04d.png"
    output_filename  = "cubed_sphere_surface_gravity_waves_face$(face_number).mp4"

    # Need extra crop video filter in case we end up with odd number of pixels in width or height.
    # See: https://stackoverflow.com/a/29582287
    run(`ffmpeg -y -i $filename_pattern -c:v libx264 -vf "fps=15, crop=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p $output_filename`)

    [rm(f) for f in glob("cubed_sphere_surface_gravity_waves_face$(face_number)_*.png")]

    return nothing
end
