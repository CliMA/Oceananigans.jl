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

function plot_cubed_sphere_field!(fig, ax, grid, var, loc; add_colorbar, kwargs...)

    Nf = grid["faces"] |> keys |> length
    Nx = grid["faces/1/Nx"]
    Ny = grid["faces/1/Ny"]
    Hx = grid["faces/1/Hx"]
    Hy = grid["faces/1/Hy"]

    ii = 1+Hx:Nx+2Hx
    jj = 1+Hy:Ny+2Hy

    for face in 1:Nf

        # λ and φ will be the cell boundaries (so there are N+1 of them) for pcolormesh.
        # Technically these coordinates are wrong for anything but (Center, Center) but
        # the plots look wonky otherwise so we'll leave these here for now...
        λ = grid["faces/$face/λᶠᶠᵃ"][ii, jj]
        φ = grid["faces/$face/φᶠᶠᵃ"][ii, jj]

        im = ax.pcolormesh(λ, φ, var[face][:, :, 1]; kwargs...)

        # Add colorbar below all the subplots.
        if add_colorbar && face == Nf
            ax_cbar = fig.add_axes([0.25, 0.1, 0.5, 0.02])
            fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        end

        ax.set_global()
    end

    return ax
end

function animate_eddying_aquaplanet(; projections=[ccrs.Robinson()])

    ## Extract data

    file = jldopen("cubed_sphere_eddying_aquaplanet.jld2")

    iterations = parse.(Int, keys(file["timeseries/t"]))

    ## Makie movie of tracer field h

    for (n, i) in enumerate(iterations)
        @info "Plotting iteration $i/$(iterations[end]) (frame $n/$(length(iterations)))..."

        u = file["timeseries/u/$i"]
        v = file["timeseries/v/$i"]
        η = file["timeseries/η/$i"]
        ζ = file["timeseries/ζ/$i"]

        t = prettytime(file["timeseries/t/$i"])
        plot_title = "Eddying aquaplanet: ζ(λ, φ) at t = $t"

        fig = plt.figure(figsize=(16, 9))
        n_subplots = length(projections)
        subplot_kwargs = (transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance)

        for (n, projection) in enumerate(projections)
            ax = fig.add_subplot(1, n_subplots, n, projection=projection)

            # plot_cubed_sphere_field!(fig, ax, file["grid"], η, (Center, Center); add_colorbar = (n == n_subplots), subplot_kwargs...)
            plot_cubed_sphere_field!(fig, ax, file["grid"], ζ, (Face, Face); add_colorbar = (n == n_subplots), vmin=-1e-7, vmax=1e-7, subplot_kwargs...)

            n_subplots == 1 && ax.set_title(plot_title)

            gl = ax.gridlines(color="gray", alpha=0.5, linestyle="--")
            gl.xlocator = mticker.FixedLocator(-180:30:180)
            gl.ylocator = mticker.FixedLocator(-80:20:80)
        end

        n_subplots > 1 && fig.suptitle(plot_title, y=0.85)

        filename = @sprintf("cubed_sphere_eddying_aquaplanet_%04d.png", n)
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    end

    close(file)

    filename_pattern = "cubed_sphere_eddying_aquaplanet_%04d.png"
    output_filename  = "cubed_sphere_eddying_aquaplanet.mp4"

    # Need extra crop video filter in case we end up with odd number of pixels in width or height.
    # See: https://stackoverflow.com/a/29582287
    run(`ffmpeg -y -i $filename_pattern -c:v libx264 -vf "fps=10, crop=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p $output_filename`)

    [rm(f) for f in glob("cubed_sphere_eddying_aquaplanet_*.png")]

    return nothing
end
