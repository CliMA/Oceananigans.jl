using Printf
using Glob
using JLD2
using PyCall

using Oceananigans: Face, Center
using Oceananigans.Utils: prettytime

plt = pyimport("matplotlib.pyplot")
cmocean = pyimport("cmocean")

plot_indices(N, i, ::Center) = (i-1) * N + 1 : i * N
plot_indices(N, i, ::Face)   = (i-1) * (N+1) + 1 : i * (N+1)

function x_plot_indices(Nx, f, loc_x)
    f == 1   && return plot_indices(Nx, 1, loc_x)
    f in 2:3 && return plot_indices(Nx, 2, loc_x)
    f in 4:5 && return plot_indices(Nx, 3, loc_x)
    f == 6   && return plot_indices(Nx, 4, loc_x)
end

function y_plot_indices(Ny, f, loc_y)
    f in 1:2 && return plot_indices(Ny, 1, loc_y)
    f in 3:4 && return plot_indices(Ny, 2, loc_y)
    f in 5:6 && return plot_indices(Ny, 3, loc_y)
end

function animate_faces(filepath, var, loc; filename_pattern, cmap, vmin, vmax, iters=nothing, make_movie=true, delete_pngs=false)

    ## Extract data

    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))

    Nf = file["grid/faces"] |> keys |> length
    Nx = file["grid/faces/1/Nx"]
    Ny = file["grid/faces/1/Ny"]

    ## Plot!

    iterations_to_plot = isnothing(iters) ? iterations : iterations[iters]

    for (n, i) in enumerate(iterations_to_plot)
        @info "Plotting iteration $i/$(iterations_to_plot[end]) (frame $n/$(length(iterations_to_plot)))..."

        fig, ax = plt.subplots(figsize=(16, 9))

        for f in 1:Nf
            data = file["timeseries/$var/$i"][f][:, :, 1]
            is = x_plot_indices(Nx, f, loc[1])
            js = y_plot_indices(Ny, f, loc[2])

            # For pcolormesh we want to specify the cell boundaries
            # so we add an extra grid point at the end. Otherwise
            # we get gaps between the faces.
            is = is.start : is.stop+1
            js = js.start : js.stop+1

            pcm = ax.pcolormesh(is, js, data'; cmap, vmin, vmax)

            f == Nf && fig.colorbar(pcm, ax=ax)
        end

        t = prettytime(file["timeseries/t/$i"])
        ax.set_title("Cubed sphere $var(λ, φ) @ iteration $i ($t)")
        ax.set_aspect("equal")

        filename = @sprintf("%s_%04d.png", filename_pattern, n)
        plt.savefig(filename, dpi=200)
        plt.close(fig)
    end

    close(file)

    if make_movie
        png_filename_pattern = "$(filename_pattern)_%04d.png"
        mp4_filename = "$filename_pattern.mp4"
        run(`ffmpeg -y -i $png_filename_pattern -c:v libx264 -vf fps=10 -pix_fmt yuv420p $mp4_filename`)
    end

    if delete_pngs
        [rm(f) for f in glob("$filename_pattern*.png")]
    end

    return nothing
end

function animate_eddying_aquaplanet_faces()
    filepath = "cubed_sphere_eddying_aquaplanet.jld2"

    animate_faces("cubed_sphere_eddying_aquaplanet.jld2", "u", (Center(), Center()), filename_pattern = "eddying_aquaplanet_u", cmap = cmocean.cm.balance, vmin = -1, vmax = +1, make_movie = true, delete_pngs = true)

    for u in ("u", "v")
        animate_faces(filepath, u, (Center(), Center()),
            filename_pattern = "eddying_aquaplanet_$u",
                        cmap = cmocean.cm.balance,
                        vmin = -1,
                        vmax = +1,
                  make_movie = true,
                 delete_pngs = true
        )
    end

    animate_faces(filepath, "η", (Center(), Center()),
        filename_pattern = "eddying_aquaplanet_eta",
                    cmap = cmocean.cm.balance,
                    vmin = -0.1,
                    vmax = +0.1,
              make_movie = false,
             delete_pngs = false
    )

    # TODO: Fix (Face(), Face()) plotting by saving all (N+1)×(N+1) values.
    animate_faces(filepath, "ζ", (Center(), Center()),
        filename_pattern = "eddying_aquaplanet_vorticity",
                 # iters = 1:6:634,
                    cmap = cmocean.cm.balance,
                    vmin = -1e-5,
                    vmax = +1e-5,
              make_movie = true,
             delete_pngs = true
    )
end
