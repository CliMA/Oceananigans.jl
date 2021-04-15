using Printf
using Glob
using JLD2
using PyCall

plt = pyimport("matplotlib.pyplot")
gridspec = pyimport("matplotlib.gridspec")
ccrs = pyimport("cartopy.crs")
cmocean = pyimport("cmocean")

## Extract data

file = jldopen("tracer_advection_over_the_poles.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

Nf = file["grid/faces"] |> keys |> length
Nx = file["grid/faces/1/Nx"]
Ny = file["grid/faces/1/Ny"]
Nz = file["grid/faces/1/Nz"]

## Plot!

for (n, i) in enumerate(iterations)
    @info "Plotting iteration $i/$(iterations[end]) (frame $n/$(length(iterations)))..."

    # fig = plt.figure(constrained_layout=true)
    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 4), (2, 0))
    ax2 = plt.subplot2grid((3, 4), (2, 1))
    ax3 = plt.subplot2grid((3, 4), (1, 1))
    ax4 = plt.subplot2grid((3, 4), (1, 2))
    ax5 = plt.subplot2grid((3, 4), (0, 2))
    ax6 = plt.subplot2grid((3, 4), (0, 3))

    ax1.set_title("face 1")
    ax1.pcolormesh(file["timeseries/h/$i"][1][:, :, 1]', cmap=cmocean.cm.thermal, vmin=0, vmax=1000)
    ax1.get_xaxis().set_visible(false)
    ax1.get_yaxis().set_visible(false)

    ax2.set_title("face 2")
    ax2.pcolormesh(file["timeseries/h/$i"][2][:, :, 1]', cmap=cmocean.cm.thermal, vmin=0, vmax=1000)
    ax2.get_xaxis().set_visible(false)
    ax2.get_yaxis().set_visible(false)

    ax3.set_title("face 3")
    ax3.pcolormesh(file["timeseries/h/$i"][3][:, :, 1]', cmap=cmocean.cm.thermal, vmin=0, vmax=1000)
    ax3.get_xaxis().set_visible(false)
    ax3.get_yaxis().set_visible(false)

    ax4.set_title("face 4")
    ax4.pcolormesh(file["timeseries/h/$i"][4][:, :, 1]', cmap=cmocean.cm.thermal, vmin=0, vmax=1000)
    ax4.get_xaxis().set_visible(false)
    ax4.get_yaxis().set_visible(false)

    ax5.set_title("face 5")
    ax5.pcolormesh(file["timeseries/h/$i"][5][:, :, 1]', cmap=cmocean.cm.thermal, vmin=0, vmax=1000)
    ax5.get_xaxis().set_visible(false)
    ax5.get_yaxis().set_visible(false)

    ax6.set_title("face 6")
    ax6.pcolormesh(file["timeseries/h/$i"][6][:, :, 1]', cmap=cmocean.cm.thermal, vmin=0, vmax=1000)
    ax6.get_xaxis().set_visible(false)
    ax6.get_yaxis().set_visible(false)

    filename = @sprintf("tracer_avection_over_the_poles_%04d.png", n)
    plt.savefig(filename, dpi=200)
    plt.close(fig)
end

close(file)

run(`ffmpeg -y -i tracer_avection_over_the_poles_%04d.png -c:v libx264 -vf fps=10 -pix_fmt yuv420p out.mp4`)

# [rm(f) for f in glob("*.png")];
