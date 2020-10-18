using NCDatasets
using Makie

#####
##### Read data
#####

ds = NCDataset("dry_convection.nc")

ρe₀ = Array(ds["ρe₀"]);
ρe  = Array(ds["ρe"]);
ρe′ = similar(ρe);

for n in 1:length(ds["time"])
    @. ρe′[:, :, :, n] = ρe[:, :, :, n] - ρe₀;
end

#####
##### Plot!
#####

α(ξ) = ξ  # Opacity/alpha along the cmap (0 <= ξ <= 1)

cmap_rgb = to_colormap(:thermal, 100)
A = α.(range(0, 1, length=length(cmap_rgb)))
cmap_rgba = RGBAf0.(cmap_rgb, A)

time_index = Node(1)

get_data(n) = ρe′[:, :, :, n]
data = @lift get_data($time_index)

function make_movie(scene)    
    n_frames = length(ds["time"])
    θ = 2π / 360

    record(scene, "dry_convection_oceananigans_makie.mp4", 1:n_frames, framerate=30) do n
        @info "frame $n/$n_frames"
        time_index[] = n
        n == 1 && zoom!(scene, (0, 0, 0), -1.25, false)
        rotate_cam!(scene, (θ, 0, 0))
    end
end

N = size(get_data(1))
scene = volume(0..N[1], 0..N[2], 0..N[3], data, colorrange=(0, 400), colormap=cmap_rgba, algorithm=:absorption, absorption=10.0f0,
               backgroundcolor=cmap_rgb[2], show_axis=false, resolution = (1920, 1080))

make_movie(scene)

