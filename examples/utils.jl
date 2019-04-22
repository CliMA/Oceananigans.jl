using Plots
using Oceananigans

using Statistics

"""
    make_vertical_slice_movie(model::Model, nc_writer::NetCDFOutputWriter,
                              var_name, Nt, Δt, var_offset=0, slice_idx=1)

Make a movie of a vertical slice produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀). A `slice_idx` can be specified to select
the index of the y-slice to be plotted (useful when plotting vertical slices
from a 3D model, it should be set to 1 for 2D xz-slice models).
"""
function make_vertical_slice_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0, slice_idx=1)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.contour(model.grid.xC, reverse(model.grid.zC), rotl90(var[:, slice_idx, :] .- 283),
                      fill=true, levels=9, linewidth=0, color=:balance,
                      clims=(-0.011, 0.011), title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
        # Plots.heatmap(model.grid.xC, model.grid.zC, rotl90(var[:, slice_idx, :]) .- var_offset,
        #               color=:balance, clims=(-0.01, 0.01), title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

"""
    make_horizontal_slice_movie(model::Model, nc_writer::NetCDFOutputWriter,
                                var_name, Nt, Δt, var_offset=0)

Make a movie of a horizontal slice produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀).
"""
function make_horizontal_slice_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.heatmap(model.grid.xC, model.grid.yC, var[:, :, 1] .- var_offset,
                      color=:balance, clims=(-0.01, 0.01),
                      title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

"""
    make_vertical_profile_movie(model::Model, nc_writer::NetCDFOutputWriter,
                                var_name, Nt, Δt, var_offset=0)

Make a movie of a vertical profile produced by `model` with output being saved by
`nc_writer`. The variable name `var_name` can be either of "u", "v", "w", "T",
or "S". `Nt` is the number of model iterations (or time steps) taken and ``Δt`
is the time step. A plotting offset `var_offset` can be specified to be
subtracted from the data before plotting (useful for plotting e.g. small
temperature perturbations around T₀).
"""
function make_vertical_profile_movie(model::Model, nc_writer::NetCDFOutputWriter, var_name, Nt, Δt, var_offset=0)
    freq = nc_writer.output_frequency
    N_frames = Int(Nt/freq)

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")
        var = read_output(nc_writer, var_name, freq*n)
        Plots.plot(var[1, 1, :] .- var_offset, model.grid.zC,
                   title="t=$(freq*n*Δt) s ($(round(freq*n*Δt/86400; digits=2)) days)")
    end

    mp4(animation, nc_writer.filename_prefix * "$(round(Int, time())).mp4", fps=30)
end

using NetCDF

function make_avg_temperature_profile_movie()
    Nt, dt = 86400, 0.5
    freq = 3600
    N_frames = Int(Nt/freq)
    filename_prefix = "convection"
    var_offset = 273.15

    Nz, Lz = 128, 100
    dz = Lz/Nz
    zC = -dz/2:-dz:-Lz

    print("Producing movie... ($N_frames frames)\n")
    Plots.gr(dpi=150)

    animation = @animate for n in 0:N_frames
        print("\rframe = $n / $N_frames   ")

        filepath = filename_prefix * lpad(freq*n, 9, "0") * ".nc"
        field_data = ncread(filepath, "T")
        ncclose(filepath)

        T_profile = mean(field_data; dims=[1,2])

        Plots.plot(reshape(T_profile, Nz) .- var_offset, zC,
                   title="t=$(freq*n*dt) s ($(round(freq*n*dt/86400; digits=2)) days)")
    end

    mp4(animation, filename_prefix * "$(round(Int, time())).mp4", fps=30)
end
