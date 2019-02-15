using Plots
using Oceananigans

function deep_convection_2d()
    Nx, Ny, Nz = 100, 100, 50
    Lx, Ly, Lz = 2000, 2000, 1000
    Nt, Δt = 50, 20

    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz), :gpu, Float32)
    @. model.forcings.FT.data[Int(Nx/10):Int(9Nx/10), Int(Ny/10):Int(9Ny/10), 1] = -0.25e-5 + 0.5e-6*rand()

    field_writer = FieldWriter(".", "deep_convection_2d", 50,
                               [model.velocities.u, model.tracers.T],
                               ["u", "T"])

    push!(model.output_writers, field_writer)

    Tx, Ty = 16, 16  # Threads per block
    Bx, By, Bz = Int(model.grid.Nx/Tx), Int(model.grid.Ny/Ty), Nz  # Blocks in grid.

    println("Threads per block: ($Tx, $Ty)")
    println("Blocks in grid:    ($Bx, $By, $Bz)")

    @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel!(Val(:GPU), model; Nt=Nt, Δt=Δt)

    # time_step!(model; Nt=Nt, Δt=Δt)
    # make_temperature_movie(model, field_writer)
end

function make_temperature_movie(model::Model, fw::FieldWriter)
    n_frames = Int(model.clock.time_step / fw.output_frequency)

    xC, zC = model.grid.xC, model.grid.zC
    Δt = 20

    print("Creating temperature movie... ($n_frames frames)\n")

    Plots.gr()
    movie = @animate for tidx in 0:n_frames
        print("\rframe = $tidx / $n_frames   ")
        temperature_xz = read_output(model, fw, "T", tidx*fw.output_frequency*Δt)
        Plots.heatmap(xC, zC, rotl90(temperature_xz[:, 1, :]) .- 283, color=:balance,
                      clims=(-0.02, 0),
                      title="T change @ t=$(tidx*fw.output_frequency*Δt)")
    end

    mp4(movie, "deep_convection_2d_$(round(Int, time())).mp4", fps = 30)
end
