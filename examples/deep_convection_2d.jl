using Plots
using Oceananigans

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

function deep_convection_2d()
    Nx, Ny, Nz = 100, 1, 50
    Lx, Ly, Lz = 2000, 1, 1000
    Nt, Δt = 2500, 20
    ΔR = 10

    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))
    @. model.forcings.FT.data[Int(Nx/10):Int(9Nx/10), 1, 1] = -0.5e-5 + 1e-6*rand()

    checkpointer = Checkpointer(".", "deep_convection_2d", 1000)
    field_writer = FieldWriter(".", "deep_convection_2d", 100,
                               [model.velocities.u, model.tracers.T],
                               ["u", "T"])

    push!(model.output_writers, checkpointer)
    push!(model.output_writers, field_writer)

    R  = SavedFields(model.grid, Nt, ΔR)

    time_step!(model; Nt=Nt, Δt=Δt, R=R)
    make_temperature_movie(model, field_writer)
end
