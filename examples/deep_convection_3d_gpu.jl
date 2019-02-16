# using Plots
using GPUifyLoops, CUDAnative, CuArrays
using Oceananigans

function deep_convection_3d_gpu()
    Nx, Ny, Nz = 128, 128, 64
    Lx, Ly, Lz = 2000, 2000, 1000
    Nt, Δt = 50, 20

    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz), :gpu, Float32)

    T_initial = zeros(Nx, Ny, Nz)
    forcing = zeros(Nx, Ny, Nz)
    i1, i2, j1, j2 = Int(round(Nx/10)), Int(round(9Nx/10)), Int(round(Ny/10)), Int(round(9Ny/10))
    
    @. T_initial[i1:i2, j1:j2, 1] += 0.01*rand()
    @. forcing[i1:i2, j1:j2, 1] = -0.25e-5
    
    model.tracers.T.data .= cu(T_initial)
    model.forcings.FT.data .= cu(forcing)

    field_writer = FieldWriter(".", "deep_convection_2d", 50,
                               [model.velocities.u, model.tracers.T],
                               ["u", "T"])

    push!(model.output_writers, field_writer)

    time_step_kernel!(model, Nt, Δt)

    # time_step!(model; Nt=Nt, Δt=Δt)
    # make_temperature_movie(model, field_writer)
end

# function make_temperature_movie(model::Model, fw::FieldWriter)
#     n_frames = Int(model.clock.time_step / fw.output_frequency)

#     xC, zC = model.grid.xC, model.grid.zC
#     Δt = 20

#     print("Creating temperature movie... ($n_frames frames)\n")

#     Plots.gr()
#     movie = @animate for tidx in 0:n_frames
#         print("\rframe = $tidx / $n_frames   ")
#         temperature_xz = read_output(model, fw, "T", tidx*fw.output_frequency*Δt)
#         Plots.heatmap(xC, zC, rotl90(temperature_xz[:, 1, :]) .- 283, color=:balance,
#                       clims=(-0.02, 0),
#                       title="T change @ t=$(tidx*fw.output_frequency*Δt)")
#     end

#     mp4(movie, "deep_convection_2d_$(round(Int, time())).mp4", fps = 30)
# end
