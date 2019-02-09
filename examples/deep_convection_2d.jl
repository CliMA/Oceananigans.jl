using Plots
using Oceananigans

function make_movies(model::Model, R::SavedFields, Nt, Δt)
    g = model.grid

    print("Creating tracer movie... ($(Int(Nt/R.ΔR)) frames)\n")

    Plots.gr()

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
                      clims=(-0.02, 0),
                      title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "deep_convection_2d_$(round(Int, time())).mp4", fps = 30)
end

function deep_convection_2d()
    Nx, Ny, Nz = 100, 1, 50
    Lx, Ly, Lz = 2000, 1, 1000
    Nt, Δt = 2500, 20
    ΔR = 10

    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))
    @. model.forcings.FT.data[Int(Nx/10):Int(9Nx/10), 1, 1] = -0.5e-5 + 1e-6*rand()

    R  = SavedFields(model.grid, Nt, ΔR)

    time_step!(model; Nt=Nt, Δt=Δt, R=R)
    make_movies(model, R, Nt, Δt)

end
