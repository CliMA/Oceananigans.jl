using Plots
using Oceananigans

function vertical_1d()
    Nx, Ny, Nz = 1, 1, 100
    Lx, Ly, Lz = 1, 1, 2000
    Nt, Δt, ΔR = 10000, 60, 100

    problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz));

    @. problem.tr.T.data[:, :, 1:Int(problem.g.Nz/2)]   = 282.99;
    @. problem.tr.T.data[:, :, Int(problem.g.Nz/2):end] = 283.01;

    R = SavedFields(problem.g, Nt, ΔR);

    time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

    make_movies(problem, R, Nt, Δt)
end

function make_movie(problem::Problem, R::SavedFields, Nt, Δt)
    g = problem.g

    print("Creating tracer movie... ($(Int(Nt/R.ΔR)) frames)\n")

    Plots.gr()

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.plot(R.T[tidx, 1, 1, :] .- 283, g.zC, title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "heavy_light_1D_$(round(Int, time())).mp4", fps = 30)
end
