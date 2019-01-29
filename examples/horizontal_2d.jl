using Plots
using Oceananigans

function make_movies(problem::Problem, R::SavedFields, Nt, Δt)
    g = problem.g

    print("Creating tracer movie... ($(Int(Nt/R.ΔR)) frames)\n")

    Plots.gr()

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.heatmap(g.xC, g.yC, R.T[tidx, :, :, 1] .- 283, color=:balance,
                      clims=(0, 0.01), title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "tracer_T_$(round(Int, time())).mp4", fps = 30)
end

function horizontal_2d()
    Nx, Ny, Nz = 100, 100, 1
    Lx, Ly, Lz = 2000, 2000, 10
    Nt, Δt, ΔR = 2500, 20, 10

    problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz));
    @. problem.tr.T.data[70:90, 10:30, 1] = 283.01;
    @. problem.U.u.data = 0.1;
    @. problem.U.v.data = 0.1;

    R  = SavedFields(problem.g, Nt, ΔR);
    time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

    make_movies(problem, R, Nt, Δt)
end
