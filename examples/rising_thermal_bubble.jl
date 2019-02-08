using Plots, Oceananigans

function make_movies(problem::Problem, R::SavedFields, Nt, Δt)
    g = problem.g

    print("Creating tracer movie... ($(Int(Nt/R.ΔR)) frames)\n")

    Plots.gr()

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
                      clims=(-0.01, 0.01),
                      # clims=(-maximum(R.T[tidx, :, 1, :] .- 283), maximum(R.T[tidx, :, 1, :] .- 283)),
                      title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "rising_thermal_bubble_$(round(Int, time())).mp4", fps = 30)
end

function rising_thermal_bubble()
    Nx, Ny, Nz = 100, 1, 100
    Lx, Ly, Lz = 2000, 2000, 2000
    Nt, Δt = 2500, 10

    problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz))

    # U.u.data[:, 1, :] .= 0.05
    g = problem.g
    hot_buble_perturbation = reshape(0.01 * exp.(-150 * ((g.xC .- g.Lx/2).^2 .+ (g.zC .+ g.Lz/2)'.^2) / (g.Lx^2 + g.Lz^2)), (g.Nx, g.Ny, g.Nz))
    @. problem.tr.T.data = 282.99 + 2*hot_buble_perturbation
    # @. problem.tr.T.data[40:60, 1, 40:60] = 283.01

    R  = SavedFields(problem.g, Nt, Δt)

    time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

    make_movies(problem, R, Nt, Δt)
end
