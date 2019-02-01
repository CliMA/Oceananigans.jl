using Plots
using Oceananigans

function make_movies(problem::Problem, R::SavedFields, Nt, Δt)
    g = problem.g

    print("Creating tracer movie... ($(Int(Nt/R.ΔR)) frames)\n")

    Plots.gr()

    # animU = @animate for tidx in 1:Int(Nt/ΔR)
    #     print("\rframe = $tidx / $(Int(Nt/ΔR))   ")
    #     Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.u[tidx, :, 1, :]), color=:balance,
    #                   clims=(-0.01, 0.01),
    #                   title="u-velocity @ t=$(tidx*ΔR*Δt)")
    # end
    # mp4(animU, "uvel_$(round(Int, time())).mp4", fps = 30)

    # animT = @animate for tidx in 1:Int(Nt/R.ΔR)
    #     print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
    #     Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
    #                   clims=(-0.02, 0.02),
    #                   title="T change @ t=$(tidx*R.ΔR*Δt)")
    # end
    # mp4(animT, "rayleigh_taylor_2d_$(round(Int, time())).mp4", fps = 30)

    default(dpi=300)

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.contourf(1:g.Nx, 1:g.Nz, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
                      clims=(-0.02, 0.02), title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "rayleigh_taylor_2d_$(round(Int, time())).mp4", fps = 30)

    # animρ = @animate for tidx in 1:Int(Nt/ΔR)
    #     print("\rframe = $tidx / $(Int(Nt/ΔR))   ")
    #     Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.ρ[tidx, :, 1, :]) .- eos.ρ₀, color=:balance,
    #                   clims=(-0.001, 0.001),
    #                   # clims=(-maximum(R.ρ[tidx, :, 1, :] .- eos.ρ₀), maximum(R.ρ[tidx, :, 1, :] .- eos.ρ₀)),
    #                   title="delta rho @ t=$(tidx*ΔR*Δt)")
    # end
    # mp4(animρ, "tracer_δρ_$(round(Int, time())).mp4", fps = 30)
end

function rayleigh_taylor_2d()
    Nx, Ny, Nz = 1024, 1, 512
    Lx, Ly, Lz = 20000, 1, 10000
    Nt, Δt = 5000, 10
    ΔR = 10

    problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz))

    ΔT = 0.02  # Temperature difference.
    ε = 0.0001  # Small temperature perturbation.

    @. problem.tr.T.data[:, :, 1:Int(problem.g.Nz/2)]   = 283 - (ΔT/2) + ε*rand()
    @. problem.tr.T.data[:, :, Int(problem.g.Nz/2):end] = 283 + (ΔT/2) + ε*rand()

    R  = SavedFields(problem.g, Nt, ΔR)

    time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

    make_movies(problem, R, Nt, Δt)
end
