# using Pkg
# Pkg.activate(".")

using Statistics, Printf
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

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
                      clims=(-0.1, 0),
                      # clims=(-maximum(R.T[tidx, :, 1, :] .- 283), maximum(R.T[tidx, :, 1, :] .- 283)),
                      title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "tracer_T_$(round(Int, time())).mp4", fps = 30)

    # animρ = @animate for tidx in 1:Int(Nt/ΔR)
    #     print("\rframe = $tidx / $(Int(Nt/ΔR))   ")
    #     Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.ρ[tidx, :, 1, :]) .- eos.ρ₀, color=:balance,
    #                   clims=(-0.001, 0.001),
    #                   # clims=(-maximum(R.ρ[tidx, :, 1, :] .- eos.ρ₀), maximum(R.ρ[tidx, :, 1, :] .- eos.ρ₀)),
    #                   title="delta rho @ t=$(tidx*ΔR*Δt)")
    # end
    # mp4(animρ, "tracer_δρ_$(round(Int, time())).mp4", fps = 30)
end

function rising_thermal_bubble()
    Nx, Ny, Nz = 100, 1, 50
    Lx, Ly, Lz = 2000, 1, 1000
    Nt, Δt = 5000, 10

    problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz))

    # U.u.data[:, 1, :] .= 0.05
    g = problem.g
    hot_buble_perturbation = reshape(0.01 * exp.(-100 * ((g.xC .- g.Lx/2).^2 .+ (g.zC .+ g.Lz/2)'.^2) / (g.Lx^2 + g.Lz^2)), (g.Nx, g.Ny, g.Nz))
    @. problem.tr.T.data = 282.99 + 2*hot_buble_perturbation

    R  = SavedFields(problem.g, 5000, 10)

    time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

    make_movies(problem, R, Nt, Δt)
end
