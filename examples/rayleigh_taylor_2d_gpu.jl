using CUDAnative, CUDAdrv, CuArrays

using Plots
using Oceananigans

function make_movies(problem::Problem, R::SavedFields, Nt, Δt)
    g = problem.g

    print("Creating tracer movie... ($(Int(Nt/R.ΔR)) frames)\n")

    Plots.gr()

    default(dpi=300)

    animT = @animate for tidx in 1:Int(Nt/R.ΔR)
        print("\rframe = $tidx / $(Int(Nt/R.ΔR))   ")
        Plots.heatmap(1:g.Nx, 1:g.Nz, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
                      clims=(-0.01, 0.01), title="T change @ t=$(tidx*R.ΔR*Δt)")
    end
    mp4(animT, "rayleigh_taylor_2d_$(round(Int, time())).mp4", fps = 30)
end

function rayleigh_taylor_2d()
    Nx, Ny, Nz = 128, 8, 64
    Lx, Ly, Lz = 3000, 200, 1500
    Nt, Δt = 25, 20
    ΔR = 1

    problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz), :gpu, Float32)
    # problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz))

    ΔT = 0.02  # Temperature difference.
    ε = 0.0001  # Small temperature perturbation.

    Tc = zeros(size(problem.tr.T.data))
    @. Tc[:, :, 1:Int(problem.g.Nz/2)]   = 283 - (ΔT/2) + ε*rand()
    @. Tc[:, :, Int(problem.g.Nz/2):end] = 283 + (ΔT/2) + ε*rand()

    Tg = cu(Tc)
    # Tg = Tc
    @. problem.tr.T.data = Tg

    @show problem.tr.T.data[:, :, 1]
    @show problem.tr.T.data[:, :, end]

    R  = SavedFields(problem.g, Nt, ΔR)

    time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

    make_movies(problem, R, Nt, Δt)
end
