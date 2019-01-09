# using Pkg
# Pkg.activate(".")

using Statistics, Printf

using FFTW

import PyPlot
using Interact, Plots

using Oceananigans

function isotropic_diffusion_2d()
    N = (100, 1, 50)
    L = (2000, 1, 1000)

    c = EarthConstants()
    eos = LinearEquationOfState()

    g = RegularCartesianGrid(N, L; dim=2, FloatType=Float64)

    U  = VelocityFields(g)
    tr = TracerFields(g)
    pr = PressureFields(g)
    G  = SourceTerms(g)
    Gp = SourceTerms(g)
    F  = ForcingFields(g)
    stmp = StepperTemporaryFields(g)
    otmp = OperatorTemporaryFields(g)

    stmp.fCC1.data .= rand(eltype(g), g.Nx, g.Ny, g.Nz)
    ssp = SpectralSolverParameters(g, stmp.fCC1, FFTW.PATIENT, verbose=true)

    U.u.data  .= 0
    U.v.data  .= 0
    U.w.data  .= 0
    tr.S.data .= 35
    tr.T.data .= 283

    @. F.FT.data[Int(g.Nx/10):Int(9g.Nx/10), 1, 1] = -0.5e-5 + 1e-6*rand()

    pHY_profile = [-eos.ρ₀*c.g*h for h in g.zC]
    pr.pHY.data .= repeat(reshape(pHY_profile, 1, 1, g.Nz), g.Nx, g.Ny, 1)

    ρ!(eos, g, tr)

    # i1, i2 = round(Int, g.Nx/2 - g.Nx/10), round(Int, g.Nx/2 + g.Nx/10)
    # k1, k2 = round(Int, g.Nz/3 + g.Nz/10), round(Int, g.Nz/3 + g.Nz/5)
    # tr.T.data[i1:i2, 1, k1:k2] .= 283.02;

    # U.u.data[:, 1, :] .= 0.05
    # hot_buble_perturbation = reshape(0.01 * exp.(-100 * ((g.xC .- g.Lx/2).^2 .+ (g.zC .+ g.Lz/2)'.^2) / (g.Lx^2 + g.Lz^2)), (g.Nx, g.Ny, g.Nz))
    # @. tr.T.data = 282.99 + 2*hot_buble_perturbation

    # U.u.data[:, 1, 1:Int(g.Nz/2)]   .=  0.1
    # U.u.data[:, 1, Int(g.Nz/2):end] .= -0.1
    #
    # tr.T.data[:, 1, 1:Int(g.Nz/2)]   .= 282.9
    # tr.T.data[:, 1, Int(g.Nz/2):end] .= 283.1
    #
    # for (i, x) in enumerate(g.xC)
    #     tr.T.data[i, 1, Int(g.Nz/2)] += i % 2 == 0 ? +0.01 : -0.01
    # end

    # U.u.data[:, 1, 1:end-1] .= 0.01

    Nt = 5000
    Δt = 10
    ΔR = 10
    R  = SavedFields(g, Nt, ΔR)

    @time time_stepping!(g, c, eos, ssp, U, tr, pr, G, Gp, F, stmp, otmp, Nt, Δt, R, ΔR)
    print("\n")

    print("Creating tracer movie... ($(Int(Nt/ΔR)) frames)\n")

    Plots.gr()

    # animU = @animate for tidx in 1:Int(Nt/ΔR)
    #     print("\rframe = $tidx / $(Int(Nt/ΔR))   ")
    #     Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.u[tidx, :, 1, :]), color=:balance,
    #                   clims=(-0.01, 0.01),
    #                   title="u-velocity @ t=$(tidx*ΔR*Δt)")
    # end
    # mp4(animU, "uvel_$(round(Int, time())).mp4", fps = 30)

    animT = @animate for tidx in 1:Int(Nt/ΔR)
        print("\rframe = $tidx / $(Int(Nt/ΔR))   ")
        Plots.heatmap(g.xC ./ 1000, g.zC ./ 1000, rotl90(R.T[tidx, :, 1, :]) .- 283, color=:balance,
                      clims=(-0.1, 0),
                      # clims=(-maximum(R.T[tidx, :, 1, :] .- 283), maximum(R.T[tidx, :, 1, :] .- 283)),
                      title="T change @ t=$(tidx*ΔR*Δt)")
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
