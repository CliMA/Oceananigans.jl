# using Pkg
# Pkg.activate(".")

using Statistics, Printf

using FFTW

import PyPlot
using Interact, Plots

using Oceananigans, Oceananigans.Operators

struct SavedFields
    u::Array{Float64,4}
    w::Array{Float64,4}
    T::Array{Float64,4}
    ρ::Array{Float64,4}
end

function SavedFields(g, Nt, ΔR)
    u = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    w = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    T = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    ρ = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    SavedFields(u, w, T, ρ)
end

function ∫dz!(g::Grid, c::PlanetaryConstants, δρ::CellField, δρz::FaceFieldZ, pHY′::CellField)
    gΔz = c.g * g.Δz
    for j in 1:g.Ny, i in 1:g.Nx
      pHY′.data[i, j, 1] = δρ.data[i, j, 1] * gΔz / 2
    end
    for k in 2:g.Nz, j in 1:g.Ny, i in 1:g.Nx
      pHY′.data[i, j, k] = pHY′.data[i, j, k-1] + (δρz.data[i, j, k] * gΔz)
    end
end

function time_stepping!(g::Grid, c::PlanetaryConstants, eos::LinearEquationOfState, ssp::SpectralSolverParameters,
                        U::VelocityFields, tr::TracerFields, pr::PressureFields, G::SourceTerms, Gp::SourceTerms, F::ForcingFields, tmp::TemporaryFields,
                        Nt, Δt, R, ΔR)
    for n in 1:Nt
        # Calculate new density and density deviation.
        δρ = tmp.fC1
        δρ!(eos, g, δρ, tr.T)
        @. tr.ρ.data = eos.ρ₀ + δρ.data

        # Calculate density at the z-faces.
        δρz = tmp.fFZ
        avgz!(g, δρ, δρz)

        # Calculate hydrostatic pressure anomaly (buoyancy).
        ∫dz!(g, c, δρ, δρz, pr.pHY′)

        # Store source terms from previous time step.
        Gp.Gu.data .= G.Gu.data
        Gp.Gv.data .= G.Gv.data
        Gp.Gw.data .= G.Gw.data
        Gp.GT.data .= G.GT.data
        Gp.GS.data .= G.GS.data

        # Calculate source terms for current time step.
        ∂xpHY′ = tmp.fFX2
        δx!(g, pr.pHY′, ∂xpHY′)
        @. ∂xpHY′.data = ∂xpHY′.data / (g.Δx * eos.ρ₀)

        @. G.Gu.data = - ∂xpHY′.data

        𝜈∇²u = tmp.fFX2
        𝜈∇²u!(g, U.u, 𝜈∇²u, 4e-2, 4e-2, tmp)

        @. G.Gu.data = G.Gu.data + 𝜈∇²u.data

        ∂ypHY′ = tmp.fFY2
        δy!(g, pr.pHY′, ∂ypHY′)
        @. ∂ypHY′.data = ∂ypHY′.data / (g.Δy * eos.ρ₀)

        @. G.Gv.data = - ∂ypHY′.data

        𝜈∇²v = tmp.fFY2
        𝜈∇²v!(g, U.v, 𝜈∇²v, 4e-2, 4e-2, tmp)

        @. G.Gv.data = G.Gv.data + 𝜈∇²v.data

        𝜈∇²w = tmp.fFZ2
        𝜈∇²w!(g, U.w, 𝜈∇²w, 4e-2, 4e-2, tmp)

        @. G.Gw.data = 𝜈∇²w.data

        ∇uT = tmp.fC4
        div_flux!(g, U.u, U.v, U.w, tr.T, ∇uT, tmp)

        @. G.GT.data = -∇uT.data

        κ∇²T = tmp.fC4
        κ∇²!(g, tr.T, κ∇²T, 4e-2, 4e-2, tmp)

        @. G.GT.data = G.GT.data + κ∇²T.data

        ∇uS = tmp.fC4
        div_flux!(g, U.u, U.v, U.w, tr.S, ∇uS, tmp)
        @. G.GS.data = -∇uS.data

        κ∇²S = tmp.fC4
        κ∇²!(g, tr.S, κ∇²S, 4e-2, 4e-2, tmp)

        @. G.GS.data = G.GS.data + κ∇²S.data

        χ = 0.1  # Adams-Bashforth (AB2) parameter.
        @. G.Gu.data = (1.5 + χ)*G.Gu.data - (0.5 + χ)*Gp.Gu.data
        @. G.Gv.data = (1.5 + χ)*G.Gv.data - (0.5 + χ)*Gp.Gv.data
        @. G.Gw.data = (1.5 + χ)*G.Gw.data - (0.5 + χ)*Gp.Gw.data
        @. G.GT.data = (1.5 + χ)*G.GT.data - (0.5 + χ)*Gp.GT.data
        @. G.GS.data = (1.5 + χ)*G.GS.data - (0.5 + χ)*Gp.GS.data

        RHS = tmp.fCC1
        ϕ   = tmp.fCC2
        div!(g, G.Gu, G.Gv, G.Gw, RHS, tmp)
        # @time solve_poisson_3d_ppn!(g, RHS, ϕ)
        solve_poisson_3d_ppn_planned!(ssp, g, RHS, ϕ)
        @. pr.pNHS.data = real(ϕ.data)

        # div!(g, G.Gu, G.Gv, G.Gw, RHS, tmp)
        # RHSr = real.(RHS.data)
        # RHS_rec = laplacian3d_ppn(pr.pNHS.data) ./ (g.Δx)^2  # TODO: This assumes Δx == Δy == Δz.
        # error = RHS_rec .- RHSr
        # @info begin
        #     string("Fourier-spectral solver diagnostics:\n",
        #             @sprintf("RHS:     min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
        #                      minimum(RHSr), maximum(RHSr), mean(RHSr), mean(abs.(RHSr)), std(RHSr)),
        #             @sprintf("RHS_rec: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
        #                      minimum(RHS_rec), maximum(RHS_rec), mean(RHS_rec), mean(abs.(RHS_rec)), std(RHS_rec)),
        #             @sprintf("error:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
        #                      minimum(error), maximum(error), mean(error), mean(abs.(error)), std(error))
        #             )
        # end

        ∂xpNHS, ∂ypNHS, ∂zpNHS = tmp.fFX, tmp.fFY, tmp.fFZ

        δx!(g, pr.pNHS, ∂xpNHS)
        δy!(g, pr.pNHS, ∂ypNHS)
        δz!(g, pr.pNHS, ∂zpNHS)

        @. ∂xpNHS.data = ∂xpNHS.data / (g.Δx)
        @. ∂ypNHS.data = ∂ypNHS.data / (g.Δy)
        @. ∂zpNHS.data = ∂zpNHS.data / (g.Δz)

        @. U.u.data  = U.u.data  + (G.Gu.data - ∂xpNHS.data) * Δt
        @. U.v.data  = U.v.data  + (G.Gv.data - ∂ypNHS.data) * Δt
        @. U.w.data  = U.w.data  + (G.Gw.data - ∂zpNHS.data) * Δt
        @. tr.T.data = tr.T.data + (G.GT.data * Δt)
        @. tr.S.data = tr.S.data + (G.GS.data * Δt)

        div_u1 = tmp.fC1
        div!(g, U.u, U.v, U.w, div_u1, tmp)

        print("\rt = $(n*Δt) / $(Nt*Δt)   ")
        if n % ΔR == 0
            # names = ["u", "v", "w", "T", "S", "Gu", "Gv", "Gw", "GT", "GS",
            #          "pHY", "pHY′", "pNHS", "ρ", "∇·u"]
            # print("t = $(n*Δt) / $(Nt*Δt)\n")
            # for (i, Q) in enumerate([U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data,
            #               G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data,
            #               pr.pHY.data, pr.pHY′.data, pr.pNHS.data, tr.ρ.data, div_u1])
            #     @printf("%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
            #             lpad(names[i], 4), minimum(Q), maximum(Q), mean(Q), mean(abs.(Q)), std(Q))
            # end

            Ridx = Int(n/ΔR)
            R.u[Ridx, :, :, :] .= U.u.data
            # Rv[n, :, :, :] = copy(vⁿ)
            R.w[Ridx, :, :, :] .= U.w.data
            R.T[Ridx, :, :, :] .= tr.T.data
            # RS[n, :, :, :] = copy(Sⁿ)
            R.ρ[Ridx, :, :, :] .= tr.ρ.data
            # RpHY′[n, :, :, :] = copy(pʰʸ′)
            # R.pNHS[Ridx, :, :, :] = copy(pⁿʰ⁺ˢ)
        end
    end
end

function main()
    N = (100, 1, 100)
    L = (20000, 1, 20000)

    c = EarthConstants()
    eos = LinearEquationOfState()

    g = RegularCartesianGrid(N, L; dim=2, FloatType=Float64)

    U  = VelocityFields(g)
    tr = TracerFields(g)
    pr = PressureFields(g)
    G  = SourceTerms(g)
    Gp = SourceTerms(g)
    F  = ForcingFields(g)
    tmp = TemporaryFields(g)

    tmp.fCC1.data .= rand(eltype(g), g.Nx, g.Ny, g.Nz)
    ssp = SpectralSolverParameters(g, tmp.fCC1, FFTW.PATIENT)

    U.u.data  .= 0.01
    U.v.data  .= 0
    U.w.data  .= 0
    tr.S.data .= 35
    tr.T.data .= 282.99

    pHY_profile = [-eos.ρ₀*c.g*h for h in g.zC]
    pr.pHY.data .= repeat(reshape(pHY_profile, 1, 1, g.Nz), g.Nx, g.Ny, 1)

    ρ!(eos, g, tr)

    tr.T.data[Int(g.Nx/2)-5:Int(g.Nx/2)+5, 1, 15:25] .= 283.01;

    Nt = 5000
    Δt = 10
    ΔR = 25
    R  = SavedFields(g, Nt, ΔR)

    time_stepping!(g, c, eos, ssp, U, tr, pr, G, Gp, F, tmp, Nt, Δt, R, ΔR)
    print("\n")

    print("Creating tracer movie... ($(Nt/ΔR) frames)\n")

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
                      clims=(-0.01, 0.01),
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
