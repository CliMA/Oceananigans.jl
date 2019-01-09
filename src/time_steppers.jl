using Oceananigans.Operators

function time_stepping!(g::Grid, c::PlanetaryConstants, eos::LinearEquationOfState, ssp::SpectralSolverParameters,
                        U::VelocityFields, tr::TracerFields, pr::PressureFields, G::SourceTerms, Gp::SourceTerms, F::ForcingFields,
                        stmp::StepperTemporaryFields, otmp::OperatorTemporaryFields,
                        Nt, Δt, R, ΔR)

    κh = 4e-2  # Horizontal Laplacian heat diffusion [m²/s]. diffKhT in MITgcm.
    κv = 4e-2  # Vertical Laplacian heat diffusion [m²/s]. diffKzT in MITgcm.
    𝜈h = 4e-2  # Horizontal eddy viscosity [Pa·s]. viscAh in MITgcm.
    𝜈v = 4e-2  # Vertical eddy viscosity [Pa·s]. viscAz in MITgcm.
    for n in 1:Nt
        # Calculate new density and density deviation.
        δρ = stmp.fC1
        δρ!(eos, g, δρ, tr.T)
        @. tr.ρ.data = eos.ρ₀ + δρ.data

        # Calculate density at the z-faces.
        δρz = stmp.fFZ
        avgz!(g, δρ, δρz)

        # Calculate hydrostatic pressure anomaly (buoyancy).
        ∫δρgdz!(g, c, δρ, δρz, pr.pHY′)

        # Store source terms from previous time step.
        Gp.Gu.data .= G.Gu.data
        Gp.Gv.data .= G.Gv.data
        Gp.Gw.data .= G.Gw.data
        Gp.GT.data .= G.GT.data
        Gp.GS.data .= G.GS.data

        # Calculate source terms for current time step.
        u∇u = stmp.fFX
        u∇u!(g, U, u∇u, otmp)
        @. G.Gu.data = -u∇u.data

        ∂xpHY′ = stmp.fFX
        δx!(g, pr.pHY′, ∂xpHY′)
        @. ∂xpHY′.data = ∂xpHY′.data / (g.Δx * eos.ρ₀)
        @. G.Gu.data += - ∂xpHY′.data

        𝜈∇²u = stmp.fFX
        𝜈∇²u!(g, U.u, 𝜈∇²u, 𝜈h, 𝜈v, otmp)
        @. G.Gu.data += 𝜈∇²u.data

        ###
        u∇v = stmp.fFY
        u∇v!(g, U, u∇v, otmp)
        @. G.Gv.data = -u∇v.data

        ∂ypHY′ = stmp.fFY
        δy!(g, pr.pHY′, ∂ypHY′)
        @. ∂ypHY′.data = ∂ypHY′.data / (g.Δy * eos.ρ₀)
        @. G.Gv.data += - ∂ypHY′.data

        𝜈∇²v = stmp.fFY
        𝜈∇²v!(g, U.v, 𝜈∇²v, 𝜈h, 𝜈v, otmp)
        @. G.Gv.data += 𝜈∇²v.data

        u∇w = stmp.fFZ
        u∇w!(g, U, u∇w, otmp)
        @. G.Gw.data = -u∇w.data

        𝜈∇²w = stmp.fFZ
        𝜈∇²w!(g, U.w, 𝜈∇²w, 𝜈h, 𝜈v, otmp)
        @. G.Gw.data += 𝜈∇²w.data

        ∇uT = stmp.fC1
        div_flux!(g, U.u, U.v, U.w, tr.T, ∇uT, otmp)
        @. G.GT.data = -∇uT.data

        κ∇²T = stmp.fC1
        κ∇²!(g, tr.T, κ∇²T, κh, κv, otmp)
        @. G.GT.data += κ∇²T.data

        @. G.GT.data += F.FT.data

        ∇uS = stmp.fC1
        div_flux!(g, U.u, U.v, U.w, tr.S, ∇uS, otmp)
        @. G.GS.data = -∇uS.data

        κ∇²S = stmp.fC1
        κ∇²!(g, tr.S, κ∇²S, κh, κv, otmp)
        @. G.GS.data += κ∇²S.data

        χ = 0.1  # Adams-Bashforth (AB2) parameter.
        @. G.Gu.data = (1.5 + χ)*G.Gu.data - (0.5 + χ)*Gp.Gu.data
        @. G.Gv.data = (1.5 + χ)*G.Gv.data - (0.5 + χ)*Gp.Gv.data
        @. G.Gw.data = (1.5 + χ)*G.Gw.data - (0.5 + χ)*Gp.Gw.data
        @. G.GT.data = (1.5 + χ)*G.GT.data - (0.5 + χ)*Gp.GT.data
        @. G.GS.data = (1.5 + χ)*G.GS.data - (0.5 + χ)*Gp.GS.data

        RHS = stmp.fCC1
        ϕ   = stmp.fCC2
        div!(g, G.Gu, G.Gv, G.Gw, RHS, otmp)
        # @time solve_poisson_3d_ppn!(g, RHS, ϕ)
        solve_poisson_3d_ppn_planned!(ssp, g, RHS, ϕ)
        @. pr.pNHS.data = real(ϕ.data)

        # div!(g, G.Gu, G.Gv, G.Gw, RHS, otmp)
        # RHSr = real.(RHS.data)
        # RHS_rec = laplacian3d_ppn(pr.pNHS.data) ./ (g.Δx)^2  # TODO: This assumes Δx == Δy == Δz.
        # error = RHS_rec .- RHSr
        # @printf("RHS:     min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(RHSr), maximum(RHSr), mean(RHSr), mean(abs.(RHSr)), std(RHSr))
        # @printf("RHS_rec: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(RHS_rec), maximum(RHS_rec), mean(RHS_rec), mean(abs.(RHS_rec)), std(RHS_rec))
        # @printf("error:   min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n", minimum(error), maximum(error), mean(error), mean(abs.(error)), std(error))

        ∂xpNHS, ∂ypNHS, ∂zpNHS = stmp.fFX, stmp.fFY, stmp.fFZ

        δx!(g, pr.pNHS, ∂xpNHS)
        δy!(g, pr.pNHS, ∂ypNHS)
        δz!(g, pr.pNHS, ∂zpNHS)

        @. ∂xpNHS.data = ∂xpNHS.data / g.Δx
        @. ∂ypNHS.data = ∂ypNHS.data / g.Δy
        @. ∂zpNHS.data = ∂zpNHS.data / g.Δz

        @. U.u.data  = U.u.data  + (G.Gu.data - ∂xpNHS.data) * Δt
        @. U.v.data  = U.v.data  + (G.Gv.data - ∂ypNHS.data) * Δt
        @. U.w.data  = U.w.data  + (G.Gw.data - ∂zpNHS.data) * Δt
        @. tr.T.data = tr.T.data + (G.GT.data * Δt)
        @. tr.S.data = tr.S.data + (G.GS.data * Δt)

        div_u1 = stmp.fC1
        div!(g, U.u, U.v, U.w, div_u1, otmp)

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
