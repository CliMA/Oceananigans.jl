using GPUifyLoops, CUDAnative, CuArrays
using Oceananigans.Operators

using Test  # for debugging

function time_step!(model::Model; Nt, Δt)
    metadata = model.metadata
    cfg = model.configuration
    bc = model.boundary_conditions
    g = model.grid
    c = model.constants
    eos = model.eos
    ssp = model.ssp
    U = model.velocities
    tr = model.tracers
    pr = model.pressures
    G = model.G
    Gp = model.Gp
    F = model.forcings
    stmp = model.stepper_tmp
    otmp = model.operator_tmp
    clock = model.clock

    model_start_time = clock.time
    model_end_time = model_start_time + Nt*Δt

    # Write out initial state.
    if clock.time_step == 0
        for output_writer in model.output_writers
            write_output(model, output_writer)
        end
        for diagnostic in model.diagnostics
            run_diagnostic(model, diagnostic)
        end
    end

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

        avg_x_v = stmp.fC1
        avg_xy_v = stmp.fFX
        avgx!(g, U.v, avg_x_v)
        avgy!(g, avg_x_v, avg_xy_v)
        @. G.Gu.data += c.f * avg_xy_v.data

        ∂xpHY′ = stmp.fFX
        δx!(g, pr.pHY′, ∂xpHY′)
        @. ∂xpHY′.data = ∂xpHY′.data / (g.Δx * eos.ρ₀)
        @. G.Gu.data += - ∂xpHY′.data

        𝜈∇²u = stmp.fFX
        𝜈∇²u!(g, U.u, 𝜈∇²u, cfg.𝜈h, cfg.𝜈v, otmp)
        @. G.Gu.data += 𝜈∇²u.data

        if bc.bottom_bc == :no_slip
            @. G.Gu.data[:, :, 1] += - (1/g.Δz) * (cfg.𝜈v * U.u[:, :, 1] / (g.Δz / 2))
            @. G.Gu.data[:, :, end] += - (1/g.Δz) * (cfg.𝜈v * U.u[:, :, end] / (g.Δz / 2))
        end

        u∇v = stmp.fFY
        u∇v!(g, U, u∇v, otmp)
        @. G.Gv.data = -u∇v.data

        avg_y_u = stmp.fC1
        avg_xy_u = stmp.fFY
        avgy!(g, U.u, avg_y_u)
        avgx!(g, avg_y_u, avg_xy_u)
        @. G.Gv.data += - c.f * avg_xy_u.data

        ∂ypHY′ = stmp.fFY
        δy!(g, pr.pHY′, ∂ypHY′)
        @. ∂ypHY′.data = ∂ypHY′.data / (g.Δy * eos.ρ₀)
        @. G.Gv.data += - ∂ypHY′.data

        𝜈∇²v = stmp.fFY
        𝜈∇²v!(g, U.v, 𝜈∇²v, cfg.𝜈h, cfg.𝜈v, otmp)
        @. G.Gv.data += 𝜈∇²v.data

        if bc.bottom_bc == :no_slip
            @. G.Gv.data[:, :, 1] += - (1/g.Δz) * (cfg.𝜈v * U.v[:, :, 1] / (g.Δz / 2))
            @. G.Gv.data[:, :, end] += - (1/g.Δz) * (cfg.𝜈v * U.v[:, :, end] / (g.Δz / 2))
        end

        u∇w = stmp.fFZ
        u∇w!(g, U, u∇w, otmp)
        @. G.Gw.data = -u∇w.data

        𝜈∇²w = stmp.fFZ
        𝜈∇²w!(g, U.w, 𝜈∇²w, cfg.𝜈h, cfg.𝜈v, otmp)
        @. G.Gw.data += 𝜈∇²w.data

        ∇uT = stmp.fC1
        div_flux!(g, U.u, U.v, U.w, tr.T, ∇uT, otmp)
        @. G.GT.data = -∇uT.data

        κ∇²T = stmp.fC1
        κ∇²!(g, tr.T, κ∇²T, cfg.κh, cfg.κv, otmp)
        @. G.GT.data += κ∇²T.data

        @. G.GT.data += F.FT.data

        ∇uS = stmp.fC1
        div_flux!(g, U.u, U.v, U.w, tr.S, ∇uS, otmp)
        @. G.GS.data = -∇uS.data

        κ∇²S = stmp.fC1
        κ∇²!(g, tr.S, κ∇²S, cfg.κh, cfg.κv, otmp)
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

        if metadata.arch == :cpu
            # @time solve_poisson_3d_ppn!(g, RHS, ϕ)
            solve_poisson_3d_ppn_planned!(ssp, g, RHS, ϕ)
            @. pr.pNHS.data = real(ϕ.data)
        elseif metadata.arch == :gpu
            solve_poisson_3d_ppn_gpu!(g, RHS, ϕ)
            @. pr.pNHS.data = real(ϕ.data)
        end

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

        # div_u1 = stmp.fC1
        # div!(g, U.u, U.v, U.w, div_u1, otmp)

        clock.time += Δt
        clock.time_step += 1
        print("\rmodel.clock.time = $(clock.time) / $model_end_time   ")

        for output_writer in model.output_writers
            if clock.time_step % output_writer.output_frequency == 0
                println()
                write_output(model, output_writer)
            end
        end

        for diagnostic in model.diagnostics
            if clock.time_step % diagnostic.diagnostic_frequency == 0
                run_diagnostic(model, diagnostic)
            end
        end
    end
end

# time_step_elementwise!(model::Model; Nt, Δt) = time_step_kernel!(Val(:CPU), model; Nt=Nt, Δt=Δt)

# function time_step_elementwise!(model::Model; Nt, Δt)
#     Tx, Ty = 16, 16  # Threads per block
#     Bx, By, Bz = Int(model.grid.Nx/Tx), Int(model.grid.Ny/Ty), Nz  # Blocks in grid.

#     # println("Threads per block: ($Tx, $Ty)")
#     # println("Blocks in grid:    ($Bx, $By, $Bz)")

#     @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel!(Val(:GPU), A, B)
# end

include("operators/ops_regular_cartesian_grid_elementwise.jl")

function time_step_kernel!(model::Model, Nt, Δt)
    metadata = model.metadata
    cfg = model.configuration
    bc = model.boundary_conditions
    g = model.grid
    c = model.constants
    eos = model.eos
    ssp = model.ssp
    U = model.velocities
    tr = model.tracers
    pr = model.pressures
    G = model.G
    Gp = model.Gp
    F = model.forcings
    stmp = model.stepper_tmp
    otmp = model.operator_tmp
    clock = model.clock

    model_start_time = clock.time
    model_end_time = model_start_time + Nt*Δt

    if clock.time_step == 0
        for output_writer in model.output_writers
            write_output(model, output_writer)
        end
        for diagnostic in model.diagnostics
            run_diagnostic(model, diagnostic)
        end
    end
    
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Lx, Ly, Lz = g.Lx, g.Ly, g.Lz
    Δx, Δy, Δz = g.Δx, g.Δy, g.Δz
    
    model_true = Model((Nx, Ny, Nz), (Lx, Ly, Lz), :cpu, Float32)
    
    T_initial = 293.15 .* ones(Nx, Ny, Nz)
    forcing = zeros(Nx, Ny, Nz)
    i1, i2, j1, j2 = Int(round(Nx/10)), Int(round(9Nx/10)), Int(round(Ny/10)), Int(round(9Ny/10))
    @. T_initial[i1:i2, j1:j2, 1] += 0.01
    @. forcing[i1:i2, j1:j2, 1] = -0.25e-5
    @. model_true.tracers.T.data = T_initial
    @. model_true.forcings.FT.data = forcing
    
    (typeof(@test Δx ≈ model_true.grid.Δx) == Test.Pass) && println("OK: Δx")
    (typeof(@test Δy ≈ model_true.grid.Δy) == Test.Pass) && println("OK: Δy")
    (typeof(@test Δz ≈ model_true.grid.Δz) == Test.Pass) && println("OK: Δz")
    (typeof(@test tr.T.data ≈ model_true.tracers.T.data) == Test.Pass) && println("OK: Initial T")
    (typeof(@test F.FT.data ≈ model_true.forcings.FT.data) == Test.Pass) && println("OK: T forcing")
    
    # Field references.
    δρ = stmp.fC1
    RHS = stmp.fCC1
    ϕ   = stmp.fCC2

    # Constants.
    gΔz = c.g * g.Δz
    χ = 0.1  # Adams-Bashforth (AB2) parameter.
    fCor = c.f

    Tx, Ty = 16, 16  # Threads per block
    Bx, By, Bz = Int(Nx/Tx), Int(Ny/Ty), Nz  # Blocks in grid.
    
    kx² = cu(zeros(g.Nx, 1))
    ky² = cu(zeros(g.Ny, 1))
    kz² = cu(zeros(g.Nz, 1))

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end
    
    factors = 2 * exp.(collect(-1im*π*(0:Nz-1) / (2*Nz)))
    dct_factors = cu(repeat(reshape(factors, 1, 1, Nz), Nx, Ny, 1))
    
    bfactors = 0.5 * exp.(collect(1im*π*(0:Nz-1) / (2*Nz)))
    idct_bfactors = cu(repeat(reshape(bfactors, 1, 1, Nz), Nx, Ny, 1))

    println("Threads per block: ($Tx, $Ty)")
    println("Blocks in grid:    ($Bx, $By, $Bz)")
    
    RHS_cpu = CellField(ModelMetadata(:cpu, Float32), model.grid, Complex{Float32})
    ϕ_cpu = CellField(ModelMetadata(:cpu, Float32), model.grid, Complex{Float32})

    for n in 1:Nt
        println("Time stepping true model...")
        time_step!(model_true; Nt=1, Δt=Δt)
        println()
        
        print("1 "); @time @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part1!(Val(:GPU), gΔz, Nx, Ny, Nz, tr.ρ.data, δρ.data, tr.T.data, pr.pHY′.data, eos.ρ₀, eos.βT, eos.T₀)
        
        ###
        (typeof(@test model_true.tracers.ρ.data ≈ tr.ρ.data) == Test.Pass) && println("OK: Time stepping ρ")
        (typeof(@test model_true.pressures.pHY′.data ≈ pr.pHY′.data) == Test.Pass) && println("OK: Time stepping pHY′")
        ###
        
        print("2 "); @time @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part2!(Val(:GPU), fCor, χ, eos.ρ₀, cfg.κh, cfg.κv, cfg.𝜈h, cfg.𝜈v, Nx, Ny, Nz, Δx, Δy, Δz,
                                                                                      U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data, pr.pHY′.data,
                                                                                      G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data,
                                                                                      Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data, F.FT.data)
        
        ###
        Gu_t, Gv_t, Gw_t, GT_t, GS_t = model_true.G.Gu, model_true.G.Gv, model_true.G.Gw, model_true.G.GT, model_true.G.GS
        
        Gu_min1, Gu_max1, Gu_avg1, Gu_std1 = minimum(Gu_t.data), maximum(Gu_t.data), mean(Gu_t.data), std(Gu_t.data)
        Gu_min2, Gu_max2, Gu_avg2, Gu_std2 = minimum(G.Gu.data), maximum(G.Gu.data), mean(G.Gu.data), std(G.Gu.data)
        println("Gu_cpu: min=$Gu_min1, max=$Gu_max1, mean=$Gu_avg1, std=$Gu_std1")
        println("Gu_gpu: min=$Gu_min2, max=$Gu_max2, mean=$Gu_avg2, std=$Gu_std2")
        mfactoru = mean(filter(!isinf, filter(!isnan, Gu_t.data ./ Array(G.Gu.data))))
        println("mfactoru_mean=$mfactoru")

        Gv_min1, Gv_max1, Gv_avg1, Gv_std1 = minimum(Gv_t.data), maximum(Gv_t.data), mean(Gv_t.data), std(Gv_t.data)
        Gv_min2, Gv_max2, Gv_avg2, Gv_std2 = minimum(G.Gv.data), maximum(G.Gv.data), mean(G.Gv.data), std(G.Gv.data)
        println("Gv_cpu: min=$Gv_min1, max=$Gv_max1, mean=$Gv_avg1, std=$Gv_std1")
        println("Gv_gpu: min=$Gv_min2, max=$Gv_max2, mean=$Gv_avg2, std=$Gv_std2")
        mfactorv = mean(filter(!isinf, filter(!isnan, Gv_t.data ./ Array(G.Gv.data))))
        println("mfactorv_mean=$mfactorv")

        Gw_min1, Gw_max1, Gw_avg1, Gw_std1 = minimum(Gw_t.data), maximum(Gw_t.data), mean(Gw_t.data), std(Gw_t.data)
        Gw_min2, Gw_max2, Gw_avg2, Gw_std2 = minimum(G.Gw.data), maximum(G.Gw.data), mean(G.Gw.data), std(G.Gw.data)
        println("Gw_cpu: min=$Gw_min1, max=$Gw_max1, mean=$Gw_avg1, std=$Gw_std1")
        println("Gw_gpu: min=$Gw_min2, max=$Gw_max2, mean=$Gw_avg2, std=$Gw_std2")

        GT_min1, GT_max1, GT_avg1, GT_std1 = minimum(GT_t.data), maximum(GT_t.data), mean(GT_t.data), std(GT_t.data)
        GT_min2, GT_max2, GT_avg2, GT_std2 = minimum(G.GT.data), maximum(G.GT.data), mean(G.GT.data), std(G.GT.data)
        println("GT_cpu: min=$GT_min1, max=$GT_max1, mean=$GT_avg1, std=$GT_std1")
        println("GT_gpu: min=$GT_min2, max=$GT_max2, mean=$GT_avg2, std=$GT_std2")

        GS_min1, GS_max1, GS_avg1, GS_std1 = minimum(GS_t.data), maximum(GS_t.data), mean(GS_t.data), std(GS_t.data)
        GS_min2, GS_max2, GS_avg2, GS_std2 = minimum(G.GS.data), maximum(G.GS.data), mean(G.GS.data), std(G.GS.data)
        println("GS_cpu: min=$GS_min1, max=$GS_max1, mean=$GS_avg1, std=$GS_std1")
        println("GS_gpu: min=$GS_min2, max=$GS_max2, mean=$GS_avg2, std=$GS_std2")
        
        # (typeof(@test Gu_t.data ≈ model.G.Gu.data) == Test.Pass) && println("OK: Gu")
        Gu_dis = sum(.!(Gu_t.data .≈ Array(model.G.Gu.data))); println("Gu disagreement: $Gu_dis/$(Nx*Ny*Nz)");
        # (typeof(@test Gv_t.data ≈ model.G.Gv.data) == Test.Pass) && println("OK: Gv")
        Gv_dis = sum(.!(Gv_t.data .≈ Array(model.G.Gv.data))); println("Gv disagreement: $Gv_dis/$(Nx*Ny*Nz)");
        # (typeof(@test Gw_t.data ≈ model.G.Gw.data) == Test.Pass) && println("OK: Gw")
        Gw_dis = sum(.!(Gw_t.data .≈ Array(model.G.Gw.data))); println("Gw disagreement: $Gw_dis/$(Nx*Ny*Nz)");
        # (typeof(@test GT_t.data ≈ model.G.GT.data) == Test.Pass) && println("OK: GT")
        GT_dis = sum(.!(GT_t.data .≈ Array(model.G.GT.data))); println("GT disagreement: $GT_dis/$(Nx*Ny*Nz)");
        # (typeof(@test GS_t.data ≈ model.G.GS.data) == Test.Pass) && println("OK: GS")
        GS_dis = sum(.!(GS_t.data .≈ Array(model.G.GS.data))); println("GS disagreement: $GS_dis/$(Nx*Ny*Nz)");
        ###
        
        print("3 "); @time @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part3!(Val(:GPU), Nx, Ny, Nz, Δx, Δy, Δz, G.Gu.data, G.Gv.data, G.Gw.data, RHS.data)
        
        # println("Nonhydrostatic pressure correction step...")
        # @time solve_poisson_3d_ppn_gpu!(g, RHS, ϕ)
        # print("P "); @time solve_poisson_3d_ppn_gpu!(Tx, Ty, Bx, By, Bz, g, RHS, ϕ, kx², ky², kz²)
        # @. pr.pNHS.data = real(ϕ.data)

        RHS_cpu.data .= Array(RHS.data)
        solve_poisson_3d_ppn!(g, RHS_cpu, ϕ_cpu)
        pr.pNHS.data .= cu(real.(ϕ_cpu.data))
        
        ###
        pNHS_t = model_true.pressures.pNHS
        pNHS_min1, pNHS_max1, pNHS_avg1, pNHS_std1 = minimum(pNHS_t.data), maximum(pNHS_t.data), mean(pNHS_t.data), std(pNHS_t.data)
        pNHS_min2, pNHS_max2, pNHS_avg2, pNHS_std2 = minimum(pr.pNHS.data), maximum(pr.pNHS.data), mean(pr.pNHS.data), std(pr.pNHS.data)
        println("pNHS_cpu: min=$pNHS_min1, max=$pNHS_max1, mean=$pNHS_avg1, std=$pNHS_std1")
        println("pNHS_gpu: min=$pNHS_min2, max=$pNHS_max2, mean=$pNHS_avg2, std=$pNHS_std2")
        
        # (typeof(@test Gu_t.data ≈ model.G.Gu.data) == Test.Pass) && println("OK: Gu")
        ##
        
        print("4 ");
        @time @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part4!(Val(:GPU), Nx, Ny, Nz, Δx, Δy, Δz, Δt,
                                                                           U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data, pr.pNHS.data,
                                                                           G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data,
                                                                           Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data)
        
        # Store source terms from previous time step.
        # @. Gp.Gu.data = G.Gu.data
        # @. Gp.Gv.data = G.Gv.data
        # @. Gp.Gw.data = G.Gw.data
        # @. Gp.GT.data = G.GT.data
        # @. Gp.GS.data = G.GS.data

        clock.time += Δt
        clock.time_step += 1
        # print("\rmodel.clock.time = $(clock.time) / $model_end_time   ")
        println("\rmodel.clock.time = $(clock.time) / $model_end_time   ")
        
        for output_writer in model.output_writers
            if clock.time_step % output_writer.output_frequency == 0
                write_output(model, output_writer)
            end
        end
        
        for diagnostic in model.diagnostics
            if clock.time_step % diagnostic.diagnostic_frequency == 0
                run_diagnostic(model, diagnostic)
            end
        end
    end
end

@inline δρ(eos::LinearEquationOfState, T::CellField, i, j, k) = - eos.ρ₀ * eos.βT * (T.data[i, j, k] - eos.T₀)
@inline δρ(ρ₀, βT, T₀, T, i, j, k) = @inbounds -ρ₀ * βT * (T[i, j, k] - T₀)

function time_step_kernel_part1!(::Val{Dev}, gΔz, Nx, Ny, Nz, ρ, δρ, T, pHY′, ρ₀, βT, T₀) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate new density and density deviation.
                # @inbounds δρ[i, j, k] = δρ(ρ₀, βT, T₀, T, i, j, k)
                # @inbounds  ρ[i, j, k] = ρ₀ + δρ(ρ₀, βT, T₀, T, i, j, k)
                
                @inbounds δρ[i, j, k] = -ρ₀*βT * (T[i, j, k] - T₀)
                @inbounds  ρ[i, j, k] = ρ₀ + δρ[i, j, k]

                # Calculate hydrostatic pressure anomaly (buoyancy): ∫δρgdz
                # @inbounds pHY′[i, j, 1] = δρ(ρ₀, βT, T₀, T, i, j, 1) * 0.5f0 * gΔz
                # for k′ in 2:k
                #     @inbounds pHY′[i, j, k] += (δρ(ρ₀, βT, T₀, T, i, j, k′-1) - δρ(eos, T, i, j, k′)) * gΔz
                # end
                
                # ∫δρgdz = δρ(ρ₀, βT, T₀, T, i, j, 1) * 0.5f0 * gΔz
                # for k′ in 2:k
                #     ∫δρgdz += (δρ(ρ₀, βT, T₀, T, i, j, k′-1) - δρ(eos, T, i, j, k′)) * gΔz
                # end
                
                ∫δρ = (-ρ₀*βT*(T[i, j, 1]-T₀))
                for k′ in 2:k
                    ∫δρ += ((-ρ₀*βT*(T[i, j, k′-1]-T₀)) + (-ρ₀*βT*(T[i, j, k′]-T₀)))
                end
                @inbounds pHY′[i, j, k] = 0.5f0 * gΔz * ∫δρ
            end
        end
    end

    @synchronize
end

function time_step_kernel_part2!(::Val{Dev}, fCor, χ, ρ₀, κh, κv, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, u, v, w, T, S, pHY′, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, FT) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gpu[i, j, k] = Gu[i, j, k]
                @inbounds Gpv[i, j, k] = Gv[i, j, k]
                @inbounds Gpw[i, j, k] = Gw[i, j, k]
                @inbounds GpT[i, j, k] = GT[i, j, k]
                @inbounds GpS[i, j, k] = GS[i, j, k]
                
                # Calculate source terms for current time step.
                # @inbounds G.Gu.data[i, j, k] = -u∇u(g, U, i, j, k) + c.f*avg_xy(g, U.v, i, j, k) - δx_c2f(g, pr.pHY′, i, j, k) / (g.Δx * eos.ρ₀) + 𝜈∇²u(g, U.u, cfg.𝜈h, cfg.𝜈v)
                # @inbounds G.Gv.data[i, j, k] = -u∇v(g, U, i, j, k) - c.f*avg_xy(g, U.u, i, j, k) - δy_c2f(g, pr.pHY′, i, j, k) / (g.Δy * eos.ρ₀) + 𝜈∇²v(g, U.v, cfg.𝜈h, cfg.𝜈v)
                # @inbounds G.Gw.data[i, j, k] = -u∇w(g, U, i, j, k)                                                                               + 𝜈∇²w(g, U.w, cfg.𝜈h, cfg.𝜈v)
                 
                @inbounds Gu[i, j, k] = -u∇u(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + fCor*avg_xy(v, Nx, Ny, i, j, k) - δx_c2f(pHY′, Nx, i, j, k) / (Δx * ρ₀) + 𝜈∇²u(u, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                @inbounds Gv[i, j, k] = -u∇v(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) - fCor*avg_xy(u, Nx, Ny, i, j, k) - δy_c2f(pHY′, Ny, i, j, k) / (Δy * ρ₀) + 𝜈∇²v(v, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                @inbounds Gw[i, j, k] = -u∇w(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)                                                                           + 𝜈∇²w(w, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)

                # @inbounds G.GT.data[i, j, k] = -div_flux(g, U, tr.T, i, j, k) + κ∇²(g, tr.T, i, j, k) + F.FT.data[i, j, k]
                # @inbounds G.GS.data[i, j, k] = -div_flux(g, U, tr.S, i, j, k) + κ∇²(g, tr.S, i, j, k)

                @inbounds GT[i, j, k] = -div_flux(u, v, w, T, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + κ∇²(T, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + FT[i, j, k]
                @inbounds GS[i, j, k] = -div_flux(u, v, w, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + κ∇²(S, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)

                # @inbounds G.Gu.data[i, j, k] = (1.5f0 + χ)*G.Gu.data[i, j, k] - (0.5f0 + χ)*Gp.Gu.data[i, j, k]
                # @inbounds G.Gv.data[i, j, k] = (1.5f0 + χ)*G.Gv.data[i, j, k] - (0.5f0 + χ)*Gp.Gv.data[i, j, k]
                # @inbounds G.Gw.data[i, j, k] = (1.5f0 + χ)*G.Gw.data[i, j, k] - (0.5f0 + χ)*Gp.Gw.data[i, j, k]
                # @inbounds G.GT.data[i, j, k] = (1.5f0 + χ)*G.GT.data[i, j, k] - (0.5f0 + χ)*Gp.GT.data[i, j, k]
                # @inbounds G.GS.data[i, j, k] = (1.5f0 + χ)*G.GS.data[i, j, k] - (0.5f0 + χ)*Gp.GS.data[i, j, k]

                @inbounds Gu[i, j, k] = (1.5f0 + χ)*Gu[i, j, k] - (0.5f0 + χ)*Gpu[i, j, k]
                @inbounds Gv[i, j, k] = (1.5f0 + χ)*Gv[i, j, k] - (0.5f0 + χ)*Gpv[i, j, k]
                @inbounds Gw[i, j, k] = (1.5f0 + χ)*Gw[i, j, k] - (0.5f0 + χ)*Gpw[i, j, k]
                @inbounds GT[i, j, k] = (1.5f0 + χ)*GT[i, j, k] - (0.5f0 + χ)*GpT[i, j, k]
                @inbounds GS[i, j, k] = (1.5f0 + χ)*GS[i, j, k] - (0.5f0 + χ)*GpS[i, j, k]
            end
        end
    end

    @synchronize
end

function time_step_kernel_part3!(::Val{Dev}, Nx, Ny, Nz, Δx, Δy, Δz, Gu, Gv, Gw, RHS) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # @inbounds RHS[i, j, k] = div(g, G.Gu, G.Gv, G.Gw, i, j, k)
                @inbounds RHS[i, j, k] = div_f2c(Gu, Gv, Gw, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
            end
        end
    end

    @synchronize
end

function time_step_kernel_part4!(::Val{Dev}, Nx, Ny, Nz, Δx, Δy, Δz, Δt, u, v, w, T, S, pNHS, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds u[i, j, k] = u[i, j, k] + (Gu[i, j, k] - (δx_c2f(pNHS, Nx, i, j, k) / Δx)) * Δt
                @inbounds v[i, j, k] = v[i, j, k] + (Gv[i, j, k] - (δy_c2f(pNHS, Ny, i, j, k) / Δy)) * Δt
                @inbounds w[i, j, k] = w[i, j, k] + (Gw[i, j, k] - (δz_c2f(pNHS, Nz, i, j, k) / Δz)) * Δt
                @inbounds T[i, j, k] = T[i, j, k] + (GT[i, j, k] * Δt)
                @inbounds S[i, j, k] = S[i, j, k] + (GS[i, j, k] * Δt)
                
                #@inbounds Gpu[i, j, k] = Gu[i, j, k]
                #@inbounds Gpv[i, j, k] = Gv[i, j, k]
                #@inbounds Gpw[i, j, k] = Gw[i, j, k]
                #@inbounds GpT[i, j, k] = GT[i, j, k]
                #@inbounds GpS[i, j, k] = GS[i, j, k]
            end
        end
    end

    @synchronize
end
