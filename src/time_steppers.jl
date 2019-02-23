@hascuda using GPUifyLoops, CUDAnative, CuArrays

using Oceananigans.Operators

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

include("operators/ops_regular_cartesian_grid_elementwise.jl")

function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        value, units = t / 1e9, "s"
    end
    return string(@sprintf("%.3f", value), " ", units)
end

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

    kx² = CuArray{Float64}(undef, Nx)
    ky² = CuArray{Float64}(undef, Ny)
    kz² = CuArray{Float64}(undef, Nz)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    # Exponential factors required to calculate the DCT on the GPU.
    factors = 2 * exp.(collect(-1im*π*(0:Nz-1) / (2*Nz)))
    dct_factors = CuArray{Complex{Float64}}(repeat(reshape(factors, 1, 1, Nz), Nx, Ny, 1))

    # "Backward" exponential factors required to calculate the IDCT on the GPU.
    bfactors = exp.(collect(1im*π*(0:Nz-1) / (2*Nz)))
    bfactors[1] *= 0.5
    idct_bfactors = CuArray{Complex{Float64}}(repeat(reshape(bfactors, 1, 1, Nz), Nx, Ny, 1))

    println("Threads per block: ($Tx, $Ty)")
    println("Blocks in grid:    ($Bx, $By, $Bz)")

    for n in 1:Nt
        t1 = time_ns(); # Timing the time stepping loop.

        @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part1!(Val(:GPU), gΔz, Nx, Ny, Nz, tr.ρ.data, δρ.data, tr.T.data, pr.pHY′.data, eos.ρ₀, eos.βT, eos.T₀)

        @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part2!(Val(:GPU), fCor, χ, eos.ρ₀, cfg.κh, cfg.κv, cfg.𝜈h, cfg.𝜈v, Nx, Ny, Nz, Δx, Δy, Δz,
                                                                                      U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data, pr.pHY′.data,
                                                                                      G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data,
                                                                                      Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data, F.FT.data)

        @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part3!(Val(:GPU), Nx, Ny, Nz, Δx, Δy, Δz, G.Gu.data, G.Gv.data, G.Gw.data, RHS.data)

        solve_poisson_3d_ppn_gpu!(Tx, Ty, Bx, By, Bz, g, RHS, ϕ, kx², ky², kz², dct_factors, idct_bfactors)
        @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) idct_permute!(Val(:GPU), Nx, Ny, Nz, ϕ.data, pr.pNHS.data)

        @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part4!(Val(:GPU), Nx, Ny, Nz, Δx, Δy, Δz, Δt,
                                                                           U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data, pr.pNHS.data,
                                                                           G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data,
                                                                           Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data)

        clock.time += Δt
        clock.time_step += 1
        print("\rmodel.clock.time = $(clock.time) / $model_end_time   ")

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

        t2 = time_ns();
        print(prettytime(t2 - t1))
    end
end

@inline δρ(eos::LinearEquationOfState, T::CellField, i, j, k) = - eos.ρ₀ * eos.βT * (T.data[i, j, k] - eos.T₀)
@inline δρ(ρ₀, βT, T₀, T, i, j, k) = @inbounds -ρ₀ * βT * (T[i, j, k] - T₀)

function time_step_kernel_part1!(::Val{Dev}, gΔz, Nx, Ny, Nz, ρ, δρ, T, pHY′, ρ₀, βT, T₀) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds δρ[i, j, k] = -ρ₀*βT * (T[i, j, k] - T₀)
                @inbounds  ρ[i, j, k] = ρ₀ + δρ[i, j, k]

                ∫δρ = (-ρ₀*βT*(T[i, j, 1]-T₀))
                for k′ in 2:k
                    ∫δρ += ((-ρ₀*βT*(T[i, j, k′-1]-T₀)) + (-ρ₀*βT*(T[i, j, k′]-T₀)))
                end
                @inbounds pHY′[i, j, k] = 0.5 * gΔz * ∫δρ
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

                @inbounds Gu[i, j, k] = -u∇u(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + fCor*avg_xy(v, Nx, Ny, i, j, k) - δx_c2f(pHY′, Nx, i, j, k) / (Δx * ρ₀) + 𝜈∇²u(u, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                @inbounds Gv[i, j, k] = -u∇v(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) - fCor*avg_xy(u, Nx, Ny, i, j, k) - δy_c2f(pHY′, Ny, i, j, k) / (Δy * ρ₀) + 𝜈∇²v(v, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                @inbounds Gw[i, j, k] = -u∇w(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)                                                                           + 𝜈∇²w(w, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)

                @inbounds GT[i, j, k] = -div_flux(u, v, w, T, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + κ∇²(T, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + FT[i, j, k]
                @inbounds GS[i, j, k] = -div_flux(u, v, w, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) + κ∇²(S, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)

                @inbounds Gu[i, j, k] = (1.5 + χ)*Gu[i, j, k] - (0.5 + χ)*Gpu[i, j, k]
                @inbounds Gv[i, j, k] = (1.5 + χ)*Gv[i, j, k] - (0.5 + χ)*Gpv[i, j, k]
                @inbounds Gw[i, j, k] = (1.5 + χ)*Gw[i, j, k] - (0.5 + χ)*Gpw[i, j, k]
                @inbounds GT[i, j, k] = (1.5 + χ)*GT[i, j, k] - (0.5 + χ)*GpT[i, j, k]
                @inbounds GS[i, j, k] = (1.5 + χ)*GS[i, j, k] - (0.5 + χ)*GpS[i, j, k]
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
                # Calculate divergence of thApplying permutation which is the first step in the DCT.
                if CUDAnative.ffs(k) == 1  # isodd(k)
                    @inbounds RHS[i, j, convert(UInt32, CUDAnative.floor(k/2) + 1)] = div_f2c(Gu, Gv, Gw, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                else
                    @inbounds RHS[i, j, convert(UInt32, Nz - CUDAnative.floor((k-1)/2))] = div_f2c(Gu, Gv, Gw, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                end
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
            end
        end
    end

    @synchronize
end

function idct_permute!(::Val{Dev}, Nx, Ny, Nz, ϕ, pNHS) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k <= Nz/2
                    @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
                else
                    @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
                end
            end
        end
    end

    @synchronize
end
