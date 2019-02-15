using GPUifyLoops, CUDAnative, CuArrays
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

@inline δρ(eos::LinearEquationOfState, T::CellField, i, j, k) = - eos.ρ₀ * eos.βT * (T.data[i, j, k] - eos.T₀)
@inline δρ(ρ₀, βT, T₀, T, i, j, k) = - ρ₀ * βT * (T[i, j, k] - T₀)

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

    # Field references.
    δρ = stmp.fC1
    RHS = stmp.fCC1
    ϕ   = stmp.fCC2

    # Constants.
    gΔz = c.g * g.Δz
    χ = 0.1  # Adams-Bashforth (AB2) parameter.

    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Δx, Δy, Δz = g.Δx, g.Δy, g.Δz

    Tx, Ty = 16, 16  # Threads per block
    Bx, By, Bz = Int(Nx/Tx), Int(Ny/Ty), Nz  # Blocks in grid.

    println("Threads per block: ($Tx, $Ty)")
    println("Blocks in grid:    ($Bx, $By, $Bz)")

    for n in 1:Nt
        println("Launching kernel 1...")
        @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part1!(Val(:GPU), Nx, Ny, Nz, tr.ρ.data, δρ.data, tr.T.data, pr.pHY′.data, eos.ρ₀, eos.βT, eos.T₀)

        println("Launching kernel 2...")
        @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) time_step_kernel_part2!(Val(:GPU), Nx, Ny, Nz, Δx, Δy, Δz, U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data, G.Gu.data, G.Gv.data, G.Gw.data)

        # println("Launching kernel 3...")
        # time_step_kernel_part3!(Val(:GPU), g, G, RHS)

        # println("Nonhydrostatic pressure correction step...")
        # solve_poisson_3d_ppn_gpu!(g, RHS, ϕ)
        # @. pr.pNHS.data = real(ϕ.data)

        # println("Launching kernel 4...")
        # time_step_kernel_part4!(Val(:GPU), g, G, RHS)

        # Store source terms from previous time step.
        @. Gp.Gu.data = G.Gu.data
        @. Gp.Gv.data = G.Gv.data
        @. Gp.Gw.data = G.Gw.data
        @. Gp.GT.data = G.GT.data
        @. Gp.GS.data = G.GS.data

        clock.time += Δt
        clock.time_step += 1
        print("\rmodel.clock.time = $(clock.time) / $model_end_time   ")
    end
end

# δρ should be an array.
function time_step_kernel_part1!(::Val{Dev}, Nx, Ny, Nz, ρ, δρ, T, pHY′, ρ₀, βT, T₀) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate new density and density deviation.
                @inbounds δρ[i, j, k] = -ρ₀*βT * (T[i, j, k] - T₀)
                @inbounds  ρ[i, j, k] = ρ₀ + δρ[i, j, k]

                # # Calculate hydrostatic pressure anomaly (buoyancy): ∫δρg dz
                # @inbounds pHY′[i, j, 1] = δρ(ρ₀, βT, T₀, T, i, j, k) * 0.5f0 * gΔz
                # for k′ in 2:k
                #   @inbounds pHY′[i, j, k] += (δρ(ρ₀, βT, T₀, T, i, j, k′-1) - δρ(eos, T, i, j, k′)) * gΔz
                # end
            end
        end
    end

    @synchronize
end

function time_step_kernel_part2!(::Val{Dev}, Nx, Ny, Nz, Δx, Δy, Δz, u, v, w, T, S, Gu, Gv, Gw) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate source terms for current time step.
                # @inbounds G.Gu.data[i, j, k] = -u∇u(g, U, i, j, k) + c.f*avg_xy(g, U.v, i, j, k) - δx_c2f(g, pr.pHY′, i, j, k) / (g.Δx * eos.ρ₀) + 𝜈∇²u(g, U.u, cfg.𝜈h, cfg.𝜈v)
                # @inbounds G.Gv.data[i, j, k] = -u∇v(g, U, i, j, k) - c.f*avg_xy(g, U.u, i, j, k) - δy_c2f(g, pr.pHY′, i, j, k) / (g.Δy * eos.ρ₀) + 𝜈∇²v(g, U.v, cfg.𝜈h, cfg.𝜈v)
                # @inbounds G.Gw.data[i, j, k] = -u∇w(g, U, i, j, k)                                                                               + 𝜈∇²w(g, U.w, cfg.𝜈h, cfg.𝜈v)
                @inbounds Gu[i, j, k] = u∇u(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)

                # @inbounds G.GT.data[i, j, k] = -div_flux(g, U, tr.T, i, j, k) + κ∇²(g, tr.T, i, j, k) + F.FT.data[i, j, k]
                # @inbounds G.GS.data[i, j, k] = -div_flux(g, U, tr.S, i, j, k) + κ∇²(g, tr.S, i, j, k)

                # @inbounds G.Gu.data[i, j, k] = (1.5f0 + χ)*G.Gu.data[i, j, k] - (0.5f0 + χ)*Gp.Gu.data[i, j, k]
                # @inbounds G.Gv.data[i, j, k] = (1.5f0 + χ)*G.Gv.data[i, j, k] - (0.5f0 + χ)*Gp.Gv.data[i, j, k]
                # @inbounds G.Gw.data[i, j, k] = (1.5f0 + χ)*G.Gw.data[i, j, k] - (0.5f0 + χ)*Gp.Gw.data[i, j, k]
                # @inbounds G.GT.data[i, j, k] = (1.5f0 + χ)*G.GT.data[i, j, k] - (0.5f0 + χ)*Gp.GT.data[i, j, k]
                # @inbounds G.GS.data[i, j, k] = (1.5f0 + χ)*G.GS.data[i, j, k] - (0.5f0 + χ)*Gp.GS.data[i, j, k]
            end
        end
    end

    @synchronize
end

function time_step_kernel_part3!(::Val{Dev}, g, G, RHS) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds RHS[i, j, k] = div(g, G.Gu, G.Gv, G.Gw, i, j, k)
            end
        end
    end

    @synchronize
end

function time_step_kernel_part4!(::Val{Dev}, g, G, RHS) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds  U.u.data[i, j, k] =  U.u.data[i, j, k] + (G.Gu.data[i, j, k] - (δx_c2f(g, pr.pNHS, i, j, k) / g.Δx)) * Δt
                @inbounds  U.v.data[i, j, k] =  U.v.data[i, j, k] + (G.Gv.data[i, j, k] - (δy_c2f(g, pr.pNHS, i, j, k) / g.Δy)) * Δt
                @inbounds  U.w.data[i, j, k] =  U.w.data[i, j, k] + (G.Gw.data[i, j, k] - (δz_c2f(g, pr.pNHS, i, j, k) / g.Δz)) * Δt
                @inbounds tr.T.data[i, j, k] = tr.T.data[i, j, k] + (G.GT.data[i, j, k] * Δt)
                @inbounds tr.S.data[i, j, k] = tr.S.data[i, j, k] + (G.GS.data[i, j, k] * Δt)
            end
        end
    end

    @synchronize
end
