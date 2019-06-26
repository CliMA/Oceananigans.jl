@hascuda using CUDAnative, CuArrays

import GPUifyLoops: @launch, @loop, @unroll, @synchronize

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model::Model{A}, Nt, Δt) where A <: Architecture
    clock = model.clock
    model_start_time = clock.time
    model_end_time = model_start_time + Nt*Δt

    if clock.iteration == 0
        for output_writer in model.output_writers
            write_output(model, output_writer)
        end
        for diagnostic in model.diagnostics
            run_diagnostic(model, diagnostic)
        end
    end

    arch = A()

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    # Unpack model fields
         grid = model.grid
        clock = model.clock
          eos = model.eos
    constants = model.constants
            U = model.velocities
           tr = model.tracers
           pr = model.pressures
      forcing = model.forcing
      closure = model.closure
    poisson_solver = model.poisson_solver

    bcs = model.boundary_conditions
    G = model.G
    Gp = model.Gp

    # We can use the same array for the right-hand-side RHS and the solution ϕ.
    RHS, ϕ = poisson_solver.storage, poisson_solver.storage

    gΔz = model.constants.g * model.grid.Δz
    fCor = model.constants.f

    uvw = U.u.data, U.v.data, U.w.data
    TS = tr.T.data, tr.S.data
    Guvw = G.Gu.data, G.Gv.data, G.Gw.data

    # Source terms at current (Gⁿ) and previous (G⁻) time steps.
    Gⁿ = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data
    G⁻ = Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    tb = (threads=(Tx, Ty), blocks=(Bx, By, Bz))
    FT = eltype(grid)

    # Field tuples for fill_halo_regions.
    u_ft = (:u, bcs.u, U.u.data)
    v_ft = (:v, bcs.v, U.v.data)
    w_ft = (:w, bcs.w, U.w.data)
    T_ft = (:T, bcs.T, tr.T.data)
    S_ft = (:S, bcs.S, tr.S.data)
    Gu_ft = (:u, bcs.u, G.Gu.data)
    Gv_ft = (:v, bcs.v, G.Gv.data)
    Gw_ft = (:w, bcs.w, G.Gw.data)
    pHY′_ft = (:w, bcs.w, pr.pHY′.data)
    pNHS_ft = (:w, bcs.w, pr.pNHS.data)

    uvw_ft = (u_ft, v_ft, w_ft)
    TS_ft = (T_ft, S_ft)
    Guvw_ft = (Gu_ft, Gv_ft, Gw_ft)

    for n in 1:Nt
        χ = ifelse(model.clock.iteration == 0, FT(-0.5), FT(0.125)) # Adams-Bashforth (AB2) parameter.

        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) store_previous_source_terms!(grid, Gⁿ..., G⁻...)
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By)     update_buoyancy!(grid, constants, eos, tr.T.data, pr.pHY′.data)
                                                                  fill_halo_regions(grid, uvw_ft..., TS_ft..., pHY′_ft)
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_interior_source_terms!(grid, constants, eos, closure, uvw..., TS..., pr.pHY′.data, Gⁿ..., forcing)
                                                                  calculate_boundary_source_terms!(model)
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) adams_bashforth_update_source_terms!(grid, Gⁿ..., G⁻..., χ)
                                                                  fill_halo_regions(grid, Guvw_ft...)
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_poisson_right_hand_side!(arch, grid, Δt, uvw..., Guvw..., RHS)
                                                                  solve_for_pressure!(arch, model)
                                                                  fill_halo_regions(grid, pNHS_ft)
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) update_velocities_and_tracers!(grid, uvw..., TS..., pr.pNHS.data, Gⁿ..., G⁻..., Δt)
                                                                  fill_halo_regions(grid, uvw_ft...)
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By)     compute_w_from_continuity!(grid, uvw...)

        clock.time += Δt
        clock.iteration += 1

        for diagnostic in model.diagnostics
            (clock.iteration % diagnostic.diagnostic_frequency) == 0 && run_diagnostic(model, diagnostic)
        end

        for output_writer in model.output_writers
            (clock.iteration % output_writer.output_frequency) == 0 && write_output(model, output_writer)
        end
    end

    return nothing
end

time_step!(model; Nt, Δt) = time_step!(model, Nt, Δt)

function solve_for_pressure!(::CPU, model::Model)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage

    solve_poisson_3d!(model.poisson_solver, model.grid)
    data(model.pressures.pNHS) .= real.(ϕ)
end

function solve_for_pressure!(::GPU, model::Model)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage

    Tx, Ty = 16, 16  # Not sure why I have to do this. Will be superseded soon.
    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    solve_poisson_3d!(Tx, Ty, Bx, By, Bz, model.poisson_solver, model.grid)
    @launch device(GPU()) threads=(Tx, Ty) blocks=(Bx, By, Bz) idct_permute!(model.grid, ϕ, model.pressures.pNHS.data)
end

"""Store previous source terms before updating them."""
function store_previous_source_terms!(grid::Grid, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gpu[i, j, k] = Gu[i, j, k]
                @inbounds Gpv[i, j, k] = Gv[i, j, k]
                @inbounds Gpw[i, j, k] = Gw[i, j, k]
                @inbounds GpT[i, j, k] = GT[i, j, k]
                @inbounds GpS[i, j, k] = GS[i, j, k]
            end
        end
    end
    @synchronize
end

"Update the hydrostatic pressure perturbation pHY′ and buoyancy δρ."
function update_buoyancy!(grid::Grid, constants, eos, T, pHY′)
    gΔz = constants.g * grid.Δz

    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = 0.5 * gΔz * δρ(eos, T, i, j, 1)
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = pHY′[i, j, k-1] + gΔz * 0.5 * (δρ(eos, T, i, j, k-1) + δρ(eos, T, i, j, k))
            end
        end
    end

    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_interior_source_terms!(grid::Grid, constants, eos, closure, u, v, w, T, S, pHY′, Gu, Gv, Gw, GT, GS, F)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    grav = constants.g
    fCor = constants.f
    ρ₀ = eos.ρ₀

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # u-momentum equation
                @inbounds Gu[i, j, k] = (-u∇u(grid, u, v, w, i, j, k)
                                            + fv(grid, v, fCor, i, j, k)
                                            - δx_c2f(grid, pHY′, i, j, k) / (Δx * ρ₀)
                                            + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, eos, grav, u, v, w, T, S)
                                            + F.u(grid, u, v, w, T, S, i, j, k))

                # v-momentum equation
                @inbounds Gv[i, j, k] = (-u∇v(grid, u, v, w, i, j, k)
                                            - fu(grid, u, fCor, i, j, k)
                                            - δy_c2f(grid, pHY′, i, j, k) / (Δy * ρ₀)
                                            + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, eos, grav, u, v, w, T, S)
                                            + F.v(grid, u, v, w, T, S, i, j, k))

                # w-momentum equation: comment about how pressure and buoyancy are handled
                @inbounds Gw[i, j, k] = (-u∇w(grid, u, v, w, i, j, k)
                                            + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, eos, grav, u, v, w, T, S)
                                            + F.w(grid, u, v, w, T, S, i, j, k))

                # temperature equation
                @inbounds GT[i, j, k] = (-div_flux(grid, u, v, w, T, i, j, k)
                                            + ∇_κ_∇ϕ(i, j, k, grid, T, closure, eos, grav, u, v, w, T, S)
                                            + F.T(grid, u, v, w, T, S, i, j, k))

                # salinity equation
                @inbounds GS[i, j, k] = (-div_flux(grid, u, v, w, S, i, j, k)
                                            + ∇_κ_∇ϕ(i, j, k, grid, S, closure, eos, grav, u, v, w, T, S)
                                            + F.S(grid, u, v, w, T, S, i, j, k))
            end
        end
    end

    @synchronize
end

function adams_bashforth_update_source_terms!(grid::Grid{FT}, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, χ) where FT
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gu[i, j, k] = (FT(1.5) + χ)*Gu[i, j, k] - (FT(0.5) + χ)*Gpu[i, j, k]
                @inbounds Gv[i, j, k] = (FT(1.5) + χ)*Gv[i, j, k] - (FT(0.5) + χ)*Gpv[i, j, k]
                @inbounds Gw[i, j, k] = (FT(1.5) + χ)*Gw[i, j, k] - (FT(0.5) + χ)*Gpw[i, j, k]
                @inbounds GT[i, j, k] = (FT(1.5) + χ)*GT[i, j, k] - (FT(0.5) + χ)*GpT[i, j, k]
                @inbounds GS[i, j, k] = (FT(1.5) + χ)*GS[i, j, k] - (FT(0.5) + χ)*GpS[i, j, k]
            end
        end
    end
    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_poisson_right_hand_side!(::CPU, grid::Grid, Δt, u, v, w, Gu, Gv, Gw, RHS)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
            end
        end
    end

    @synchronize
end

function calculate_poisson_right_hand_side!(::GPU, grid::Grid, Δt, u, v, w, Gu, Gv, Gw, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation
                # which is the first step in the DCT.
                if CUDAnative.ffs(k) == 1  # isodd(k)
                    @inbounds RHS[i, j, convert(UInt32, CUDAnative.floor(k/2) + 1)] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
                else
                    @inbounds RHS[i, j, convert(UInt32, Nz - CUDAnative.floor((k-1)/2))] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
                end
            end
        end
    end

    @synchronize
end

function idct_permute!(grid::Grid, ϕ, pNHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
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


function update_velocities_and_tracers!(grid::Grid, u, v, w, T, S, pNHS, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, Δt)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds u[i, j, k] = u[i, j, k] + (Gu[i, j, k] - (δx_c2f(grid, pNHS, i, j, k) / grid.Δx)) * Δt
                @inbounds v[i, j, k] = v[i, j, k] + (Gv[i, j, k] - (δy_c2f(grid, pNHS, i, j, k) / grid.Δy)) * Δt
                @inbounds T[i, j, k] = T[i, j, k] + (GT[i, j, k] * Δt)
                @inbounds S[i, j, k] = S[i, j, k] + (GS[i, j, k] * Δt)
            end
        end
    end

    @synchronize
end

"Compute the vertical velocity w from the continuity equation."
function compute_w_from_continuity!(grid::Grid, u, v, w)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds w[i, j, 1] = 0
            @unroll for k in 2:grid.Nz
                @inbounds w[i, j, k] = w[i, j, k-1] + grid.Δz * ∇h_u(i, j, k-1, grid, u, v)
            end
        end
    end

    @synchronize
end

"Apply boundary conditions by modifying the source term G."
function calculate_boundary_source_terms!(model::Model{A}) where A <: Architecture
    arch = A()

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    grid = model.grid
    clock = model.clock
    eos =  model.eos
    closure = model.closure
    bcs = model.boundary_conditions
    U = model.velocities
    tr = model.tracers
    G = model.G

    grav = model.constants.g
    t, iteration = clock.time, clock.iteration
    u, v, w, T, S = U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data
    Gu, Gv, Gw, GT, GS = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    coord = :z #for coord in (:x, :y, :z) when we are ready to support more coordinates.

    u_x_bcs = getproperty(bcs.u, coord)
    v_x_bcs = getproperty(bcs.v, coord)
    w_x_bcs = getproperty(bcs.w, coord)
    T_x_bcs = getproperty(bcs.T, coord)
    S_x_bcs = getproperty(bcs.S, coord)

    # Apply boundary conditions in the vertical direction.

    # *Note*: for vertical boundaries in xz or yz, the transport coefficients should be evaluated at
    # different locations than the ones speciifc below, which are specific to boundaries in the xy-plane.

    apply_bcs!(arch, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, grid, u, Gu, ν₃₃.ffc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, grid, v, Gv, ν₃₃.fcf,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    #apply_bcs!(arch, Val(coord), Bx, By, Bz, w_x_bcs.left, w_x_bcs.right, grid, w, Gw, ν₃₃.cff,
    #    closure, eos, grav, t, iteration, u, v, w, T, S)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, T_x_bcs.left, T_x_bcs.right, grid, T, GT, κ₃₃.ccc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, S_x_bcs.left, S_x_bcs.right, grid, S, GS, κ₃₃.ccc,
        closure, eos, grav, t, iteration, u, v, w, T, S)

    return nothing
end
