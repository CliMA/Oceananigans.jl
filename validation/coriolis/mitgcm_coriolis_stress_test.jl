# MITgcm Coriolis stress test — same domain/topography as the Oceananigans version
# Runs MITgcm via MITgcm.jl with different selectCoriScheme values.
# Momentum advection is OFF so only the Coriolis term is active.
#
# selectCoriScheme controls the pure Coriolis discretization in mom_vi_coriolis.F:
#   0 = simple average (no hFac weighting)
#   1 = Jamart wet-point average: sum(v*dx*hFacS) / sum(hFacS)
#   2 = hFac weighted average: 0.25*sum(v*dx*hFacS) * recip_hFacW
#   3 = energy-conserving with hFac weighted average
#
# Requires: MITgcm source (set MITGCM_DIR env variable or place at ../../../MITgcm)

using MITgcm
using Printf
using JLD2
using CairoMakie

#####
##### Grid and physics (must match Oceananigans test)
#####

Nx, Ny, Nz = 60, 60, 1
H    = 500.0
Δx   = 0.5  # degrees
Δy   = 0.5
Δz   = H / Nz
Δt   = 300.0
stop_time    = 30 * 86400.0  # 30 days [s]
save_interval = 12 * 3600.0  # 12 hours [s]

#####
##### Topography and initial conditions (identical to Oceananigans)
#####

function stress_bottom(λ, φ)
    z = -H
    φ < 35 && return 0.0
    in_island = (10 < λ < 20) && (40 < φ < 50)
    in_strait = (13.5 < λ < 15.0) && (44 < φ < 46)
    in_island && !in_strait && return 0.0
    (5 < λ < 7) && (44 < φ < 46) && return 0.0
    (24.5 < λ < 25.5) && (47.5 < φ < 48.5) && return 0.0
    (2.5 < λ < 3.5) && (35 < φ < 42) && return 0.0
    if 35 ≤ φ ≤ 37
        mod(λ, 4.0) < 1.0 && return -H
        return 0.0
    end
    return z
end

function u_init(λ, φ, z)
    U_jet = 0.5; φ_jet = 45.0; σ_jet = 3.0
    return U_jet * exp(-(φ - φ_jet)^2 / (2 * σ_jet^2))
end

function v_init(λ, φ, z)
    λ₀, φ₀, σ, V₀ = 7.0, 45.0, 2.5, 0.2
    r² = (λ - λ₀)^2 + (φ - φ₀)^2
    return -V₀ * (λ - λ₀) / σ * exp(-r² / (2σ^2))
end

#####
##### MITgcm input file generation
#####

function write_bin(filename, data)
    open(filename, "w") do io
        for x in vec(Float64.(data))
            write(io, hton(reinterpret(UInt64, x)))
        end
    end
end

function generate_mitgcm_files(base_dir)
    code_dir  = joinpath(base_dir, "code")
    input_dir = joinpath(base_dir, "input")
    mkpath(code_dir)
    mkpath(input_dir)

    # SIZE.h
    write(joinpath(code_dir, "SIZE.h"), """
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER (
     &  sNx=$Nx, sNy=$Ny, OLx=4, OLy=4,
     &  nSx=1, nSy=1, nPx=1, nPy=1,
     &  Nx=sNx*nSx*nPx, Ny=sNy*nSy*nPy, Nr=$Nz)
      PARAMETER ( MAX_OLX=OLx, MAX_OLY=OLy )
""")

    # packages.conf — need mom_vecinv for vectorInvariantMomentum
    write(joinpath(code_dir, "packages.conf"), "gfd\nmom_vecinv\n")

    # eedata
    write(joinpath(input_dir, "eedata"), " &EEPARMS\n /\n")

    # data.pkg
    write(joinpath(input_dir, "data.pkg"), " &PACKAGES\n /\n")

    # Bathymetry
    bathy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        λ = (i - 0.5) * Δx   # cell center longitude
        φ = 30.0 + (j - 0.5) * Δy  # cell center latitude
        bathy[i, j] = stress_bottom(λ, φ)
    end
    
    write_bin(joinpath(input_dir, "bathymetry.bin"), bathy)

    # Initial u, v (Nx × Ny × Nz) — evaluate at staggered C-grid locations
    uvel = zeros(Nx, Ny, Nz)
    vvel = zeros(Nx, Ny, Nz)
    rc = [-Δz * (k - 0.5) for k in 1:Nz]
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # u at west face: λ = (i-1)*Δx, φ = 30 + (j-0.5)*Δy
        λ_u = (i - 1) * Δx
        φ_u = 30.0 + (j - 0.5) * Δy
        uvel[i, j, k] = u_init(λ_u, φ_u, rc[k])
        # v at south face: λ = (i-0.5)*Δx, φ = 30 + (j-1)*Δy
        λ_v = (i - 0.5) * Δx
        φ_v = 30.0 + (j - 1) * Δy
        vvel[i, j, k] = v_init(λ_v, φ_v, rc[k])
    end
    write_bin(joinpath(input_dir, "uVelInit.bin"), uvel)
    write_bin(joinpath(input_dir, "vVelInit.bin"), vvel)

    # data namelist (selectCoriScheme will be overwritten per run)
    write_data_file(input_dir; selectCoriScheme=0)

    return code_dir, input_dir
end

function write_data_file(dir; selectCoriScheme=0)
    nsteps = Int(stop_time / Δt)
    tref = join(["$Nz*20."], ", ")
    sref = join(["$Nz*35."], ", ")
    delR = join(["$Nz*$(Δz)"], ", ")

    write(joinpath(dir, "data"), """
 &PARM01
 tRef=$tref,
 sRef=$sref,
 viscAr=0.01,
 no_slip_sides=.FALSE.,
 no_slip_bottom=.FALSE.,
 bottomDragLinear=1.E-04,
 implicitFreeSurface=.TRUE.,
 vectorInvariantMomentum=.TRUE.,
 momAdvection=.FALSE.,
 selectCoriScheme=$selectCoriScheme,
 staggerTimeStep=.TRUE.,
 readBinaryPrec=64,
 writeBinaryPrec=64,
 /
 &PARM02
 cg2dMaxIters=200,
 cg2dTargetResWunit=1.E-13,
 /
 &PARM03
 deltaT=$(Δt),
 nTimeSteps=$nsteps,
 dumpFreq=0.,
 chkptFreq=0.,
 pChkptFreq=0.,
 monitorFreq=86400.,
 /
 &PARM04
 usingSphericalPolarGrid=.TRUE.,
 delX=$Nx*$(Δx),
 delY=$Ny*$(Δy),
 delR=$delR,
 xgOrigin=0.,
 ygOrigin=30.,
 /
 &PARM05
 bathyFile='bathymetry.bin',
 uVelInitFile='uVelInit.bin',
 vVelInitFile='vVelInit.bin',
 /
""")
end

#####
##### Build MITgcm library
#####

base_dir = joinpath(@__DIR__, "mitgcm_stress_test")
code_dir, input_dir = generate_mitgcm_files(base_dir)

mitgcm_dir = get(ENV, "MITGCM_DIR",
                 joinpath(@__DIR__, "..", "..", "..", "MITgcm"))

if !isdir(mitgcm_dir)
    error("MITgcm source not found at $mitgcm_dir — set MITGCM_DIR.")
end

@info "Building MITgcm library..."
build_result = build_mitgcm_library(mitgcm_dir;
                                     output_dir=base_dir,
                                     code_dir, input_dir)
lib_path = build_result.library_path
run_dir  = build_result.run_dir

#####
##### Run all Coriolis schemes (no momentum advection)
#####

# MITgcm selectCoriScheme (in mom_vi_coriolis.F):
#   0 = simple average, no hFac weighting
#   1 = Jamart wet-point average: sum(v*dx*hFacS)/sum(hFacS)
#   2 = hFac weighted average: 0.25*sum(v*dx*hFacS) * recip_hFacW
#   3 = energy-conserving with hFac weighted average
#   4 = 
mitgcm_schemes = [(0, "Simple"), (1, "Jamart"), (2, "hFac"), (3, "EnCons")]

nsteps     = Int(stop_time / Δt)
save_every = Int(save_interval / Δt)
Nsaves     = nsteps ÷ save_every

results = Dict{String, Any}()

for (scheme_id, label) in mitgcm_schemes
    @info "Running MITgcm scheme: $label (selectCoriScheme=$scheme_id)"

    # Write data namelist with this scheme
    write_data_file(run_dir; selectCoriScheme=scheme_id)

    # Clean up any leftover pickup files
    for f in readdir(run_dir)
        (startswith(f, "pickup") || startswith(f, "T.") || startswith(f, "S.") ||
         startswith(f, "U.") || startswith(f, "V.") || startswith(f, "Eta.")) &&
            rm(joinpath(run_dir, f); force=true)
    end

    ocean = MITgcmOceanSimulation(lib_path, run_dir; verbose=false)
    lib = ocean.library

    # Pre-allocate snapshot storage
    η_snaps = zeros(Nx, Ny, Nsaves)
    u_snaps = zeros(Nx, Ny, Nsaves)
    v_snaps = zeros(Nx, Ny, Nsaves)
    t_snaps = zeros(Nsaves)
    snap_idx = 0

    wall_clock = time_ns()

    for s in 1:nsteps
        step!(lib)

        if s % save_every == 0
            refresh_state!(ocean)
            snap_idx += 1
            η_snaps[:, :, snap_idx] .= ocean.etan
            u_snaps[:, :, snap_idx] .= ocean.uvel[:, :, 1]
            v_snaps[:, :, snap_idx] .= ocean.vvel[:, :, 1]
            t_snaps[snap_idx] = s * Δt
        end

        if s % (save_every * 5) == 0
            refresh_state!(ocean)
            elapsed = (time_ns() - wall_clock) * 1e-9
            u_max = maximum(abs, ocean.uvel)
            v_max = maximum(abs, ocean.vvel)
            @printf("[%4s] t=%.1f days  max|u|=%.3e  max|v|=%.3e  (%.1fs)\n",
                    label, s * Δt / 86400, u_max, v_max, elapsed)
            wall_clock = time_ns()

            if !isfinite(u_max) || u_max > 100
                @warn "$label: BLOW-UP at t=$(s*Δt/86400) days"
                break
            end
        end
    end

    refresh_state!(ocean)
    results[label] = (; etan=copy(ocean.etan),
                        uvel=copy(ocean.uvel),
                        vvel=copy(ocean.vvel),
                        η_snaps=η_snaps[:, :, 1:snap_idx],
                        u_snaps=u_snaps[:, :, 1:snap_idx],
                        v_snaps=v_snaps[:, :, 1:snap_idx],
                        t_snaps=t_snaps[1:snap_idx])

    finalize!(lib)
end

# Save results
jldsave(joinpath(@__DIR__, "mitgcm_coriolis_results.jld2"); results, Nx, Ny, Nz, Δx, Δy)
@info "Saved mitgcm_coriolis_results.jld2"

#####
##### Summary
#####

println("\n" * "="^60)
println("MITgcm CORIOLIS-ONLY STRESS TEST SUMMARY (momAdvection=.FALSE.)")
println("="^60)

for (_, label) in mitgcm_schemes
    r = results[label]
    u_max = maximum(abs, r.uvel)
    v_max = maximum(abs, r.vvel)
    η_max = maximum(abs, r.etan)
    status = isfinite(u_max) && u_max < 100 ? "OK" : "BLOW-UP"
    @printf("  %-4s: max|u|=%8.4f  max|v|=%8.4f  max|η|=%8.4f  [%s]\n",
            label, u_max, v_max, η_max, status)
end
println("="^60)
