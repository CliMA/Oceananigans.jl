# Accuracy gate for tracer advection on the OctaHEALPix SphericalShellGrid.
#
# Unlike test_spherical_shell_tracer_advection.jl (which checks conservation + stability),
# this verifies CORRECTNESS against the exact solution of solid-body rotation, with error
# that must CONVERGE under grid refinement. It is the gate that distinguishes "runs and
# conserves mass" from "advects correctly."
#
# Exact solution: solid-body rotation is a rigid translation of the blob along a great
# circle. For u_extrinsic = cosλ sinφ, v_extrinsic = -sinλ (Ω⃗ = -x̂, Ω = u₀/R = 1) the
# blob center is ĉ(t) = (0, cos t, -sin t). The mass centroid must ride ĉ(t): cross-track
# ≈ 0 AND along-track phase error → 0 with resolution.

using Oceananigans
using Oceananigans.Advection: WENO, cell_advection_timescale
using Oceananigans.Fields: interpolate
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_tracer_tendencies!, compute_transport_velocities!
using Oceananigans.Operators: Az⁻¹ᶜᶜᶜ, Vᶜᶜᶜ, horizontal_transport_flux_div_xyᶜᶜᶜ
using Test

function tilted_solid_body_tracer(λ, φ, t, σ)
    q = cosd(φ) * sind(λ) * cos(t) - sind(φ) * sin(t)
    return exp(-(acos(clamp(q, -1, 1)) / σ)^2 / 2)
end

function volume_weighted_tracer_mass(model)
    grid = model.grid
    c = model.tracers.c
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    mass = zero(FT)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mass += Vᶜᶜᶜ(i, j, k, grid) * c[i, j, k]
    end

    return mass
end

function maximum_abs_transport_w(model, k)
    grid = model.grid
    w = model.transport_velocities.w
    FT = eltype(grid)
    Nx, Ny, _ = size(grid)
    maximum_w = zero(FT)

    for j in 1:Ny, i in 1:Nx
        maximum_w = max(maximum_w, abs(w[i, j, k]))
    end

    return maximum_w
end

function maximum_abs_horizontal_transport_divergence(model)
    grid = model.grid
    u, v, _ = model.transport_velocities
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    maximum_divergence = zero(FT)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        divergence = horizontal_transport_flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v) *
                     Az⁻¹ᶜᶜᶜ(i, j, k, grid)
        maximum_divergence = max(maximum_divergence, abs(divergence))
    end

    return maximum_divergence
end

# returns (mass_drift, cross_track, final_phase_error_deg, max_phase_error_deg)
# after one full revolution.
function solid_body_com_error(N; T = 2π, σ = 0.5)
    FT = Float64
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              z = (zero(FT), one(FT)), radius = one(FT), halo = (3, 3, 3))
    Nx, Ny, Nz = size(grid)
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
        coriolis = nothing, free_surface = nothing, momentum_advection = nothing,
        tracer_advection = WENO(FT; order = 5))
    set!(model, u = (λ, φ, z) -> cosd(λ) * sind(φ), v = (λ, φ, z) -> -sind(λ))
    set!(model, c = (λ, φ, z) -> exp(-(acos(clamp(cosd(φ) * sind(λ), -one(FT), one(FT))) / σ)^2 / 2))

    # NOTE: keep all flattenings column-major (i-fastest) so cell c, (λ,φ), and V are
    # paired correctly. vec(2D array) is column-major; build Vc the same way (comma, then
    # vec) — NOT `for i in 1:Nx for j in 1:Ny` (that is j-fastest and transposes the pairing).
    λ = vec(Array(λnodes(grid, Center(), Center(), Center())))
    φ = vec(Array(φnodes(grid, Center(), Center(), Center())))
    X = @. cosd(φ) * cosd(λ); Y = @. cosd(φ) * sind(λ); Z = @. sind(φ)
    Vc = vec([Vᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny])

    mass() = sum(Vc .* vec(Array(interior(model.tracers.c))[:, :, 1]))
    M0 = mass()

    function com_error()
        t = model.clock.time
        c = vec(Array(interior(model.tracers.c))[:, :, 1])
        m = Vc .* c
        M = sum(m)
        nx = sum(m .* X) / M
        ny = sum(m .* Y) / M
        nz = sum(m .* Z) / M
        nrm = sqrt(nx^2 + ny^2 + nz^2)
        nx /= nrm
        ny /= nrm
        nz /= nrm

        dphase = mod(atan(nz, ny) - atan(-sin(t), cos(t)) + π, 2π) - π
        return (M - M0) / M0, abs(nx), abs(rad2deg(dphase))
    end

    dt = convert(FT, 1//5) * cell_advection_timescale(grid, model.velocities)
    sample_interval = T / 16
    next_sample_time = zero(T)
    final_drift, final_cross_track, final_phase_error = com_error()
    max_phase_error = final_phase_error

    while model.clock.time < T
        step_Δt = min(dt, T - model.clock.time)
        time_step!(model, step_Δt)

        if model.clock.time >= next_sample_time || model.clock.time >= T
            final_drift, final_cross_track, final_phase_error = com_error()
            max_phase_error = max(max_phase_error, final_phase_error)
            next_sample_time += sample_interval
        end
    end

    return final_drift, final_cross_track, final_phase_error, max_phase_error
end

function local_solid_body_tracer_error(N; target = 1.24, σ = 0.5)
    FT = Float64
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              z = (zero(FT), one(FT)), radius = one(FT), halo = (3, 3, 3))
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
                                        coriolis = nothing, free_surface = nothing,
                                        momentum_advection = nothing,
                                        tracer_advection = WENO(FT; order = 5))

    set!(model, u = (λ, φ, z) -> cosd(λ) * sind(φ),
                v = (λ, φ, z) -> -sind(λ),
                c = (λ, φ, z) -> tilted_solid_body_tracer(λ, φ, zero(FT), σ))

    dt = convert(FT, 1//5) * cell_advection_timescale(grid, model.velocities)

    while model.clock.time < target
        time_step!(model, min(dt, target - model.clock.time))
    end

    t = model.clock.time
    c = model.tracers.c
    λ = Array(λnodes(grid, Center(), Center(), Center()))
    φ = Array(φnodes(grid, Center(), Center(), Center()))

    native_l1 = zero(FT)
    native_l2 = zero(FT)
    native_mass = zero(FT)
    native_l∞ = zero(FT)

    for j in 1:grid.Ny, i in 1:grid.Nx
        difference = c[i, j, 1] - tilted_solid_body_tracer(λ[i, j], φ[i, j], t, σ)
        volume = Vᶜᶜᶜ(i, j, 1, grid)
        native_l1 += volume * abs(difference)
        native_l2 += volume * difference^2
        native_mass += volume
        native_l∞ = max(native_l∞, abs(difference))
    end

    interpolated_l1 = zero(FT)
    interpolated_l2 = zero(FT)
    interpolated_l∞ = zero(FT)
    interpolated_count = 0

    for φᵢ in -89:1:89, λᵢ in -179:1:179
        λ′ = convert(FT, λᵢ)
        φ′ = convert(FT, φᵢ)
        difference = interpolate((λ′, φ′, convert(FT, 1//2)), c) -
                     tilted_solid_body_tracer(λ′, φ′, t, σ)
        interpolated_l1 += abs(difference)
        interpolated_l2 += difference^2
        interpolated_l∞ = max(interpolated_l∞, abs(difference))
        interpolated_count += 1
    end

    return (native_l1 / native_mass,
            sqrt(native_l2 / native_mass),
            native_l∞,
            interpolated_l1 / interpolated_count,
            sqrt(interpolated_l2 / interpolated_count),
            interpolated_l∞)
end

function uniform_tracer_transport_errors(N; target = 1.24)
    FT = Float64
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              z = (zero(FT), one(FT)), radius = one(FT), halo = (3, 3, 3))
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
                                        coriolis = nothing, free_surface = nothing,
                                        momentum_advection = nothing,
                                        tracer_advection = WENO(FT; order = 5))

    set!(model, u = (λ, φ, z) -> cosd(λ) * sind(φ),
                v = (λ, φ, z) -> -sind(λ),
                c = (λ, φ, z) -> one(FT))

    compute_tracer_tendencies!(model)

    Gc = model.timestepper.Gⁿ.c
    max_tendency = zero(FT)

    for j in 1:grid.Ny, i in 1:grid.Nx
        max_tendency = max(max_tendency, abs(Gc[i, j, 1]))
    end

    dt = convert(FT, 1//5) * cell_advection_timescale(grid, model.velocities)

    while model.clock.time < target
        time_step!(model, min(dt, target - model.clock.time))
    end

    c = model.tracers.c
    max_error = zero(FT)
    minimum_c = one(FT)
    maximum_c = one(FT)

    for j in 1:grid.Ny, i in 1:grid.Nx
        cᵢⱼ = c[i, j, 1]
        max_error = max(max_error, abs(cᵢⱼ - one(FT)))
        minimum_c = min(minimum_c, cᵢⱼ)
        maximum_c = max(maximum_c, cᵢⱼ)
    end

    return max_tendency, max_error, minimum_c, maximum_c
end

function zonal_transport_divergence(N)
    FT = Float64
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              z = (zero(FT), one(FT)), radius = one(FT), halo = (3, 3, 3))

    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
                                        coriolis = nothing, free_surface = nothing,
                                        momentum_advection = nothing,
                                        tracer_advection = WENO(FT; order = 5))

    set!(model, u = (λ, φ, z) -> cosd(φ),
                v = (λ, φ, z) -> zero(FT),
                c = (λ, φ, z) -> one(FT))

    compute_transport_velocities!(model, model.free_surface)

    return maximum_abs_horizontal_transport_divergence(model)
end

function multilayer_rigid_lid_transport_metrics(N, Nz; steps = 11)
    FT = Float64
    half = convert(FT, 1//2)
    tenth = convert(FT, 1//10)
    σ = half
    z = collect(range(-one(FT), stop = zero(FT), length = Nz + 1))
    grid = SphericalShellGrid(CPU(), FT; mapping = OctaHEALPixMapping(N),
                              size = (2N, 2N, Nz), z,
                              radius = one(FT), halo = (3, 3, 3))

    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing,
                                        coriolis = nothing, free_surface = nothing,
                                        momentum_advection = nothing,
                                        tracer_advection = WENO(FT; order = 5))

    vertical_structure = z -> one(FT) + half * z

    set!(model, u = (λ, φ, z) -> vertical_structure(z) * cosd(λ) * sind(φ),
                v = (λ, φ, z) -> -vertical_structure(z) * sind(λ),
                c = (λ, φ, z) -> (one(FT) + tenth * z) *
                                  exp(-(acos(clamp(cosd(φ) * sind(λ), -one(FT), one(FT))) / σ)^2 / 2))

    initial_mass = volume_weighted_tracer_mass(model)
    compute_transport_velocities!(model, model.free_surface)

    bottom_w = maximum_abs_transport_w(model, 1)
    top_w = maximum_abs_transport_w(model, Nz + 1)
    interior_w = maximum(maximum_abs_transport_w(model, k) for k in 2:Nz)

    dt = convert(FT, 1//5) * cell_advection_timescale(grid, model.velocities)

    for _ in 1:steps
        time_step!(model, dt)
    end

    final_mass = volume_weighted_tracer_mass(model)
    mass_drift = (final_mass - initial_mass) / initial_mass

    return bottom_w, top_w, interior_w, mass_drift
end

@testset "OctaHEALPix tracer transport zonal divergence guard" begin
    @test zonal_transport_divergence(16) < 1e-10
end

@testset "OctaHEALPix uniform tracer free-stream preservation" begin
    tendency₁₆, error₁₆, minimum₁₆, maximum₁₆ = uniform_tracer_transport_errors(16)
    tendency₃₂, error₃₂, minimum₃₂, maximum₃₂ = uniform_tracer_transport_errors(32)

    @info "uniform tracer over-pole transport: max tendency $(tendency₁₆), $(tendency₃₂); max error $(error₁₆), $(error₃₂); extrema [$(minimum₁₆), $(maximum₁₆)], [$(minimum₃₂), $(maximum₃₂)]"

    @test tendency₁₆ < 1e-12
    @test tendency₃₂ < 1e-12
    @test error₁₆ < 1e-12
    @test error₃₂ < 1e-12
    @test abs(minimum₁₆ - 1) < 1e-12
    @test abs(maximum₁₆ - 1) < 1e-12
    @test abs(minimum₃₂ - 1) < 1e-12
    @test abs(maximum₃₂ - 1) < 1e-12
end

@testset "OctaHEALPix multilayer rigid-lid tracer transport" begin
    bottom_w, top_w, interior_w, mass_drift = multilayer_rigid_lid_transport_metrics(16, 4)

    @info "multilayer rigid-lid transport: bottom_w=$(bottom_w), top_w=$(top_w), interior_w=$(interior_w), mass_drift=$(mass_drift)"

    @test bottom_w ≤ 100eps(Float64)
    @test top_w ≤ 1e-12
    @test interior_w > 1e-8
    @test abs(mass_drift) < 1e-12
end

@testset "OctaHEALPix tracer advection ACCURACY (solid-body rotation vs exact)" begin
    drift₁₆, cross₁₆, phase₁₆, max_phase₁₆ = solid_body_com_error(16)
    drift₃₂, cross₃₂, phase₃₂, max_phase₃₂ = solid_body_com_error(32)
    tolerance₁₆ = rad2deg(2 * sqrt(π) / 16)
    tolerance₃₂ = rad2deg(2 * sqrt(π) / 32)

    @info "COM accuracy: phase_err N=16 = $(phase₁₆)°, N=32 = $(phase₃₂)°; max = $(max_phase₁₆)°, $(max_phase₃₂)°  (cross-track $(cross₁₆), $(cross₃₂))"

    # conservation + on-trajectory (these already pass)
    @test abs(drift₁₆) < 1e-10
    @test abs(drift₃₂) < 1e-10
    @test cross₁₆ < 1e-6
    @test cross₃₂ < 1e-6

    # CORRECTNESS: the COM must track the exact great circle for a full revolution.
    # The tolerance is two nominal OctaHEALPix grid spacings Δθ = sqrt(π) / N.
    @test phase₁₆ < tolerance₁₆
    @test phase₃₂ < tolerance₃₂
    @test max_phase₁₆ < tolerance₁₆
    @test max_phase₃₂ < tolerance₃₂
    # The local-field gate below is the refinement check for the full tracer
    # solution. Keep this COM gate tied to Greg's stated acceptance criterion:
    # one full revolution returns to the initial position within two nominal
    # OctaHEALPix grid spacings, with no cross-track drift.
end

@testset "OctaHEALPix tracer local-field accuracy at pole crossing" begin
    native_l1₁₆, native_l2₁₆, native_l∞₁₆, interpolated_l1₁₆, interpolated_l2₁₆, interpolated_l∞₁₆ =
        local_solid_body_tracer_error(16)
    native_l1₃₂, native_l2₃₂, native_l∞₃₂, interpolated_l1₃₂, interpolated_l2₃₂, interpolated_l∞₃₂ =
        local_solid_body_tracer_error(32)
    native_l1₆₄, native_l2₆₄, native_l∞₆₄, interpolated_l1₆₄, interpolated_l2₆₄, interpolated_l∞₆₄ =
        local_solid_body_tracer_error(64)

    @info "local tracer error at t≈1.24: native L∞ $(native_l∞₁₆), $(native_l∞₃₂), $(native_l∞₆₄); interpolated L∞ $(interpolated_l∞₁₆), $(interpolated_l∞₃₂), $(interpolated_l∞₆₄)"

    @test native_l1₃₂ < native_l1₁₆
    @test native_l1₆₄ < native_l1₃₂
    @test native_l2₃₂ < native_l2₁₆
    @test native_l2₆₄ < native_l2₃₂
    @test native_l∞₃₂ < native_l∞₁₆
    @test native_l∞₆₄ < native_l∞₃₂

    @test interpolated_l1₃₂ < interpolated_l1₁₆
    @test interpolated_l1₆₄ < interpolated_l1₃₂
    @test interpolated_l2₃₂ < interpolated_l2₁₆
    @test interpolated_l2₆₄ < interpolated_l2₃₂
    @test interpolated_l∞₃₂ < interpolated_l∞₁₆
    @test interpolated_l∞₆₄ < interpolated_l∞₃₂
end
