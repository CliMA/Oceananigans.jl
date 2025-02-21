using Oceananigans
include("NN_closure_global_Ri_nof_BBLRifirstzone510_train62newstrongSO_20seed_Ri8020_round3.jl")
include("xin_kai_vertical_diffusivity_local_2step_train56newstrongSO.jl")
pushfirst!(LOAD_PATH, @__DIR__)
using Statistics

using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, TKEDissipationVerticalDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10
using NVTX

function CATKE_ocean_closure()
    mixing_length = CATKEMixingLength(Cᵇ=0.28)
    turbulent_kinetic_energy_equation = CATKEEquation()
    return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)
end

# number of grid points
function setup_model(Nxy, Nz, closure)
    Δz = 8
    Nz = Nz
    Lz = Δz * Nz

    Nx = Ny = Nxy
    Lx = 4000kilometers
    Ly = 6000kilometers

    grid = RectilinearGrid(GPU(), Float64,
                           topology = (Bounded, Bounded, Bounded),
                           size = (Nx, Ny, Nz),
                           halo = (3, 3, 3),
                              x = (-Lx/2, Lx/2),
                              y = (-Ly/2, Ly/2),
                              z = (-Lz, 0))

    @info "Built a grid: $grid."

    T_north = 0
    T_south = 30
    T_mid = (T_north + T_south) / 2
    ΔT = T_south - T_north

    S_north = 34
    S_south = 37
    S_mid = (S_north + S_south) / 2

    τ₀ = 1e-4

    μ_drag = 1/30days
    μ_T = 1/8days

    @inline T_initial(x, y, z) = 10 + 20 * (1 + z / Lz)

    @inline surface_u_flux(x, y, t) = -τ₀ * cos(2π * y / Ly)

    surface_u_flux_bc = FluxBoundaryCondition(surface_u_flux)

    @inline u_drag(x, y, t, u) = @inbounds -μ_drag * Lz * u
    @inline v_drag(x, y, t, v) = @inbounds -μ_drag * Lz * v

    u_drag_bc  = FluxBoundaryCondition(u_drag; field_dependencies=:u)
    v_drag_bc  = FluxBoundaryCondition(v_drag; field_dependencies=:v)

    u_bcs = FieldBoundaryConditions(   top = surface_u_flux_bc, 
                                    bottom = u_drag_bc,
                                     north = ValueBoundaryCondition(0),
                                     south = ValueBoundaryCondition(0))

    v_bcs = FieldBoundaryConditions(   top = FluxBoundaryCondition(0),
                                    bottom = v_drag_bc,
                                      east = ValueBoundaryCondition(0),
                                      west = ValueBoundaryCondition(0))

    @inline T_ref(y) = T_mid - ΔT / Ly * y
    @inline surface_T_flux(x, y, t, T) = μ_T * Δz * (T - T_ref(y))
    surface_T_flux_bc = FluxBoundaryCondition(surface_T_flux; field_dependencies=:T)
    T_bcs = FieldBoundaryConditions(top = surface_T_flux_bc)

    @inline S_ref(y) = (S_north - S_south) / Ly * y + S_mid
    @inline S_initial(x, y, z) = S_ref(y)
    @inline surface_S_flux(x, y, t, S) = μ_T * Δz * (S - S_ref(y))
    surface_S_flux_bc = FluxBoundaryCondition(surface_S_flux; field_dependencies=:S)
    S_bcs = FieldBoundaryConditions(top = surface_S_flux_bc)

    coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=45, radius=6371e3)

    if closure isa CATKEVerticalDiffusivity
        tracers = (:T, :S, :e)
    elseif closure isa TKEDissipationVerticalDiffusivity
        tracers = (:T, :S, :e, :ϵ)
    else
        tracers = (:T, :S)
    end

    advection_scheme = WENO(grid = grid)

    @info "Building a model..."

    # This is a weird bug. If a model is not initialized with a closure other than XinKaiVerticalDiffusivity,
    # the code will throw a CUDA: illegal memory access error for models larger than a small size.
    # This is a workaround to initialize the model with a closure other than XinKaiVerticalDiffusivity first,
    # then the code will run without any issues.
    model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
        momentum_advection = advection_scheme,
        tracer_advection = advection_scheme,
        buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
        coriolis = coriolis,
        closure = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5),
        tracers = tracers,
        boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
    );
    
    model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
        momentum_advection = advection_scheme,
        tracer_advection = advection_scheme,
        buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
        coriolis = coriolis,
        closure = closure,
        tracers = tracers,
        boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
    );

    noise(z) = rand() * exp(z / 8)

    T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(z)
    S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(z)
    
    set!(model, T=T_initial_noisy, S=S_initial_noisy)
    update_state!(model)
    return model
end

function benchmark_timestep(Nxy, Nz, closure_str)
    Δt = 5minutes
    @info "Benchmarking $closure_str closure with $Nxy horizontal, $Nz vertical grid points"

    if closure_str == "CATKE"
        closure = CATKE_ocean_closure()
    elseif closure_str == "k_epsilon"
        closure = TKEDissipationVerticalDiffusivity()
    else
        closure = (XinKaiLocalVerticalDiffusivity(), NNFluxClosure(GPU()))
    end

    model = setup_model(Nxy, Nz, closure);

    for _ in 1:3
        time_step!(model, Δt)
    end

    for _ in 1:200
        NVTX.@range "$(closure_str), Nxy $Nxy, Nz $Nz" begin
            time_step!(model, Δt)
        end
    end
end

Nxys = [32, 48, 64, 96, 128]
Nz = 192

for closure_str in ["NN", "CATKE", "k_epsilon"], Nxy in Nxys
    benchmark_timestep(Nxy, Nz, closure_str)
end

Nxy = 128
Nzs = [32, 48, 64, 96, 128, 192, 256]

for closure_str in ["NN", "CATKE", "k_epsilon"], Nz in Nzs
    benchmark_timestep(Nxy, Nz, closure_str)
end