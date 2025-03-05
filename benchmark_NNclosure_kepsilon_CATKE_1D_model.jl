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
function setup_model(Nz, closure)
    Δz = 8
    Lz = Δz * Nz

    grid = RectilinearGrid(GPU(),
                           topology = (Flat, Flat, Bounded),
                           size = Nz,
                           halo = 3,
                           z = (-Lz, 0))

    @info "Built a grid: $grid."
    
    dTdz = 0.014
    dSdz = 0.0021

    T_surface = 25
    S_surface = 37

    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(2e-4))
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(3e-4))
    S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(3e-4))

    f₀ = 1e-4
    coriolis = FPlane(f=f₀)

    T_initial(z) = dTdz * z + T_surface
    S_initial(z) = dSdz * z + S_surface

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
        boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
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
        boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
    );

    #####
    ##### Initial conditions
    #####

    # resting initial condition
    noise(z) = rand() * exp(z / 8)

    T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
    S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)

    set!(model, T=T_initial_noisy, S=S_initial_noisy)
    update_state!(model)
    return model
end

function benchmark_timestep(N, closure_str)
    Δt = 5minutes

    if closure_str == "CATKE"
        closure = CATKE_ocean_closure()
    elseif closure_str == "k_epsilon"
        closure = TKEDissipationVerticalDiffusivity()
    else
        closure = (XinKaiLocalVerticalDiffusivity(), NNFluxClosure(GPU()))
    end

    model = setup_model(N, closure);

    @info "Benchmarking $closure_str closure with $N grid points"

    for _ in 1:3
        time_step!(model, Δt)
    end

    for _ in 1:200
        NVTX.@range "$(closure_str), N $N" begin
            time_step!(model, Δt)
        end
    end
end

Ns = [32, 48, 64, 96, 128]

for closure_str in ["NN", "CATKE", "k_epsilon"], N in Ns
    benchmark_timestep(N, closure_str)
end