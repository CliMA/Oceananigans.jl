using Reactant

using Test
using OffsetArrays
using Oceananigans
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.Utils: launch!
using SeawaterPolynomials: TEOS10EquationOfState
using KernelAbstractions: @kernel, @index
using Random

using Oceananigans.TimeSteppers: update_state!

using CUDA

if get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU"
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

OceananigansReactantExt = Base.get_extension(Oceananigans, :OceananigansReactantExt)
bottom_height(x, y) = - 0.5

function test_reactant_model_correctness(GridType, ModelType, grid_kw, model_kw)
    r_arch = ReactantState()
    r_grid = GridType(r_arch; grid_kw...)

    r_model = ModelType(r_grid; model_kw...)

    Nsteps = ConcreteRNumber(3)
    @time "  Compiling r_time_step!" begin
        r_time_step! = @compile sync=true my_time_step!(r_model)
    end

    return r_model
end

function my_time_step!(model)
    # Be paranoid and update state at iteration 0
    @trace if model.clock.iteration == 0
        update_state!(model, [])
    end
end

@info "Performing Reactanigans RectilinearGrid simulation tests..."
Nx, Ny, Nz = (10, 10, 10) # number of cells
halo = (7, 7, 7)
z = (-1, 0)
rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
hydrostatic_model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))

@info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
test_reactant_model_correctness(RectilinearGrid,
                                HydrostaticFreeSurfaceModel,
                                rectilinear_kw,
                                hydrostatic_model_kw)

@info "Done!"