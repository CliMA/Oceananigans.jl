using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
import Oceananigans.Architectures: arch_array

@inline arch_array(::CPU, A) = A
@inline arch_array(::GPU, A) = CuArray(A)

@testset "Immersed boundaries with hydrostatic free surface models" begin
    @info "Testing immersed boundaries vertical integrals"

    for arch in archs
        Nx = 5
        Ny = 5

        underlying_grid = RectilinearGrid(arch,
                                          size = (Nx, Ny, 3),
                                          extent = (Nx, Ny, 3),
                                          topology = (Periodic, Periodic, Bounded))

        # B for bathymetry
        B = [-3. for i=1:Nx, j=1:Ny ]
        B[2:Nx-1,2:Ny-1] .= [-2. for i=2:Nx-1, j=2:Ny-1 ]
        B[3:Nx-2,3:Ny-2] .= [-1. for i=3:Nx-2, j=3:Ny-2 ]

        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(B))

        model = HydrostaticFreeSurfaceModel(grid = grid,
                                            free_surface = ImplicitFreeSurface(),
                                            buoyancy = nothing,
                                            tracers = nothing,
                                            closure = nothing)

        x_ref = arch_array(arch, [3.0  3.0  3.0  3.0  3.0
                                  3.0  2.0  2.0  2.0  2.0
                                  3.0  2.0  1.0  1.0  2.0
                                  3.0  2.0  2.0  2.0  2.0
                                  3.0  3.0  3.0  3.0  3.0]')

        y_ref = arch_array(arch, [3.0  3.0  3.0  3.0  3.0
                                  3.0  2.0  2.0  2.0  3.0
                                  3.0  2.0  1.0  2.0  3.0
                                  3.0  2.0  1.0  2.0  3.0
                                  3.0  2.0  2.0  2.0  3.0]')

        fs = model.free_surface
        vertically_integrated_lateral_areas = fs.implicit_step_solver.vertically_integrated_lateral_areas

        ∫Axᶠᶜᶜ = vertically_integrated_lateral_areas.xᶠᶜᶜ
        ∫Ayᶜᶠᶜ = vertically_integrated_lateral_areas.yᶜᶠᶜ

        ∫Axᶠᶜᶜ = interior(∫Axᶠᶜᶜ)
        ∫Ayᶜᶠᶜ = interior(∫Ayᶜᶠᶜ)

        Ax_ok = ∫Axᶠᶜᶜ[:, :, 1] ≈ x_ref
        Ay_ok = ∫Ayᶜᶠᶜ[:, :, 1] ≈ y_ref

        @test (Ax_ok & Ay_ok)
    end
end

