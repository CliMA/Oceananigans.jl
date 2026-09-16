include("dependencies_for_runtests.jl")

using Random
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom,
                                       immersed_peripheral_node, immersed_inactive_node, mask_immersed_field!
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ

#####
##### Boundary / halo handling of the SplitExplicitFreeSurface barotropic corrector.
#####
##### The barotropic correction is masked at solid velocity faces. The mask must catch every
##### immersed wall (including walls buried at a domain edge) while leaving the exterior halo
##### and wet domain edges untouched — otherwise it zeroes velocity halos (which leaked into
##### the interior and broke the hydrostatic regression) or walls off open boundaries.
#####
##### Dynamical invariants are checked from a *random* initial state and a single step, so the
##### masking is shown to actively zero what it should rather than passing from a zero start.
#####
##### Grid predicates and depth integrals are evaluated on a CPU copy of the grid, and fields
##### are pulled to the host with `Array(interior(...))`, so the tests run on CPU and GPU.
#####

# Random initial condition (host arrays moved to the field's architecture, halos included).
function randomize!(field)
    arch = architecture(field)
    parent(field) .= on_architecture(arch, randn(size(parent(field))...))
    return field
end

function randomize_prognostic_state!(model)
    randomize!(model.velocities.u)
    randomize!(model.velocities.v)
    for c in model.tracers
        randomize!(c)
    end
    randomize!(model.free_surface.displacement)
    return model
end

build_model(grid; substeps=10, extend_halos=true, boundary_conditions=NamedTuple()) =
    HydrostaticFreeSurfaceModel(grid; free_surface=SplitExplicitFreeSurface(grid; substeps, extend_halos),
                                buoyancy=BuoyancyTracer(), tracers=:b, boundary_conditions)

# After a random init + one step: immersed velocities are zero, the interior is alive, and the
# depth-integrated velocity equals the barotropic transport (the corrector's core invariant).
function check_common_invariants(model)
    grid = on_architecture(CPU(), model.grid)
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    U = Array(interior(model.free_surface.barotropic_velocities.U))
    V = Array(interior(model.free_surface.barotropic_velocities.V))
    Nx, Ny, Nz = size(grid)
    f, c = Face(), Center()

    @test all(u[i, j, k] == 0 for i in 1:size(u, 1), j in 1:size(u, 2), k in 1:size(u, 3) if immersed_peripheral_node(i, j, k, grid, f, c, c))
    @test all(v[i, j, k] == 0 for i in 1:size(v, 1), j in 1:size(v, 2), k in 1:size(v, 3) if immersed_peripheral_node(i, j, k, grid, c, f, c))

    @test maximum(abs, u) > 0
    @test maximum(abs, v) > 0

    ∫u(i, j) = sum(Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k] for k in 1:Nz)
    ∫v(i, j) = sum(Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k] for k in 1:Nz)
    @test maximum(abs(∫u(i, j) - U[i, j, 1]) for i in 2:Nx, j in 1:Ny; init=0.0) < 1e-12
    @test maximum(abs(∫v(i, j) - V[i, j, 1]) for i in 1:Nx, j in 2:Ny; init=0.0) < 1e-12

    return nothing
end

bump(x, y) = -0.5 - 0.4 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.05)

@testset "SplitExplicitFreeSurface boundary handling [$arch]" for arch in archs

    @testset "Corrector mask predicates" begin
        # immersed_peripheral isolates one-sided immersed walls but misses a wall buried at a
        # domain edge; immersed_inactive catches those and — unlike raw inactive_node — stays
        # false in the exterior halo. Their union is the corrector mask.
        underlying = RectilinearGrid(arch, size=(6, 1, 1), x=(0, 6), y=(0, 1), z=(0, 1),
                                     topology=(Bounded, Periodic, Bounded))
        ibg = ImmersedBoundaryGrid(underlying, GridFittedBottom((x, y) -> x > 5 ? 1.0 : 0.0)) # cell 6 = land at east edge
        grid = on_architecture(CPU(), ibg)
        f, c = Face(), Center()
        mask(i) = immersed_peripheral_node(i, 1, 1, grid, f, c, c) | immersed_inactive_node(i, 1, 1, grid, f, c, c)

        @test  immersed_inactive_node(7, 1, 1, grid, f, c, c)     # edge-buried wall caught by inactive...
        @test !immersed_peripheral_node(7, 1, 1, grid, f, c, c)   # ... and missed by peripheral
        @test mask(6)   # wet/solid interface wall masked
        @test mask(7)   # edge-buried wall masked
        @test !mask(8)  # deep exterior halo NOT masked
        @test immersed_inactive_node(1, 1, 1, on_architecture(CPU(), underlying), f, c, c) == false # non-immersed fallback
    end

    @testset "No spurious masking on a non-immersed grid" begin
        # Raw inactive_node was true throughout the exterior halo, which zeroed velocity halos
        # and broke the Bounded hydrostatic regression. The corrector mask must be empty here.
        grid = on_architecture(CPU(), RectilinearGrid(arch, size=(6, 6, 2), x=(0, 6), y=(0, 6), z=(-1, 0),
                                                      topology=(Bounded, Bounded, Bounded)))
        f, c = Face(), Center()
        maskU(i, j) = immersed_peripheral_node(i, j, 1, grid, f, c, c) | immersed_inactive_node(i, j, 1, grid, f, c, c)
        maskV(i, j) = immersed_peripheral_node(i, j, 1, grid, c, f, c) | immersed_inactive_node(i, j, 1, grid, c, f, c)
        @test !any(maskU(i, j) for i in -1:grid.Nx+3, j in -1:grid.Ny+3)
        @test !any(maskV(i, j) for i in -1:grid.Nx+3, j in -1:grid.Ny+3)
    end

    @testset "mask_immersed_field! zeroes immersed walls" begin
        underlying = RectilinearGrid(arch, size=(6, 1, 2), x=(0, 6), y=(0, 1), z=(-1, 0),
                                     topology=(Bounded, Periodic, Bounded))
        grid = ImmersedBoundaryGrid(underlying, GridFittedBottom((x, y) -> 2 < x < 3 ? 0.0 : -1.0)) # column 3 = land
        u = XFaceField(grid)
        set!(u, (x, y, z) -> 1)
        mask_immersed_field!(u)
        ui = Array(interior(u))[:, 1, 1]
        @test ui[3] == 0 && ui[4] == 0        # land-column interface faces zeroed
        @test all(ui[[1, 2, 5, 6, 7]] .== 1)  # wet faces untouched
    end

    @testset "Random-init invariants — $(name)" for (name, grid) in (
        ("non-immersed, closed",  RectilinearGrid(arch, size=(8, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0), topology=(Bounded, Bounded, Bounded))),
        ("immersed, closed", ImmersedBoundaryGrid(RectilinearGrid(arch, size=(8, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                                 topology=(Bounded, Bounded, Bounded)), GridFittedBottom(bump))),
        ("periodic x, closed y", RectilinearGrid(arch, size=(8, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0), topology=(Periodic, Bounded, Bounded))),
        ("immersed, periodic x", ImmersedBoundaryGrid(RectilinearGrid(arch, size=(8, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                                 topology=(Periodic, Bounded, Bounded)), GridFittedBottom(bump))),
       )

        model = build_model(grid)
        randomize_prognostic_state!(model)
        time_step!(model, 1e-3)
        check_common_invariants(model)

        # The normal velocity at a closed (NormalFlow{Nothing}) wall is zeroed per level by the
        # boundary-condition fill — no flow through the wall at any depth, even from a random start.
        u = Array(interior(model.velocities.u))
        v = Array(interior(model.velocities.v))
        TX, TY, _ = topology(grid)
        if TX == Bounded
            @test all(view(u, 1, :, :) .== 0)
            @test all(view(u, 9, :, :) .== 0)
        end
        if TY == Bounded
            @test all(view(v, :, 1, :) .== 0)
            @test all(view(v, :, 9, :) .== 0)
        end
    end

    @testset "Periodic: extended vs non-extended halos agree" begin
        function run(extend_halos)
            grid = RectilinearGrid(arch, size=(16, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                   topology=(Periodic, Bounded, Bounded))
            model = build_model(grid; extend_halos)
            set!(model, b = (x, y, z) -> z + 0.02 * sinpi(2x) * exp(-(y - 0.5)^2 / 0.05))
            for _ in 1:15
                time_step!(model, 0.004)
            end
            return Array(interior(model.velocities.u)), Array(interior(model.free_surface.displacement))
        end
        u_ext, η_ext = run(true)
        u_non, η_non = run(false)
        @test u_ext ≈ u_non
        @test η_ext ≈ η_non
    end

    @testset "Open boundary with prescribed normal flow is not walled off" begin
        underlying = RectilinearGrid(arch, size=(8, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                     topology=(Bounded, Bounded, Bounded))
        U_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0.5))
        model = build_model(underlying; boundary_conditions=(; U=U_bcs))
        U = model.free_surface.barotropic_velocities.U
        grid = on_architecture(CPU(), underlying)
        f, c = Face(), Center()

        @test typeof(model.free_surface).parameters[1].name.name == :LocalHaloFilling
        @test !any(immersed_peripheral_node(1, j, 1, grid, f, c, c) |
                   immersed_inactive_node(1, j, 1, grid, f, c, c) for j in 1:grid.Ny)
    end

    @testset "Free surface does not evolve on land at an open boundary" begin
        underlying = RectilinearGrid(arch, size=(8, 8, 4), x=(0, 1), y=(0, 1), z=(-1, 0),
                                     topology=(Bounded, Bounded, Bounded))

        ibg = ImmersedBoundaryGrid(underlying, GridFittedBottom((x, y) -> (x < 1/8 && y < 0.5) ? 0.0 : -1.0))
        U_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(0.5))
        model = build_model(ibg; boundary_conditions=(; U=U_bcs))

        for _ in 1:10
            time_step!(model, 1e-3)
        end

        grid = on_architecture(CPU(), ibg)
        η = Array(interior(model.free_surface.displacement))
        Nx, Ny, Nz = size(grid)
        c = Center()

        solid(i, j) = immersed_peripheral_node(i, j, Nz, grid, c, c, c)
        @test all(η[i, j, 1] == 0 for i in 1:Nx, j in 1:Ny if solid(i, j))
        @test maximum(abs(η[i, j, 1]) for i in 1:Nx, j in 1:Ny if !solid(i, j); init=0.0) > 0
    end
end
