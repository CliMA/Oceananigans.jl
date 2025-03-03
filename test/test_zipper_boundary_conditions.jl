include("dependencies_for_runtests.jl")

using OrthogonalSphericalShellGrids: Zipper

@testset "Zipper boundary conditions..." begin
    grid = TripolarGrid(size = (10, 10, 1))
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    c = CenterField(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)

    @test c.boundary_conditions.north.classification isa Zipper
    @test u.boundary_conditions.north.classification isa Zipper
    @test v.boundary_conditions.north.classification isa Zipper

    # The velocity fields are reversed at the north boundary 
    # boundary_conditions.north.condition == -1, while the tracer
    # is not: boundary_conditions.north.condition == 1
    @test c.boundary_conditions.north.condition == 1
    @test u.boundary_conditions.north.condition == -1
    @test v.boundary_conditions.north.condition == -1

    set!(c, 1)
    set!(u, 1)
    set!(v, 1)

    fill_halo_regions!(c)
    fill_halo_regions!(u)   
    fill_halo_regions!(v)

    north_boundary_c = view(c.data, :, Ny+1:Ny+Hy, 1)
    north_boundary_v = view(v.data, :, Ny+1:Ny+Hy, 1)
    @test all(north_boundary_c .== 1)
    @test all(north_boundary_v .== -1)

    # U is special, because periodicity is hardcoded in the x-direction
    north_interior_boundary_u = view(u.data, 2:Nx-1, Ny+1:Ny+Hy, 1)
    @test all(north_interior_boundary_u .== -1)

    north_boundary_u_left  = view(u.data, 1, Ny+1:Ny+Hy, 1)
    north_boundary_u_right = view(u.data, Nx+1, Ny+1:Ny+Hy, 1)
    @test all(north_boundary_u_left  .== 1)
    @test all(north_boundary_u_right .== 1)

    bottom(x, y) = rand()

    grid = TripolarGrid(size = (10, 10, 1))
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

    bottom_height = grid.immersed_boundary.bottom_height

    @test interior(bottom_height, :, 10, 1) == interior(bottom_height, 10:-1:1, 10, 1)

    c = CenterField(grid)
    u = XFaceField(grid)

    set!(c, (x, y, z) -> x)
    set!(u, (x, y, z) -> x)

    fill_halo_regions!(c)
    fill_halo_regions!(u)

    @test interior(c, :, 10, 1) ==   interior(c, 10:-1:1, 10, 1)
    # For x face fields the first element is unique and we remove the 
    # north pole that is exactly at Nx+1
    left_side  = interior(u, 2:5, 10, 1)
    right_side = interior(u, 7:10, 10, 1)

    # The sign of velocities is opposite between different sides at the north boundary
    @test left_side == - reverse(right_side)
end
