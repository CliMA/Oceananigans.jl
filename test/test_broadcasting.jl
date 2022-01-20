include("dependencies_for_runtests.jl")

@testset "Field broadcasting" begin
    @info "  Testing broadcasting with fields..."

    for arch in archs

        #####
        ##### Basic functionality tests
        #####
        
        grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
        a, b, c = [CenterField(grid) for i = 1:3]

        Nx, Ny, Nz = size(a)

        a .= 1
        @test all(a .== 1) 

        b .= 2

        c .= a .+ b
        @test all(c .== 3)

        c .= a .+ b .+ 1
        @test all(c .== 4)

        # Halo regions
        fill_halo_regions!(c, arch) # Does not happen by default in broadcasting now
        @test c[1, 1, 0] == 4
        @test c[1, 1, Nz+1] == 4

        #####
        ##### Broadcasting with interpolation
        #####
        
        three_point_grid = RectilinearGrid(arch, size=(1, 1, 3), extent=(1, 1, 1))

        a2 = CenterField(three_point_grid)

        b2_bcs = FieldBoundaryConditions(grid, (Center, Center, Face), top=OpenBoundaryCondition(0), bottom=OpenBoundaryCondition(0))
        b2 = ZFaceField(three_point_grid, boundary_conditions=b2_bcs)

        b2 .= 1
        fill_halo_regions!(b2, arch) # sets b2[1, 1, 1] = b[1, 1, 4] = 0

        @test b2[1, 1, 1] == 0
        @test b2[1, 1, 2] == 1
        @test b2[1, 1, 3] == 1
        @test b2[1, 1, 4] == 0

        a2 .= b2
        @test a2[1, 1, 1] == 0.5
        @test a2[1, 1, 2] == 1.0
        @test a2[1, 1, 3] == 0.5

        a2 .= b2 .+ 1
        @test a2[1, 1, 1] == 1.5
        @test a2[1, 1, 2] == 2.0
        @test a2[1, 1, 3] == 1.5

        #####
        ##### Broadcasting with ReducedField
        #####
        
        for loc in [
                    (Nothing, Center, Center),
                    (Center, Nothing, Center),
                    (Center, Center, Nothing),
                    (Center, Nothing, Nothing),
                    (Nothing, Center, Nothing),
                    (Nothing, Nothing, Center),
                    (Nothing, Nothing, Nothing),
                   ]

            @info "    Testing broadcasting to location $loc..."

            r, p, q = [Field(loc, grid) for i = 1:3]

            r .= 2 
            @test all(r .== 2) 

            p .= 3 

            q .= r .* p
            @test all(q .== 6) 

            q .= r .* p .+ 1
            @test all(q .== 7) 
        end


        #####
        ##### Broadcasting with arrays
        #####

        two_two_two_grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))

        c = CenterField(two_two_two_grid)
        random_column = arch_array(arch, reshape(rand(2), 1, 1, 2))

        c .= random_column # broadcast to every horizontal column in c

        c_cpu = Array(interior(c))
        random_column_cpu = Array(random_column)

        @test all(c_cpu[1, 1, :] .== random_column_cpu[:])
        @test all(c_cpu[2, 1, :] .== random_column_cpu[:])
        @test all(c_cpu[1, 2, :] .== random_column_cpu[:])
        @test all(c_cpu[2, 2, :] .== random_column_cpu[:])
    end
end

