include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: φnode, λnode, halo_size
using Oceananigans.Utils: Iterate, getregion
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!

function get_range_of_indices(operation, index, Nx, Ny)

    if operation == :endpoint && index == :first
        range_x = 1
        range_y = 1
    elseif operation == :endpoint && index == :last
        range_x = Nx
        range_y = Ny
    elseif operation == :subset && index == :first # here index is the index to skip
        range_x = 2:Nx
        range_y = 2:Ny
    elseif operation == :subset && index == :last # here index is the index to skip
        range_x = 1:Nx-1
        range_y = 1:Ny-1
    else
        range_x = 1:Nx
        range_y = 1:Ny
    end

    return range_x, range_y
end

function get_halo_data(field, ::West, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    _, range_y = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[-Hx+1:0, range_y, k_index]
end

function get_halo_data(field, ::East, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    _, range_y = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[Nx+1:Nx+Hx, range_y, k_index]
end

function get_halo_data(field, ::North, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    range_x, _ = get_range_of_indices(operation, index, Nx, Ny)
    
    return field.data[range_x, Ny+1:Ny+Hy, k_index]
end

function get_halo_data(field, ::South, k_index=1; operation=nothing, index=:all)
    Nx, Ny, _ = size(field)
    Hx, Hy, _ = halo_size(field.grid)

    range_x, _ = get_range_of_indices(operation, index, Nx, Ny)

    return field.data[range_x, -Hy+1:0, k_index]
end

# Solid body rotation
R = 1        # sphere's radius
U = 1        # velocity scale
φʳ = 0       # Latitude pierced by the axis of rotation
α  = 90 - φʳ # Angle between axis of rotation and north pole (degrees)
Ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

function create_test_data(grid; variable_location="cca")

    Nx, Ny, Nz = size(grid)
    
    if variable_location == "cca"
        Location = Center, Center, Center
        LocationWithParenthesis = Center(), Center()
    elseif variable_location == "fca"
        Location = Face, Center, Center
        LocationWithParenthesis = Face(), Center()
    elseif variable_location == "cfa"
        Location = Center, Face, Center
        LocationWithParenthesis = Center(), Face()
    elseif variable_location == "ffa"
        Location = Face, Face, Center
        LocationWithParenthesis = Face(), Face()
    end 
    
    Ψ = Field{Location...}(grid)
    
    for region in 1:6

        for i in 1:Nx, j in 1:Ny
            λAtNode = λnode(i, j, getregion(grid, region), LocationWithParenthesis...)
            φAtNode = φnode(i, j, getregion(grid, region), LocationWithParenthesis...)
            getregion(Ψ, region).data[i, j, 1] = Ψᵣ(λAtNode, φAtNode, 0)
        end
        
    end
    
    return Ψ
    
end

function create_parent_test_data(grid; variable_location="cca")
    
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)
    
    if variable_location == "cca"
        Location = Center, Center, Center
        LocationWithParenthesis = Center(), Center()
    elseif variable_location == "fca"
        Location = Face, Center, Center
        LocationWithParenthesis = Face(), Center()
    elseif variable_location == "cfa"
        Location = Center, Face, Center
        LocationWithParenthesis = Center(), Face()
    elseif variable_location == "ffa"
        Location = Face, Face, Center
        LocationWithParenthesis = Face(), Face()
    end 
    
    Ψ = Field{Location...}(grid)
    
    for region in 1:6

        for i in 1-Hx:Nx+Hx, j in 1-Hy:Ny+Hy
            λAtNode = λnode(i, j, getregion(grid, region), LocationWithParenthesis...)
            φAtNode = φnode(i, j, getregion(grid, region), LocationWithParenthesis...)
            getregion(Ψ, region).data[i, j, 1] = Ψᵣ(λAtNode, φAtNode, 0)
        end
        
    end
    
    return Ψ
    
end

create_c_test_data(grid) = create_test_data(grid; variable_location="cca")
create_u_test_data(grid) = create_test_data(grid; variable_location="fca")
create_v_test_data(grid) = create_test_data(grid; variable_location="cfa")
create_Ψ_test_data(grid) = create_test_data(grid; variable_location="ffa")

create_c_parent_test_data(grid) = create_parent_test_data(grid; variable_location="cca")
create_u_parent_test_data(grid) = create_parent_test_data(grid; variable_location="fca")
create_v_parent_test_data(grid) = create_parent_test_data(grid; variable_location="cfa")
create_Ψ_parent_test_data(grid) = create_parent_test_data(grid; variable_location="ffa")

@testset "Testing conformal cubed sphere fill halos at cca locations" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos at cca locations [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3, z_halo = 1)
            data = create_c_test_data(grid)
            parent_data = create_c_parent_test_data(grid)

            fill_halo_regions!(data)
            
            Hx, Hy, Hz = halo_size(grid)

            # Confirm that the cca halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin
            
                switch_device!(grid, 1)
                # @test get_halo_data(getregion(data, 1),  West()) == get_halo_data(getregion(parent_data, 1),  West())
                @test get_halo_data(getregion(data, 1),  East()) == get_halo_data(getregion(parent_data, 1),  East())
                @test get_halo_data(getregion(data, 1), South()) == get_halo_data(getregion(parent_data, 1), South())
                # @test get_halo_data(getregion(data, 1), North()) == get_halo_data(getregion(parent_data, 1), North())
                
                switch_device!(grid, 2)
                @test get_halo_data(getregion(data, 2),  West()) == get_halo_data(getregion(parent_data, 2),  West())
                # @test get_halo_data(getregion(data, 2),  East()) == get_halo_data(getregion(parent_data, 2),  East())
                # @test get_halo_data(getregion(data, 2), South()) == get_halo_data(getregion(parent_data, 2), South())
                @test get_halo_data(getregion(data, 2), North()) == get_halo_data(getregion(parent_data, 2), North())
                
                switch_device!(grid, 3)
                # @test get_halo_data(getregion(data, 3),  West()) == get_halo_data(getregion(parent_data, 3),  West())
                @test get_halo_data(getregion(data, 3),  East()) == get_halo_data(getregion(parent_data, 3),  East())
                @test get_halo_data(getregion(data, 3), South()) == get_halo_data(getregion(parent_data, 3), South())
                # @test get_halo_data(getregion(data, 3), North()) == get_halo_data(getregion(parent_data, 3), North())
                
                switch_device!(grid, 4)
                @test get_halo_data(getregion(data, 4),  West()) == get_halo_data(getregion(parent_data, 4),  West())
                # @test get_halo_data(getregion(data, 4),  East()) == get_halo_data(getregion(parent_data, 4),  East())
                # @test get_halo_data(getregion(data, 4), South()) == get_halo_data(getregion(parent_data, 4), South())
                @test get_halo_data(getregion(data, 4), North()) == get_halo_data(getregion(parent_data, 4), North())
                
                switch_device!(grid, 5)
                # @test get_halo_data(getregion(data, 5),  West()) == get_halo_data(getregion(parent_data, 5),  West())
                @test get_halo_data(getregion(data, 5),  East()) == get_halo_data(getregion(parent_data, 5),  East())
                @test get_halo_data(getregion(data, 5), South()) == get_halo_data(getregion(parent_data, 5), South())
                # @test get_halo_data(getregion(data, 5), North()) == get_halo_data(getregion(parent_data, 5), North())
                
                switch_device!(grid, 6)
                @test get_halo_data(getregion(data, 6),  West()) == get_halo_data(getregion(parent_data, 6),  West())
                # @test get_halo_data(getregion(data, 6),  East()) == get_halo_data(getregion(parent_data, 6),  East())
                # @test get_halo_data(getregion(data, 6), South()) == get_halo_data(getregion(parent_data, 6), South())
                @test get_halo_data(getregion(data, 6), North()) == get_halo_data(getregion(parent_data, 6), North())

            end # CUDA.@allowscalar
            
        end
    end
end

@testset "Testing conformal cubed sphere fill halos at fca locations" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos at fca locations [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3, z_halo = 1)
            data = create_u_test_data(grid)
            parent_data = create_u_parent_test_data(grid)

            fill_halo_regions!(data)
            
            Hx, Hy, Hz = halo_size(grid)

            # Confirm that the fca halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin
                
                switch_device!(grid, 1)
                # @test get_halo_data(getregion(data, 1),  West()) == get_halo_data(getregion(parent_data, 1),  West())
                @test get_halo_data(getregion(data, 1),  East()) == get_halo_data(getregion(parent_data, 1),  East())
                @test get_halo_data(getregion(data, 1), South()) == get_halo_data(getregion(parent_data, 1), South())
                # @test get_halo_data(getregion(data, 1), North()) == get_halo_data(getregion(parent_data, 1), North())
                
                switch_device!(grid, 2)
                @test get_halo_data(getregion(data, 2),  West()) == get_halo_data(getregion(parent_data, 2),  West())
                # @test get_halo_data(getregion(data, 2),  East()) == get_halo_data(getregion(parent_data, 2),  East())
                # @test get_halo_data(getregion(data, 2), South()) == get_halo_data(getregion(parent_data, 2), South())
                @test get_halo_data(getregion(data, 2), North()) == get_halo_data(getregion(parent_data, 2), North())
                
                switch_device!(grid, 3)
                # @test get_halo_data(getregion(data, 3),  West()) == get_halo_data(getregion(parent_data, 3),  West())
                @test get_halo_data(getregion(data, 3),  East()) == get_halo_data(getregion(parent_data, 3),  East())
                @test get_halo_data(getregion(data, 3), South()) == get_halo_data(getregion(parent_data, 3), South())
                # @test get_halo_data(getregion(data, 3), North()) == get_halo_data(getregion(parent_data, 3), North())
                
                switch_device!(grid, 4)
                @test get_halo_data(getregion(data, 4),  West()) == get_halo_data(getregion(parent_data, 4),  West())
                # @test get_halo_data(getregion(data, 4),  East()) == get_halo_data(getregion(parent_data, 4),  East())
                # @test get_halo_data(getregion(data, 4), South()) == get_halo_data(getregion(parent_data, 4), South())
                @test get_halo_data(getregion(data, 4), North()) == get_halo_data(getregion(parent_data, 4), North())
                
                switch_device!(grid, 5)
                # @test get_halo_data(getregion(data, 5),  West()) == get_halo_data(getregion(parent_data, 5),  West())
                @test get_halo_data(getregion(data, 5),  East()) == get_halo_data(getregion(parent_data, 5),  East())
                @test get_halo_data(getregion(data, 5), South()) == get_halo_data(getregion(parent_data, 5), South())
                # @test get_halo_data(getregion(data, 5), North()) == get_halo_data(getregion(parent_data, 5), North())
                
                switch_device!(grid, 6)
                @test get_halo_data(getregion(data, 6),  West()) == get_halo_data(getregion(parent_data, 6),  West())
                # @test get_halo_data(getregion(data, 6),  East()) == get_halo_data(getregion(parent_data, 6),  East())
                # @test get_halo_data(getregion(data, 6), South()) == get_halo_data(getregion(parent_data, 6), South())
                @test get_halo_data(getregion(data, 6), North()) == get_halo_data(getregion(parent_data, 6), North())
                
            end # CUDA.@allowscalar
            
        end
    end
end

@testset "Testing conformal cubed sphere fill halos at cfa locations" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos at cfa locations [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3, z_halo = 1)
            data = create_v_test_data(grid)
            parent_data = create_v_parent_test_data(grid)

            fill_halo_regions!(data)
            
            Hx, Hy, Hz = halo_size(grid)

            # Confirm that the fca halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin
                
                switch_device!(grid, 1)
                # @test get_halo_data(getregion(data, 1),  West()) == get_halo_data(getregion(parent_data, 1),  West())
                @test get_halo_data(getregion(data, 1),  East()) == get_halo_data(getregion(parent_data, 1),  East())
                @test get_halo_data(getregion(data, 1), South()) == get_halo_data(getregion(parent_data, 1), South())
                # @test get_halo_data(getregion(data, 1), North()) == get_halo_data(getregion(parent_data, 1), North())
                
                switch_device!(grid, 2)
                @test get_halo_data(getregion(data, 2),  West()) == get_halo_data(getregion(parent_data, 2),  West())
                # @test get_halo_data(getregion(data, 2),  East()) == get_halo_data(getregion(parent_data, 2),  East())
                # @test get_halo_data(getregion(data, 2), South()) == get_halo_data(getregion(parent_data, 2), South())
                @test get_halo_data(getregion(data, 2), North()) == get_halo_data(getregion(parent_data, 2), North())
                
                switch_device!(grid, 3)
                # @test get_halo_data(getregion(data, 3),  West()) == get_halo_data(getregion(parent_data, 3),  West())
                @test get_halo_data(getregion(data, 3),  East()) == get_halo_data(getregion(parent_data, 3),  East())
                @test get_halo_data(getregion(data, 3), South()) == get_halo_data(getregion(parent_data, 3), South())
                # @test get_halo_data(getregion(data, 3), North()) == get_halo_data(getregion(parent_data, 3), North())
                
                switch_device!(grid, 4)
                @test get_halo_data(getregion(data, 4),  West()) == get_halo_data(getregion(parent_data, 4),  West())
                # @test get_halo_data(getregion(data, 4),  East()) == get_halo_data(getregion(parent_data, 4),  East())
                # @test get_halo_data(getregion(data, 4), South()) == get_halo_data(getregion(parent_data, 4), South())
                @test get_halo_data(getregion(data, 4), North()) == get_halo_data(getregion(parent_data, 4), North())
                
                switch_device!(grid, 5)
                # @test get_halo_data(getregion(data, 5),  West()) == get_halo_data(getregion(parent_data, 5),  West())
                @test get_halo_data(getregion(data, 5),  East()) == get_halo_data(getregion(parent_data, 5),  East())
                @test get_halo_data(getregion(data, 5), South()) == get_halo_data(getregion(parent_data, 5), South())
                # @test get_halo_data(getregion(data, 5), North()) == get_halo_data(getregion(parent_data, 5), North())
                
                switch_device!(grid, 6)
                @test get_halo_data(getregion(data, 6),  West()) == get_halo_data(getregion(parent_data, 6),  West())
                # @test get_halo_data(getregion(data, 6),  East()) == get_halo_data(getregion(parent_data, 6),  East())
                # @test get_halo_data(getregion(data, 6), South()) == get_halo_data(getregion(parent_data, 6), South())
                @test get_halo_data(getregion(data, 6), North()) == get_halo_data(getregion(parent_data, 6), North())
                
            end # CUDA.@allowscalar
            
        end
    end
end

@testset "Testing conformal cubed sphere fill halos at ffa locations" begin
    for FT in float_types
        for arch in archs
            @info "  Testing fill halos at ffa locations [$FT, $(typeof(arch))]..."

            Nx, Ny, Nz = 9, 9, 1

            grid = ConformalCubedSphereGrid(arch, FT; panel_size = (Nx, Ny, Nz), z = (0, 1), radius = 1, horizontal_direction_halo = 3, z_halo = 1)
            data = create_Ψ_test_data(grid)
            parent_data = create_Ψ_parent_test_data(grid)

            fill_halo_regions!(data)
            
            Hx, Hy, Hz = halo_size(grid)

            # Confirm that the ffa halos were filled according to connectivity described at ConformalCubedSphereGrid docstring.
            CUDA.@allowscalar begin
                
                # switch_device!(grid, 1)
                # @test get_halo_data(getregion(data, 1),  West()) == get_halo_data(getregion(parent_data, 1),  West())
                # @test get_halo_data(getregion(data, 1),  East()) == get_halo_data(getregion(parent_data, 1),  East())
                # @test get_halo_data(getregion(data, 1), South()) == get_halo_data(getregion(parent_data, 1), South())
                # @test get_halo_data(getregion(data, 1), North()) == get_halo_data(getregion(parent_data, 1), North())
                
                # switch_device!(grid, 2)
                # @test get_halo_data(getregion(data, 2),  West()) == get_halo_data(getregion(parent_data, 2),  West())
                # @test get_halo_data(getregion(data, 2),  East()) == get_halo_data(getregion(parent_data, 2),  East())
                # @test get_halo_data(getregion(data, 2), South()) == get_halo_data(getregion(parent_data, 2), South())
                # @test get_halo_data(getregion(data, 2), North()) == get_halo_data(getregion(parent_data, 2), North())
                
                # switch_device!(grid, 3)
                # @test get_halo_data(getregion(data, 3),  West()) == get_halo_data(getregion(parent_data, 3),  West())
                # @test get_halo_data(getregion(data, 3),  East()) == get_halo_data(getregion(parent_data, 3),  East())
                # @test get_halo_data(getregion(data, 3), South()) == get_halo_data(getregion(parent_data, 3), South())
                # @test get_halo_data(getregion(data, 3), North()) == get_halo_data(getregion(parent_data, 3), North())
                
                # switch_device!(grid, 4)
                # @test get_halo_data(getregion(data, 4),  West()) == get_halo_data(getregion(parent_data, 4),  West())
                # @test get_halo_data(getregion(data, 4),  East()) == get_halo_data(getregion(parent_data, 4),  East())
                # @test get_halo_data(getregion(data, 4), South()) == get_halo_data(getregion(parent_data, 4), South())
                # @test get_halo_data(getregion(data, 4), North()) == get_halo_data(getregion(parent_data, 4), North())
                
                # switch_device!(grid, 5)
                # @test get_halo_data(getregion(data, 5),  West()) == get_halo_data(getregion(parent_data, 5),  West())
                # @test get_halo_data(getregion(data, 5),  East()) == get_halo_data(getregion(parent_data, 5),  East())
                # @test get_halo_data(getregion(data, 5), South()) == get_halo_data(getregion(parent_data, 5), South())
                # @test get_halo_data(getregion(data, 5), North()) == get_halo_data(getregion(parent_data, 5), North())
                
                # switch_device!(grid, 6)
                # @test get_halo_data(getregion(data, 6),  West()) == get_halo_data(getregion(parent_data, 6),  West())
                # @test get_halo_data(getregion(data, 6),  East()) == get_halo_data(getregion(parent_data, 6),  East())
                # @test get_halo_data(getregion(data, 6), South()) == get_halo_data(getregion(parent_data, 6), South())
                # @test get_halo_data(getregion(data, 6), North()) == get_halo_data(getregion(parent_data, 6), North())
                
            end # CUDA.@allowscalar
            
        end
    end
end