using Oceananigans.Fields: KernelComputedField
using KernelAbstractions: @kernel, @index

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1),
                            topology=(Periodic, Periodic, Bounded))

@kernel function double!(doubled_field, grid, single_field)
    i, j, k = @index(Global, NTuple)
    doubled_field[i, j, k] = 2 * single_field[i, j, k]
end

@kernel function multiply!(multiplied_field, grid, field, multiple)
    i, j, k = @index(Global, NTuple)
    multiplied_field[i, j, k] = multiple * field[i, j, k]
end

for arch in archs
    @testset "KernelComputedField [$(typeof(arch))]" begin
        @info "  Testing KernelComputedField..."

        single_field = Field(Center, Center, Center, arch, grid)

        doubled_field = KernelComputedField(Center, Center, Center,
                                            double!, arch, grid,
                                            field_dependencies = single_field)

        # Test that the constructor worked
        @test doubled_field isa KernelComputedField

        multiple = 3
        multiplied_field = KernelComputedField(Center, Center, Center,
                                               multiply!, arch, grid,
                                               field_dependencies = doubled_field,
                                               parameters = multiple)

        # Test that the constructor worked
        @test multiplied_field isa KernelComputedField

        set!(single_field, π)

        compute!(doubled_field)
        @test doubled_field.data[1, 1, 1] == 2π

        set!(doubled_field, 0)
        compute!(multiplied_field)

        @test doubled_field.data[1, 1, 1] == 2π
        @test multiplied_field.data[1, 1, 1] == multiple * 2 * π
    end
end
