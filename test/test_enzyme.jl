using Oceananigans
using Enzyme
using Oceananigans.Fields: FunctionField
using Oceananigans: architecture
using KernelAbstractions

# Required presently
Enzyme.API.runtimeActivity!(true)

EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true

f(grid) = CenterField(grid)

@testset "Enzyme Unit Tests" begin
    arch=CPU()
    FT=Float64

    N = 100
    topo = (Periodic, Flat, Flat)
    grid = RectilinearGrid(arch, FT, topology=topo, size=N, halo=2, x=(-1, 1), y=(-1, 1), z=(-1, 1))
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, typeof(Const(grid)))

    tape, primal, shadow = fwd(Const(f), Const(grid) )

    @show tape, primal, shadow

    @test size(primal) == size(shadow)
end

function set_initial_condition_via_launch!(model_tracer, amplitude)
    # Set initial condition
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[]
    temp = Base.broadcasted(Base.identity, FunctionField((Center, Center, Center), cᵢ, model_tracer.grid))

    temp = convert(Base.Broadcast.Broadcasted{Nothing}, temp)
    grid = model_tracer.grid
    arch = architecture(model_tracer)

    param = Oceananigans.Utils.KernelParameters(size(model_tracer), map(Oceananigans.Fields.offset_index, model_tracer.indices))
    Oceananigans.Utils.launch!(arch, grid, param, Oceananigans.Fields._broadcast_kernel!, model_tracer, temp)

    return nothing
end

function set_initial_condition!(model, amplitude)
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[] * exp(-z^2 / 0.02 - (x^2 + y^2) / 0.05)
    set!(model, c=cᵢ)

    return nothing
end

@testset "Enzyme + Oceananigans Initialization Broadcast Kernel" begin

    Enzyme.API.looseTypeAnalysis!(true)

    Nx = Ny = 64
    Nz = 8

    x = y = (-π, π)
    z = (-0.5, 0.5)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracers = :c,
                                        buoyancy = nothing)

    model_tracer = model.tracers.c

    amplitude = 1.0
    amplitude = Ref(amplitude)

    # This has a "width" of 0.1
    cᵢ(x, y, z) = amplitude[]
    temp = Base.broadcasted(Base.identity, FunctionField((Center, Center, Center), cᵢ, model_tracer.grid))

    temp = convert(Base.Broadcast.Broadcasted{Nothing}, temp)
    grid = model_tracer.grid
    arch = architecture(model_tracer)

    param = Oceananigans.Utils.KernelParameters(size(model_tracer), map(Oceananigans.Fields.offset_index, model_tracer.indices))

    dmodel_tracer = Enzyme.make_zero(model_tracer)
    # Test the individual kernel launch
    @test try
        autodiff(Enzyme.Reverse,
                Oceananigans.Utils.launch!,
                Const(arch),
                Const(grid),
                Const(param),
                Const(Oceananigans.Fields._broadcast_kernel!),
                Duplicated(model_tracer, dmodel_tracer),
                Const(temp))
        true
    catch
        @warn "Failed to differentiate Oceananigans.Utils.launch!"
        false
    end

    # Test out differentiation of the broadcast infrastructure
    @test try
        autodiff(Enzyme.Reverse,
                set_initial_condition_via_launch!,
                Duplicated(model_tracer, dmodel_tracer),
                Active(1.0))
        true
    catch
        @warn "Failed to differentiate set_initial_condition_via_launch!"
        false
    end

    # Test differentiation of the high-level set interface
    @test try
        autodiff(Enzyme.Reverse,
                set_initial_condition!,
                Duplicated(model_tracer, dmodel_tracer),
                Active(1.0))
        true
    catch
        @warn "Failed to differentiate set_initial_condition!"
        false
    end

    Enzyme.API.looseTypeAnalysis!(false)
end
