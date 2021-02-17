using Oceananigans: AbstractModel
using Oceananigans.Grids
using Oceananigans.Utils: tupleit

struct KernelComputedField{X, Y, Z, S, A, G, K, C, F, P} <: AbstractField{X, Y, Z, A, G}
                   data :: A
                   grid :: G
                 kernel :: K
    boundary_conditions :: C
     field_dependencies :: F
             parameters :: P
                 status :: S

    """
        KernelComputedField(loc, kernel, grid)

    Example
    =======

    ```julia
    using KernelAbstractions: @index, @kernel
    using Oceananigans.Fields: AveragedField, KernelComputedField, compute!
    using Oceananigans.Grids: Center, Face
    
    @inline ψ²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2
    @kernel function compute_variance!(var, grid, ϕ, Φ)
        i, j, k = @index(Global, NTuple)
    
        @inbounds var[i, j, k] = ψ′²(i, j, k, grid, ϕ, Φ)
    end
    
    u, v, w = model.velocities
    
    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))
    
    u′² = KernelComputedField(Face, Center, Center, compute_variance!, model; field_dependencies=(u, U,))
    v′² = KernelComputedField(Center, Face, Center, compute_variance!, model; field_dependencies=(v, V,))
    w′² = KernelComputedField(Center, Center, Face, compute_variance!, model; field_dependencies=(w, 0,))
    
    compute!(u′²)
    compute!(v′²)
    compute!(w′²)
    ```
    """
    function KernelComputedField{X, Y, Z}(kernel::K, arch, grid;
                                          boundary_conditions = ComputedFieldBoundaryConditions(grid, (X, Y, Z)),
                                          field_dependencies = (),
                                          parameters::P = nothing,
                                          data = nothing,
                                          recompute_safely = true) where {X, Y, Z, K, P}

        field_dependencies = tupleit(field_dependencies)

        if isnothing(data)
            data = new_data(arch, grid, (X, Y, Z))
        end

        # Use FieldStatus if we want to avoid always recomputing
        status = recompute_safely ? nothing : FieldStatus(0.0)

        G = typeof(grid)
        A = typeof(data)
        S = typeof(status)
        F = typeof(field_dependencies)
        C = typeof(boundary_conditions)

        return new{X, Y, Z, S,
                   A, G, K, C, F, P}(data, grid, kernel, boundary_conditions, field_dependencies, parameters)
    end
end

KernelComputedField(X, Y, Z, kernel, model::AbstractModel; kwargs...) =
    KernelComputedField{X, Y, Z}(kernel, model.architecture, model.grid; kwargs...)

KernelComputedField(X, Y, Z, kernel, arch::AbstractArchitecture, grid::AbstractGrid; kwargs...) =
    KernelComputedField{X, Y, Z}(kernel, arch, grid; kwargs...)

function compute!(kcf::KernelComputedField{X, Y, Z}) where {X, Y, Z}

    for dependency in kcf.field_dependencies
        compute!(dependency)
    end

    arch = architecture(kcf.data)

    workgroup, worksize = work_layout(kcf.grid,
                                      :xyz,
                                      location=(X, Y, Z),
                                      include_right_boundaries=true)

    compute_kernel! = kcf.kernel(device(arch), workgroup, worksize)

    event = isnothing(kcf.parameters) ?
        compute_kernel!(kcf.data, kcf.grid, kcf.field_dependencies...) :
        compute_kernel!(kcf.data, kcf.grid, kcf.field_dependencies..., kcf.parameters)

    wait(device(arch), event)

    fill_halo_regions!(kcf, arch)

    return nothing
end

compute!(field::KernelComputedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(field, time)

Adapt.adapt_structure(to, kcf::KernelComputedField) = Adapt.adapt(to, kcf.data)
