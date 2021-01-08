using Oceananigans.Grids

struct KernelComputedField{X, Y, Z, A, G, K, F, P} <: AbstractField{X, Y, Z, A, G}
                  data :: A
                  grid :: G
                kernel :: K
    field_dependencies :: F
            parameters :: P

    """
        KernelComputedField(loc, kernel, grid)

    Example
    =======

    ```julia

    using KernelAbstractions: @index, @kernel
    using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ

    @inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds (ψ[i, j, k] - Ψ[i, j, k])^2

    @kernel function compute_tke!(tke, grid, u, v, w, U, V)
        i, j, k = @index(Global, NTuple)

        @inbounds tke[i, j, k] = (ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) + 
                                  ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
                                  ℑzᵃᵃᶜ(i, j, k, grid, ψ², w)
                                 ) / 2
    end

    u, v, w = model.velocities

    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(u, dims=(1, 2))

    tke = KernelComputedField(compute_tke!, model; field_dependencies=(u, v, w, U, V)) 

    compute!(tke)
    ```
    """
    function KernelComputedField{X, Y, Z}(kernel::K, model;
                                          field_dependencies::F=(),
                                          parameters::P=nothing,
                                          data=nothing) where {X, Y, Z, K, F, P}

        if isnothing(data)
            data = new_data(model.architecture, model.grid, (X, Y, Z))
        end

        G = typeof(model.grid)
        A = typeof(data)

        return KernelComputedField{X, Y, Z, A, G, K, F, P}(data, grid, kernel, field_dependencies, parameters)
    end
end

function compute!(kcf::KernelComputedField{X, Y, Z}, time) where {X, Y, Z}

    for dependency in kcf.field_dependencies
        compute!(dependency, time)
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

    return nothing
end

compute!(kcf::KernelComputedField) = compute!(kcf, nothing)

Adapt.adapt_structure(to, kcf::KernelComputedField) = Adapt.adapt(to, kcf.data)
