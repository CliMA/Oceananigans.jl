using Oceananigans.Grids: prettysummary

struct KernelFunctionOperation{LX, LY, LZ, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
    kernel_function :: K
    grid :: G
    arguments :: D

    @doc """
        KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid, arguments...)
    
    Construct a `KernelFunctionOperation` at location `(LX, LY, LZ)` on `grid` with `arguments`.
    
    `kernel_function` is called with
    
    ```julia
    kernel_function(i, j, k, grid, arguments...)
    ```
    
    Note that `compute!(kfo::KernelFunctionOperation)` calls `compute!` on all `kfo.arguments`.
    
    Examples
    ========
    
    Construct a `KernelFunctionOperation` that returns random numbers:
    
    ```jldoctest kfo
    using Oceananigans

    grid = RectilinearGrid(size=(1, 8, 8), extent=(1, 1, 1));

    random_kernel_function(i, j, k, grid) = rand(); # use CUDA.rand on the GPU
    
    kernel_op = KernelFunctionOperation{Center, Center, Center}(random_kernel_function, grid)

    # output

    KernelFunctionOperation at (Center, Center, Center)
    ├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
    ├── kernel_function: random_kernel_function (generic function with 1 method)
    └── arguments: ()
    ```
    
    Construct a `KernelFunctionOperation` using the vertical vorticity operator used internally
    to compute vertical vorticity on all grids:
    
    ```jldoctest kfo
    using Oceananigans.Operators: ζ₃ᶠᶠᶜ # called with signature ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

    model = HydrostaticFreeSurfaceModel(; grid);
    
    u, v, w = model.velocities;
    
    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)

    # output

    KernelFunctionOperation at (Face, Face, Center)
    ├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
    ├── kernel_function: ζ₃ᶠᶠᶜ (generic function with 1 method)
    └── arguments: ("1×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU", "1×8×8 Field{Center, Face, Center} on RectilinearGrid on CPU")
    ```
    """
    function KernelFunctionOperation{LX, LY, LZ}(kernel_function::K,
                                                 grid::G, 
                                                 arguments...) where {LX, LY, LZ, K, G}
        T = eltype(grid)
        D = typeof(arguments)
        return new{LX, LY, LZ, G, T, K, D}(kernel_function, grid, arguments)
    end

end

@inline Base.getindex(κ::KernelFunctionOperation, i, j, k) = κ.kernel_function(i, j, k, κ.grid, κ.arguments...)
indices(κ::KernelFunctionOperation) = interpolate_indices(κ.arguments...; loc_operation = location(κ))
compute_at!(κ::KernelFunctionOperation, time) = Tuple(compute_at!(d, time) for d in κ.arguments)

"Adapt `KernelFunctionOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, κ::KernelFunctionOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(Adapt.adapt(to, κ.kernel_function),
                                        Adapt.adapt(to, κ.grid),
                                        Tuple(Adapt.adapt(to, a) for a in κ.arguments)...)

Base.show(io::IO, kfo::KernelFunctionOperation) =
    print(io,
      summary(kfo), '\n',
      "├── grid: ", summary(kfo.grid), '\n',
      "├── kernel_function: ", prettysummary(kfo.kernel_function), '\n',
      "└── arguments: ", if isempty(kfo.arguments)
                             "()"
                         else
                             Tuple(string(prettysummary(a)) for a in kfo.arguments[1:end-1])...,
                             prettysummary(kfo.arguments[end])
                         end
)
