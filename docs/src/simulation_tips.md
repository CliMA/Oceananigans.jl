# Simulation tips

Oceananigans attemps to optimize computations as much as possible "behind the scenes".
Yet Oceananigans' flexibility places some responsibility on users to ensure high performance simulations,
especially for complex setups with user-defined forcing functions, boundary condition functions, and diagnostics.
Furthermore, in case of more complex GPU runs, some details could
sometimes prevent your simulation from running altogether. While Julia knowledge is obviously
desirable here, a user that is unfamiliar with Julia can get away with efficient simulations by
learning a few rules of thumb. It is nonetheless recommended that users go through Julia's
[performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/), which contains more
in-depth explanations of some of the aspects discussed here.

## General (CPU/GPU) simulation tips

### Avoid global variables whenever possible

In general using a [global
variable](https://docs.julialang.org/en/v1/manual/variables-and-scoping/#Global-Scope) (which can be
loosely defined as a variable defined in the main script) inside functions slows down the code. One
way to circumvent this is to always [use local variables or pass them as arguments to
functions](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables). This
helps the compiler optimize the code.

Another way around this is to [define global variables as constants whenever
possible](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables). One
thing to keep in mind when doing this is that when a `const` is defined, its value can't be changed
until you restart the Julia session. So this latter approach is good for production-ready code, but
may be undesirable in the early stages of development while you still have to change the parameters
of the simulation for exploration.

It is especially important to avoid global variables in functions that are meant to be executed in
GPU kernels (such as functions defining boundary conditions and forcings). Otherwise the Julia GPU
compiler can fail with obscure errors. This is explained in more detail in the GPU simulation tips
section below.

### Consider inlining small functions

Inlining is when the compiler [replaces a function call with the body of the function that is being
called before compiling](https://en.wikipedia.org/wiki/Inline_expansion). The advantage of inlining
(which in julia can be done with the [`@inline`
macro](https://docs.julialang.org/en/v1/devdocs/meta/)) is that gets rid of the time spent calling
the function. The Julia compiler automatically makes some calls as to what functions it should or
shouldn't inline, but you can force a function to be inlined by including the macro `@inline` before
its definition. This is more suited for small functions that are called often. Here's an example of
an implementation of the Heaviside function that forces it to be inlined:

```julia
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
```

In practice it's hard to say whether inlining a function will bring runtime benefits _with
certainty_, since Julia and KernelAbstractions.jl (needed for GPU runs) already inline some
functions automatically. However, it is generally a good idea to at least investigate this aspect in
your code as the benefits can potentially be significant.

## GPU simulation tips

Running on GPUs can be very different from running on CPUs. Oceananigans makes most of the necessary
changes in the background, so that for very simple simulations changing between CPUs and GPUs is
just a matter of changing the `architecture` argument in the model from `CPU()` to `GPU()`. However,
for more complex simulations some care needs to be taken on the part of the user. While knowledge of
GPU computing (and Julia) is again desirable, an inexperienced user can also achieve high efficiency
in GPU simulations by following a few simple principles.

### Global variables that need to be used in GPU computations need to be defined as constants or passed as parameters

Any global variable that needs to be accessed by the GPU needs to be a constant or the simulation
will crash. This includes any variables that are referenced as global variables in functions
used for forcing of boundary conditions. For example,

```julia
T₀ = 20 # ᵒC
surface_temperature(x, y, t) = T₀ * sin(2π / 86400 * t)
T_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(surface_temperature))
```

will throw an error if run on the GPU (and will run more slowly than it should on the CPU).
Replacing the first line above with

```julia
const T₀ = 20 # ᵒC
```

fixes the issue by indicating to the compiler that `T₀` will not change.

Note that the _literal_ `2π / 86400` is not an issue -- it's only the
_variable_ `T₀` that must be declared `const`.

Alternatively, passing the variable as a parameter to `GradientBoundaryCondition` also works:

```julia
T₀ = 20 # ᵒC
surface_temperature(x, y, t, p) = p.T₀ * sin(2π / 86400 * t)
T_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(surface_temperature, parameters=(T₀=T₀,)))
```

### Complex diagnostics using computed `Field`s may not work on GPUs

`Field`s are the most convenient way to calculate diagnostics for your simulation. They will
always work on CPUs, but when their complexity is high (in terms of number of abstract operations)
the compiler can't translate them into GPU code and they fail for GPU runs. (This limitation is summarized 
in [this Github issue](https://github.com/CliMA/Oceananigans.jl/issues/1886) and contributions are welcome.)
For example, in the example below, calculating `u²` works in both CPUs and GPUs, but calculating 
`ε` will not compile on GPUs when we call the command `compute!`:

```julia
using Oceananigans
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, closure=ScalarDiffusivity(ν=1e-6))
u, v, w = model.velocities
ν = model.closure.ν
u² = Field(u^2)
ε = Field(ν*(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2))
compute!(u²)
compute!(ε)
```

There are a few ways to work around this issue.
One is to compute `ε` in steps by nesting computed `Field`s,
```julia
ddx² = Field(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2)
ddy² = Field(∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2)
ddz² = Field(∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2)
ε = Field(ν * (ddx² + ddy² + ddz²))
compute!(ε)
```

This method increases the computational cost since it requires computing and storing 3 intermediate terms.
`ε` may also be calculated via `KernelFunctionOperations`s, which
requires explicitly building a "kernel function" from low-level Oceananigans
operators.

```julia
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation

@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = @inbounds (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2

function isotropic_viscous_dissipation_rate_ccc(i, j, k, grid, u, v, w, ν)
    Σˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, u)^2
    Σʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, v)^2
    Σᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, w)^2

    Σˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Σˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Σʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    return ν * 2 * (Σˣˣ² + Σʸʸ² + Σᶻᶻ² + 2 * (Σˣʸ² + Σˣᶻ² + Σʸᶻ²))
end

ε_op = KernelFunctionOperation{Center, Center, Center}(isotropic_viscous_dissipation_rate_ccc,
                                                       grid;
                                                       computed_dependencies=(u, v, w, ν))

ε = Field(ε_op)

compute!(ε)
```

Writing kernel functions like `isotropic_viscous_dissipation_rate_ccc`
requires understanding the C-grid, but incurs only one iteration over the domain.

`KernelFunctionOperation`s for some diagnostics common to large eddy simulation are defined in
[Oceanostics.jl](https://github.com/tomchor/Oceanostics.jl/blob/3b8f67338656557877ef8ef5ebe3af9e7b2974e2/src/TurbulentKineticEnergyTerms.jl#L35-L57),

```julia
using Oceanostics: IsotropicPseudoViscousDissipationRate
ε = IsotropicViscousDissipationRate(model, u, v, w, ν)
compute!(ε)
```
[Start an issue on Github](https://github.com/CliMA/Oceananigans.jl/issues/new) if more help is needed.


### Try to decrease the memory-use of your runs

GPU runs are sometimes memory-limited. A state-of-the-art Tesla V100 GPU has 32GB of
memory --- enough memory for simulations with about 100 million points, or grids a bit smaller
than 512 × 512 × 512. (The maximum grid size depends on some user-specified factors,
like the number of passive tracers or computed diagnostics.)
For large simulations on the GPU, careful management of memory allocation may be required:

- Use the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) command
  line utility to monitor the memory usage of the GPU. It should tell you how much memory there is
  on your GPU and how much of it you're using and you can run it from Julia with the command `` run(`nvidia-smi`) ``.

- Try to use higher-order advection schemes. In general when you use a higher-order scheme you need
  fewer grid points to achieve the same accuracy that you would with a lower-order one. Oceananigans
  provides two high-order advection schemes: 5th-order WENO method (WENO5) and 3rd-order upwind.

- Manually define scratch space to be reused in diagnostics. By default, every time a user-defined
  diagnostic is calculated the compiler reserves a new chunk of memory for that calculation, usually
  called scratch space. In general, the more diagnostics, the more scratch space needed and the bigger
  the memory requirements. However, if you explicitly create a scratch space and pass that same
  scratch space for as many diagnostics as you can, you minimize the memory requirements of your
  calculations by reusing the same chunk of memory. As an example, you can see scratch space being
  created
  [here](https://github.com/CliMA/LESbrary.jl/blob/cf31b0ec20219d5ad698af334811d448c27213b0/examples/three_layer_constant_fluxes.jl#L380-L383)
  and then being used in calculations
  [here](https://github.com/CliMA/LESbrary.jl/blob/cf31b0ec20219d5ad698af334811d448c27213b0/src/TurbulenceStatistics/first_through_third_order.jl#L109-L112).

### Arrays in GPUs are usually different from arrays in CPUs

Oceananigans.jl uses [`CUDA.CuArray`](https://cuda.juliagpu.org/stable/usage/array/) to store 
data for GPU computations. One limitation of `CuArray`s compared to the `Array`s used for 
CPU computations is that `CuArray` elements in general cannot be accessed outside kernels
launched through CUDA.jl or KernelAbstractions.jl. (You can learn more about GPU kernels 
[here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and 
[here](https://cuda.juliagpu.org/stable/usage/overview/#Kernel-programming-with-@cuda).)
Doing so requires individual elements to be copied from or to the GPU for processing,
which is very slow and can result in huge slowdowns. For this reason, Oceananigans.jl disables CUDA
scalar indexing by default. See the [scalar indexing](https://juliagpu.github.io/CUDA.jl/dev/usage/workflow/#UsageWorkflowScalar)
section of the CUDA.jl documentation for more information on scalar indexing.

For example if can be difficult to just view a `CuArray` since Julia needs to access 
its elements to do that. Consider the example below:

```julia
julia> using Oceananigans, Adapt

julia> grid = RectilinearGrid(GPU(); size=(1, 1, 1), extent=(1, 1, 1))
1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 1×1×1 halo
├── Periodic x ∈ [0.0, 1.0)  regularly spaced with Δx=1.0
├── Periodic y ∈ [0.0, 1.0)  regularly spaced with Δy=1.0
└── Bounded  z ∈ [-1.0, 0.0] regularly spaced with Δz=1.0

julia> model = NonhydrostaticModel(; grid)
NonhydrostaticModel{GPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 1×1×1 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing

julia> typeof(model.velocities.u.data)
OffsetArrays.OffsetArray{Float64, 3, CUDA.CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}}

julia> adapt(Array, model.velocities.u.data)
3×3×3 OffsetArray(::Array{Float64, 3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

Notice that in order to view the `CuArray` that stores values for `u` we needed to transform
it into a regular `Array` first using `Adapt.adapt`. If we naively try to view the `CuArray`
without that step we get an error:

```julia
julia> model.velocities.u.data
3×3×3 OffsetArray(::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
Error showing value of type OffsetArrays.OffsetArray{Float64, 3, CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}}:
ERROR: Scalar indexing is disallowed.
```

Here `CUDA.jl` throws an error because scalar `getindex` is not `allowed`. There are ways to
overcome this limitation and allow scalar indexing (more about that 
in the [CUDA.jl documentation](https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar)), but this option
can be very slow on GPUs, so it is advised to only use this last method when using the REPL or 
prototyping --- never in production-ready scripts.

You might also need to keep these differences in mind when using arrays
to define initial conditions, boundary conditions or
forcing functions on a GPU. To learn more about working with `CuArray`s, see the
[array programming](https://juliagpu.github.io/CUDA.jl/dev/usage/array/) section
of the CUDA.jl documentation.
