# Simulation tips

In Oceananigans we try to do most of the optimizing behind the scenes, that way the average user
doesn't have to worry about details when setting up a simulation. However, there's just so much
optimization that can be done in the source code. Because of Oceananigans script-based interface,
the user has to be aware of some things when writing the simulation script in order to take full
advantage of Julia's speed. Furthermore, in case of more complex GPU runs, some details could
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


### Variables that need to be used in GPU computations need to be defined as constants

Any global variable that needs to be accessed by the GPU needs to be a constant or the simulation
will crash. This includes any variables used in forcing functions and boundary conditions. For
example, if you define a boundary condition like the example below and run your simulation on a GPU
you'll get an error.

```julia
dTdz = 0.01 # K m⁻¹
T_bcs = TracerBoundaryConditions(grid,
                                 bottom = GradientBoundaryCondition(dTdz))
```

However, if you define `dTdz` as a constant by replacing the first line with `const dTdz = 0.01`,
then (provided everything else is done properly) your run will be successful.


### Complex diagnostics using `ComputedField`s may not work on GPUs

`ComputedField`s are the most convenient way to calculate diagnostics for your simulation. They will
always work on CPUs, but when their complexity is high (in terms of number of abstract operations)
the compiler can't translate them into GPU code and they fail for GPU runs. (This limitation is discussed 
in [this Github issue](https://github.com/CliMA/Oceananigans.jl/issues/1241) and contributors are welcome.)
For example, in the example below, calculating `u²` works in both CPUs and GPUs, but calculating 
`KE` may not work in some GPUs:

```julia
u, v, w = model.velocities
u² = ComputedField(u^2)
KE = ComputedField((u^2 + v^2 + w^2)/2)
compute!(u²)
compute!(KE)
```

Assuming `compute!(KE)` fails for your GPU, there are two approaches to 
bypass this issue. The first is to nest `ComputedField`s. For example,
we can make `KE` be successfully computed on GPUs by defining it as
```julia
u, v, w = model.velocities
u² = ComputedField(u^2)
v² = ComputedField(v^2)
w² = ComputedField(w^2)
u²plusv² = ComputedField(u² + v²)
KE = ComputedField((u²plusv² + w²)/2)
compute!(KE)
```

This is a simple workaround that is especially suited for the development stage of a simulation.
However, when running this, the code will iterate over the whole domain 5 times to calculate `KE`
(one for each computed field defined), which is not very efficient.

A different way to calculate `KE` is by using `KernelComputedField`s, where the
user manually specifies the computing kernel to the compiler. The advantage of this method is that
it's more efficient (the code will only iterate once over the domain in order to calculate `KE`),
but the disadvantage is that this requires that the has some knowledge of Oceananigans operations
and how they should be performed on a C-grid. For example calculating `KE` with this approach would
look like this:

```julia
using Oceananigans.Operators
using KernelAbstractions: @index, @kernel
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: KernelComputedField

@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2

@kernel function kinetic_energy_ccc!(tke, grid, u, v, w)
    i, j, k = @index(Global, NTuple)
    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, ψ², u) + # Calculates u^2 using function ψ² and then interpolates in x to grid center
                              ℑyᵃᶜᵃ(i, j, k, grid, ψ², v) + # Calculates v^2 using function ψ² and then interpolates in y to grid center
                              ℑzᵃᵃᶜ(i, j, k, grid, ψ², w)   # Calculates w^2 using function ψ² and then interpolates in z to grid center
                             ) / 2
end

KE = KernelComputedField(Center, Center, Center, kinetic_energy_ccc!, model;
                         computed_dependencies=(u, v, w))
```


It may be useful to know that there are some kernels already defined for commonly-used diagnostics
in packages that are companions to Oceananigans. For example
[Oceanostics.jl](https://github.com/tomchor/Oceanostics.jl/blob/13d2ba5c48d349c5fce292b86785ce600cc19a88/src/TurbulentKineticEnergyTerms.jl#L23-L30)
and
[LESbrary.jl](https://github.com/CliMA/LESbrary.jl/blob/master/src/TurbulenceStatistics/shear_production.jl).
Users should first look there before writing any kernel by hand and are always welcome to [start an
issue on Github](https://github.com/CliMA/Oceananigans.jl/issues/new) if they need help to write a
different kernel.



### Try to decrease the memory-use of your runs

GPU runs are generally memory-limited. As an example, a state-of-the-art Tesla V100 GPU has 32GB of
memory, which is enough to fit, on average, a simulation with about 100 million points --- a bit
smaller than a 512-cubed simulation. (The precise number depends on many other things, such as the
number of tracers simulated, as well as the diagnostics that are calculated.) This means that it is
especially important to be mindful of the size of your runs when running Oceananigans on GPUs and it
is generally good practice to decrease the memory required for your runs. Below are some useful tips
to achieve this

- Use the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) command
  line utility to monitor the memory usage of the GPU. It should tell you how much memory there is
  on your GPU and how much of it you're using.
- Try to use higher-order advection schemes. In general when you use a higher-order scheme you need
  fewer grid points to achieve the same accuracy that you would with a lower-order one. Oceananigans
  provides two high-order advection schemes: 5th-order WENO method (WENO5) and 3rd-order upwind.
- Manually define scratch space to be reused in diagnostics. By default, every time a user-defined
  diagnostic is calculated the compiler reserves a new chunk of memory for that calculation, usually
  called scratch space. In general, the more diagnostics, the more scratch space needed and the bigger
  the memory requirements. However, if you explicitly create a scratch space and pass that same
  scratch space for as many diagnostics as you can, you minimize the memory requirements of your
  calculations by reusing the same memory chunk. As an example, you can see scratch space being
  created
  [here](https://github.com/CliMA/LESbrary.jl/blob/cf31b0ec20219d5ad698af334811d448c27213b0/examples/three_layer_ constant_fluxes.jl#L380-L383)
  and then being used in calculations
  [here](https://github.com/CliMA/LESbrary.jl/blob/cf31b0ec20219d5ad698af334811d448c27213b0/src/TurbulenceStatistics/first_through_third_order.jl#L109-L112).



### Arrays in GPUs are usually different from arrays in CPUs

On the CPU Oceananigans.jl uses regular `Array`s, but on the GPU it has to use `CuArray`s
from the CUDA.jl package. While deep down both are arrays, their implementations are different
and both can behave very differently. For example if can be difficult just view a `CuArray`.
Consider the example below:

```julia
julia> using Oceananigans; using Adapt

julia> grid = RegularRectilinearGrid(size=(1,1,1), extent=(1,1,1))
RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1.0, 0.0]
                 topology: (Periodic, Periodic, Bounded)
  resolution (Nx, Ny, Nz): (1, 1, 1)
   halo size (Hx, Hy, Hz): (1, 1, 1)
grid spacing (Δx, Δy, Δz): (1.0, 1.0, 1.0)

julia> model = IncompressibleModel(grid=grid, architecture=GPU())
IncompressibleModel{GPU, Float64}(time = 0 seconds, iteration = 0) 
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── tracers: (:T, :S)
├── closure: IsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing

julia> typeof(model.velocities.u.data)
OffsetArrays.OffsetArray{Float64,3,CUDA.CuArray{Float64,3}}

julia> adapt(Array, model.velocities.u.data)
3×3×3 OffsetArray(::Array{Float64,3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
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
3×3×3 OffsetArray(::CUDA.CuArray{Float64,3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
Error showing value of type OffsetArrays.OffsetArray{Float64,3,CUDA.CuArray{Float64,3}}:
ERROR: scalar getindex is disallowed
```

Here `CUDA.jl` throws an error because scalar `getindex` is not `allowed`. Another way
around this limitation is to allow scalar operations:

```julia
julia> using CUDA; CUDA.allowscalar(true)

julia> model.velocities.u.data
3×3×3 OffsetArray(::CuArray{Float64,3}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
[:, :, 0] =
┌ Warning: Performing scalar operations on GPU arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`
└ @ GPUArrays ~/.julia/packages/GPUArrays/WV76E/src/host/indexing.jl:43
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

Notice the warning we get when we do this. Scalar operations on GPUs can be very slow, so it is
advised to only use this last method when using the REPL or prototyping --- never in
production-ready scripts.


You might also need to keep these differences in mind when using arrays
to `set!` initial conditions or when using arrays to provide boundary conditions and
forcing functions. To learn more about working with `CuArray`s, see the
[array programming](https://juliagpu.github.io/CUDA.jl/dev/usage/array/) section
of the CUDA.jl documentation.

Something to keep in mind when working with `CuArray`s is that you do not want to set or
get/access elements of a `CuArray` outside of a kernel. Doing so invokes scalar operations
in which individual elements are copied from or to the GPU for processing. This is very
slow and can result in huge slowdowns. For this reason, Oceananigans.jl disables CUDA
scalar operations by default.

See the [scalar indexing](https://juliagpu.github.io/CUDA.jl/dev/usage/workflow/#UsageWorkflowScalar)
section of the CUDA.jl documentation for more information on scalar indexing.

Sometimes you need to perform scalar operations on `CuArray`s in which case you may want
to temporarily allow scalar operations with the `CUDA.@allowscalar` macro or by calling
`CUDA.allowscalar(true)`.
