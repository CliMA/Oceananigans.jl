# Simulation tips

In Oceananigans we try to do most of the optimizing behind the scenes, that way the average user
doesn't have to worry about details when setting up a simulation. However, there's just so much
optimization that can be done in the source code. In order to take advantage of Julia's full speed
the user has to be aware of some things when writing the simulation script. Furthermore, in case of
more complex GPU runs, some details could sometimes prevent your simulation from running altogether.
While Julia knowledge is obviously desirable here, a user that is unfamiliar with Julia can get away
with efficient simulations by learning a few rules of thumb. It is also recommended that users go
through Julia's [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/), which
contains a more in-depth explanation of some of the aspects discussed here.


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


### Consider inlining small functions

Inlining is when the compiler [replaces a function call with the body of the function that is being
called before compiling](https://en.wikipedia.org/wiki/Inline_expansion). The advantage of inlining
(which is julia can be done with the [`@inline`
macro](https://docs.julialang.org/en/v1/devdocs/meta/)) is that gets rid of the time spent calling the
function. The Julia compiler automatically makes some calls as to what functions it should or
shouldn't inline, but you can force a function to be inlined by including the macro `@inline` before
its definition. This is more suited for small functions that are called often. Here's an example of
an implementation of the Heaviside function that forces it to be inlined:

```julia
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
```

In practice it's hard to say whether inlining a function will bring runtime benefits _with
certainty_, since Julia already inlines some small functions automatically. However, it is generally
a good idea to at least investigate this aspect in your code as the benefits can potentially be
significant.



## GPU simulation tips

Running on GPUs is very different from running on CPUs. Oceananigans makes most of the necessary
changes in the background, so that for very simple simulations changing between CPUs and GPUs is
just a matter of changing the `architecture` argument in the model from `CPU()` to `GPU()`. However,
for more complex simulations some care needs to be taken on the part of the user. While knowledge of
GPU computing (and Julia) is again desirable, the inexperienced user can also achieve efficient GPU
simulations by following a few simple principles.


### Variables that need to be used in GPU computations need to be defined as constants

Any global variable that needs to be accessed by the GPU needs to be a constant or the simulation will crash


### Complex diagnostics using `ComputedField`s may not work on GPUs

`ComputedField`s are the most convenient way to calculate diagnostics for your simulation. They will
always work on CPUs, but when their complexity is high (in terms of number of abstract operations)
the compiler can't  translate them into GPU code and they fail. For example, in the example below,
calculating `u²` works in both CPUs and GPUs, but calculating `KE` only works in CPUs:

```julia
u, v, w = model.velocities
u² = ComputedField(u^2)
KE = ComputedField((u^2 + v^2 + w^2)/2)
compute!(u²)
compute!(KE)
```

There are two approaches to bypass this issue. The first is to nest `ComputedField`s. For example,
we can make `KE` be computed on GPU by defining it as
```julia
u, v, w = model.velocities
u² = ComputedField(u^2)
v² = ComputedField(v^2)
w² = ComputedField(w^2)
uplusv = ComputedField(u² + v²)
KE = ComputedField((uplusv + w²)/2)
compute!(KE)
```

This is a simple workaround that is especially suited for the development stage of a simulation.
However, when running this on a GPU, the code will iterate over the whole domain 5 times to
calculate `KE` (one for each computed field defined), which is not very efficient.

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

GPU runs are generally memory-limited, so it's good to both keep track of and try to reduce the size of your runs. Useful tips in this regard are

Try to use higher-order schemes as you need fewer grid points to achieve the same resolution

Use nvidia-smi to monitor the memory usage of the GPU

Manually define scratch space to be reused in diagnostics, to avoid creating one scratch space for each separate diagnostic you have.


### Arrays in GPUs are usually different from arrays in CPUs

Talk about converting to CuArrays and viewing CuArrays as well!

