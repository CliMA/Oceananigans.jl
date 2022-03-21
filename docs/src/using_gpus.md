# Using GPUs

A big feature of Oceananigans is being able to run on graphical processing units (GPUs)
for increased performance. Depending on your CPU and GPU combination, speedups of >150x
are possible, for example on Google Cloud where running on GPUs is more cost-effective.
See the [performance benchmarks](@ref performance_benchmarks) for more details.

See [Architecture](@ref) for instructions on setting up a model on a GPU.

Oceananigans does not yet support distributed parallelism (multi-CPU or multi-GPU).

!!! tip "Running on GPUs"
    If you are having issues with running Oceananigans on a GPU or setting things up,
    please [open an issue](https://github.com/CLiMA/Oceananigans.jl/issues/new)
    and we'll do our best to help out!

## When to use a GPU

GPUs are very useful for running large simulations. If your simulation uses over
1,000,000 grid points, you will probably benefit significantly from running your
simulation on a GPU.

GPU simulations tend to be memory-limited. That is, you'll probably fill the GPU's
memory long before the model becomes unbearably slow. High-end GPUs such as the
Nvidia Tesla V100 only come with up to 32 GB of memory. On a GPU with 16 GB of memory,
you can run a simulation (with 2 tracers) with up to ~50 million grid points.

## Getting access to GPUs

If you don't have a GPU there are a few resources you can try to acquire a GPU from.

In general, to get good performance you'll want a GPU with good 64-bit floating point
performance although Oceananigans can be used with 32-bit floats. Most recent gaming GPUs
should work but might have poor 64-bit float performance.

If you have access to any supercomputer clusters, check to see if they have any GPUs.
See also this Stack Overflow post:
[Where can I get access to GPU cluster for educational purpose?](https://scicomp.stackexchange.com/questions/8508/where-can-i-get-access-to-gpu-cluster-for-educational-purpose)

Cloud computing providers such as Google Cloud and Amazon EC2 allow you to rent GPUs per
hour. Sometimes they offer free trials or credits that can be used towards GPUs although
they seem to be getting less common.

See the [Julia on Google Colab: Free GPU-Accelerated Shareable Notebooks](https://discourse.julialang.org/t/julia-on-google-colab-free-gpu-accelerated-shareable-notebooks/15319)
post on the Julia Discourse.

[Code Ocean](https://codeocean.com/) also has
[GPU support](https://help.codeocean.com/en/articles/1053107-gpu-support) and allows you
to spin up capsules with pretty decent Tesla K80 GPUs for free (for now) if you want to
play around with them. They may not be powerful enough for huge simulations though. You'll
want to use their "Ubuntu Linux with GPU support (18.04.3)" with the ability to compile
CUDA code. Then you'll have to install Julia manually.

## I have a GPU. Now what?

Make sure you have an Nvidia GPU that is CUDA compatible:
[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). Most
recent GPUs should be but older GPUs and many laptop GPUs may not be.

Then download and install the CUDA Toolkit:
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Once the CUDA Toolkit is installed, you might have to build Oceananigans again
```
julia>]
(v1.6) pkg> build Oceananigans
```
The ocean wind mixing and convection example is a good one to test out on the GPU.
