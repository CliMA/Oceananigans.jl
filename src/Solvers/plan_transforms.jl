using FFTW

using Oceananigans: PBC, NFBC

#=
These functions return the transforms required to solve Poisson's equation with
periodic boundary conditions or staggered Neumann boundary.

Fast Fourier transforms (FFTs) are used in the periodic dimensions and
real-to-real discrete cosine transforms are used in the wall-bounded dimensions.
Note that the DCT-II is used for the DCT and the DCT-III for the IDCT
which correspond to REDFT10 and REDFT01 in FFTW.

They operatore on an array with the shape of `A`, which is needed to plan
efficient transforms. `A` will be mutated.
=#

plan_forward_transform(A::Array, ::PBC, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_fft!(A, dims, flags=planner_flag)

plan_forward_transform(A::Array, ::NFBC, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_r2r!(A, FFTW.REDFT10, dims, flags=planner_flag)

plan_backward_transform(A::Array, ::PBC, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_ifft!(A, dims, flags=planner_flag)

plan_backward_transform(A::Array, ::NFBC, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_r2r!(A, FFTW.REDFT01, dims, flags=planner_flag)

@hascuda begin
     plan_forward_transform(::PBC, A::CuArray, dims) = plan_fft!(A, dims)
    plan_backward_transform(::PBC, A::CuArray, dims) = plan_ifft!(A, dims)
end
