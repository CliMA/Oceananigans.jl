# Navier-Stokes and tracer conservation equations

Oceananigans.jl solves the incompressible Navier-Stokes equations and an arbitrary
number of tracer conservation equations.
Physics associated with individual terms in the momentum and tracer conservation
equations --- the background rotation rate of the equation's reference frame,
gravitational effects associated with buoyant tracers under the Boussinesq
approximation[^1], generalized stresses and tracer fluxes associated with viscous and
diffusive physics, and arbitrary "forcing functions" --- are determined by the whims of the
user.

[^1]: Named after Boussinesq (1903) although used earlier by Oberbeck (1879), the Boussinesq
      approximation neglects density differences in the momentum equation except when associated
      with the gravitational term. It is an accurate approximation for many flows, and especially
      so for oceanic flows where density differences are very small. See Vallis (2017, section 2.4)
      for an oceanographic introduction to the Boussinesq equations and Vallis (2017, Section 2.A)
      for an asymptotic derivation. See Kundu (2015, Section 4.9) for an engineering
      introduction.
