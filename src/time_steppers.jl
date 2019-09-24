"""
    RK3(f, Φ, Δt)

Time-step the ODE `∂Φ/∂t = f(Φ)` using the three-stage third-order Runge-Kutta (RK) method described by
Skamarock & Klemp (2008).

This RK3 scheme is suitable for split-explicit time-stepping for the compressible Navier–Stokes equations.

Note that this RK3 scheme is not a standard Runge–Kutta scheme per se because, while it is third-order accurate for
linear equations, it is only second-order accurate for nonlinear equations.

References
==========
Skamarock & Klemp (2008), "A time-split nonhydrostatic atmospheric model for weather research and forecasting
    applications", Journal of Computational Physics 227, pp.3465-3485.
"""
function RK3(f, Φ, Δt)
    Φ′   = Φ + f(Φ)   * Δt/3
    Φ′′  = Φ + f(Φ′)  * Δt/2
    return Φ + f(Φ′′) * Δt
end

