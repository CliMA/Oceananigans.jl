"""
    HierarchicalExtrapolation()

Ghosts that continue the nodes before the boundary with a polynomial of degree up to two:

                        boundary
        │   c₋₁   │    c₀   ┃    c₁   │    c₂   │    c₃   │    c₄   │    c₅   │
                            ┃         Δ₁        Δ₂        Δ₃        Δ₄        │
               ghosts                    nodes before the boundary

```math
    \\begin{align}
    φ(a, b) & = 2ab / (a² + b²) \\\\
    G₁  & = [φ(Δ₁, Δ₂) ⋅ φ(Δ₂, Δ₃)]² \\\\
    G₂  & = [φ(κ₁, κ₂) ⋅ φ(κ₂, κ₃)]² \\\\
    σ   & = [G₂ + (1 - G₂) G₁] Δ₁ \\\\
    κ   & = G₂ κ₁ \\\\
    c₀  & = c₁ -  σ -  κ \\\\
    c₋₁ & = c₁ - 2σ - 3κ \\\\
    c₋₂ & = c₁ - 3σ - 6κ
    \\end{align}
```

with `Δⱼ = cⱼ₊₁ - cⱼ`, `κⱼ = Δⱼ - Δⱼ₊₁`, `φ = 0` unless `a` and `b` share a sign, and in general the g-th ghost `c₁₋g = c₁ - g σ - g (g + 1) κ / 2`.
Linear data, and quadratic data on five active nodes, are continued exactly, and a front next to the boundary (`G₁ = G₂ = 0`) has constant ghosts.
"""
struct HierarchicalExtrapolation end

"""
    TransportedWallValue()

Ghosts built from a wall value `cʷ`, stored in the tracer at the node beyond each wall (the inactive node beyond an immersed
wall, the halo node beyond a wall of the domain) and transported as

```math
V ∂ₜcʷ = Az max(-w, 0) (ĉ - cʷ) + Ax [u⁻ (ĉ⁻ - cʷ) - u⁺ (ĉ⁺ - cʷ)]
```

                       interior fluid
          ┌────────────┬────────────┬────────────┐
          │            │            │            │
          │            │     c₂     │            │
          │            │            │            │
          ├────────────┼──── ↓ w ───┼────────────┤
          │            │            │            │
          │            →     c₁     →            │     wall cells: the fluid cells next to the wall
          │           u⁻           u⁺            │
        ━━━━━━━━━━━━━━━━━━━━ ĉ ━━━━━━━━━━━━━━━━━━━━━   wall, where c₁, c₂, … extrapolate to ĉ
               cʷ⁻    ĉ⁻     cʷ    ĉ⁺    cʷ⁺           wall values, stored in the solid nodes beyond it

with `V`, `Ax` and `Az` the volume and face areas of the wall cell of `c₁`, `w` the velocity at its top face, so that
only the fluid entering from the interior relaxes `cʷ` towards `ĉ = c₁ - σʰ/2`, the wall value of the
`HierarchicalExtrapolation` slope `σʰ` of the line `c₁, c₂, …`, and `u⁻`, `u⁺` the velocities at its faces along the
wall, which carry `ĉ⁻`, `ĉ⁺`: the upwind wall value plus half its van Albada slope along the wall. On other walls the
velocities normal to and along the wall take the places of `w` and `u`, and a wall with two directions along it sums the
last term over both. A solid node beyond several walls, such as the corner of a step, stores a single `cʷ`: it starts from
the average of their `ĉ` and steps with the average of their tendencies.

Upwind of a face, with `Δⱼ = cⱼ₊₁ - cⱼ`, `κⱼ = Δⱼ - Δⱼ₊₁` and `φ` as in `HierarchicalExtrapolation`, the ghosts beyond
the wall are

```math
    \\begin{align}
    Δʷ  & = 2 (c₁ - cʷ) \\\\
    K   & = 3/2 (Δʷ - Δ₁) \\\\
    G   & = [φ(K, κ₁) ⋅ φ(κ₁, κ₂)]² \\\\
    σ   & = G Δ₁ + (1 - G) \\, \\mathrm{minmod}(Δʷ, 2Δ₁) \\\\
    κ   & = G K \\\\
    c₀  & = c₁ -  σ -  κ \\\\
    c₋₁ & = c₁ - 2σ - 3κ \\\\
    c₋₂ & = c₁ - 3σ - 6κ
    \\end{align}
```

with `Δʷ` the slope towards `cʷ`, half a node beyond `c₁`, `K` the curvature of the quadratic through `cʷ`, `c₁` and
`c₂`, and `G`, which is 1 when `K`, `κ₁` and `κ₂` are equal and falls to 0 as they differ: the ghosts follow the
quadratic where the line is smooth and the slope towards `cʷ`, limited by `2Δ₁`, otherwise. Downwind ghosts, and those of
fields without stored wall values, extrapolate as `HierarchicalExtrapolation`.

On the walls of the domain where the tracer has a `Value`, `Gradient` or `Mixed` boundary condition, `cʷ` is the wall value
that the condition fills into the halo node beyond the wall. Along a direction with an `Open` side the ghosts extrapolate as
`HierarchicalExtrapolation`.

!!! note
    Wall values are stored only for tracers, at `(Center, Center, Center)`; velocities extrapolate as `HierarchicalExtrapolation`.
"""
struct TransportedWallValue{W}
    wall_values :: W
end

TransportedWallValue() = TransportedWallValue(nothing)

Base.summary(::HierarchicalExtrapolation) = "HierarchicalExtrapolation()"
Base.summary(::TransportedWallValue) = "TransportedWallValue()"
