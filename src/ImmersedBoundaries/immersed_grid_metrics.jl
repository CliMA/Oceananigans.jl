for metric in (
               :Δxᶜᶜᵃ,
               :Δxᶜᶠᵃ,
               :Δxᶠᶠᵃ,
               :Δxᶠᶜᵃ,

               :Δyᶜᶜᵃ,
               :Δyᶜᶠᵃ,
               :Δyᶠᶠᵃ,
               :Δyᶠᶜᵃ,

               :Δzᵃᵃᶠ,
               :Δzᵃᵃᶜ,

               :Azᶜᶜᵃ,
               :Azᶜᶠᵃ,
               :Azᶠᶠᵃ,
               :Azᶠᶜᵃ,

               :Axᶜᶜᶜ, 
               :Axᶠᶜᶜ,
               :Axᶠᶠᶜ,
               :Axᶜᶠᶜ,
               :Axᶠᶜᶠ,
               :Axᶜᶜᶠ,
               
               :Ayᶜᶜᶜ,
               :Ayᶜᶠᶜ,
               :Ayᶠᶜᶜ,
               :Ayᶠᶠᶜ,
               :Ayᶜᶠᶠ,
               :Ayᶜᶜᶠ,

               :Vᶜᶜᶜ, 
               :Vᶠᶜᶜ,
               :Vᶜᶠᶜ,
               :Vᶜᶜᶠ,
              )

    @eval begin
        import Oceananigans.Operators: $metric
        @inline $metric(i, j, k, ibg::ImmersedBoundaryGrid) = $metric(i, j, k, ibg.grid)
    end
end
