struct ModelConfiguration
    boundary_conditions::BoundaryConditions
    κh # Horizontal Laplacian heat diffusion [m²/s]. diffKhT in MITgcm.
    κv # Vertical Laplacian heat diffusion [m²/s]. diffKzT in MITgcm.
    𝜈h # Horizontal eddy viscosity [Pa·s]. viscAh in MITgcm.
    𝜈v # Vertical eddy viscosity [Pa·s]. viscAz in MITgcm.
end
