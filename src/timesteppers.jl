struct AdamsBashforthTimestepper
      G :: SourceTerms
     Gp :: SourceTerms
    tmp :: StepperTemporaryFields
      χ :: Float64
end

function AdamsBashforthTimestepper(arch, grid, χ)
       G = SourceTerms(arch, grid)
      Gp = SourceTerms(arch, grid)
     tmp = StepperTemporaryFields(arch, grid)
     return AdamsBashforthTimestepper(G, Gp, tmp, χ)
end
