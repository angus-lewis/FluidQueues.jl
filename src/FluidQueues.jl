module FluidQueues
using DiscretisedFluidQueues
import LinearAlgebra
using JSON 

CMEParams = JSON.parsefile((@__DIR__)*"/iltparams.json")

include("inverse_laplace.jl")
include("SFM_operators.jl")

end