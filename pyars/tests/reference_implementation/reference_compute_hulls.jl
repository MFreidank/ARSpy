include("ars.jl")
include("ars_defs.jl")
include("logging.jl")
include("probability_utils.jl")


S = [-2.0, 1.996, 0.998, .0, .35082725486762767, .998, .996, .0]
fS = [-4.0, 3.984016, 0.996004, 0.0, 0.1230797627579554, 0.996004, 3.984016, 4.0]
domain = [-Inf, Inf]

arsComputeHulls(S, fS, domain)


