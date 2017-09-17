include("ars.jl")
include("ars_defs.jl")
include("probability_utils.jl")


function gaussian(x; sigma=1)
    log(exp(-x ^ 2 / sigma))
end

name = ARGS[1]
a = parse(Float64, ARGS[2])
b = parse(Float64, ARGS[3])
nSamples = parse(Int64, ARGS[4])

# Change this to parse from args
domain = [-Inf, Inf]

if name == "gaussian"

   result = ars(gaussian, a, b, domain, nSamples) 

end
println(result)

