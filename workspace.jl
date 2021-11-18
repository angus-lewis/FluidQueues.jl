using Plots
include("src/FluidQueues.jl")
include("model_def.jl")
h(t) = (t>0.0) ? FluidQueues.hitting_times_cdf(model,t,1.0,2.0) : zeros(2,4)
tvec = 0.0:0.1:3
cdfs = zeros(length(tvec),4)
for (c,t) in enumerate(tvec)
    h_mat = h(t)
    cdfs[c,:] = h_mat[3:6]
end
plot(tvec,cdfs)