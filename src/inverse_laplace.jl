abstract type AbstractInverseLaplaceMethod end
struct CME <: AbstractInverseLaplaceMethod end
struct Talbot <: AbstractInverseLaplaceMethod end
struct Euler <: AbstractInverseLaplaceMethod end
struct Gaver <: AbstractInverseLaplaceMethod end

const cme = CME()
const talbot = Talbot()
const euler = Euler()
const gaver = Gaver()

function ilt(m::AbstractInverseLaplaceMethod, f_s, t, M)
    error("This inverse laplace transform method is unknown. Supported ones: cme, euler, gaver, talbot");
end

ilt(f_s::Function, T::Real, maxFnEvals::Int) = ilt(cme,f_s,T,maxFnEvals)

function ilt(m::CME,f_s::Function, T::Real, maxFnEvals::Int)
    # find the most steep CME satisfying maxFnEvals
    params = CMEParams[1];
    for i=2:length(CMEParams)
        if CMEParams[i]["cv2"]<params["cv2"] && (CMEParams[i]["n"]+1)<=maxFnEvals
            params = CMEParams[i];
        end
    end
    # compute eta and beta parameters
    eta = [params["c"]*params["mu1"]  params["a"]'*params["mu1"] + 1im*params["b"]'*params["mu1"]][:];
    beta = [1; (1).+1im*collect(1:params["n"])*params["omega"]][:] * params["mu1"];
    return 1 ./T .* sum(real(eta .* f_s.(beta./T)))
end

function ilt(m::Euler, f_s::Function, T::Real, maxFnEvals::Int)
    n_euler = Int(floor((maxFnEvals-1)/2))
    eta = [0.5 ones(1, n_euler) zeros(1, n_euler-1) 2^-Float64(n_euler)]
    for k = 1:n_euler-1
#        eta(2*n_euler-k + 1) = eta(2*n_euler-k + 2) + 2^-n_euler * nchoosek(n_euler, k);
        eta[2*n_euler-k + 1] = eta[2*n_euler-k + 2] + exp(sum(log.(1:n_euler)) .- n_euler*log(2) .- sum(log.(1:k)) - sum(log.(1:(n_euler-k))))
    end
    k = collect(0:2*n_euler)'
    beta = n_euler*log(10)/3 .+ 1im*pi*k
    eta  = (10^((n_euler)/3))*(1 .-mod.(k, 2)*2) .* eta
    return 1 ./T .* sum(real(eta .* f_s.(beta./T)))
end

function ilt(m::Gaver,f_s::Function, T::Real, maxFnEvals::Int)
    if mod(maxFnEvals,2)==1
        maxFnEvals = maxFnEvals - 1
    end
    ndiv2 = maxFnEvals/2
    eta = zeros(1,maxFnEvals)
    beta = zeros(1,maxFnEvals)
    for k = 1:maxFnEvals # itration index
        inside_sum = 0.0
        for j = floor((k+1)/2):min(k,ndiv2) #  eta summation index
#            inside_sum=inside_sum+((j^((ndiv2+1))/factorial(ndiv2))*(nchoosek(ndiv2, j)*nchoosek(2*j, j)*nchoosek(j, k-j)));           
            inside_sum=inside_sum+exp((ndiv2+1)*log(j) - sum(log.(1:(ndiv2-j))) + sum(log.(1:2*j)) - 2*sum(log.(1:j)) - sum(log.(1:(k-j))) - sum(log.(1:(2*j-k))))
        end
        eta[k]=log(2.0)*(-1)^(k+ndiv2)*inside_sum
        beta[k] = k * log(2.0)
    end  
    return 1 ./T .* sum(real(eta .* f_s.(beta./T)))
end

function ilt(m::Talbot, f_s::Function, t::Real, M::Int)

    # ilt = talbot_inversion(f_s, t, [M])
    #
    # Returns an approximation to the inverse Laplace transform of function
    # handle f_s evaluated at a value t using Talbot's method as
    # summarized in the source below.
    # 
    # f_s: a function of s
    # t:   a time at which to evaluate f(t) 
    # M:   Optional, number of terms to sum for each t (64 is a good guess);
    #      highly oscillatory functions require higher M, but this can grow
    #      unstable; see test_talbot.m for an example of stability.
    # 
    # Abate, Joseph, and Ward Whitt. "A Unified Framework for Numerically 
    # Inverting Laplace Transforms." INFORMS Journal of Computing, vol. 18.4 
    # (2006): 408-421. Print.
    # 
    # Also online: http://www.columbia.edu/~ww2040/allpapers.html.

        
        # Vectorized Talbot's algorithm
        
        k = 1:(M-1); # Iteration index
        
        # Calculate delta for every index.
        delta = zeros(Complex,1, M);
        delta[1] = 2*M/5;
        delta[2:end] = 2*pi/5 * k .* (cot.(pi/M*k).+1im);
        
        # Calculate gamma for every index.
        gamma = zeros(Complex,1, M);
        gamma[1] = 0.5*exp.(delta[1]);
        gamma[2:end] =    (1 .+ 1im*pi/M*k.*(1 .+cot.(pi/M*k).^2)-1im*cot.(pi/M*k)) .* exp.(delta[2:end]);
        
        
        # Finally, calculate the inverse Laplace transform for each given time.
        ilt = 0.4./t .* sum(real( gamma .* f_s.(delta./t)));

        return ilt
end

