struct UnboundedFluidQueue <: Model
    T::Array{Float64,2}
    S::PhaseSet
    function UnboundedFluidQueue(T::Array{Float64,2}, S::PhaseSet)
        DiscretisedFluidQueues._fluid_queue_checks(T,S)
        !(all(sum(T,dims=2).≈0.0))&&@warn "row sums of T should be 0"
        return new(T,S)
    end
end

"""
    UnboundedFluidQueue(T::Array{Float64,2},c::Array{Float64,1},b::Float64)

Alias to `UnboundedFluidQueue(T,PhaseSet(c))`.
"""
UnboundedFluidQueue(T::Array{Float64,2},c::Array{Float64,1}) = UnboundedFluidQueue(T,PhaseSet(c))

"""
    _model_dicts(model::Model) 

    input: a Model object
outputs:
     - SDict: a dictionary with keys `"+","-","0","bullet"`
    and corresponding values `findall(model.C .> 0)`, `findall(model.C .< 0)`,
    `findall(model.C .== 0)`, `findall(model.C .!= 0)`, respectively.
     - TDict: a dictionary of submatrices of `T` with keys
    `"ℓm"` with ``ℓ,m∈{+,-,0,bullet}`` and corresponding values
    `model.T[S[ℓ],S[m]]`.
"""
function _model_dicts(model::Model) 
    nPhases = n_phases(model)
    SDict = Dict{String,Array}("S" => 1:nPhases)
    SDict["+"] = findall(rates(model) .> 0)
    SDict["-"] = findall(rates(model) .< 0)
    SDict["0"] = findall(rates(model) .== 0)
    SDict["bullet"] = findall(rates(model) .!= 0)

    TDict = Dict{String,Array}("T" => model.T)
    for ℓ in ["+" "-" "0" "bullet"], m in ["+" "-" "0" "bullet"]
        TDict[ℓ*m] = model.T[SDict[ℓ], SDict[m]]
    end

    return SDict, TDict
end

"""
Construct and evaluate ``Ψ(s)`` for a triditional SFM.

Uses newtons method to solve the Ricatti equation
``D⁺⁻(s) + Ψ(s)D⁻⁺(s)Ψ(s) + Ψ(s)D⁻⁻(s) + D⁺⁺(s)Ψ(s) = 0.``

    psi_fun_x( model::Model; s = 0, MaxIters = 1000, err = 1e-8)

# Arguments
- `model`: a Model object
- `s::Real`: a value to evaluate the LST at
- `MaxIters::Int`: the maximum number of iterations of newtons method
- `err::Float64`: an error tolerance for terminating newtons method. Terminates
    when `max(Ψ_{n} - Ψ{n-1}) .< eps`.

# Output
- `Ψ(s)::Array{Float64,2}` the matrix ``Ψ``
"""
function psi_fun_x( model::Model; s = 0.0, MaxIters = 1000, err = 1e-8)
    SDict, TDict = _model_dicts(model)

    T00inv = inv(TDict["00"] - s * LinearAlgebra.I)
    # construct the generator Q(s)
    Q =
        (1 ./ abs.(rates(model)[SDict["bullet"]])) .* (
            TDict["bulletbullet"] - s * LinearAlgebra.I -
            TDict["bullet0"] * T00inv * TDict["0bullet"]
        )

    # model_without_zero_phases = @suppress_err BoundedFluidQueue(Q,model.S[SDict["bullet"]],model.b)

     ~, QDict = make_QDict(Q,model.S[SDict["bullet"]]) #_model_dicts(model_without_zero_phases)

    Ψ = zeros(Float64, length(SDict["+"]), length(SDict["-"]))
    A = QDict["++"]
    B = QDict["--"]
    D = QDict["+-"]
    # use netwons method to solve the Ricatti equation
    for n in 1:MaxIters
        Ψ = LinearAlgebra.sylvester(A,B,D)
        if maximum(abs.(sum(Ψ,dims=2).-1)) < err
            break
        end
        A = QDict["++"] + Ψ * QDict["-+"]
        B = QDict["--"] + QDict["-+"] * Ψ
        D = QDict["+-"] - Ψ * QDict["-+"] * Ψ
    end

    return Ψ
end

function make_QDict(Q,S_without_zero)
    SDict = Dict{String,Array}("S" => 1:length(S_without_zero))
    SDict["+"] = findall(rates(S_without_zero) .> 0)
    SDict["-"] = findall(rates(S_without_zero) .< 0)
    SDict["0"] = findall(rates(S_without_zero) .== 0)
    SDict["bullet"] = findall(rates(S_without_zero) .!= 0)

    TDict = Dict{String,Array}("T" => Q)
    for ℓ in ["+" "-" "0" "bullet"], m in ["+" "-" "0" "bullet"]
        TDict[ℓ*m] = Q[SDict[ℓ], SDict[m]]
    end

    return SDict, TDict
end

"""
Construct the vector ``ξ`` containing the distribution of the phase at the time
when ``X(t)`` first hits `0`.

    xi_x( model::Model, Ψ::Array)

# Arguments
- `model`: a Model object
- `Ψ`: an array as output from `psi_fun_x`

# Output
- the vector `ξ`
"""
function xi_x( model::Model, Ψ::Array)
    # the system to solve is [ξ 0](-[B₋₋ B₋₀; B₀₋ B₀₀])⁻¹[B₋₊; B₀₊]Ψ = ξ
    # writing this out and using block inversion (as described on wikipedia)
    # we can solve this in the following way
    SDict, TDict = _model_dicts(model)

    T00inv = inv(TDict["00"])
    invT₋₋ =
        inv(TDict["--"] - TDict["-0"] * T00inv * TDict["0-"])
    invT₋₀ = -invT₋₋ * TDict["-0"] * T00inv

    A =
        -(
            invT₋₋ * TDict["-+"] * Ψ + invT₋₀ * TDict["0+"] * Ψ +
            LinearAlgebra.I
        )
    b = zeros(1, size(TDict["--"], 1))
    A[:, 1] .= 1.0 # normalisation conditions
    b[1] = 1.0 # normalisation conditions

    ξ = b / A

    return ξ
end

"""
Construct the stationary distribution of the SFM

    stationary_distribution_x( model::Model, Ψ::Array, ξ::Array)

# Arguments
- `model`: a Model object
- `Ψ`: an array as output from `psi_fun_x`
- `ξ`: an array as returned from `xi_x`

# Output
- `pₓ::Array{Float64,2}`: the point masses of the SFM
- `πₓ(x)` a function with two methods
    - `πₓ(x::Real)`: for scalar inputs, returns the stationary density evaluated
        at `x` in all phases.
    - `πₓ(x::Array)`: for array inputs, returns an array with the same shape
        as is output by Coeff2Dist.
- `K::Array{Float64,2}`: the matrix in the exponential of the density.
"""
function stationary_distribution_x( model::Model, Ψ::Array, ξ::Array)
    # using the same block inversion trick as in xi_x
    SDict, TDict = _model_dicts(model)
    
    T00inv = inv(TDict["00"])
    invT₋₋ =
        inv(TDict["--"] - TDict["-0"] * T00inv * TDict["0-"])
    invT₋₀ = -invT₋₋ * TDict["-0"] * T00inv

    Q =
        (1 ./ abs.(rates(model)[SDict["bullet"]])) .* (
            TDict["bulletbullet"] -
            TDict["bullet0"] * T00inv * TDict["0bullet"]
        )

    # model_without_zero_phases = @suppress_err BoundedFluidQueue(Q,model.S[SDict["bullet"]],model.b)

    # ~, QDict = _model_dicts(model_without_zero_phases)
     ~, QDict = make_QDict(Q,model.S[SDict["bullet"]])
    
    K = QDict["++"] + Ψ * QDict["-+"]

    A = -[invT₋₋ invT₋₀]

    # unnormalised values
    αpₓ = ξ * A

    απₓ = αpₓ *
        [TDict["-+"]; TDict["0+"]] *
        -inv(K) *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(rates(model)[SDict["bullet"]]))

    απₓ0 = -απₓ * [TDict["+0"];TDict["-0"]] * T00inv

    # normalising constant
    α = sum(αpₓ) + sum(απₓ) + sum(απₓ0)

    # normalised values
    # point masses
    pₓ = αpₓ/α
    # density method for scalar x-values
    idx = [findall(rates(model).>0);
        findall(rates(model).<0);
        findall(rates(model).==0)]
    function πₓ(x::Real)
        out = zeros(n_phases(model))
        out[idx] = (pₓ *
        [TDict["-+"]; TDict["0+"]] *
        exp(K*x) *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(rates(model)[SDict["bullet"]])) *
        [LinearAlgebra.I(sum(rates(model) .!= 0)) [TDict["+0"];TDict["-0"]] * -T00inv])
        return out 
    end
    # density method for arrays so that πₓ returns an array with the same shape
    # as is output by Coeff2Dist
    function πₓ(x::Array)
        temp = πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), n_phases(model))
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    # CDF method for scalar x-values
    function Πₓ(x::Real)
        out = zeros(n_phases(model))
        out[idx] = [zeros(1,sum(rates(model).>0)) pₓ] .+
        pₓ *
        [TDict["-+"]; TDict["0+"]] *
        (exp(K*x) - LinearAlgebra.I) / K *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(rates(model)[SDict["bullet"]])) *
        [LinearAlgebra.I(sum(rates(model) .!= 0)) [TDict["+0"];TDict["-0"]] * -T00inv]
        return out 
    end
    # CDF method for arrays so that Πₓ returns an array with the same shape
    # as is output by Coeff2Dist
    function Πₓ(x::Array)
        temp = Πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), n_phases(model))
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    return pₓ, πₓ, Πₓ, K
end

function stationary_distribution_x( model::Model)
    Ψ = psi_fun_x( model)
    ξ = xi_x( model, Ψ)
    pₓ, πₓ, Πₓ, K = stationary_distribution_x(model,Ψ,ξ)
    return pₓ, πₓ, Πₓ, K
end
rate_reverse(model::BoundedFluidQueue) = BoundedFluidQueue(model.T,-rates(model),model.P_lwr,model.P_upr,model.b)

function expected_crossings(model::BoundedFluidQueue)
    rate_reverse_model = rate_reverse(model)
    ~, TDict = _model_dicts(model)
    ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    K = TDict["++"] + Ψ*TDict["-+"]
    K̃ = T̃Dict["++"] + Ψ̃*T̃Dict["-+"]
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    N(x) = [I₊ exp(K*model.b)*Ψ; exp(K̃*model.b)*Ψ̃ I₋]\[exp(K*x) zeros(n₊,n₋); zeros(n₋,n₊) exp(K̃*(model.b-x))]*
        [I₊ Ψ; Ψ̃ I₋]
    return N
end

function int_expected_crossings(model::BoundedFluidQueue)
    rate_reverse_model = rate_reverse(model)
    ~, TDict = _model_dicts(model)
    ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    K = TDict["++"] + Ψ*TDict["-+"]
    K̃ = T̃Dict["++"] + Ψ̃*T̃Dict["-+"]
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    N(x) = [I₊ exp(K*model.b)*Ψ; exp(K̃*model.b)*Ψ̃ I₋]\[K\(exp(K*x)-I₊) zeros(n₊,n₋); zeros(n₋,n₊) K̃\(exp(K̃*model.b)-exp(K̃*(model.b-x)))]*
        [I₊ Ψ; Ψ̃ I₋]
    return N
end

function hat_matrices(model::BoundedFluidQueue)
    rate_reverse_model = rate_reverse(model)
    ~, TDict = _model_dicts(model)
    # ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    U = TDict["--"] + TDict["-+"]*Ψ
    Ũ = TDict["++"] + TDict["+-"]*Ψ̃
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    b = model.b
    Λ̂₁₁ = (I₊ - Ψ*Ψ̃)*exp(Ũ*b)/(I₊ - Ψ*exp(U*b)*Ψ̃*exp(Ũ*b))
    Ψ̂₁₂ = (Ψ - exp(Ũ*b)*Ψ*exp(U*b))/(I₋ - Ψ̃*exp(Ũ*b)*Ψ*exp(U*b))
    Ψ̂₂₁ = (Ψ̃ - exp(U*b)*Ψ̃*exp(Ũ*b))/(I₊ - Ψ*exp(U*b)*Ψ̃*exp(Ũ*b))
    Λ̂₂₂ = (I₋ - Ψ̃*Ψ)*exp(U*b)/(I₋ - Ψ̃*exp(Ũ*b)*Ψ*exp(U*b))
    return Λ̂₁₁, Ψ̂₁₂, Ψ̂₂₁, Λ̂₂₂
end

function H_matrix(model)
    Λ̂₁₁, Ψ̂₁₂, Ψ̂₂₁, Λ̂₂₂ = hat_matrices(model)
    hat_matrix = [Λ̂₁₁ Ψ̂₁₂; Ψ̂₂₁ Λ̂₂₂]
    S, TDict = _model_dicts(model)
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    n = length(model.S)
    Ĥ = hat_matrix*[zeros(n₊,n₊) model.P_upr[:,S["-"]]; model.P_lwr[:,S["+"]] zeros(n₋,n₋)] + 
        hat_matrix*[model.P_upr[:,[i∉S["-"] for i in 1:n]] zeros(n₊,n-n₊); zeros(n₋,n-n₋) model.P_lwr[:,[i∉S["+"] for i in 1:n]]]*
            [-inv(model.T[[i∉S["-"] for i in 1:n],[i∉S["-"] for i in 1:n]]) zeros(n-n₋,n-n₊); 
             zeros(n-n₊,n-n₋) -inv(model.T[[i∉S["+"] for i in 1:n],[i∉S["+"] for i in 1:n]])]*
                [zeros(n-n₋,n₊) model.T[[i∉S["-"] for i in 1:n],S["-"]];
                model.T[[i∉S["+"] for i in 1:n],S["+"]] zeros(n-n₊,n₋)]
    return Ĥ
end

function nu(model)
    A = H_matrix(model) - LinearAlgebra.I
    A[:,1] .= 1.0
    b = zeros(1,size(A,1))
    b[1] = 1.0
    return b/A
end

function normalising_constant_matrix(model)
    rate_reverse_model = rate_reverse(model)
    ~, TDict = _model_dicts(model)
    ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    K = TDict["++"] + Ψ*TDict["-+"]
    K̃ = T̃Dict["++"] + Ψ̃*T̃Dict["-+"]
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    if abs(LinearAlgebra.det(K))>sqrt(eps()) && abs(LinearAlgebra.det(K̃))>sqrt(eps())
        N = [I₊ exp(K*model.b)*Ψ; exp(K̃*model.b)*Ψ̃ I₋]\[Matrix(K)\(exp(K*model.b)-I₊) zeros(n₊,n₋); zeros(n₋,n₊) Matrix(K̃)\(exp(K̃*model.b)-I₋)]*
            [I₊ Ψ; Ψ̃ I₋]
    elseif abs(LinearAlgebra.det(K))>sqrt(eps())
        N = Matrix([I₊ exp(K*model.b)*Ψ; exp(K̃*model.b)*Ψ̃ I₋])\Matrix([Matrix(K)\(exp(K*model.b)-I₊) zeros(n₊,n₋); zeros(n₋,n₊) model.b*I₋])*
            [I₊ Ψ; Ψ̃ I₋]
    elseif abs(LinearAlgebra.det(K̃))>sqrt(eps())
        N = [I₊ exp(K*model.b)*Ψ; exp(K̃*model.b)*Ψ̃ I₋]\[model.b*I₊ zeros(n₊,n₋); zeros(n₋,n₊) Matrix(K̃)\(exp(K̃*model.b)-I₋)]*
            [I₊ Ψ; Ψ̃ I₋]
    else
        N = [I₊ exp(K*model.b)*Ψ; exp(K̃*model.b)*Ψ̃ I₋]\[model.b*I₊ zeros(n₊,n₋); zeros(n₋,n₊) model.b*I₋]*
            [I₊ Ψ; Ψ̃ I₋]
    end
    return N
end

function B_matrices(model::BoundedFluidQueue)
    Λ̂₁₁, Ψ̂₁₂, Ψ̂₂₁, Λ̂₂₂ = hat_matrices(model)
    S, ~ = _model_dicts(model)
    n = n_phases(model)
    B₁ₒ = (LinearAlgebra.I - Ψ̂₁₂*model.P_lwr[:,S["+"]])\Ψ̂₁₂*model.P_lwr[:,[i∉S["+"] for i in 1:n]]
    B₂ₒ = (LinearAlgebra.I - Ψ̂₂₁*model.P_upr[:,S["-"]])\Ψ̂₂₁*model.P_upr[:,[i∉S["-"] for i in 1:n]]
    B₁₁ = (LinearAlgebra.I - Ψ̂₁₂*model.P_lwr[:,S["+"]])\Λ̂₁₁
    A₂₂ = (LinearAlgebra.I - Ψ̂₂₁*model.P_upr[:,S["-"]])\Λ̂₂₂
    return B₁ₒ, B₂ₒ, B₁₁, A₂₂
end

function omega_matrices(model::BoundedFluidQueue)
    B₁ₒ, B₂ₒ, B₁₁, A₂₂ = B_matrices(model)
    n = n_phases(model)
    S, ~ = _model_dicts(model)
    Ω̂₁ₒ = (LinearAlgebra.I - B₁₁*model.P_upr[:,S["-"]]*A₂₂*model.P_lwr[:,S["+"]])\
        (B₁₁*model.P_upr[:,[i∉S["-"] for i in 1:n]] + B₁₁*model.P_upr[:,S["-"]]*B₂ₒ)
    Ω₂ₒ = (LinearAlgebra.I - A₂₂*model.P_lwr[:,S["+"]]*B₁₁*model.P_upr[:,S["-"]])\
        (A₂₂*model.P_lwr[:,[i∉S["+"] for i in 1:n]] + A₂₂*model.P_lwr[:,S["+"]]*B₁ₒ)
    Ω₁ₒ = (LinearAlgebra.I - B₁₁*model.P_upr[:,S["-"]]*A₂₂*model.P_lwr[:,S["+"]])\
        (B₁₁*model.P_upr[:,S["-"]]*A₂₂*model.P_lwr[:,S["+"]]*B₁ₒ +B₁₁*model.P_upr[:,S["-"]]*A₂₂*model.P_lwr[:,[i∉S["+"] for i in 1:n]])
    Ω̂₂ₒ = (LinearAlgebra.I - A₂₂*model.P_lwr[:,S["+"]]*B₁₁*model.P_upr[:,S["-"]])\
        (A₂₂*model.P_lwr[:,S["+"]]*B₁₁*model.P_upr[:,S["-"]]*B₂ₒ + A₂₂*model.P_lwr[:,S["+"]]*B₁₁*model.P_upr[:,[i∉S["-"] for i in 1:n]])
    return Ω̂₁ₒ, Ω₂ₒ, Ω₁ₒ, Ω̂₂ₒ
end 

function point_mass(model)
    B₁ₒ, B₂ₒ, B₁₁, A₂₂ = B_matrices(model)
    Ω̂₁ₒ, Ω₂ₒ, Ω₁ₒ, Ω̂₂ₒ = omega_matrices(model)
    n = n_phases(model)
    S, ~ = _model_dicts(model)
    Z₁₁ = model.T[[i∉S["-"] for i in 1:n],[i∉S["-"] for i in 1:n]] + model.T[[i∉S["-"] for i in 1:n],S["-"]]*(B₂ₒ+Ω̂₂ₒ) 
    Z₁₂ = model.T[[i∉S["-"] for i in 1:n],S["-"]]*Ω₂ₒ
    Z₂₁ = model.T[[i∉S["+"] for i in 1:n],S["+"]]*Ω̂₁ₒ 
    Z₂₂ = model.T[[i∉S["+"] for i in 1:n],[i∉S["+"] for i in 1:n]] + model.T[[i∉S["+"] for i in 1:n],S["+"]]*(B₁ₒ+Ω₁ₒ) 
    Z = [Z₁₁ Z₁₂; Z₂₁ Z₂₂]
    Z[:,1] .= 1.0
    b = zeros(1,size(Z,1))
    b[1] = 1.0
    return b/Z
end

function preprocess(model::Model)
    S,T = _model_dicts(model)   
    not_zero = [i∉S["0"] for i in 1:n_phases(model)]
    C = rates(model)[not_zero]
    Cinv = 1.0./abs.(C)
    Q = Cinv.*model.T[not_zero,not_zero] + 
        Cinv.*model.T[not_zero,S["0"]]*(-model.T[S["0"],S["0"]])^-1*model.T[S["0"],not_zero]
    P_lwr = model.P_lwr[:,not_zero]
    P_upr = model.P_lwr[:,not_zero]
    return BoundedFluidQueue(Q,C,P_lwr,P_upr,model.b)
end

"""
This might be the least robust piece of code ever written. Use with caution. Answers you get might
just be straight up wrong without warning. Be careful. 
"""
function stationary_distribution_x(model::BoundedFluidQueue)
    ν = nu(model)
    S, T = _model_dicts(model)
    Cinv = LinearAlgebra.diagm(1.0./[rates(model)[S["+"]];abs.(rates(model)[S["-"]])])
    n = n_phases(model)

    # density matrix 
    N = expected_crossings(model)

    # cdf matrix
    int_N = int_expected_crossings(model)

    # boundary probabilities
    p = point_mass(model)
    n_upr = n-length(S["-"])
    p_upr = p[1:n_upr]'
    p_lwr = p[n_upr+1:end]'
    # p_to_density = [p_lwr*model.T[[i∉S["+"] for i in 1:n],S["+"]] p_upr*model.T[[i∉S["-"] for i in 1:n],S["-"]]]
    k = sum(p)

    # normalising constant 
    # integral over denisty matrix
    N_const = int_N(model.b)# DiscretisedFluidQueues.normalising_constant_matrix(model)
    # integrals over densities for + and -
    plus_minus_m_norm_vec = ν*N_const*Cinv
    # intetegral over zero phases
    zero_norm_vec = plus_minus_m_norm_vec*[T["+0"];T["-0"]]*(-inv(T["00"]))
    # the constant 
    z = sum(plus_minus_m_norm_vec) + sum(zero_norm_vec) #+ sum(p)
    
    if (sum(p_lwr)==0.0)&&(sum(p_upr)==0.0)
        cz = z
        F0 = zeros(n)
        Fb = zeros(n)
    elseif (sum(p_upr)==0.0)
        vec0₋ = (ν*N(0)*Cinv)[length(S["+"])+1:end]
        π0₋ = vec0₋.*model.P_lwr[:,[i∉S["+"] for i in 1:n]]
        z0 = sum(π0₋)
        
        k0 = sum(p_lwr*model.T[[i∉S["+"] for i in 1:n],S["+"]])

        r0 = z0/k0

        cz = z + k*r0
        cp = cz/r0

        p /= cp
        p_upr = p[1:n_upr]'
        p_lwr = p[n_upr+1:end]'

        F0 = zeros(n)
        Fb = zeros(n)
        counter0 = 0
        counterb = 0
        for i in 1:n
            if rates(model)[i]<=0 
                counter0 += 1
                F0[i] = p_lwr[counter0]
            end
            if rates(model)[i]>=0
                counterb += 1
                Fb[i] = p_upr[counterb]
            end
        end
    else 
        vec0₊ = (ν*N(model.b)*Cinv)[1:length(S["+"])]
        π0₊ = vec0₊.*model.P_upr[:,[i∉S["-"] for i in 1:n]]
        z0 = sum(π0₊)
        
        k0 = sum(p_upr*model.T[[i∉S["-"] for i in 1:n],S["-"]])

        r0 = z0/k0

        cz = z + k*r0
        cp = cz/r0

        p /= cp
        p_upr = p[1:n_upr]'
        p_lwr = p[n_upr+1:end]'

        F0 = zeros(n)
        Fb = zeros(n)
        counter0 = 0
        counterb = 0
        for i in 1:n
            if rates(model)[i]<=0 
                counter0 += 1
                F0[i] = p_lwr[counter0]
            end
            if rates(model)[i]>=0
                counterb += 1
                Fb[i] = p_upr[counterb]
            end
        end
    end
    # p /= z
    # p_upr = p[1:n_upr]'
    # p_lwr = p[n_upr+1:end]'
    # p_to_density = [p_lwr*model.T[[i∉S["+"] for i in 1:n],S["+"]] p_upr*model.T[[i∉S["-"] for i in 1:n],S["-"]]]

    idx = [findall(rates(model).>0);
        findall(rates(model).<0);
        findall(rates(model).==0)]
    function πₓ(x)
        vec = ν*N(x)*Cinv
        ϕ̂ = vec*[T["+0"];T["-0"]]*(-inv(T["00"]))
        out = [vec ϕ̂]./cz
        return out[idx]
    end
    function Πₓ(x)
        vec = ν*int_N(x)*Cinv
        ϕ̂ = vec*[T["+0"];T["-0"]]*(-inv(T["00"]))
        out = [vec ϕ̂]./cz
        return (F0 + out[idx])*(zero(x)<=x) + Fb*(model.b<=x)
    end
    return πₓ, Πₓ
end

## HITTING TIMES ## 
_error_on_neg(x) = (x>=0.0) ? true : throw(DomainError("x must be positive"))
"""
Returns G₋₋(s,x), G₊₋(s,x)
"""
function draining_times_lst(model::Model)
    Ψ(s::ComplexF64) = psi_fun_x(model,s=s)
    S,T = _model_dicts(model)
    B(s::ComplexF64) = (-1.0./rates(model)[S["-"]])*(T["--"]-s*LinearAlgebra.I + T["-+"]*Ψ(s) - T["-0"]*inv(T["00"]-s*LinearAlgebra.I)*(T["0-"] + T["0+"]*Ψ(s)))
    G₋₋(s::ComplexF64,x::Float64) = _error_on_neg(x)&&exp(B(s)*x)
    G₊₋(s::ComplexF64,x::Float64) = Ψ(s)*G₋₋(s,x)
    G₋₋(s::Real,x::Real) = G₋₋(Complex{Float64}(s),Float64(x))
    G₊₋(s::Real,x::Real) = G₊₋(Complex{Float64}(s),Float64(x))
    return G₋₋, G₊₋
end 
function draining_times_lst(model::Model,s,x)
    _error_on_neg(x)
    Ψ = psi_fun_x(model,s=s)
    S,T = _model_dicts(model)
    B = (-1.0./rates(model)[S["-"]])*(T["--"]-s*LinearAlgebra.I + T["-+"]*Ψ - T["-0"]*inv(T["00"]-s*LinearAlgebra.I)*(T["0-"] + T["0+"]*Ψ))
    G₋₋ = exp(B*x)
    G₊₋ = Ψ*G₋₋
    return G₋₋, G₊₋
end 
"""
Returns H₊₊(s,x), H₋₊(s,x)
"""
filling_times_lst(model::Model) = draining_times_lst(rate_reverse(model))
filling_times_lst(model::Model,s,x) = draining_times_lst(rate_reverse(model),s,x)

function draining_filling_times_lst(model::Model,s,x,y)
    _error_on_neg(x)
    _error_on_neg(y)
    _error_on_neg(y-x)
    Ψ = psi_fun_x(model,s=s)
    S,T = _model_dicts(model)
    the_inv = inv(T["00"]-s*LinearAlgebra.I)
    B = (-1.0./rates(model)[S["-"]])*(T["--"]-s*LinearAlgebra.I + T["-+"]*Ψ - T["-0"]*the_inv*(T["0-"] + T["0+"]*Ψ))
    G₋₋y = exp(B*y)
    G₊₋y = Ψ*G₋₋y
    G₋₋x = exp(B*x)
    G₊₋x = Ψ*G₋₋x

    Ξ = psi_fun_x(rate_reverse(model),s=s)
    # S,T = _model_dicts(model)
    A = (1.0./rates(model)[S["+"]])*(T["++"]-s*LinearAlgebra.I + T["+-"]*Ξ - T["+0"]*the_inv*(T["0+"] + T["0-"]*Ξ))
    H₊₊y = exp(A*y)
    H₋₊y = Ξ*H₊₊y
    H₊₊z = exp(A*(y-x))
    H₋₊z = Ξ*H₊₊z

    n₊, n₋ = size(G₊₋y)

    Gy = [zeros(n₊,n₊) G₊₋y; zeros(n₋,n₊) G₋₋y]
    Gx = [zeros(n₊,n₊) G₊₋x; zeros(n₋,n₊) G₋₋x]
    Hy= [H₊₊y zeros(n₊,n₋); H₋₊y zeros(n₋,n₋)]
    Hz= [H₊₊z zeros(n₊,n₋); H₋₊z zeros(n₋,n₋)]
    lst = [Gx Hz]/[LinearAlgebra.I Hy; Gy LinearAlgebra.I]
    return lst
end

function build_H_lst(model::Model)
    H₊₊, H₋₊ = filling_times_lst(model)
    S, ~ = _model_dicts(model)
    n₋, n₊ = length(S["-"]), length(S["+"])
    H(s::ComplexF64,x::Float64) = _error_on_neg(x)&&[H₊₊(s,x) zeros(n₊,n₋); H₋₊(s,x) zeros(n₋,n₋)]
    H(s::Real,x::Real) = H(Complex{Float64}(s),Float64(x)) 
    return H
end
function build_H_lst(model::Model,s,x)
    H₊₊, H₋₊ = filling_times_lst(model,s,x)
    S, ~ = _model_dicts(model)
    n₋, n₊ = length(S["-"]), length(S["+"])
    H = [H₊₊ zeros(n₊,n₋); H₋₊ zeros(n₋,n₋)]
    return H
end

function build_G_lst(model::Model)
    G₋₋, G₊₋ = draining_times_lst(model)
    S, ~ = _model_dicts(model)
    n₋, n₊ = length(S["-"]), length(S["+"])
    G(s::ComplexF64,x::Float64) = [zeros(n₊,n₊) G₊₋(s,x); zeros(n₋,n₊) G₋₋(s,x)]
    G(s::Real,x::Real) = G(Complex{Float64}(s),Float64(x)) 
    return G
end
function build_G_lst(model::Model,s,x)
    G₋₋, G₊₋ = draining_times_lst(model,s,x)
    S, ~ = _model_dicts(model)
    n₋, n₊ = length(S["-"]), length(S["+"])
    G = [zeros(n₊,n₊) G₊₋; zeros(n₋,n₊) G₋₋]
    return G
end

function hitting_times_lst(model::Model)
    G = build_G_lst(model)
    H = build_H_lst(model)
    lst(s::ComplexF64,x::Float64,y::Float64) = [G(s,x) H(s,y-x)]/[LinearAlgebra.I H(s,y); G(s,y) LinearAlgebra.I]
    lst(s::Real,x::Real,y::Real) = lst(Complex{Float64}(s),Float64(x),Float64(y))
    return lst
end 
function hitting_times_lst(model::Model,s,x,y)
    Gx = build_G_lst(model,s,x)
    Gy = build_G_lst(model,s,y)
    Hz = build_H_lst(model,s,y-x)
    Hy = build_H_lst(model,s,y)
    lst = [Gx Hz]/[LinearAlgebra.I Hy; Gy LinearAlgebra.I]
    return lst
end 

for fun_name in (:filling, :draining)
    @eval function $(Symbol(fun_name,"_times_fun"))(model::Model,t,x)
        fun = ilt(cme,s->$(Symbol(fun_name,"_times_lst"))(model,s,x),t,n)
        return fun
    end    
end

function hitting_times_cdf(model,t,x,y,n=21)
    return ilt(cme,s->draining_filling_times_lst(model,s,x,y)./s,t,n)
end
