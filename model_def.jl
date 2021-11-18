T_reflecting = [-1.1 1.1; 1.0 -1.0]
c_reflecting = [1.0; -1.0]
b_reflecting = 10.0

P_upr_reflecting = [0.0 1.0]
P_lwr_reflecting = [1.0 0.0] # both reflecting boundaries

model = BoundedFluidQueue(T_reflecting,c_reflecting,P_lwr_reflecting,P_upr_reflecting,b_reflecting)
