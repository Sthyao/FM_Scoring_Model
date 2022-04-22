from logging import basicConfig
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def mat_structure(len1,len2,d):
    x_m = []
    for i in range(len1+len2):
        start = i*d+1
        end = (i+1)*d+1
        vars = symbols('x_' + str(start) + ':' + str(end))
        x_m.append(vars)
    return Matrix(x_m)

def SRG(m_init,d=2):
    
    #d is the dimension of implicit vectors
    m,n = m_init.shape
    xvars = mat_structure(m,n,d)
    ans_list = []
    d_list = []

    #Represents implicit vectors belonging to rows and columns, respectively
    ind1 = [x for x in range(m)]
    ind2 = [x for x in range(m,n+m)]

    #To avoid singular matrices, we need to modify the constraints
    
    fx = sum([s**2 for s in (xvars.row(ind1)*(xvars.row(ind2).T) - m_init)])
    hxs = [sum(xvars.row(ind1)*(xvars.row(ind2).T) - m_init),
    sum(xvars.row(ind1).col(0))-m,    
    sum(xvars.row(ind1).col(1))-m]

    xcurr = [1.5,0.5, 0.5, 0.3, 1.5, 1.7, 1, 1.5, 1, 3, 2, 1, 3, 1,3,2]
    xcurr = np.array(xcurr)
    x_l1 = np.zeros(len(xcurr))
    x_l1[-1] = 0.01
    x_l1[-2] = 0.01
    x_l2 = np.zeros(len(xcurr))
    
    # Parameter initializations
    # The parameters of the fastest descent and the parameters of the gradient method

    alpha_fast = 1
    speed_fast = 0.4
    
    alpha_norm = 0.003									
    
    max_iter = 10
    max_outer_iter = 2
    eps = 0.001

    #Generates a list of stored derivatives
    dfx = np.array([diff(fx, xvar) for xvar in xvars])
    dhxs = np.array([[diff(hx, xvar) for xvar in xvars] for hx in hxs])
    nonbasic_vars = len(xvars) - len(hxs)
    basic_vars = len(hxs)
    opt_sols = []

    #Divide constrained and unconstrained variables
    constrain_ind = [x for x in range(m*2-basic_vars,m)]
    nonconstrain_ind = [x for x in range((m+n)*2)]

    for outer_iter in range(max_outer_iter):

        #print( '\n\nOuter loop iteration: {0}, optimal solution: {1}'.format(outer_iter + 1, xcurr))
        ans_list.append([float(xcurr[i]) for i in range(len(xcurr))])
        opt_sols.append(fx.subs(zip(xvars, xcurr)))

        #The gradient of the objective function, numeralization
        #The gradient of the constraint function,numeralization
        delta_f = np.array([df.subs(zip(xvars, xcurr)) for df in dfx])
        delta_h = np.array([[dh.subs(zip(xvars, xcurr)) for dh in dhx] for dhx in dhxs])	
        #J,C stored constrained and unconstrained gradients respectively, numerically
        J = np.array([dhx[constrain_ind] for dhx in delta_h])								
        C = np.array([dhx[nonconstrain_ind]] for dhx in delta_h])
        #same to J,C
        delta_f_bar = delta_f[constrain_ind]
        delta_f_cap = delta_f[nonconstrain_ind]
        #Find the inverse of constrained gradient matrix
        J_inv = np.linalg.inv(np.array(J, dtype=float))
        #dz = delta f - delta f * J^-1 * C
        delta_f_tilde = delta_f_cap - delta_f_bar.dot(J_inv.dot(C))
        
        #First judgment, there's no gradient and you just exit
        if abs(delta_f_tilde[0]) <= eps:
            break

        #Direction of search in current iteration, D is the gradient that we put together
        d_bar = - delta_f_tilde.T 									
        d_cap = - J_inv.dot(C.dot(d_bar))
        d = np.concatenate((d_bar, d_cap)).T
        d_list.append([float(d[i]) for i in range(len(d))])

        alpha = alpha_fast
        #alpha = alpha_norm

        while alpha > 0.000001:
        #count1 = 0
        #while count1 < 100:
            #count1 += 1
            #print( '\nAlpha value: {0}\n'.format(alpha))
            v = xcurr.T + alpha * d
            
            #Upper and lower limit conditions
            v[-1] = -0.01 if v[-1] < -0.01 else v[-1]
            v[-2] = -0.01 if v[-2] < -0.01 else v[-2]
            v[-1] = 0.01 if v[-1] > 0.01 else v[-1]
            v[-2] = 0.01 if v[-2] > 0.01 else v[-2]
            
            #print(v)
            v_bar = v[constrain_ind]
            v_cap = v[nonconstrain_ind]
            flag = False

            #Fast gradient descent
            for iters in range(max_iter):
                #print( 'Iteration: {0}, optimal solution obtained at x = {1}'.format(iters + 1, v))
                h = np.array([hx.subs(zip(xvars, v)) for hx in hxs])
                # Check if hx satisfies all constraints, it means our constraints are satisfied in the form of derivatives
                #print(h)
                if all([abs(h_i) < eps for h_i in h]):
                    #Better or not				
                    if fx.subs(zip(xvars, xcurr)) <= fx.subs(zip(xvars, v)):
                        alpha = alpha * speed_fast
                        break
                    else:
                        xcurr = v 						
                        flag = True
                        break

                delta_h_v = np.array([[dh.subs(zip(xvars, v)) for dh in dhx] for dhx in dhxs])
                J_inv_v = np.linalg.inv(np.array([dhx[nonconstrain_ind] for dhx in delta_h_v], dtype=float))
                v_next_cap = v_cap - J_inv_v.dot(h)   
                print(v_next_cap)
                #v_next_cap[-1] = -0.01 if v_next_cap[-1] < -0.01 else v_next_cap[-1]
                #v_next_cap[-2] = -0.01 if v_next_cap[-2] < -0.01 else v_next_cap[-2]
                #v_next_cap[-1] = 0.01 if v_next_cap[-1] > 0.01 else v_next_cap[-1]
                #v_next_cap[-2] = 0.01 if v_next_cap[-2] > 0.01 else v_next_cap[-2]
                if abs(np.linalg.norm(np.array(v_cap - v_next_cap, dtype=float), 1)) > eps:
                    v_cap = v_next_cap
                    v = np.concatenate((v_bar, v_cap))
                else:
                    v_cap = v_next_cap
                    v = np.concatenate((v_bar, v_cap))
                    h = np.array([hx.subs(zip(xvars, v)) for hx in hxs])
                    if all([abs(h_i) < eps for h_i in h]):
                        #Search for lower values of alpha, in steepest descent method
                        if fx.subs(zip(xvars, xcurr)) <= fx.subs(zip(xvars, v)):
                            alpha = alpha * speed_fast				
                            break
                        else:
                            xcurr = v
                            flag = True
                            break
                    else:
                        alpha = alpha * speed_fast
                        break
            if flag == True:
                break

    print( '\n\nFinal solution obtained is: {0}'.format(xcurr))
    #print( 'Value of the function at this point: {0}\n'.format(fx.subs(zip(xvars, xcurr))))

    #Plot the solutions obtained after every iteration
    plt.plot(opt_sols, 'ro')							
    plt.show()
    #print(d_list)
    #print(ans_list)


if __name__ == '__main__':
    mat=np.array([[211.5, 233.5, 400.5],
       [113.4, 1134.3, 2333.3],
       [100.6, 421.7, 7000.7],
       [350.5, 300.5, 533.5]])
    SRG(mat)
