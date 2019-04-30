import torch
import numpy as np
import osqp


def PSp(x):
    m,n=x.shape
    l=np.ones([m,1])
    t0 = (x-l/2)*np.sqrt(m)/(2*np.linalg.norm(x-l/2, 2))
    out = l/2 + t0

    return out

       
#udpdate z1,z2,v_admm
def admm_update1(layer,alpha): 
    N=layer.v.data.cpu().numpy().size
    layer.v_np=layer.v.data.cpu().numpy().reshape(N,1) 
    
    # update z1    
    # Create an OSQP object
    prob = osqp.OSQP()
    layer.q = -layer.v_np-layer.y1/layer.rho   #verbose=False

    prob.setup(layer.P, layer.q, layer.E, layer.l, layer.u, alpha=1.0, verbose=False)
    res = prob.solve()
    layer.z1=res.x
    layer.z1=layer.z1.reshape(N,1)

    
    # update z2   
    layer.z2=PSp(layer.v_np+layer.y2/layer.rho)

    
    # update grad_v   
    #grad_vadmm
    grad_vadmm=layer.y1+layer.y2+layer.rho*(2*layer.v_np-layer.z1-layer.z2) 

    
    #update grad_v
    grad_vnew=alpha*grad_vadmm[:,0] + (1-alpha)*(layer.v.grad.data.cpu().numpy().reshape(N,))
    v_shape=layer.v.grad.data.cpu().numpy().shape
    layer.v.grad.data=torch.from_numpy(grad_vnew.reshape(v_shape)).float().cuda() 



#update y1,y2,rho  
def admm_update2(layer,rho_flag):
    N=layer.v.data.cpu().numpy().size
    layer.v_np=layer.v.data.cpu().numpy().reshape(N,1) 
           
    # update y1,y2
    layer.y1 = layer.y1 + layer.rho*(layer.v_np-layer.z1)
    layer.y2 = layer.y2 + layer.rho*(layer.v_np-layer.z2)      
    
    # update rho
    if rho_flag is True:
        if layer.rho<layer.rho_maximum:
            layer.rho = layer.rho*layer.mu
            
                     
