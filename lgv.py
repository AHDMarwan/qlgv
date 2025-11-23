import torch
import math

def gradient(params, node, x=None):
    """Computes the gradient w.r.t. params (L, W, G) using the parameter-shift rule (pi/2 shift)."""
    
    # Handle 2D input (L, W) by treating it as (L, W, G=1)
    params_3d = params.unsqueeze(2) if params.dim() == 2 else params
        
    if params_3d.dim() != 3:
        raise ValueError(f"Parameters tensor must be 2D (L, W) or 3D (L, W, G), got shape {params.shape}")
        
    L, W, G = params_3d.shape
    grads_3d = torch.zeros_like(params_3d)
    
    # --- Check for scalar output ---
    try:
        is_scalar = not torch.is_tensor(node(params, x=x)) or node(params, x=x).dim() == 0
    except:
        is_scalar = True 
        
    if not is_scalar:
        raise ValueError("Gradient calculation (Parameter-Shift) is only valid for scalar outputs (expectation values).")

    # --- Parameter-Shift Loop (L, W, G) ---
    for l in range(L):
        for w in range(W):
            for g in range(G): 
                p_plus = params_3d.clone()
                p_minus = params_3d.clone()
                
                # Apply shifts to the specific (l, w, g) index
                p_plus[l, w, g] += math.pi/2
                p_minus[l, w, g] -= math.pi/2
                
                # Squeeze back to original 2D shape for the node call if G=1
                p_plus_in = p_plus.squeeze(2) if G == 1 else p_plus
                p_minus_in = p_minus.squeeze(2) if G == 1 else p_minus
                
                # Compute QNode outputs
                outp = node(p_plus_in, x=x)
                outm = node(p_minus_in, x=x)
                
                # Apply Parameter-Shift formula
                grads_3d[l, w, g] = 0.5 * (outp - outm)
            
    # Squeeze the output gradient back to 2D if the input was 2D
    return grads_3d.squeeze(2) if params.dim() == 2 else grads_3d

def compute_lgv(node, sampler, n_samples=5, data=None):
    """Computes the Local Gradient Variance (LGV) over sampled parameters."""
    grads=[]
    for _ in range(n_samples):
        p = sampler() # Sampler returns L x W x G
        x = None 
        if data is not None and len(data) > 0:
            x = data[torch.randint(0,len(data),(1,)).item()]
            
        grads.append(gradient(p,node,x))
        
    # Stack flattened gradients and compute variance
    G_flat = torch.stack([g.reshape(-1) for g in grads])
    # The variance of the gradient components (LGV)
    return torch.var(G_flat, dim=0, unbiased=True).reshape(grads[0].shape)

def compute_shot_noise(node, params, repeats=2, data=None):
    """Estimates the variance due to finite measurement shots (shot noise) at a fixed parameter point."""
    grads=[]
    for _ in range(repeats):
        x = None 
        if data is not None and len(data) > 0:
            x = data[torch.randint(0,len(data),(1,)).item()]
            
        grads.append(gradient(params,node,x).reshape(-1))
        
    G_flat = torch.stack(grads)
    return torch.var(G_flat, dim=0, unbiased=True).reshape(params.shape)