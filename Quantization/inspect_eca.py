import sys
import torch
import torch.nn as nn
sys.path.append('/workspace/LS-YOLO')

def inspect():
    print("=== CHECK DPU COMPATIBILITY (ECA & OTHER LAYERS) ===")
    m = torch.load('/workspace/best_qat.pt', map_location='cpu')
    m = m.get('model', m).float()
    
    unsupported_ops = []
    eca_found = False

    for name, module in m.named_modules():
        # Check ECA
        if module.__class__.__name__ == 'ECA':
            if not eca_found:
                print("\n[INFO] --- ECA LAYER STRUCTURE ---")
                print(f"Name: {name}")
                print(module)
                eca_found = True
            
        # Check functions not well-supported by DPU
        if isinstance(module, nn.Conv1d):
            unsupported_ops.append((name, 'Conv1d (Recommend using Conv2d instead)'))
        elif isinstance(module, nn.Hardswish):
            unsupported_ops.append((name, 'Hardswish (DPU INT8 not well-supported, recommend ReLU/LeakyReLU)'))
        elif isinstance(module, nn.Hardsigmoid):
            unsupported_ops.append((name, 'Hardsigmoid (Should use standard Sigmoid)'))

    print("\n[INFO] --- LAYERS AT RISK OF BEING PUSHED TO CPU (SUBGRAPH SPLIT) ---")
    if unsupported_ops:
        for name, reason in unsupported_ops:
            print(f" - {name}: {reason}")
        print("\n[RECOMMENDATION]:")
        print("  1. In ECA layer, replace Conv1d with Conv2d(kernel_size=(1, k))")
        print("  2. If Hardswish is present, replace with ReLU or LeakyReLU.")
    else:
        print("Excellent! No Conv1d or layers that easily break Subgraph detected.")

if __name__ == "__main__":
    inspect()
