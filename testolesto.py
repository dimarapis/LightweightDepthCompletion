import torch
import numpy as np
from features.decnet_sanity import inverse_depth_norm
from models.sparse_guided_depth import DecnetDepthRefinement ,RgbGuideDepth, DecnetSparseIncorporated


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_shapes():
    test_data_pred = torch.randint(0, 256, (1, 1, 352, 608)).to(device)
    test_data_sparse = torch.randint(0, 256, (1, 1, 352, 608)).to(device)
    test_data_pred = test_data_pred.to(torch.float32)
    test_data_sparse = test_data_sparse.to(torch.float32)

    model = RgbGuideDepth(True)
    #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location=device))
    model.load_state_dict(torch.load('./weights/DecnetModule_19.pth', map_location=device))
    model.to(device)

    print(model.feature_extractor)

    refinement_model = DecnetDepthRefinement()
    refinement_model.to(device)
    refinement_model.eval()

    new_pred = refinement_model(test_data_pred,test_data_sparse)

    print(new_pred.shape)



def gpu_timings(models):
    
    decnet_model = DecnetSparseIncorporated()
    decnet_model.to(device)
    decnet_model.eval()
    
    model = RgbGuideDepth(True)
    #model.load_state_dict(torch.load('./weights/nn_final_base.pth', map_location=device))
    model.to(device)
    model.eval()
    
    #encodepart = model.feature_extractor()


    refinement_model = DecnetDepthRefinement()
    #refinement_model.load_state_dict(torch.load('./weights/nn_final_ref.pth', map_location=device))
    refinement_model.to(device)
    refinement_model.eval()
    
    for modelo in models:
        print("Calculating inference for models...")
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 100
        timings = np.zeros((repetitions, 1))

        # GPU warm-up
        for _ in range(20):
            test_data_rgb = torch.randint(0, 256, (1, 3, 352, 608)).to(device)
            test_data_sparse = torch.randint(0, 256, (1, 1, 352, 608)).to(device)
            test_data_rgb = test_data_rgb.to(torch.float32)
            test_data_sparse = test_data_sparse.to(torch.float32)
            if modelo == 'DecnetSparseIncorporated':
                pred = decnet_model(test_data_rgb,test_data_sparse)
            
            
            if modelo == 'Encoder':
                    
                y_eighth = model.feature_extractor(test_data_rgb)
                
                
            if modelo == 'Basemodel' or modelo == 'Refinement':
                rgb_half, y_half, sparse_half, y, inv_pred = model(test_data_rgb,test_data_sparse)
                    
                pred = inverse_depth_norm(80.0,inv_pred)
                
            if modelo == 'Refinement':
                    
                #refined_pred = refinement_model(rgb_half, test_data_rgb, y_half, y, sparse_half, test_data_sparse, pred)
                refined_pred = refinement_model(pred,test_data_sparse)

        # Measure performance 
        with torch.no_grad():
            for rep in range(repetitions):
                
                test_data_rgb = torch.randint(0, 256, (1, 3, 352, 608)).to(device)
                test_data_sparse = torch.randint(0, 256, (1, 1, 352, 608)).to(device)
                test_data_rgb = test_data_rgb.to(torch.float32)
                test_data_sparse = test_data_sparse.to(torch.float32)
                
                
                
                starter.record()
                
                if modelo == 'DecnetSparseIncorporated':
                    pred = decnet_model(test_data_rgb,test_data_sparse)

                    
                
                if modelo == 'Encoder':
                    
                    y_eighth = model.feature_extractor(test_data_rgb)
                
                
                if modelo == 'Basemodel' or modelo == 'Refinement':
                    rgb_half, y_half, sparse_half, y, inv_pred = model(test_data_rgb,test_data_sparse)
                    
                    pred = inverse_depth_norm(80.0,inv_pred)
                
                if modelo == 'Refinement':
                    
                    #refined_pred = refinement_model(rgb_half, test_data_rgb, y_half, y, sparse_half, test_data_sparse, pred)
                    refined_pred = refinement_model(pred,test_data_sparse)

                ender.record()

                # Wait for GPU to sync
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        # Calculate mean and std
        mean_time = np.sum(timings) / repetitions
        std_time = np.std(timings)
        #print(f'{modelo} model timings calculation...\n')
        
        print(f'{modelo} model timings calculation\nMean time to process {repetitions} frames: {mean_time}, with std_deviation of: {std_time}')

gpu_timings(['DecnetSparseIncorporated','Encoder','Basemodel','Refinement'])

#print_shapes()