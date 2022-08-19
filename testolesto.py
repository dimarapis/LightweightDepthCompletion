import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from models.sparse_guided_depth import DecnetDepthRefinement

test_data_pred = torch.randint(0, 256, (1, 1, 352, 608)).to(device)
test_data_sparse = torch.randint(0, 256, (1, 1, 352, 608)).to(device)
test_data_pred = test_data_pred.to(torch.float32)
test_data_sparse = test_data_sparse.to(torch.float32)


refinement_model = DecnetDepthRefinement()
refinement_model.to(device)
refinement_model.eval()

new_pred = refinement_model(test_data_pred,test_data_sparse)

print(new_pred.shape)