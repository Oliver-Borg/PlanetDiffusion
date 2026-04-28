
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
import lpips
from pytorch_msssim import ssim, ms_ssim
import torch.nn.functional as F

class Metric:

    def reset(self):
        raise NotImplementedError

    def add(self, outputs, targets):
        # outputs and targets (in batches)
        raise NotImplementedError

    def total(self):
        # Return dictionary of metrics
        raise NotImplementedError

class TrainingMetrics(Metric):
    '''
    This metric class provides extra metrics for training.
    This includes mse, lpips, ssim, ms-ssim, and fid.
    # TODO: Fix fid and lpips
    '''

    def __init__(self, device='cpu') -> None:
        self.device = device
        # self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_model = None
        self.reset()
    
    def reset(self):
        self.output_tensors = []
        self.target_tensors = []

    def add(self, outputs, targets):
        self.output_tensors.append(outputs)
        self.target_tensors.append(targets)

    def total(self):
        running_mse = 0
        running_ssim = 0
        running_ms_ssim = 0
        running_lpips = 0

        for output, target in zip(self.output_tensors, self.target_tensors):
            assert output.shape == target.shape
            running_mse += F.mse_loss(output, target).item()
            running_ssim += ssim(output, target, data_range=1).item()
            running_ms_ssim += ms_ssim(output, target, data_range=1).item()
            if self.lpips_model is not None:
                assert output.shape[1] == 1 or output.shape[1] == 3
                if output.shape[1] == 1:
                    running_lpips += self.lpips_model(torch.cat([output]*3, dim=1), torch.cat([target]*3, dim=1)).mean()
                else:
                    running_lpips += self.lpips_model(output, target).item()
        count = len(self.output_tensors)
        metrics = {
            'mse': running_mse/count,
            'ssim': running_ssim/count,
            'ms_ssim': running_ms_ssim/count,
            # 'fid': calculate_fid_given_tensors(
            #     self.output_tensors,
            #     self.target_tensors,
            #     batch_size=2,
            #     device=self.device,
            #     dims=2048,
            # )
        }
        if self.lpips_model is not None:
            metrics['lpips'] = running_lpips/count
        return metrics
        


class ClassificationMetrics(Metric):

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []

    def add(self, outputs, targets):
        _, predictions = torch.max(outputs, dim=1)

        for prediction, target in zip(predictions, targets):
            self.all_predictions.append(prediction.item())
            self.all_targets.append(target.item())

        # TODO return something for live updates?

    def total(self):
        data = dict(
            y_true=self.all_targets,
            y_pred=self.all_predictions
        )

        accuracy = accuracy_score(**data)
        balanced_accuracy = balanced_accuracy_score(**data)
        prec, rec, f1, _ = precision_recall_fscore_support(
            **data, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        }
