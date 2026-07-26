import sys
import os
print(f"path : {os.getcwd()}")
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from torchinfo import summary
from tabulate import tabulate
import gc
from datetime import datetime
from pathlib import Path
from src.common.resources import log_process_memory


class Train:
    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, config, gpu_id, checkpoint_path=None):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.gpu_id = gpu_id
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.save_every = config.get('save_every', 1)
        self.patience = config.get('patience', 5)
        self.best_loss = float('inf')
        self.start_epoch = 0
        self.counter = 0

        # Load checkpoint if provided
        if checkpoint_path:
            self.initialize_model()
            self.start_epoch, self.best_loss, self.optimizer = self.load_checkpoint(checkpoint_path)
            self.ddp_model = DDP(self.model, find_unused_parameters=True, device_ids=[gpu_id])
            #self.ddp_model = DDP(self.model, device_ids=[gpu_id])
        else :     
            self.initialize_model()
            self.ddp_model = DDP(self.model, find_unused_parameters=True, device_ids=[gpu_id])
            #self.ddp_model = DDP(self.model, device_ids=[gpu_id])
            self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=config.get('learning_rate', 1e-3), weight_decay=config.get('weight_decay', 1e-5))
            
    def initialize_model(self):
        """Method to initialize model in each subclass"""
        pass
    
    
 
    def train(self, mlflow=None):
        """
        Shared training logic. Subclasses will define their own `process_batch`.
        """
        folder_run = ""
        if dist.get_rank() == 0 :
            folder_run = f"{self.config['data_save']}/mlruns/{self.config['experiment_id']}/{self.config['run_id']}/artifacts"
            print('Start Training')
            print(folder_run)
        n_epochs = self.config['n_epochs']
        
        # Nombre total de paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Nombre total de paramètres : {total_params}")

        # Nombre de paramètres entraînables
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Nombre de paramètres entraînables : {trainable_params}")
        # Dictionnaires pour stocker les normes moyennes des poids et gradients pour chaque couche
        weight_norms_dict = {name: [] for name, param in self.ddp_model.named_parameters() if 'weight' in name}
        bias_norms_dict = {name: [] for name, param in self.ddp_model.named_parameters() if 'bias' in name}
        grad_norms_dict = {name: [] for name, param in self.ddp_model.named_parameters()}
        epochs_list = []
        for epoch in range(self.start_epoch, n_epochs):
            actual_epoch = epoch + 1
            # Dictionnaires temporaires pour les normes d'une époque
            epoch_weight_norms = {name: 0 for name in weight_norms_dict.keys()}
            epoch_bias_norms = {name: 0 for name in bias_norms_dict.keys()}
            epoch_grad_norms = {name: 0 for name in grad_norms_dict.keys()}
            batch_count = 1
            
            train_epoch_loss, train_epoch_unormalized, train_epoch_mse, train_epoch_pearson, train_epoch_criterion, train_epoch_phonemes = 0, 0, 0, 0, 0, 0
            validation_epoch_loss, validation_epoch_unormalized, validation_epoch_mse, validation_epoch_pearson, validation_epoch_criterion, validation_epoch_phonemes = 0, 0, 0, 0, 0, 0
            for batch in tqdm(self.train_dataloader):
                state = 'train'
                batch_loss, batch_unormalized, batch_mse, batch_pearson, batch_criterion, batch_phonemes = self.process_batch(batch)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                
                train_epoch_loss += batch_loss.item()
                train_epoch_unormalized += batch_unormalized
                train_epoch_mse += batch_mse
                train_epoch_pearson += batch_pearson
                train_epoch_criterion += batch_criterion
                train_epoch_phonemes += batch_phonemes
                # Log des normes des poids, biais et gradients
                epoch_weight_norms, epoch_bias_norms, epoch_grad_norms, batch_count = self._log_norms(
                epoch_weight_norms, epoch_bias_norms, epoch_grad_norms, batch_count)
            torch.cuda.empty_cache()
            #self.log_memory(stage=f"Fin epoch {epoch}")
            # Average metrics over the batches
            train_epoch_loss /= len(self.train_dataloader)
            train_epoch_unormalized /= len(self.train_dataloader)
            train_epoch_mse /= len(self.train_dataloader)
            train_epoch_pearson /= len(self.train_dataloader)
            train_epoch_criterion /= len(self.train_dataloader)
            train_epoch_phonemes /= len(self.train_dataloader)
            weight_norms_dict, bias_norms_dict, grad_norms_dict = self.mean_norms(
            epoch_weight_norms, epoch_bias_norms, epoch_grad_norms, weight_norms_dict, bias_norms_dict, grad_norms_dict, batch_count)
            self._log_metrics(mlflow, state, epoch, train_epoch_loss, train_epoch_unormalized, train_epoch_mse, train_epoch_pearson, train_epoch_criterion, train_epoch_phonemes)
            self._print_results(state, actual_epoch, n_epochs, train_epoch_loss, train_epoch_unormalized, train_epoch_mse, train_epoch_pearson, train_epoch_criterion, train_epoch_phonemes)
            # Enregistrer les époques pour le graphique
            epochs_list.append(epoch + 1)
            # Perform validation and checkpointing
            if epoch % self.save_every == 0:
                state = 'validation'
                self.ddp_model.eval()
                with torch.no_grad():
                    for batch in self.validation_dataloader:
                        batch_loss, batch_unormalized, batch_mse, batch_pearson, batch_criterion, batch_phonemes = self.process_batch(batch)
                        
                    
                        validation_epoch_loss += batch_loss.item()
                        validation_epoch_unormalized += batch_unormalized
                        validation_epoch_mse += batch_mse
                        validation_epoch_pearson += batch_pearson
                        validation_epoch_criterion += batch_criterion
                        validation_epoch_phonemes += batch_phonemes
                    torch.cuda.empty_cache()
                    #self.log_memory(stage=f"Fin validation {epoch}")
                # Average metrics over the batches
                validation_epoch_loss /= len(self.validation_dataloader)
                validation_epoch_unormalized /= len(self.validation_dataloader)
                validation_epoch_mse /= len(self.validation_dataloader)
                validation_epoch_pearson /= len(self.validation_dataloader)
                validation_epoch_criterion /= len(self.validation_dataloader)
                validation_epoch_phonemes /= len(self.validation_dataloader)
                
                self._log_metrics(mlflow, state, epoch, validation_epoch_loss, validation_epoch_unormalized, validation_epoch_mse, validation_epoch_pearson, validation_epoch_criterion, validation_epoch_phonemes)
                self._print_results(state, actual_epoch, n_epochs, validation_epoch_loss, validation_epoch_unormalized, validation_epoch_mse, validation_epoch_pearson, validation_epoch_criterion, validation_epoch_phonemes)
                print('Begin save')
                self.save_checkpoint(folder_run, validation_epoch_loss, epoch)
                print('Begin Stoppig')
                early_stop = self.early_stopping(validation_epoch_loss, epoch)
                print('End Stoppig')
                stop_training_tensor = torch.tensor([0], dtype=torch.int).cuda()
                print('End stop_training_tensor')
                if dist.get_rank() == 0 and early_stop :
                    print(f"Loss function doesn't increase.Training stop at epoch: {actual_epoch}")
                    stop_training_tensor[0] = 1
                dist.broadcast(stop_training_tensor, src=0)
                if stop_training_tensor.item() == 1:
                    break
        # print(f"End training")
        # self.log_memory(stage=f"End training")
        del self.train_dataloader
        del self.validation_dataloader
        #report_memory()
        # print(self.train_dataloader)
        # print(f"\n Aprés le kill : \n")
        # del self.train_dataloader
        # del self.validation_dataloader
        # n = gc.collect()
        # print(f"{n} objets ont été collectés.")
        # self.log_memory(stage=f"After kill")
        # report_memory()

        if dist.get_rank() == 0:
            self.save_model(folder_run)
            #self.log_memory(stage=f"Model saved")
            print('model saved')
            self.best_ddp_model.eval()
            #self.log_memory(stage=f"Model eval()")
            for batch in self.test_dataloader:
                result_test, y_array, y_pred_array = self.test_batch(batch, folder_run)
                self._print_test(result_test, folder_run)
                self._save_files(result_test, folder_run)
                self._print_test_mri(result_test, folder_run)
            self._log_metrics_test(result_test)
            #self.log_memory(stage=f"test finished")
            for name in weight_norms_dict.keys():
                self.plot_and_save(name, epochs_list, weight_norms_dict[name], bias_norms_dict.get(name, None), grad_norms_dict[name], folder_run)
            #self.log_memory(stage=f"Plot finished")
            frames = result_test['all_frames']
            phonemes = result_test['all_phonemes']
            rmse = result_test['rmse_contours_mean_per_image_all_articulators']
            rmse_mm = result_test['rmse_contours_mean_all_images_all_articulators'] * 1.62
            print("RMSE mean per image (mm) : ", rmse_mm)
            self._export_training_result(folder_run, result_test, rmse_mm)
            if not self.config.get('skip_outlier_detection', False):
                raise RuntimeError("Outlier detection is not included in the minimal ST-5 branch.")
            if not self.config.get('skip_test_plots', False):
                raise RuntimeError("Test plotting is not included in the minimal ST-5 branch.")
            mlflow.end_run()

    def _unsupported_batch(self, batch):
        """Abstract hook shared by the two subclass batch entry points."""
        raise NotImplementedError

    process_batch = _unsupported_batch
    test_batch = _unsupported_batch

    def _log_metrics(self, mlflow, state, epoch, loss, unormalized, mse, pearson, criterion, phonemes):
        """
        Log training metrics using MLFlow.
        """
        if dist.get_rank() == 0:
            mlflow.log_metric(f"{state}_loss", loss, step=epoch)
            mlflow.log_metric(f"{state}_unormalized", unormalized, step=epoch)
            mlflow.log_metric(f"{state}_mse", mse, step=epoch)
            mlflow.log_metric(f"{state}_pearson", pearson, step=epoch)
            mlflow.log_metric(f"{state}_criterion", criterion, step=epoch)
            mlflow.log_metric(f"{state}_phonemes", phonemes, step=epoch)
    
    def _log_metrics_test(self, result_test): 
        mlflow.log_metric(f"test_rmse_contours_mean_all_sequences_all_articulators", result_test['rmse_contours_mean_all_sequences_all_articulators'])
        # mlflow.log_metric(f"test_rmse", result_test['rmse_mean'])
        # mlflow.log_metric(f"test_loss_rmse", result_test['mse_mean'])
        # mlflow.log_metric(f"test_loss_mse", result_test['pearson_mean'])
        # mlflow.log_metric(f"test_loss", result_test['loss_mean'])      
        mlflow.log_metric(f"test_rmse_contours_mean_all_images_all_articulators", result_test['rmse_contours_mean_all_images_all_articulators'])
        mlflow.log_metric(f"test_rmse_contours_median_all_images_all_articulators", result_test['rmse_contours_median_all_images_all_articulators'])
        mlflow.log_metric(f"test_rmse_contours_std_all_images_all_articulators", result_test['rmse_contours_std_all_images_all_articulators'])
        mlflow.log_metric(f"test_rmse_contours_mean_all_points_all_articulators", result_test['rmse_contours_mean_all_points_all_articulators'])
        mlflow.log_metric(f"test_rmse_contours_median_all_points_all_articulators", result_test['rmse_contours_median_all_points_all_articulators'])
        mlflow.log_metric(f"test_rmse_contours_std_all_points_all_articulators", result_test['rmse_contours_std_all_points_all_articulators'])


        for index, articulator in enumerate(self.config['classes']): 
            # Sequences
            mlflow.log_metric(f"rmse_contours_mean_all_sequences_{articulator}", result_test['rmse_contours_mean_all_sequences_per_articulator'][index])
            mlflow.log_metric(f"rmse_contours_median_all_sequences_{articulator}", result_test['rmse_contours_median_all_sequences_per_articulator'][index])
            mlflow.log_metric(f"rmse_contours_std_all_sequences_{articulator}", result_test['rmse_contours_std_all_sequences_per_articulator'][index])
            
            # Images 
            mlflow.log_metric(f"rmse_contours_mean_all_images_{articulator}", result_test['rmse_contours_mean_all_images_per_articulator'][index])
            mlflow.log_metric(f"rmse_contours_median_all_images_{articulator}", result_test['rmse_contours_median_all_images_per_articulator'][index])
            mlflow.log_metric(f"rmse_contours_std_all_image_{articulator}", result_test['rmse_contours_std_all_image_per_articulator'][index])

            # Point
            mlflow.log_metric(f"rmse_contours_mean_all_point_{articulator}", result_test['rmse_contours_mean_all_point_per_articulator'][index])
            mlflow.log_metric(f"rmse_contours_median_all_point_{articulator}", result_test['rmse_contours_median_all_point_per_articulator'][index])
            mlflow.log_metric(f"rmse_contours_std_all_point_{articulator}", result_test['rmse_contours_std_all_point_per_articulator'][index])

    
    def _print_results(self, state, actual_epoch, n_epochs, loss, unormalized, mse, pearson, criterion, phonemes) :
        if dist.get_rank() == 0:
            print(f"Global Metrics {state} for epoch {actual_epoch}/{n_epochs} :")
            global_metrics = [
                ["Loss values (mse or mse_cross)", loss],
                ["Loss values Unormalized", unormalized],
                ["RMSE metrics", mse],
                ["Pearson metrics", pearson],
                ["Criterion pearson metrics", criterion],
                ["cross entropy (phonemes) metrics", phonemes]
            ]
            print(tabulate(global_metrics, tablefmt="grid"))
    
    def _print_test(self, result_test, folder_run):
        if dist.get_rank() == 0:
            print("Global Loss Metrics Test :\n")

            headers = ["Articulator", "RMSE (mm)", "Median (mm)", "Std (mm)"]
            data_seq = []
            for index, articulator in enumerate(self.config['classes']):
                data_seq.append([
                    articulator,
                    result_test['rmse_contours_mean_all_sequences_per_articulator'][index] * 1.62,
                    result_test['rmse_contours_median_all_sequences_per_articulator'][index] * 1.62,
                    result_test['rmse_contours_std_all_sequences_per_articulator'][index] * 1.62
                ])
            data_seq.append([
                "Overall RMSE (mm)",
                result_test['rmse_contours_mean_all_sequences_all_articulators'] * 1.62,
                result_test['rmse_contours_median_all_sequences_all_articulators']* 1.62,
                result_test['rmse_contours_std_all_sequences_all_articulators'] *1.62
            ])
            print("Calculate with Sequences:")
            print(tabulate(data_seq, headers=headers, tablefmt="grid"))

            print("\nCalculate with Images:")
            data_image = []
            for index, articulator in enumerate(self.config['classes']):
                data_image.append([
                    articulator,
                    result_test['rmse_contours_mean_all_images_per_articulator'][index] * 1.62,
                    result_test['rmse_contours_median_all_images_per_articulator'][index] * 1.62,
                    result_test['rmse_contours_std_all_image_per_articulator'][index] * 1.62
                ])
            data_image.append([
                'Overall RMSE (mm per image)',
                result_test['rmse_contours_mean_all_images_all_articulators'] * 1.62,
                result_test['rmse_contours_median_all_images_all_articulators'] * 1.62,
                result_test['rmse_contours_std_all_images_all_articulators'] * 1.62
            ])
            print(tabulate(data_image, headers=headers, tablefmt="grid"))


            print("\nCalculate with Points:")
            data_point = []
            for index, articulator in enumerate(self.config['classes']):
                data_point.append([
                    articulator,
                    result_test['rmse_contours_mean_all_point_per_articulator'][index] * 1.62,
                    result_test['rmse_contours_median_all_point_per_articulator'][index] * 1.62,
                    result_test['rmse_contours_std_all_point_per_articulator'][index] * 1.62
                ])
            data_point.append([
                "Overall RMSE (mm per point)",
                result_test['rmse_contours_mean_all_points_all_articulators'] * 1.62,
                result_test['rmse_contours_median_all_points_all_articulators'] * 1.62,
                result_test['rmse_contours_std_all_points_all_articulators'] * 1.62
            ])
            print(tabulate(data_point, headers=headers, tablefmt="grid"))

            print("\nTraining finished successfully\n" + "-" * 30)
            


                    
    def _save_files(self, result_test, folder_run):
        # RMSE mean per point
        rmse_values = result_test['rmse_contours_per_point_per_articulator']  # Shape (nbr_articulators, 100)
        output_file = f"{folder_run}/rmse_per_point.txt"
        # Writing to the file
        with open(output_file, "w") as file:
            # Writing the header with dynamic articulator names
            header = "point " + " | ".join([f"{self.config['classes'][i]}" for i in range(rmse_values.shape[0])]) + "\n"
            file.write(header)
            
            # Writing the RMSE values for each point
            for i in range(rmse_values.shape[1]):
                row = f"{i+1} | " + " | ".join([str(rmse_values[j, i]) for j in range(rmse_values.shape[0])]) + "\n"
                file.write(row)
                
        
        # RMSE mean per image
        mean_values = result_test['rmse_contours_mean_per_image_all_articulators']
        all_phonemes = result_test['all_phonemes']
        all_frames = result_test['all_frames'] 
        output_file = f"{folder_run}/rmse_per_image.txt"
        
        has_frames = all_frames is not None
        if has_frames:
            with open(output_file, "w") as file:
                file.write("Index| Number image | mean | mean(mm) | Phonemes\n")  # Header
                for i, (frame, mean, phoneme) in enumerate(zip(all_frames, mean_values, all_phonemes)):
                    file.write(f"{i}| {int(frame[0])}_S{int(frame[1])}_{frame[2]} | {mean:.6f} | {mean*1.62:.6f} | {phoneme}\n")
        else:
            print("Skipping per-image RMSE files because frame ids are unavailable.", flush=True)

        
        #rmse mean per image per articulator
        if has_frames:
            for index, articulator in enumerate(self.config['classes']):
                output_file_articulator = f"{folder_run}/rmse_per_image_{articulator}.txt"
                means_per_image = result_test['rmse_contours_per_image_per_articulator'][index]
                with open(output_file_articulator, "w") as file:
                    file.write("Number image | mean | mean(mm) | Phonemes\n")  # Header
                    for i, (frame, mean, phoneme) in enumerate(zip(all_frames, means_per_image, all_phonemes)):
                        file.write(f"{int(frame[0])}_S{int(frame[1])}_{frame[2]} | {mean:.6f} | {mean*1.62:.6f} | {phoneme}\n")

        for index, articulator in enumerate(self.config['classes']):
                    with open(f"{folder_run}/rmse_per_phoneme_{articulator}.txt", "w") as file:
                        file.write("Phoneme\tMean(mm)\tMedian(mm)\tStd(mm)\n")
                        for phoneme, stats in result_test['rmse_contours_per_phoneme_per_articulator'][index].items():
                            file.write(f"{phoneme}\t{stats['mean']*1.62:.4f}\t{stats['median']*1.62:.4f}\t{stats['std']*1.62:.4f}\n")
                        file.write("\n")
                        
        with open(f"{folder_run}/rmse_per_phoneme_all.txt", "w") as file:
            file.write("Phoneme\tMean(mm)\tMedian(mm)\tStd(mm)\n")
            for phoneme, stats in result_test['rmse_contours_phoneme_all_articulators'].items():
                file.write(f"{phoneme}\t{stats['mean']*1.62:.4f}\t{stats['median']*1.62:.4f}\t{stats['std']*1.62:.4f}\n")
            file.write("\n") 


    def _print_test_mri(self, result_test, folder_run):
        mlflow.log_metric(f"rmse_mris_mean_all_images", result_test['rmse_mris_mean_all_images'])
        mlflow.log_metric(f"rmse_mris_median_all_images", result_test['rmse_mris_median_all_images'])
        mlflow.log_metric(f"rmse_mris_std_all_images", result_test['rmse_mris_std_all_images'])
        mlflow.log_metric(f"ssim_scores_all_images", result_test['ssim_scores_all_images'])
        headers = ["ssim scores","RMSE (mm)", "Median (mm)", "Std (mm)"]
        data_seq = []
        data_seq.append([
            result_test['ssim_scores_all_images'],
            result_test['rmse_mris_mean_all_images'] * 1.62,
            result_test['rmse_mris_median_all_images']* 1.62,
            result_test['rmse_mris_std_all_images'] *1.62
        ])
        print("Metrics of MRI images:")
        print(tabulate(data_seq, headers=headers, tablefmt="grid"))
        
        output_file_articulator = f"{folder_run}/MRI_metrics.txt"
        rmse = result_test['rmse_mris_per_image']
        ssim = result_test['ssim_scores_per_image']
        all_phonemes = result_test['all_phonemes']
        all_frames = result_test['all_frames'] 
        
        
        with open(output_file_articulator, "w") as file:
                file.write("Number image | SSIM | RMSE(mm) | Phonemes\n")  # Header
                if all_frames is not None:
                    for i, (frame, one_ssim, one_rmse, phoneme) in enumerate(zip(all_frames, ssim, rmse, all_phonemes)):
                        file.write(f"{int(frame[0])}_S{int(frame[1])}_{frame[2]} | {one_ssim:.6f} | {one_rmse*1.62:.6f} | {phoneme}\n")

    
    def _metric_float(self, result_test, key, scale=1.0):
        value = result_test.get(key)
        if value is None:
            return None
        return float(value) * scale

    def _safe_result_name(self, value):
        return str(value).replace("/", "_").replace(" ", "_")

    def _link_or_copy(self, source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if not source_path.exists():
            return None
        if destination_path.exists() or destination_path.is_symlink():
            destination_path.unlink()
        try:
            destination_path.symlink_to(source_path.resolve())
        except OSError:
            import shutil
            shutil.copy2(source_path, destination_path)
        return str(destination_path)

    def _export_training_result(self, folder_run, result_test, rmse_mm):
        result_root = Path(self.config.get("results_dir", Path(self.config["data_save"]) / "results"))
        run_group = self._safe_result_name(self.config.get("folder_save", self.config.get("experiment_name", "training_run")))
        run_name = self._safe_result_name(self.config.get("run_name", self.config.get("run_id", "run")))
        output_dir = result_root / run_group / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_dir = Path(folder_run)
        linked_files = {}
        for filename in (
            "best_model.pth",
            "last_model.pth",
            "final_model.pth",
            "final_dict.pth",
            "config.yaml",
            "datasets.txt",
            "rmse_per_image.txt",
            "rmse_per_point.txt",
            "MRI_metrics.txt",
        ):
            linked = self._link_or_copy(artifact_dir / filename, output_dir / filename)
            if linked is not None:
                linked_files[filename] = linked

        metrics = {
            "sequence_overall_rmse_mm": self._metric_float(result_test, "rmse_contours_mean_all_sequences_all_articulators", 1.62),
            "image_overall_rmse_mm": self._metric_float(result_test, "rmse_contours_mean_all_images_all_articulators", 1.62),
            "point_overall_rmse_mm": self._metric_float(result_test, "rmse_contours_mean_all_points_all_articulators", 1.62),
            "mri_image_overall_rmse_mm": self._metric_float(result_test, "rmse_mris_mean_all_images", 1.62),
            "mri_ssim_all_images": self._metric_float(result_test, "ssim_scores_all_images", 1.0),
            "rmse_mean_per_image_mm_printed": float(rmse_mm),
        }
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_name": self.config.get("experiment_name"),
            "run_name": self.config.get("run_name"),
            "folder_save": self.config.get("folder_save"),
            "experiment_id": self.config.get("experiment_id"),
            "run_id": self.config.get("run_id"),
            "artifact_dir": str(artifact_dir),
            "results_dir": str(output_dir),
            "best_model": linked_files.get("best_model.pth"),
            "metrics": metrics,
            "linked_artifacts": linked_files,
            "post_train_inference": {
                "status": "not_run_by_train_loop",
                "note": (
                    "Use scripts/inversion_si.py infer session with this best_model "
                    "path; inference outputs can be placed under this results_dir."
                ),
            },
        }
        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with (output_dir / "training_summary.md").open("w", encoding="utf-8") as handle:
            handle.write("# Training Summary\n\n")
            handle.write(f"- results_dir: `{output_dir}`\n")
            handle.write(f"- artifact_dir: `{artifact_dir}`\n")
            handle.write(f"- best_model: `{summary['best_model']}`\n")
            for key, value in metrics.items():
                handle.write(f"- {key}: `{value}`\n")
        print(f"Training result summary exported to: {output_dir}", flush=True)


    def log_memory(self, stage=""):
        log_process_memory(stage)



    def inference(self):
        raise RuntimeError("Inference plotting is not included in the minimal ST-5 branch.")
        
    def early_stopping(self, score, epoch):
        early_stop = False
        if dist.get_rank() == 0:
            if self.best_loss > score :
                print('Best_model changed at epoch : ', epoch)
                self.best_ddp_model = self.ddp_model
                self.best_model = self.model
                self.best_loss = score
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    early_stop = True
        return early_stop

    def save_model(self, path):
            torch.save(self.best_ddp_model, f'{path}/final_model.pth')  # Use model.module to access the underlying model in DDP
            torch.save(self.best_ddp_model.module.state_dict(), f'{path}/final_dict.pth')
            #self.log_memory(stage=f"function save")

    def save_checkpoint(self, path, loss, epoch):
            if dist.get_rank() == 0:
                checkpoint = {
                'model_state_dict': self.ddp_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': loss
                }
                torch.save(checkpoint, f'{path}/last_model.pth')
                if self.best_loss>loss:
                    print('Best_model saved at epoch : ', epoch)
                    checkpoint = {
                    'model_state_dict': self.ddp_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': loss
                    }
                    torch.save(checkpoint, f'{path}/best_model.pth')

                
    
    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.gpu_id}')
        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found")

        checkpoint = torch.load(checkpoint_path)
        
        # Remove the 'module.' prefix from the state dict keys if necessary
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')  # Strip the 'module.' prefix
            new_state_dict[new_key] = state_dict[key]

        # Load the modified state dict into the model
        self.model.load_state_dict(new_state_dict)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if key == 'step':
                        # step tensors are allowed to be on CPU in float32
                        state[key] = value.float().cpu()
                    else:
                        # Ensure all other tensors match the model's device and dtype
                        state[key] = value.to(self.gpu_id)
        # # Load the state dictionary into the model
        # model.load_state_dict(state_dict)
        return checkpoint['epoch'], checkpoint['best_loss'], optimizer
    
    def _load_autoencoder(self):
        model_path = self.config['autoencoderdir']
        return torch.load(model_path, map_location=torch.device(self.gpu_id))
    
    def _log_norms(self, epoch_weight_norms, epoch_bias_norms, epoch_grad_norms, batch_count):
        if dist.get_rank() == 0:
            for name, param in self.ddp_model.named_parameters():
                if param.requires_grad:
                    if 'weight' in name:
                        epoch_weight_norms[name] += param.data.norm().item()
                    if 'bias' in name:
                        epoch_bias_norms[name] += param.data.norm().item()
                    epoch_grad_norms[name] += param.grad.norm().item() if param.grad is not None else 0
            batch_count += 1
        return epoch_weight_norms, epoch_bias_norms, epoch_grad_norms, batch_count

    def mean_norms(self, epoch_weight_norms, epoch_bias_norms, epoch_grad_norms, weight_norms_dict, bias_norms_dict, grad_norms_dict, batch_count):
        for name in weight_norms_dict.keys():
            weight_norms_dict[name].append(epoch_weight_norms[name] / batch_count)
        for name in bias_norms_dict.keys():
            bias_norms_dict[name].append(epoch_bias_norms[name] / batch_count)
        for name in grad_norms_dict.keys():
            grad_norms_dict[name].append(epoch_grad_norms[name] / batch_count)
        return weight_norms_dict, bias_norms_dict, grad_norms_dict

    def plot_and_save(self, layer_name, epochs_list, weight_norms, bias_norms, grad_norms, output_dir):
        if dist.get_rank() == 0:
            plt.figure()
            plt.plot(epochs_list, weight_norms, label='Poids', color='blue')
            if bias_norms is not None:
                plt.plot(epochs_list, bias_norms, label='Biais', color='green')
            plt.plot(epochs_list, grad_norms, label='Gradients', color='red')
            plt.title(f"Normes des poids, biais et des gradients pour {layer_name}")
            plt.xlabel('Époque')
            plt.ylabel('Norme')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True)
            
            plot_path = f"{output_dir}/{layer_name}_norms.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

def sizeof(obj):
    """Calcule la taille approximative d'un objet."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    else:
        return sys.getsizeof(obj)

def pretty_size(bytes, suffix="B"):
    """Formate une taille en octets de manière lisible."""
    for unit in ['','K','M','G','T']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}{suffix}"
        bytes /= 1024
    return f"{bytes:.2f} P{suffix}"

def report_memory():
    print("\n--- Mémoire utilisée par les variables ---")
    total = 0
    for obj in gc.get_objects():
        try:
            size = sizeof(obj)
            if size > 1e5:  # Ignore les petits objets < 100 Ko
                print(f"{type(obj)}: {pretty_size(size)}")
                total += size
        except:
            pass
    print(f"\nTotal estimé: {pretty_size(total)}")
