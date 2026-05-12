import sys
sys.path.append("CMC_utils")

import os
import shutil
# import torch
import hydra
import logging
import pandas as pd
from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate
from CMC_utils import save_load, metrics
from CMC_utils.pipelines import supervised_learning_main, supervised_tabular_missing_main, multimodal_early_fusion_supervised_learning_main, multimodal_joint_fusion_supervised_learning_main, multimodal_late_fusion_supervised_learning_main
from tqdm import tqdm
log = logging.getLogger(__name__)


@hydra.main(version_base="v1.3", config_path="confs", config_name="config")
def main(cfg: DictConfig) -> None:
    clinical_relative_path = "MIMIC_clinical_42_multilabel_classification_with_missing_generation/multilabelstratifiedkfold_multilabelholdout/predictions/naimcat_noimputation_normalize_categoricalencode_features_MCAR_drop_samples"
    CXR_relative_path = "MIMIC_CXR_42_multilabel_classification/multilabelstratifiedkfold_multilabelholdout/predictions/resnet"

    outputs_path = os.path.dirname(cfg.paths.experiment)

    new_outputs_path = os.path.join(cfg.paths.experiment+"_processed")
    new_predictions_path = os.path.join(new_outputs_path, 'predictions', os.path.basename(cfg.paths.predictions)).replace("_complete", "")
    new_results_path = os.path.join(new_outputs_path, 'results', os.path.basename(cfg.paths.predictions)).replace("_complete", "")
    os.makedirs(new_predictions_path, exist_ok=True)
    os.makedirs(new_results_path, exist_ok=True)

    clinical_predictions_path = os.path.join(outputs_path, clinical_relative_path)
    CXR_predictions_path = os.path.join(outputs_path, CXR_relative_path)

    for train_missing_fraction in tqdm(cfg.missing_percentages):
        if isinstance(train_missing_fraction, (list, ListConfig)):
            train_missing_percentage = "-".join([f"{int(100 * train_missing_fraction[i])}" for i in range(len(train_missing_fraction))])
            clinical_perc, CXR_perc = train_missing_percentage.split("-")
        else:
            train_missing_percentage = int(100 * train_missing_fraction)
            clinical_perc, CXR_perc = train_missing_percentage, train_missing_percentage

        new_preds_path = os.path.join(new_predictions_path, f'{train_missing_percentage}')
        if not os.path.exists(new_preds_path):
            os.makedirs(new_preds_path, exist_ok=True)

        shutil.copytree(os.path.join(cfg.paths.predictions, f'{train_missing_percentage}'), new_preds_path, dirs_exist_ok=True)
        for p in os.listdir(new_preds_path):
            if os.path.isdir(os.path.join(new_preds_path, p)):
                shutil.rmtree(os.path.join(new_preds_path, p))

        for fold in tqdm(range(cfg.test_cv.n_splits)):
            CXR_preds = save_load.load_params_table(os.path.join(CXR_predictions_path, CXR_perc, "0", f"{fold}_0_test.csv"), index_col=0)
            clinical_preds = save_load.load_params_table(os.path.join(clinical_predictions_path, clinical_perc, "0", f"{fold}_0_test.csv"), index_col=0)

            for test_missing_fraction in cfg.get("test_missing_percentages", cfg.missing_percentages):
                if isinstance(test_missing_fraction, (list, ListConfig)):
                    test_missing_percentage = "-".join([f"{int(100 * test_missing_fraction[i])}" for i in range(len(test_missing_fraction))])
                else:
                    test_missing_percentage = int(100 * test_missing_fraction)

                new_test_preds_path = os.path.join(new_preds_path, f'{test_missing_percentage}')
                if not os.path.exists(new_test_preds_path):
                    os.makedirs(new_test_preds_path, exist_ok=True)

                multimodal_predictions_path = f'{cfg.paths.predictions}/{train_missing_percentage}/{test_missing_percentage}/{fold}_0_test.csv'
                multimodal_preds = save_load.load_params_table(multimodal_predictions_path, index_col=0)

                clinical_rate, CXR_rate = test_missing_percentage.split('-')
                missing_masks_paths = [os.path.join(cfg.paths.missing_masks.replace("_complete", ""), f'MIMIC_clinical_{fold}_0_test_{clinical_rate}.csv'), os.path.join(cfg.paths.missing_masks.replace("_complete", ""), f'MIMIC_CXR_{fold}_0_test_{CXR_rate}.csv')]
                missing_mask = pd.concat([save_load.load_table(p, index_col=0) for p in missing_masks_paths], axis=1)

                missing_mask['binary'] = missing_mask['MIMIC_clinical'].astype(str) + missing_mask['MIMIC_CXR'].astype(str)
                missing_mask['decimal'] = missing_mask['binary'].apply(lambda x: int(x, 2))

                preds = [multimodal_preds, clinical_preds, CXR_preds]
                preds_all = pd.concat(preds, keys=range(len(preds)))

                preds_model_selection = preds_all.loc[list(zip(missing_mask.decimal.values, multimodal_preds.index.values))].droplevel(0)
                save_load.save_table(preds_model_selection, f"{fold}_0_test.csv", new_test_preds_path, index=True)

    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params={})
    metrics.compute_missing_performance(cfg.dbs['0'].classes, new_predictions_path, new_results_path, next(iter(cfg.dbs.values())).task, performance_metrics, cfg.missing_percentages, cfg.get("test_missing_percentages", cfg.missing_percentages), decimals=max(next(iter(cfg.dbs.values())).get("decimals", 2), 2))

    log.info(f"Job finished")


if __name__ == '__main__':
    main()
