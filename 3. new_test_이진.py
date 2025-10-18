import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aeon.datasets import load_classification
from tsml_eval.experiments import get_classifier_by_name, run_classification_experiment
from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.utils.resampling import resample_data   # ✅ 추가

from joblib import Parallel, delayed

# 실험 설정
UCR_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\112UCR_10"
RESULTS_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\generated_results"
NUM_REPEATS = 30
CLASSIFIERS = ["shape"]
DATASETS = [d for d in os.listdir(UCR_PATH) if os.path.isdir(os.path.join(UCR_PATH, d))]
N_JOBS = 8  # 전체 CPU 사용


def run_single_experiment(clf_name, dataset, resample_id, ucr_root, results_path):
    clf = get_classifier_by_name(clf_name)
    clf_folder = clf.__class__.__name__

    if hasattr(clf, "distance"):
        clf_folder = f"{clf_folder}_{clf.distance}"

    result_file = os.path.join(
        results_path,
        clf_folder,
        "Predictions",
        dataset,
        f"testResample{resample_id}.csv"
    )

    if os.path.exists(result_file):
        print(f"⏩ 스킵: {dataset}, {clf_folder}, resample_id={resample_id}")
        return

    try:
        print(f"🚀 실행 중: {clf_folder}, {dataset}, resample_id={resample_id}")
        # 원래 split 로드
        X_train, y_train = load_classification(dataset, split="train", extract_path=ucr_root)
        X_test, y_test = load_classification(dataset, split="test", extract_path=ucr_root)

        # ✅ resample_id > 0이면 새로운 split 생성
        if resample_id > 0:
            X_train, y_train, X_test, y_test = resample_data(
                X_train, y_train, X_test, y_test, random_state=resample_id
            )

        # 분류기 실행
        run_classification_experiment(
            X_train, y_train,
            X_test, y_test,
            clf,
            results_path,
            dataset_name=dataset,
            resample_id=resample_id,
            classifier_name=clf_folder,
        )
    except Exception as e:
        print(f"⚠️ 오류 발생: {clf_folder}, {dataset}, resample_id={resample_id} - {e}")


def run_parallel_experiments(datasets, classifiers, ucr_root, results_path, num_repeats, n_jobs=-1):
    tasks = [
        (clf, dataset, rid)
        for clf in classifiers
        for dataset in datasets
        for rid in range(num_repeats)
    ]

    Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(clf, dataset, rid, ucr_root, results_path)
        for clf, dataset, rid in tasks
    )


def summarize_results_by_resample(datasets, classifiers, results_path, num_repeats):
    """
    Dataset x Resample 단위로 accuracy, balanced_accuracy, log_loss를 포함한 DataFrame 생성
    """
    all_results = {}

    for clf_name in classifiers:
        clf_obj = get_classifier_by_name(clf_name)
        clf_folder = clf_obj.__class__.__name__
        if hasattr(clf_obj, "distance"):
            clf_folder = f"{clf_folder}_{clf_obj.distance}"

        print(f"▶ 분류기: {clf_folder}")

        records = []

        for dataset in datasets:
            for resample_id in range(num_repeats):
                result_file = os.path.join(
                    results_path,
                    clf_folder,
                    "Predictions",
                    dataset,
                    f"testResample{resample_id}.csv"
                )
                try:
                    cr = load_classifier_results(result_file)
                    records.append({
                        "Dataset": dataset,
                        "Resample": f"Resample_{resample_id}",
                        "accuracy": cr.accuracy,
                        "balanced_accuracy": cr.balanced_accuracy,
                        "auroc": cr.auroc_score,
                        "log_loss": cr.log_loss
                    })
                except Exception as e:
                    print(f"⚠ 결과 로딩 실패: {result_file} - {e}")
                    records.append({
                        "Dataset": dataset,
                        "Resample": f"Resample_{resample_id}",
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "auroc": np.nan,
                        "log_loss": np.nan
                    })

            df = pd.DataFrame(records)
            # ✅ dataset별 csv 저장
            df.to_csv(
                rf"{results_path}\summary_results_{dataset}_{clf_folder}.csv",
                index=False
            )
            all_results[clf_folder] = df

    return all_results


def main():
    print("✅ 고정 데이터셋 사용:", DATASETS)

    print("\n⚡ 병렬 실험 실행 중...")
    run_parallel_experiments(DATASETS, CLASSIFIERS, UCR_PATH, RESULTS_PATH, NUM_REPEATS, N_JOBS)

    print("\n📊 결과 요약 중...")
    all_results = summarize_results_by_resample(DATASETS, CLASSIFIERS, RESULTS_PATH, NUM_REPEATS)
    for clf_name, df in all_results.items():
        print(f"\n📄 {clf_name} 결과:")
        print(df)


if __name__ == "__main__":
    main()
