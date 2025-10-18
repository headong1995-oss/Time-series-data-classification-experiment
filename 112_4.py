import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aeon.datasets import load_classification
from tsml_eval.experiments import get_classifier_by_name, run_classification_experiment
from tsml_eval.evaluation.storage import load_classifier_results

from joblib import Parallel, delayed
import urllib.error
from sklearn.model_selection import train_test_split

# 실험 설정
UCR_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\112UCR"
RESULTS_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\generated_results"
NUM_REPEATS = 30
CLASSIFIERS = ["dtw", "ee"]
DATASETS = [name for name in os.listdir(UCR_PATH)
            if os.path.isdir(os.path.join(UCR_PATH, name))]
print("✅ 자동으로 감지된 데이터셋:", DATASETS)

N_JOBS = -1  # 전체 CPU 사용

def _load_ucr_txt_file_custom(file_path):
    X_data = []
    y_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                label = parts[0]
                series_values = [float(x) for x in parts[1:] if x]

                if series_values:
                    y_data.append(label)
                    X_data.append(series_values)
        
        if not X_data:
            return None, None

        X = np.array(X_data, dtype=float)
        y = np.array(y_data)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        return X, y
    
    except Exception as e:
        print(f"❌ {file_path} 파일 로드 중 오류 발생: {e}")
        return None, None

def run_single_experiment(clf_name, dataset, resample_id, ucr_root, results_path):
    clf = get_classifier_by_name(clf_name)
    clf_folder = clf.__class__.__name__

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
        
        # 1. 먼저 aeon의 load_classification을 이용해 표준 ts 파일을 로드하려고 시도
        try:
            X_train, y_train = load_classification(dataset, split="train", extract_path=ucr_root)
            X_test, y_test = load_classification(dataset, split="test", extract_path=ucr_root)
            print(f"ℹ️ {dataset}: aeon 로더로 .ts 파일을 성공적으로 로드했습니다.")
        except Exception:
            # 2. .ts 파일 로드에 실패하면 .txt 파일을 로드하려고 시도
            train_txt_path = os.path.join(ucr_root, dataset, f"{dataset}_TRAIN.txt")
            test_txt_path = os.path.join(ucr_root, dataset, f"{dataset}_TEST.txt")
            
            X_train, y_train = _load_ucr_txt_file_custom(train_txt_path)
            X_test, y_test = _load_ucr_txt_file_custom(test_txt_path)
            
            if X_train is not None and X_test is not None:
                print(f"ℹ️ {dataset}: 맞춤형 로더로 .txt 파일을 성공적으로 로드했습니다.")
            else:
                raise ValueError("데이터셋 파일 로드 실패")

        # 로드된 데이터가 유효한지 확인
        if X_train is None or X_test is None:
            raise ValueError("데이터셋 로드 실패")

        # 재샘플링을 위해 데이터를 합쳐서 무작위로 분할
        X_full = np.concatenate((X_train, X_test), axis=0)
        y_full = np.concatenate((y_train, y_test), axis=0)

        X_train_resample, X_test_resample, y_train_resample, y_test_resample = train_test_split(
            X_full, y_full, test_size=0.3, random_state=resample_id, stratify=y_full
        )

        run_classification_experiment(
            X_train_resample, y_train_resample,
            X_test_resample, y_test_resample,
            clf,
            results_path,
            dataset_name=dataset,
            resample_id=resample_id
        )
    except urllib.error.HTTPError as he:
        print(f"⚠️ 오류 발생: {clf_folder}, {dataset}, resample_id={resample_id} - {he}")
        print("💡 401 오류는 로컬에 데이터셋이 없어서 온라인 다운로드를 시도했으나 권한 문제로 실패했을 때 발생합니다.")
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
    all_results = {}

    for clf_name in classifiers:
        clf_folder = get_classifier_by_name(clf_name).__class__.__name__
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
        df.to_csv(rf"C:\Users\chomi\tsml-eval\bakeoff\generated_results\summary_results_by_resample_{clf_folder}.csv", index=False)
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
