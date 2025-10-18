import os
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from tsml_eval.experiments import get_classifier_by_name, run_classification_experiment
from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.utils.resampling import resample_data
from joblib import Parallel, delayed

# 실험 설정
UCR_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\112UCR"
RESULTS_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\generated_results"
NUM_REPEATS = 5
CLASSIFIERS = ["grail"]
DATASETS = [d for d in os.listdir(UCR_PATH) if os.path.isdir(os.path.join(UCR_PATH, d))]
N_JOBS = 4


def _load_ucr_txt_file(file_path):
    """
    UCR 표준 .txt 파일을 직접 읽어 X (데이터)와 y (라벨)로 반환합니다.
    (헤더가 없거나 탭으로 구분된 파일도 처리하도록 수정됨)
    """
    X_data = []
    y_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # 헤더 섹션 건너뛰기
            data_start_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("@data"):
                    data_start_index = i + 1
                    break

            for line in lines[data_start_index:]:
                line = line.strip()
                if not line:
                    continue

                parts = None
                if '\t' in line:
                    parts = line.split('\t')
                elif ',' in line:
                    parts = line.split(',')
                else:
                    continue  # 알 수 없는 형식의 라인은 건너뜁니다.
                
                if parts:
                    label = parts[0]
                    series_values = parts[1:]
                    try:
                        series = [float(x) for x in series_values if x]
                        if series:
                            y_data.append(label)
                            X_data.append(series)
                    except ValueError:
                        continue  # 데이터 값 변환 실패 시 건너뜁니다.

        if not X_data:
            return None, None

        X = np.array(X_data, dtype=float)
        y = np.array(y_data)

        # aeon의 load_classification 결과와 형식을 맞추기 위해 3차원으로 변환
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return X, y
    
    except Exception as e:
        print(f"❌ {file_path} 파일 로드 중 오류 발생: {e}")
        return None, None


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

        # 로컬 .txt 파일 존재 여부 확인 후 로드
        train_txt_path = os.path.join(ucr_root, dataset, f"{dataset}_TRAIN.txt")
        test_txt_path = os.path.join(ucr_root, dataset, f"{dataset}_TEST.txt")
        
        if os.path.exists(train_txt_path) and os.path.exists(test_txt_path):
            print(f"ℹ️ {dataset}: .txt 파일이 존재합니다. 로컬에서 직접 로드합니다.")
            X_train, y_train = _load_ucr_txt_file(train_txt_path)
            X_test, y_test = _load_ucr_txt_file(test_txt_path)
            
            if X_train is None or X_test is None:
                raise ValueError("로컬 .txt 파일 로드 실패.")
        else:
            print(f"ℹ️ {dataset}: .txt 파일이 없습니다. aeon.datasets.load_classification을 사용합니다.")
            X_train, y_train = load_classification(dataset, split="train", extract_path=ucr_root)
            X_test, y_test = load_classification(dataset, split="test", extract_path=ucr_root)

        # 데이터 형태(shape) 검증 및 조정
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)
        
        if X_train.ndim == 3 and X_train.shape[1] == 1:
            X_train = np.squeeze(X_train, axis=1)
            X_test = np.squeeze(X_test, axis=1)

        # resample_id > 0이면 새로운 split 생성
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
