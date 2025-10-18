import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aeon.datasets import load_classification
from tsml_eval.experiments import get_classifier_by_name, run_classification_experiment
from tsml_eval.evaluation.storage import load_classifier_results
from aeon.visualisation import plot_critical_difference

from joblib import Parallel, delayed

# 실험 설정
UCR_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\112UCR"
RESULTS_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\generated_results"
NUM_REPEATS = 30
CLASSIFIERS = ["ee", "dtw", "pf", "grail"]
DATASETS = ["TwoLeadECG", "Coffee", "UMD"]
N_JOBS = -1  # 전체 CPU 사용


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
        X_train, y_train = load_classification(dataset, split="train", extract_path=ucr_root)
        X_test, y_test = load_classification(dataset, split="test", extract_path=ucr_root)

        run_classification_experiment(
            X_train, y_train,
            X_test, y_test,
            clf,
            results_path,
            dataset_name=dataset,
            resample_id=resample_id
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


def summarize_results(datasets, classifiers, results_path, num_repeats):
    summary = []
    for clf_name in classifiers:
        clf_folder = get_classifier_by_name(clf_name).__class__.__name__

        for dataset in datasets:
            accuracies = []
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
                    accuracies.append(cr.accuracy)
                except Exception as e:
                    print(f"⚠️ 결과 로딩 실패: {result_file} - {e}")

            avg_acc = np.mean(accuracies) if accuracies else None
            summary.append({
                "Classifier": clf_folder,
                "Dataset": dataset,
                "Mean Accuracy": avg_acc,
                "Num Results": len(accuracies)
            })

    df = pd.DataFrame(summary)
    df.to_csv("summary_results.csv", index=False)
    return df


def load_benchmark_results(datasets):
    benchmark_paths = {
        "1NN-DTW": r"C:\Users\chomi\tsml-eval\bakeoff\publications\y2023\tsc_bakeoff\results\distance\1NN-DTW_accuracy.csv",
        "EE":      r"C:\Users\chomi\tsml-eval\bakeoff\publications\y2023\tsc_bakeoff\results\distance\EE_accuracy.csv",
        "GRAIL":   r"C:\Users\chomi\tsml-eval\bakeoff\publications\y2023\tsc_bakeoff\results\distance\GRAIL_accuracy.csv",
        "PF":      r"C:\Users\chomi\tsml-eval\bakeoff\publications\y2023\tsc_bakeoff\results\distance\PF_accuracy.csv",
    }

    benchmarks = {}
    for name, path in benchmark_paths.items():
        try:
            df = pd.read_csv(path, index_col=0)
            filtered = df[df.index.isin(datasets)]
            benchmarks[name] = filtered.mean(axis=1).to_dict()
        except Exception as e:
            print(f"⚠️ 벤치마크 로딩 실패: {name} - {e}")

    return benchmarks


def merge_and_compare(summary_df, benchmarks):
    name_map = {
        "ElasticEnsemble": "EE",
        "KNeighborsTimeSeriesClassifier": "1NN-DTW",
        "ProximityForest": "PF",
        "GRAILClassifier": "GRAIL",
    }

    for row in summary_df.itertuples():
        method = name_map.get(row.Classifier, row.Classifier)
        if method not in benchmarks:
            benchmarks[method] = {}
        benchmarks[method][row.Dataset] = row._3  # Mean Accuracy

    df = pd.DataFrame(benchmarks).T[summary_df["Dataset"].unique()]
    df.to_csv("benchmark_comparison.csv")
    return df


def main():
    print("✅ 고정 데이터셋 사용:", DATASETS)

    print("\n⚡ 병렬 실험 실행 중...")
    run_parallel_experiments(DATASETS, CLASSIFIERS, UCR_PATH, RESULTS_PATH, NUM_REPEATS, N_JOBS)

    print("\n📊 결과 요약 중...")
    summary_df = summarize_results(DATASETS, CLASSIFIERS, RESULTS_PATH, NUM_REPEATS)
    print(summary_df)

    print("\n📁 벤치마크 로딩 중...")
    benchmarks = load_benchmark_results(DATASETS)

    print("\n📌 성능 비교:")
    compare_df = merge_and_compare(summary_df, benchmarks)
    print(compare_df)

    print("\n📈 CD Diagram 시각화...")
    plt, _ = plot_critical_difference(compare_df.values, list(compare_df.index))
    plt.show()


if __name__ == "__main__":
    main()
