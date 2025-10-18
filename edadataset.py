import os
import numpy as np
import pandas as pd
from aeon.datasets import load_classification, load_from_tsv_file

# 실험 설정 (원본 코드에서 가져옴)
# UCR_PATH는 실제 데이터셋 경로로 변경해야 합니다.
UCR_PATH = r"C:\Users\chomi\tsml-eval\bakeoff\Missing_value_and_variable_length_datasets_adjusted"
# 실제 데이터셋 디렉토리 목록을 가져옵니다.
DATASETS = [d for d in os.listdir(UCR_PATH) if os.path.isdir(os.path.join(UCR_PATH, d))]

def summarize_data_statistics(datasets, ucr_root):
    """
    각 데이터셋에 대한 통계 정보(평균, 표준편차, 크기, 길이, 클래스 수)를
    계산하여 DataFrame으로 반환합니다.
    """
    results_list = []

    print("🔎 데이터셋 EDA 진행 중...")

    for dataset in datasets:
        print(f"  - {dataset} 데이터 로딩 및 통계 계산 중...")
        try:
            # 학습 및 테스트 데이터셋 파일 경로를 구성합니다.
            train_path_ts = os.path.join(ucr_root, dataset, f"{dataset}_TRAIN.ts")
            test_path_ts = os.path.join(ucr_root, dataset, f"{dataset}_TEST.ts")
            train_path_tsv = os.path.join(ucr_root, dataset, f"{dataset}_TRAIN.tsv")
            test_path_tsv = os.path.join(ucr_root, dataset, f"{dataset}_TEST.tsv")
            
            # 파일 확장자에 따라 적절한 함수를 사용하여 데이터셋을 로드합니다.
            if os.path.exists(train_path_ts):
                X_train, y_train = load_classification(dataset, split="train", extract_path=ucr_root)
                X_test, y_test = load_classification(dataset, split="test", extract_path=ucr_root)
            elif os.path.exists(train_path_tsv):
                X_train, y_train = load_from_tsv_file(train_path_tsv)
                X_test, y_test = load_from_tsv_file(test_path_tsv)
            else:
                raise FileNotFoundError(f"지원하는 파일 형식(.ts 또는 .tsv)을 찾을 수 없습니다.")

            # 학습 데이터셋 크기
            train_size = X_train.shape[0]
            # 테스트 데이터셋 크기
            test_size = X_test.shape[0]
            # 시계열 길이 (다변량 시계열도 고려)
            series_length = X_train.shape[-1]
            # 클래스 개수
            n_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

            # 모든 데이터를 하나로 합칩니다.
            combined_data = np.concatenate((X_train, X_test), axis=0)

            # 다변량 시계열 데이터의 경우를 처리
            if combined_data.ndim > 2:
                combined_data = combined_data.reshape(-1, combined_data.shape[-1])

            # 전체 데이터의 평균과 표준편차를 계산합니다.
            data_mean = np.mean(combined_data)
            data_std = np.std(combined_data)

            # 결과를 리스트에 저장
            results_list.append({
                "Dataset": dataset,
                "Train Size": train_size,
                "Test Size": test_size,
                "Length": series_length,
                "Num Classes": n_classes,
                "Mean": data_mean,
                "Standard Deviation": data_std
            })
        except Exception as e:
            print(f"⚠️ {dataset} 처리 중 오류 발생: {e}")
            results_list.append({
                "Dataset": dataset,
                "Train Size": np.nan,
                "Test Size": np.nan,
                "Length": np.nan,
                "Num Classes": np.nan,
                "Mean": np.nan,
                "Standard Deviation": np.nan
            })

    # 결과를 pandas DataFrame으로 변환
    results_df = pd.DataFrame(results_list)
    return results_df

# 함수 실행
if __name__ == "__main__":
    eda_results_df = summarize_data_statistics(DATASETS, UCR_PATH)
    print("\n📊 최종 EDA 요약 결과:")
    print(eda_results_df)

    # DataFrame을 CSV 파일로 저장
    output_filename = "data_eda_summary.csv"
    eda_results_df.to_csv(output_filename, index=False)
    print(f"\n✅ EDA 요약 결과가 '{output_filename}' 파일로 저장되었습니다.")
