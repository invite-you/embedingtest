# embedingtest

## CLI 사용 방법

`cli.py` 스크립트는 설정한 확장자 목록(기본값: `.txt`)에 해당하는 파일을 읽고
[`cluster_agent.DocumentClusterAgent`](cluster_agent.py)가 생성한 문장 군집 결과를 그대로 출력합니다.
각 클러스터는 중심에 가장 가까운 문장을 최대 3개까지 담으므로,
터미널에서 곧바로 "상위 대표 문장" 리스트를 확인할 수 있습니다.

```bash
python cli.py <파일_또는_폴더_경로>
python cli.py data --model qwen/qwen2.5-embedding --device cpu --dtype float32 --config config/agent.yaml
```

예시:

```bash
python cli.py data/단일주제_인공지능.txt
python cli.py data
```

* 파일 경로를 전달하면 해당 파일을 검증한 뒤 대표 문장 블록(클러스터)을 출력합니다.
* 폴더 경로를 전달하면 재귀적으로 허용 확장자를 가진 파일을 찾아 사전순으로 처리합니다.
* `--model` 옵션으로 Hugging Face 모델 이름을 바꿀 수 있습니다.
* `--device`로 임베딩 모델을 올릴 디바이스(`cpu`, `cuda`, `auto`)를 지정해 CPU·GPU 간 전환을 강제할 수 있습니다.
* `--dtype`에 `float16`, `torch.float32` 등 torch dtype을 넘기면 모델 로딩과 `torch.autocast`가 동일한 dtype으로 맞춰집니다.
* `--config`로 `config/agent.yaml` 외의 설정 파일을 지정하면 문장 수 제한, 군집 개수 등을 즉시 바꿀 수 있습니다.
* 임베딩 단계에서는 `StageTimer`가 토큰화·디바이스 전송·추론·후처리까지 세부 타임스템프(Δ/Σ 시간)를 출력해 병목 지점을 즉시 확인할 수 있습니다.
* 허용된 확장자 파일이 없거나 존재하지 않는 경로를 지정하면 오류 메시지를 반환합니다.

### 설정값 커스터마이징

`cli.py` 상단의 `CLI_CONFIG`에서 다음 항목을 조정해 필요에 맞게 확장자 및 인코딩을 바꿀 수 있습니다.

| 항목 | 설명 | 기본값 |
| --- | --- | --- |
| `allowed_extensions` | 읽을 파일 확장자 목록 (대소문자 무시) | `(".txt",)` |
| `preferred_encodings` | 순차적으로 시도할 인코딩 목록 | `("utf-8", "cp949")` |
| `embedding_model_name` | Transformers에서 다운로드할 임베딩 모델 이름 | `"qwen/qwen3-embedding-4b"` |
| `device` | 모델을 올릴 디바이스 (`cpu`, `cuda`, `auto` 등) | 시스템 CUDA 가용성에 따라 `cuda` 또는 `cpu` |
| `torch_dtype` | 모델 및 autocast에 사용할 torch dtype | CUDA 사용 시 `torch.float16`, 아니면 `torch.float32` |

예를 들어 `.md`도 함께 읽고 싶다면 `allowed_extensions=(".txt", ".md")`처럼 수정하면 됩니다.

CLI 내부에서는 여전히 동일한 임베딩 파이프라인(토큰화→추론→평균 풀링→정규화)을 통해 문장 벡터를 만든 뒤, 곧바로 군집화 결과만 출력합니다. 디버깅이 필요하면 `print_file_contents` 헬퍼를 직접 호출해 JSON 임베딩 요약을 확인할 수 있습니다.
## 문장 군집 요약

`cluster_agent.DocumentClusterAgent`는 `config/agent.yaml`에서 로드한 구성값을 바탕으로
다음 단계를 수행합니다.

1. 파일의 텍스트를 읽고 5MB 이상이면 `TEXT_TOO_LARGE` 상태로 즉시 종료합니다.
2. 문장 단위로 텍스트를 분할합니다. 문장이 부족하면 클러스터링 대신 균등 간격으로 표본을 추립니다.
3. `EmbeddingService`로 문장별 임베딩을 얻고, `min(max_clusters, ceil(sqrt(n)))`개 K-means 클러스터를 생성합니다.
4. 각 클러스터에서 중심에 가장 가까운 문장을 최대 3개까지 골라 대표 문장 목록을 만듭니다. 구성원이 2개 미만인 약한 군집은 생략합니다.
5. 대표 문장 목록을 합쳐 요약 문장을 만들고, 간단한 키워드·언어 정보를 붙입니다.

`python cli.py data`만으로도 각 파일별 대표 블록을 터미널에서 곧바로 확인할 수 있습니다.

## 폴더 단위 파일 클러스터링 CLI

`file_cluster_cli.py`는 폴더 경로를 입력받아 각 파일의 전체 텍스트를 `mdbr-leaf-ir` 임베딩으로 변환하고,
HDBSCAN을 사용해 유사한 파일끼리 자동으로 묶어 줍니다.
HDBSCAN이 노이즈로 판단한 파일도 `noise`라는 별도 클러스터로 함께 출력되므로 모든 파일이 어느 그룹에 속했는지 한눈에 확인할 수 있습니다.

```bash
python file_cluster_cli.py data --extensions .txt .md --min-cluster-size 2 --min-samples 1
```

* `--extensions`로 대상 확장자를 원하는 만큼 지정할 수 있습니다.
* 기본 임베딩 모델은 `mdbr-leaf-ir`이며, `--model`로 다른 이름을 전달하면 즉시 교체됩니다.
* `--min-cluster-size`, `--min-samples`를 조절해 클러스터링 민감도를 바꿀 수 있습니다.
* 실행 전 `pip install hdbscan numpy`로 필요한 라이브러리를 설치해 주세요.

