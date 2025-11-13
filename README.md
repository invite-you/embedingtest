# embedingtest

## CLI 사용 방법

`cli.py` 스크립트는 설정한 확장자 목록(기본값: `.txt`)에 해당하는 파일을 한 번에 출력하고,
각 텍스트를 줄바꿈(우선) 또는 문장 부호 단위로 쪼갠 뒤
[`qwen3-embedding-4b`](https://huggingface.co/qwen/qwen3-embedding-4b) 모델로 임베딩합니다.
`cluster_agent.py`는 이렇게 얻은 문장 임베딩을 K-means 기반으로 군집화하고,
각 클러스터에서 대표 문장을 골라 앞뒤 문맥까지 3줄 블록으로 묶어줍니다.

```bash
python cli.py <파일_또는_폴더_경로>
python cli.py data --model qwen/qwen2.5-embedding --device cpu --dtype float32
python -c "from pathlib import Path; from cluster_agent import run_cluster_cli; run_cluster_cli(Path('data'))"  # 문장 군집 결과
```

예시:

```bash
python cli.py data/단일주제_인공지능.txt
python cli.py data
```

* 파일 경로를 전달하면 해당 파일을 검증한 뒤 내용을 출력하고, 분할된 각 줄/문장의 임베딩 결과 JSON을 터미널에 함께 출력합니다.
* 폴더 경로를 전달하면 재귀적으로 허용 확장자를 가진 파일을 찾아 사전순으로 출력·임베딩합니다.
* `--model` 옵션으로 Hugging Face 모델 이름을 바꿀 수 있습니다.
* `--device`로 임베딩 모델을 올릴 디바이스(`cpu`, `cuda`, `auto`)를 지정해 CPU·GPU 간 전환을 강제할 수 있습니다.
* `--dtype`에 `float16`, `torch.float32` 등 torch dtype을 넘기면 모델 로딩과 `torch.autocast`가 동일한 dtype으로 맞춰집니다.
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

임베딩 결과는 JSON 형식으로 출력되며 분할된 텍스트마다 벡터가 포함됩니다.

```jsonc
{
  "source_path": "원본 파일 경로",
  "model_name": "사용한 모델 이름",
  "segment_count": 2,
  "segments": [
    {
      "index": 0,
      "text": "첫 문장입니다.",
      "vector_length": 1536,
      "vector": [0.0, 0.01, ...]
    },
    {
      "index": 1,
      "text": "두 번째 문장입니다!",
      "vector_length": 1536,
      "vector": [0.02, -0.03, ...]
    }
  ]
}
```

### 임베딩 계산 단계 요약

CLI가 `qwen3-embedding-4b` 같은 모델을 호출할 때는 다음 절차를 따릅니다.

1. **토큰화 및 전처리** – 입력 텍스트를 토큰으로 바꾸고, 매우 긴 문장은 `max_length=512`로 잘라 GPU 메모리 사용량을 제한합니다. 동시에 배치 차원에서 패딩을 맞춰 모델이 안정적으로 처리할 수 있는 PyTorch 텐서를 얻습니다.
2. **추론 전용 실행** – CLI는 학습이 아닌 추론만 수행하므로 `torch.inference_mode()` 컨텍스트에서 모델을 호출해 불필요한 그래디언트 버퍼 생성을 피하고 실행 시간을 줄입니다. CUDA 장치라면 `torch.autocast(device_type="cuda", dtype=설정값)`을 함께 사용해 절반 정밀도 추론으로 VRAM을 절약합니다.
3. **출력 검증** – 커스텀 모델 이름을 받더라도 `last_hidden_state` 같은 핵심 필드가 있는지 확인하여, 모델 호환성이 깨진 경우 즉시 오류를 보고합니다.
4. **마스킹 평균 풀링** – `attention_mask`로 실제 토큰 위치만 남기고 은닉 벡터를 평균 내면 문장 길이에 상관없이 비교 가능한 대표 임베딩을 얻을 수 있습니다. 패딩 토큰이 계산에 섞이지 않아 결과 품질이 높아집니다.
5. **L2 정규화** – 마지막으로 벡터 길이를 1로 맞춰 코사인 유사도 기반 검색이나 군집화에서 안정적인 거리를 보장합니다. 길이가 다른 문서 간의 값 범위를 맞추는 효과도 있습니다.

이 과정을 통해 길이가 다른 텍스트도 일관된 스케일의 임베딩으로 변환되며, 줄바꿈/문장 단위로 세분화된 JSON 벡터를 그대로 검색·클러스터링에 활용할 수 있습니다.
## 문장 군집 요약

`cluster_agent.DocumentClusterAgent`는 `config/agent.yaml`에서 로드한 구성값을 바탕으로
다음 단계를 수행합니다.

1. 파일의 텍스트를 읽고 5MB 이상이면 `TEXT_TOO_LARGE` 상태로 즉시 종료합니다.
2. 문장 단위로 텍스트를 분할합니다. 문장이 부족하면 클러스터링 대신 균등 간격으로 표본을 추립니다.
3. `EmbeddingService`로 문장별 임베딩을 얻고, `min(max_clusters, ceil(sqrt(n)))`개 K-means 클러스터를 생성합니다.
4. 각 클러스터에서 중심에 가장 가까운 문장을 대표 문장으로 고르고, 앞뒤 1문장을 함께 묶어 3줄 블록을 만듭니다.
5. 대표 블록을 합쳐 요약 문장을 만들고, 간단한 키워드·언어 정보를 붙입니다.

`run_cluster_cli(Path('data'))`처럼 호출하면 각 파일별 대표 블록을 터미널에서 곧바로 확인할 수 있습니다.

