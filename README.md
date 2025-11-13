# embedingtest

## CLI 사용 방법

`cli.py` 스크립트는 설정한 확장자 목록(기본값: `.txt`)에 해당하는 파일을 한 번에 출력하고, 동일한 텍스트를 지정한 임베딩 모델(`qwen3-embedding-4b`)로 변환해 벡터 정보를 보여줍니다.

```bash
python cli.py <파일_또는_폴더_경로>
```

예시:

```bash
# 기본 모델(qwen3-embedding-4b)로 실행
python cli.py data/단일주제_인공지능.txt

# 다른 임베딩 모델로 교체 실행
python cli.py data --embedding-model my-custom-model
```

* 파일 경로를 전달하면 해당 파일을 검증한 뒤 내용을 출력하고 임베딩합니다.
* 폴더 경로를 전달하면 재귀적으로 허용 확장자를 가진 파일을 찾아 사전순으로 출력·임베딩합니다.
* 허용된 확장자 파일이 없거나 존재하지 않는 경로를 지정하면 오류 메시지를 반환합니다.
* 임베딩 초기화 시 `transformers`, `torch` 패키지가 필요하며, 최초 실행 시 모델이 자동으로 다운로드됩니다.

### 설정값 커스터마이징

`cli.py` 상단의 `CLI_CONFIG`에서 다음 항목을 조정해 필요에 맞게 확장자 및 인코딩을 바꿀 수 있습니다.

| 항목 | 설명 | 기본값 |
| --- | --- | --- |
| `allowed_extensions` | 읽을 파일 확장자 목록 (대소문자 무시) | `(".txt",)` |
| `preferred_encodings` | 순차적으로 시도할 인코딩 목록 | `("utf-8", "cp949")` |
| `embedding.model_name` | 다운로드 및 사용할 임베딩 모델 이름 | `qwen3-embedding-4b` |
| `embedding.device` | `cuda`, `cpu` 등 강제할 디바이스 (`None`이면 자동 판별) | `None` |
| `embedding.max_length` | 토큰화 시 최대 길이 | `512` |
| `embedding.preview_values` | 출력 시 표시할 임베딩 앞쪽 값 개수 | `5` |

> 임베딩 기능을 사용하려면 먼저 `pip install transformers torch`로 의존성을 설치해주세요.

예를 들어 `.md`도 함께 읽고 싶다면 `allowed_extensions=(".txt", ".md")`처럼 수정하면 됩니다.
