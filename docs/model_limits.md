# Qwen3-4B-Instruct-2507 입력 한도 참고

## 공식 문서 출처
- [Qwen3 GitHub README](https://github.com/QwenLM/Qwen3/blob/main/README.md)

## 근거
- Qwen 팀이 제공한 README에는 "Qwen3-Instruct-2507" 모델이 **256K long-context understanding** 능력을 갖췄다고 명시되어 있습니다. 이는 Qwen3-2507 계열이 기본적으로 256K 토큰 입력을 처리하도록 설계되었음을 의미합니다.
- 같은 문서의 `sglang.launch_server` 예시에서는 `--context-length 262144` 값을 사용하고 있어, 256K 토큰을 262,144 토큰(2^18 * 4)로 표현하고 있음을 확인할 수 있습니다.

따라서 `Qwen/Qwen3-4B-Instruct-2507`의 최대 입력 토큰 수는 262,144로 간주할 수 있으며, 이 한도의 70%는 약 183,500 토큰입니다. 본 저장소에서는 이 값을 기반으로 발췌 길이를 자동으로 계산합니다.
