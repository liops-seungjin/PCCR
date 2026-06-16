# CloudCropper LaTeX organize notes

이 폴더는 **알고리즘 정리 PDF**를 만드는 작업 공간입니다. registration
(ICP / GICP / SDF / Gradient-SDF) 등 수식 중심의 알고리즘 노트만 PDF에
포함하며, 소프트웨어 설계 문서(architecture, format I/O, viewer, build)는
`docs/design/*.md`에 그대로 두고 여기서는 다루지 않습니다.
(설계 챕터 .tex 파일들은 chapters/에 남아 있지만 main.tex에서 제외됨)

## Structure

- `main.tex`: PDF의 진입점입니다. 장 순서와 메타데이터를 여기서 관리합니다.
- `preamble.tex`: 패키지, 색상, 코드 블록, 공통 명령을 모아둡니다.
- `chapters/`: 각 장의 본문입니다. 기존 `docs/design` 문서와 1:1에 가깝게
  매핑하되, PDF에서는 읽기 좋은 흐름으로 합쳤습니다.
- `figures/`: 아키텍처 그림, 파이프라인 그림, 캡처 이미지를 넣는 곳입니다.
- `tables/`: 큰 표를 별도 파일로 분리하고 싶을 때 쓰는 곳입니다.
- `build/`: PDF와 LaTeX 보조 파일 출력 위치입니다. Git에는 올리지 않습니다.

## Build

권장 빌드는 XeLaTeX입니다. 한글 제목과 메모를 자연스럽게 처리하기 위해
`kotex`를 사용합니다.

```bash
cd docs/organize
make
```

결과물은 `docs/organize/build/cloudcropper-organized-design.pdf`에 생성됩니다.

`latexmk`가 없다면 아래처럼 두 번 실행해 목차를 갱신할 수 있습니다.

```bash
cd docs/organize
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname=cloudcropper-organized-design main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname=cloudcropper-organized-design main.tex
```

## Editing flow

1. 기존 설계 원문은 `../design/*.md`에서 확인합니다.
2. PDF에 들어갈 요약과 결정 사항은 `chapters/*.tex`에 정리합니다.
3. 그림이 필요하면 `figures/`에 넣고 `\includegraphics`로 참조합니다.
4. 장을 추가하면 `main.tex`의 `\input{...}` 목록에 연결합니다.

