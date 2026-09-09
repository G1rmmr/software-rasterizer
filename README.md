# software-rasterizer

C++20과 SIMD를 사용하는 CPU 소프트웨어 래스터라이저입니다.
정점 변환, 클리핑, 래스터화, 조명, 그림자, SSAO, AA를 CPU에서 계산하고 MiniFB로 표시합니다.

기존 데모: https://github.com/user-attachments/assets/61df9154-b431-40e3-a6a0-6553832494ff

## 구조 읽기

[구조를 이해하기 위한 학습 안내](docs/ARCHITECTURE.md)는 소유권, 불변식, 합성, 정적 다형성, 좌표 규약을 실제 코드와 연결합니다.
“이 개념을 공부하면 이 경계가 왜 필요한지 보인다”는 방식으로 읽을 수 있습니다.

| 위치 | 책임 |
| --- | --- |
| `app/`, `AppState.hpp`, `Input.hpp` | 프로그램 구성, 입력, 창 수명, 표시·저장 |
| `resources/` | OBJ 해석과 모델·텍스처 생성 |
| `scene/` | 공유 모델 자산과 배치된 객체의 변환·가시성 |
| `graphics/Material.hpp`, `Mesh.hpp`, `Texture.hpp` | 생성 시 검증되는 렌더링 자산 |
| `graphics/Renderer.*`, `passes/`, `post/` | 패스 순서와 작업 버퍼 소유 |
| `graphics/Rasterizer.hpp` | 클리핑, 보간, 깊이 검사, 타일 래스터화 |
| `shaders/` | 정점 처리와 표면·그림자 계산 |
| `math/` | SIMD 벡터·행렬·쿼터니언 |
| `tests/` | 불변식, 경계 사례, 실제 렌더링 회귀 검사 |

`Application`이 의존성을 만들고 연결합니다. 각 `Renderer`가 executor와 framebuffer를 소유하며,
`Render()`는 모든 작업을 마친 후 반환합니다.
렌더 코어는 MiniFB를 참조하지 않아 창 없이도 테스트할 수 있습니다.

모델은 검증된 불변 자산이며 `SceneObject`가 변환과 표시 상태를 관리합니다.
`SetTransform()`은 normal 행렬과 경계구도 함께 갱신합니다.
새 모델을 배치할 때의 기본 흐름은 다음과 같습니다.

```cpp
resources::ModelFactory assets("assets");
scene::Scene scene;
const auto model = assets.LoadDiablo();
const auto instance = scene.Add(model);
scene.Get(instance).SetTransform(math::CreateScale({8.f, 8.f, 8.f}));
```

## CMake 빌드 (Windows x64)

필수 환경:

- CMake 3.21 이상
- Visual Studio 2022 또는 Build Tools 2022: **C++를 사용한 데스크톱 개발** 워크로드
- AVX2를 지원하는 x64 CPU

저장소 루트에서 PowerShell로 실행합니다. 별도의 개발자 명령 프롬프트는 필요하지 않습니다.

```powershell
cmake --preset windows-msvc
cmake --build --preset release
ctest --preset release

cmake --build --preset debug
ctest --preset debug
```

빌드 프리셋은 프로그램과 테스트를 함께 빌드합니다.
실행 파일과 `assets`는 `build/windows-msvc/bin/Release/` 또는 `bin/Debug/`에 생성됩니다.
다음 명령은 빌드 후 프로그램을 실행합니다.

```powershell
cmake --build build/windows-msvc --config Release --target run
# 디버그 실행
cmake --build build/windows-msvc --config Debug --target run
```

프로그램은 실행 파일 옆의 `assets`를 우선 찾고, 없으면 작업 폴더의 `assets`를 사용합니다.
다른 자산 디렉터리는 `--assets DIR`로 지정합니다.
자산은 빌드할 때마다 복사되므로 모델이나 텍스처 변경도 반영됩니다.
Visual Studio 솔루션은 `cmake --open build/windows-msvc`로 열 수 있습니다.

저장소의 Windows x64용 `libs/minifb.lib`를 사용하므로 추가 다운로드는 필요하지 않습니다.
MiniFB의 Release ABI에 맞춰 Debug에서도 `/MD`와 `_ITERATOR_DEBUG_LEVEL=0`을 사용합니다.
Debug의 최적화 비활성화와 디버그 심볼은 유지되지만 Debug CRT와 STL 반복자 디버깅은 사용하지 않습니다.
누락된 MiniFB PDB의 `LNK4099` 경고와 SIMD의 익명 구조체 확장에 대한 `C4201` 경고가 발생할 수 있습니다.

Release는 `/O2`, AVX2, 지원되는 경우 링크 타임 최적화를 사용합니다.
유한성 검사와 깊이 계산의 계약을 유지하기 위해 부동소수점 모드는 `/fp:precise`입니다.

## 실행과 입력

```powershell
./build/windows-msvc/bin/Release/software-rasterizer.exe --model african --toon
./build/windows-msvc/bin/Release/software-rasterizer.exe --help
```

| 입력 | 동작 |
| --- | --- |
| Space | 삼각형 → 와이어프레임 → 점 |
| Q / W / E / R | 그림자 / SSAO / AA / Toon 전환 |
| 왼쪽 마우스 드래그 | 광원 방향 변경 |
| 마우스 휠 | 카메라 거리 변경 |
| Escape | 종료 |
| 창 크기 변경 | framebuffer와 카메라 종횡비 갱신 |

보조 프로파일러 창에는 FPS, 프레임 시간, CPU 처리 시간과 SHADOW / GEOMETRY / SSAO / AA의 이름·ms 수치가 표시됩니다.
FPS와 FRAME은 창 표시·대기를 포함한 실제 루프 간격이며, CPU TOTAL은 scene 갱신과 렌더링 시간입니다.
PRESENT는 프로파일러 그리기와 두 창 갱신, WAIT는 명시적인 프레임 대기 시간입니다.
수치는 직전에 완료된 프레임 기준입니다. 그래프와 패스 막대에는 60fps에 해당하는 16.67ms 기준이 있습니다.
Debug/Release, 해상도, 카메라 거리, 작업 스레드 수와 Q/W/E/R의 ON/OFF도 텍스트로 표시합니다.
`--no-profiler`로 보조 창을 생략합니다.

## 창 없는 렌더링

실제 앱과 같은 Scene·Renderer를 사용합니다.
프레임 번호로 회전 각도를 정하므로 같은 옵션의 출력 비교가 가능합니다.

```powershell
./build/windows-msvc/bin/Release/software-rasterizer.exe --headless --frames 1 --width 640 --height 360 --model diablo --output build/diablo.bmp
./build/windows-msvc/bin/Release/software-rasterizer.exe --headless --model african --toon --output build/african.bmp
```

`--model`은 `diablo`, `african`, `cube`, `sphere`를 지원합니다.
`--workers 0`은 호출 스레드만 사용하며, 양수는 추가 작업 스레드 수입니다.
`--no-shadows`, `--no-ssao`, `--no-aa`, `--wireframe`, `--points`로 각 경로를 확인할 수 있습니다.
`--frames N`은 GUI에서도 N프레임 뒤 종료합니다. 출력은 32비트 BMP입니다.

## 성능 측정

`--benchmark`는 준비 구간 뒤 같은 회전 순서를 측정하고 평균·p95·p99 프레임 시간과 패스별 비용을 출력합니다.

```powershell
./build/windows-msvc/bin/Release/software-rasterizer.exe --benchmark --frames 600 --warmup 60
```

기본 해상도와 그림자·SSAO·AA를 유지한 최적화 내용, 측정 결과, 실제 창 표시까지 확인하는 방법은
[성능 측정 안내](docs/PERFORMANCE.md)에 정리했습니다.

## 검증 범위와 한계

CTest의 9개 항목은 수학, 래스터라이저, 자산·씬, 후처리, 셰이더, 렌더러, 앱 상태, 프로파일러, headless 실행을 검사합니다.
그 안에서 동차 클리핑, 공유 모서리의 중복 합성, 원근 보간, reflection과 shear,
잘못된 입력의 거부, 작업 예외, 홀수 해상도, 단일·병렬 실행 결과 일치 등을 확인합니다.
테스트는 Release에서도 검사를 생략하지 않습니다.

타일 병렬 렌더링과 SIMD는 유지했습니다. 점·선은 겹치는 픽셀의 경쟁을 피하도록 순차 래스터화합니다.
정렬 기반 투명 합성, 고정 범위 shadow map, 화면 공간 AO, byte 색상의 조명 계산은 학습용 근사입니다.
교차하는 투명 표면, 정확한 색 공간 처리, PBR, 범용 OBJ/MTL 지원은 완성되어 있지 않습니다.
현재 검증 대상은 Windows x64/MSVC이며 ARM/NEON 실행은 확인하지 않았습니다.
