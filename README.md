# operation
(project) A regression machine that learns a dataset based on an equation created using various operators

<a href='./docs/README_eng.md'><b>README written by English is here.</b></a>

## Introduction
이 프로젝트는 **인공지능에게 수식에 대한 직관을 부여할 수 있는가?** 라는 질문에서 시작되었습니다. 특정 항과 결과만 주어졌을 때, 인공지능이 수식을 추론할 수 있을지에 대한 실험입니다. 수식의 핵심은 operand 간의 관계(연산자)를 이해하고, 이를 통해 주어진 operand로부터 결과를 예측하는 능력을 기르는 것입니다. 이 실험은 AI가 연산자의 존재 없이도 operand 간의 규칙을 학습하고, 더 나아가 연산자 자체를 추론할 수 있는지를 알아보는 실험입니다.

먼저, 아래 data를 보고 X를 예측해보도록 합시다.

| operand1 | operand2 | result |
|----------|----------|--------|
|    1     |    2     |    3   |
|    2     |    3     |    5   |
|    3     |    4     |    7   |
|    4     |    5     |    **x**   |

위를 보면 `result = operand1 + operand2` 이므로, x는 9라고 예측할 수 있습니다. 이 예측 과정은 연산자가 주어지지 않더라도, 주어진 operand 간의 관계를 기반으로 한 추론 능력을 의미합니다. 우리가 수행하는 사고 과정은 다음과 같이 요약될 수 있습니다.

1. 각 operand를 **어떻게 연산해야** result가 나올까? (operator를 추론)
2. 여러 operand를 시도하여 일반적인 규칙(연산자)을 구성.
3. 구성한 연산자와 operand를 조합해 최종적으로 x 값을 도출.

이러한 사고방식을 인공지능이 학습할 수 있을지를 탐구하는 것이 이 프로젝트의 핵심입니다. 특히, operator를 주지 않고 operand만 주어진 데이터프레임을 학습시켜 패턴을 추출할 수 있는지를 실험할 것입니다.

## Object
이 프로젝트는 두 가지 주요 목표를 가집니다:
1. **Result 예측** : operator 없이 operand만 주어졌을 때, 인공지능이 operand 사이의 패턴을 학습하고, 이를 바탕으로 결과(result)를 예측할 수 있는가?
2. **Operator 예측** : operand와 result가 주어졌을 때, 인공지능이 그들 사이의 관계인 연산자(operator)를 추론할 수 있는가?

## 사용할 연산자

### 산술 연산자 (Arithmetic Operators)
- `+` : 덧셈
- `-` : 뺄셈
- `*` : 곱셈
- `/` : 나눗셈
- `%` : 나머지 연산
- `**` : 거듭제곱
- `//` : 정수 나눗셈 (몫)

### 비교 연산자 (Comparison Operators)
- `==` : 같음
- `!=` : 같지 않음
- `>` : 큼
- `<` : 작음
- `>=` : 크거나 같음
- `<=` : 작거나 같음

### 비트 연산자 (Bitwise Operators)
- `&` : 비트 AND
- `|` : 비트 OR
- `^` : 비트 XOR (배타적 OR)
- `<<` : 비트 왼쪽 시프트
- `>>` : 비트 오른쪽 시프트

### 특수 산술 연산자 (Special Arithmetic Operators)
- `divmod(a, b)` : a를 b로 나누고 몫과 나머지 반환
- `max(a, b)` : 두 수 중 큰 값
- `min(a, b)` : 두 수 중 작은 값
- `pow(a, b)` : a의 b 제곱
- `hypot(a, b)` : 두 수의 유클리드 거리 (직각 삼각형의 빗변 계산)

### 기타 수학 연산자 (Miscellaneous Mathematical Operators)
- `gcd(a, b)` : 최대공약수
- `lcm(a, b)` : 최소공배수
- `log(a, b)` : b를 밑으로 한 a의 로그
- `mod(a, b)` : a mod b
- `abs(a - b)` : 절대값 차

## Experiment Sections

아래 두 Section으로 experiment합니다.

### Section 1. Predict the result without operator
- n개의 operand가 feature, 계산 결과가 target으로 주어집니다.
- **operator는 주어지지 않으며**, 학습된 모델이 operand만으로 결과(result)를 예측해야 합니다.
- 각 operator에 맞는 데이터를 구성하고, operand 개수에 따라 파일을 준비합니다 (2 ~ 5개의 operand).
- 예시: 1, 2, 3이 feature, 6이 target이라면, 모델에 2, 3, 4를 입력했을 때 9가 result로 출력되어야 합니다 (+ 연산).
- 이를 통해 인공지능이 주어진 operand 간의 관계를 학습하고, 주어진 상황에서 결과를 추론할 수 있는지 실험합니다.

### Section 2. Predict the operator
- n개의 operand와 result가 feature, 연산자(operator)가 target입니다.
- **학습된 모델이 operand와 result를 입력받아 해당 연산자를 예측해야 합니다** (classification 문제로 접근).
- 다양한 연산자 중에서 주어진 operand와 result에 맞는 연산자가 무엇인지 모델이 분류해내는 과정을 포함합니다.
- 예를 들어, 1, 2, 3이 feature, 연산자가 +일 때, 모델은 이 규칙을 학습하여 2, 3, 5가 주어졌을 때 연산자로 '+'를 예측해야 합니다.

## 보고서
모든 실험이 종료된 후에는, 각 실험의 결과에 대해 상세한 **report**를 작성할 계획입니다. 보고서는 AI가 operand 간의 관계를 학습한 방법, 예측 성능, 각 연산자에 따른 정확도 등을 다룰 예정입니다. 이 실험은 연산자 없이 operand와 result만으로 수학적 규칙을 추론하는 AI 모델을 개발하는 것을 목표로 합니다.