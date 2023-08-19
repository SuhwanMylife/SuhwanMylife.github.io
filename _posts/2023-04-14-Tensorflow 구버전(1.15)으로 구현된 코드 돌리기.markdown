---
layout: post
title: "Tensorflow 구버전(1.15)으로 구현된 코드 돌리기"
tags:
  - Tensorflow
  - Docker
---

<br>

결론부터 말하면 실패했다. 여러 번 시도했음에도 불구하고 python 코드에서 웬 C++ 에러가 뜨는지…

- 배치 사이즈가 너무 크다는 피드백을 받았는데 관련해서 고민을 더 해봐야겠다.

오늘 간략하게 어떤 시도를 했고 왜 실패했는지 정리해보도록 하겠다. (정리하는 겸 마지막 시도…)

목적은 다음과 같다.

> Tensorflow 1.15버전에서 구현된 코드 실행시키기
> 

코드는 아래 GitHub 링크에서 가져왔다.

[https://github.com/mlds-lab/interp-net](https://github.com/mlds-lab/interp-net)

<br>

현재 나는 Tensorflow 1.15 코드를 돌리기 위한 환경이 필요하다. 내가 가진 것은 GPU 구동이 가능한 Docker 서버. 코드 구동을 위한 개발환경 구축을 위해 어떤 시도를 했는지 정리해보도록 하겠다.

<br>

### 첫 번째 시도: Docker Hub에서 제공하는 공식 Tensorflow 이미지 활용

원하는 이미지 활용을 위해 tag를 알아야 한다. docker hub로 가자.

검색창에 tensorflow를 치면 아래에 tensorflow/tensorflow가 나온다. 여기에 Tensorflow에서 공식적으로 제공하는 docker image가 있다.

![Untitled](https://user-images.githubusercontent.com/70688382/232229746-6a2b28c1-3e9d-4bda-8713-6ba3c88f1814.png)

찾아본 바로, Tensorflow가 Docker hub에서는 본인들이 제공하는 image를 활용하기를 권장하는 듯 하다. 환경구축 시에 참고하길 바란다.

이후, Tags로 들어가서 아래 검색창에 ‘1.15.0’을 입력 후 엔터.

![Untitled 1](https://user-images.githubusercontent.com/70688382/232229764-f5c03c3b-e04f-465e-ab7c-02ee6b29b663.png)

입력하면 여러 tag들이 나오는데, 구동하고자 하는 코드가 어디서 구동되는 코드인지 잘 확인 후에 골라야 한다. tensorflow 1버전에서는 cpu와 gpu 버전을 나눠놨는데, 코드에서 gpu를 쓰고 있고,  python 3버전을 요구했기 때문에 다음 이미지를 선택했다.

![Untitled 2](https://user-images.githubusercontent.com/70688382/232229780-c6d91b0f-1724-43c3-bb7e-e64c0e3a58e3.png)

이제 위 이미지를 활용하여 컨테이너를 만들어보자. 명령어는 다음과 같다.

```powershell
docker run -dit --gpus all --name tensorflow_docker_test \
-v ~/interp-net:/workspace tensorflow/tensorflow:1.15.0-gpu-py3
```

명령어에 대해 일일이 설명하면 내용이 불필요하게 길어진다 판단하여 필요하다 싶은 부분만 간략히 설명하도록 하겠다.

—gpus all: 컨테이너에 gpu를 연결한다.

-v: 컨테이너 외부의 디렉토리를 컨테이너 내부에서 사용 가능하게끔 한다. 위 코드 기준으로 :(콜론) 전의 경로는 컨테이너 외부의 디렉토리 경로이고, 오른쪽이 컨테이너 내부 디렉토리 경로이다.

위 명령어를 입력한 후 생긴 컨테이너에 접속하자. 명령어는 다음과 같다.

```powershell
docker exec -it tensorflow_docker_test /bin/bash
```

실행 중인 컨테이너 tensorflow_docker_test에 bash쉘로 들어가겠다는 의미로 이해하면 되겠다.

bash쉘로 들어간 후 이전에 -v로 volume 지정한 디렉토리를 찾을 수 있다.

![Untitled 3](https://user-images.githubusercontent.com/70688382/232229791-1dff34a4-fce4-4b6c-88c9-ea46ec97cbb8.png)

이후 `requirements.txt` 파일을 활용하여 코드에서 요구하는 환경으로 맞춰준다.

```powershell
pip install -r requirements.txt
```

이후 코드를 돌려봤으나 다음 에러를 보내며 돌아가지 않았다. 이전에 다른 수많은 에러가 있었으나 코드를 수정해가며 어찌저찌 해결했다. 근데 이 에러는 메모리와 관련된 데다가 C++ 단계에서의 에러라 여기서 더이상 나아가지 못했다.

![Untitled 4](https://user-images.githubusercontent.com/70688382/232229805-90b014d2-4952-41b2-a55f-e15793739818.png)


다른 곳에서 문제를 찾아보고자 했다. tensorflow image를 사용하면서 계속 걸렸던 부분이 image에서 제공하는 python 버전은 3.6버전이다. 코드에서 요구하는 버전은 3.7이다. 0.1의 차이가 유의미할까 싶긴 했는데, 아무래도 이 문제가 아닐까 싶어 해결하고자 했다.

위 문제를 해결하는 방법으로 여러가지를 생각해봤는데,

1. 컨테이너 내에서 새로운 python 버전 수정
2. Docker Python image 받아 tensorflow 설치
3. Docker Conda image 받아 conda 환경에서 개발환경 구축
4. Docker CUDA image 받아 개발환경 구축

우선 확실하진 않으나, Docker 환경에서 tensorflow 사용 시에 Docker image를 사용하라는 이유가 다른 방식으로 사용하는 경우 **라이브러리 충돌** 문제가 다발하기 때문이 아닐까 싶다. 이런 이유에서 2번 방법은 많은 어려움이 있지 않을까 싶다. 1번은 결국에 라이브러리를 새로 설치해야 하는 방법이기 때문임을 확인했기 때문에 제외. 3번은 conda 환경에서 tensorflow를 사용하는 방식을 활용하기 위한 방법이다. 3번 방법을 써보자.

3번 활용을 위해 조사하던 과정에서 각 tensorflow 버전과 호환되는 python, cuda, cudnn 등의 버전을 확인할 수 있었는데, cuda 버전을 맞출 필요가 있음을 알았다. 마침 docker에서 여러 cuda 버전의 image를 제공하고 있음을 알았기 때문에, 

1. 먼저 원하는 cuda 버전의 image를 받아 
2. conda 환경을 구축하여 
3. 원하는 버전의 python을 받아 
4. 그 위에 tensorflow를 설치하는 방법을 써보자! ~~(얼마나 돌아온거야…)~~

### 두 번째 시도: Docker CUDA 이미지 활용

역시 docker hub에 cuda를 입력한다. nvidia/cuda를 확인한 후 클릭.

![Untitled 5](https://user-images.githubusercontent.com/70688382/232229810-7ae4f4c5-4be6-4ccc-91ad-b7d70e5e8c57.png)

10.0-base 태그를 쓰기로 했다. 내가 쓰고자 하는 tensorflow 버전에 맞는 cuda 버전이 10.0이고, base는 아무래도 가장 simple한 게 좋지 않을까하는 생각에 선택했다.

![Untitled 6](https://user-images.githubusercontent.com/70688382/232229821-1c2a16c6-6061-4570-b042-855128c68283.png)

다음 명령어로 컨테이너를 만들어 준다.

```powershell
docker run -dit --gpus all --name tensorflow_docker_test_conda -v ~/interp-net_practice:/workspace nvidia/cuda:10.0-base
```

이제 conda 환경을 구축할 텐데, 여기서부터는 아래 블로그를 참고했다. 그대로 따라하면 된다. 컨테이너 안에서 수행해야 한다.

[Docker에서 Anaconda 가상환경에 tensorflow, tensorflow-gpu 설치하기](https://benghak.github.io/2019-03-21-anaconda_tensorflow/)

```powershell
conda create --name test_1 python=3.7 # 사용하고자 하는 가상환경의 python 버전
conda activate test_1 # 가상환경 활성화
conda install -c anaconda tensorflow-gpu==1.15 # 
```

여기서 conda vs. pip가 문제였는데, 일단 나는 이외의 라이브러리는 pip에서 받았다. 코드 돌릴 때는 크게 문제가 없었다. conda에서는 다른 라이브러리와의 호환성을 고려해서 설치해준다는 장점이 있다고는 하는데, 아직 이 때문에 문제가 생긴 적은 없기에…

쨌든 pip로 필요한 라이브러리 install해준 후 코드를 돌렸다. 결과는 같다.

![Untitled 4](https://user-images.githubusercontent.com/70688382/232229805-90b014d2-4952-41b2-a55f-e15793739818.png)

정리하는 과정에서 오픈톡방에서 배치 사이즈가 너무 크다는 피드백을 받았다. 여유가 되는대로 이와 관련해서 더 찾아보고 정리해야겠다.

덕분에(?) 개발환경 구축 관련해서 좀 알 수 있는 계기가 되지 않았나 싶다. 언젠간 쓸 데가 있겠지…