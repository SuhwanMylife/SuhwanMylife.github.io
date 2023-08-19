---
layout: post
title: "GitHub 블로그 만들기"
tags:
  - GitHub
  - Blog
---

<br>

## 초기 과정

우선 아래 링크의 테마 적용 과정을 활용해서 만들고 지우고를 반복했다.

[[Git] Git 블로그 만들기(3) jekyll theme 적용](https://velog.io/@dlruddms5619/Git-Git-%EB%B8%94%EB%A1%9C%EA%B7%B8-%EB%A7%8C%EB%93%A4%EA%B8%B03-jekyll-theme-%EC%A0%81%EC%9A%A9)

내가 적용한 테마는 Kiko-Now이다. 

[Kiko Now (GitHub)](https://github.com/aweekj/kiko-now)

우선 깔끔하고, 내가 원했던 기능만 최소한으로 적용돼 있어 선택했다.
github에 들어가서 Download ZIP하여 로컬에 적용하고 github에 commit하는 방식이다.

<br>

### github에 push한 후 css 깨짐 현상

그렇게 만들고 지우고를 반복하면서 잘 올린 것 같은데, css가 깨져서 html 결과물만 보이는 것이다.
구글링 후, 아래 작업을 거치고 나니 정상적으로 동작했다.

[Git blog 만들 때 jekyll 테마 적용 잘 안될 때](https://ventus.tistory.com/23)

<br>

### 블로그 꾸미기

앞서 다운받은 파일을 보면 _posts에 들어가는 게시글 외에 다른 md 확장자의 파일들을 볼 수 있다. 나의 경우 about.md와 index.markdown 파일이 있는데, 이들은 블로그의 기타 다른 페이지를 꾸미고자 할 때 수정해야 할 파일들이다.
<br>기타 다른 글꼴을 수정하거나 글자 크기 조절하는 경우에는 scss 확장자의 파일을 수정하면 된다.

<br>

### 이미지 삽입

처음에 아래 글을 보고 '뭐지? 저렇게 하면 된다고?'했는데 진짜 된다. 하고 나니 아무것도 아닌...

[Git Blog에 이미지 업로드 Tip](https://cnu-jinseop.tistory.com/21)

<br>

### 수식 출력

내가 게시하고자 하는 글에 수식이 많았는데, 이를 그대로 넣으니 Notion에서 출력되던 수식의 형태로 출력이 되지 않는 것이다. (참고로, markdown 글 작성 시 Notion으로 초안을 만든다.) <br>
찾아보니, 아래 글에서 제시한 과정을 거쳐야 제대로 출력이 가능함을 알게 됐다.

[[GitHub] Github 블로그 수식 추가 (kiko-now)](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=prt1004dms&logNo=221525385428)

<br>

위의 과정을 거치니, GitHub 블로그에 포스팅하는 과정은 어느정도 최적화가 되었다. 당분간은 내가 여태껏 공부하며 Notion에 끄적였던 내용들을 정리하며 포스팅을 해나갈 예정이다.