---
title: fcitx 行列輸入法無法使用標點符號
date: 2021-07-24 00:19:32
categories: Environment SetUp
tags: [Manjaro, InputMethod]
lang: zh-tw
mathjax:
---

設定好 fcitx 的行列輸入法時，會發現有無法正常使用標點符號的問題 (i.e. `W` + `2` 沒反應)

這部份需要更新 fcitx 的字碼表，來修改 `W` + `1` ~ `9` 的行為

<!--more-->

## 下載行列輸入法30字碼表

下載更新後的 array30.mb ([GoogleDrive](https://drive.google.com/file/d/1fDcoUeD4uBa7KtgWA8f_Ayl3nQZzud6r/view?usp=sharing) by [老刀](http://hyperrate.com/thread.php?tid=33369#33369))

> 若希望手動製作字碼表，可至官方下載字碼表原始檔 ([gcin字碼表](http://array30.sourceforge.net/files/ar30.cin))，並將其修改成 fcitx 的字碼表，再使用 fcitx 工具 `fcitx-tools` 轉為需要的格式:
> ```bash
 sudo pacman -S fcitx-tools
txt2mb array30.txt array30.mb  # Generate the table manually
 ```


## 更新字碼表

建議先將原來的字碼表做備份後，再將準備好的字碼表搬入

```bash
sudo mv /usr/share/fcitx/table/array30.mb /usr/share/fcitx/table/array30.mb.bak  # Backup
sudo mv array30.mb /usr/share/fcitx/table/array30.mb
```

## 修改 fcitx 裡候選字數的設定

這一步是為了讓配置和 Windows 一樣，將 fcitx 針對候選字數的設定變更為 10

1. 進入 fcitx 的設定 (Configure)
2. 切至全域設定 (Global Config)
3. 更新候選字數 (Candidate Word Number) 為 10


重新啟動 fcitx 或重新登入系統以載入新的字碼表就可以使用標點符號了，預設是使用 `↑` 和 `↓` 來切換上下頁
