---
title: Manjaro上安裝行列或其他中文輸入法 (fcitx + rime)
date: 2021-07-22 18:11:35
categories: Environment SetUp
tags: [Manjaro, InputMethod]
lang: zh-tw
mathjax:
---

每一次重新安裝好用來開發的Linux環境時，總會因為自己是使用較小眾的行列輸入法，常得在過程中的安裝和設定上掙扎許久...

以往這些紀錄都只會留存在我個人的 Evernote；受惠於網路甚多，現在決定開始把這些過程重新整理並分享給有需要的人

以下步驟也適用於倉頡、大易、速成、嘸蝦米等中文輸入法

<!--more-->

## 安裝fcitx相關package

``` bash
sudo pacman -S fcitx-im fcitx-chewing fcitx-table-extra fcitx-configtool fcitx-rime
```


- `fcitx-im`: fcitx + fcitx-qt5
  - fcitx: flexible Xontext-aware Input tool with eXtension
  - fcitx-qt5: fcitx Qt5 IM module
- `fcitx-chewing` : 注音輸入法
- `fcitx-table-extra` : 含行列, 倉頡, 大易，嘸蝦米等輸入法之宇碼表，其支援的字碼表細節可參考 [Fcitx5-table-extra](https://github.com/fcitx/fcitx5-table-extra)
- `fcitx-configtool` : 安裝fcitx圖形管理工具
- `fcitx-rime` : 安裝rime
  - rime 為跨平台的中文輸入法框架


安裝完成後，重新登入或啟動就可以使用了


## 若開機時無法自動啟動

非 KDE 的使用者可能會碰到開機沒有自動啟動的狀況，可以根據自身系統任選以下其中一個檔案做修改來達成開機啟動

``` bash
# ~/.xinitrc
...

export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMOFIFIERS=@im=fcitx

...

fcitx &

exec YOUR_WINDOWS_MANAGER
```


``` bash
# ~/.xprofile
...

export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMOFIFIERS=@im=fcitx
```


``` bash
# ~/.xsession
...

export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMOFIFIERS=@im=fcitx

...

YOUR_WINDOWS_MANAGER
```


## 使用方式
1. 右鍵點選 fcitx 的 Configure (在右下 icon 列或是直接在 Manjaro application launcher 裡尋找)

2. 按 + 來新增需要的輸入法
3. 選擇需要新增的輸入法 (若環境預設為英文語系或其他，記得要先將 Only Show Current Language 的勾勾點掉)
4. 新增完畢後可使用 `ctrl`+`space` 來切換