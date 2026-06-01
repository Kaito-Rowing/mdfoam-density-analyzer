# mdFOAM Density Analyzer

mdFOAM/OpenFOAM形式の計算結果をPythonで直接読み取り、密度しきい値以上の領域について体積、等価半径、蒸発完了時刻を確認するデスクトップアプリです。

ParaView と pvpython は使いません。

## セットアップ

```powershell
python -m pip install -r requirements.txt
```

## 起動

```powershell
python app.py
```

起動後、GUIの「親フォルダ」から解析対象ケースを含むフォルダを選択してください。

## 入力データ

このリポジトリにはサンプル計算データは含めていません。

アプリは以下のどちらかの構造を想定しています。

```text
parent/
  case001/
    main/
      constant/polyMesh/
      <time>/
        rhoM_water
        ...
  case002/
    main/
      ...
```

または単一ケース:

```text
case001/
  main/
    constant/polyMesh/
    <time>/
      rhoM_water
      ...
```

再構成済みフィールド `main/<time>/<field>` を優先して読みます。存在しない場合は `main/processor*/<time>/<field>` を合算します。

## 既定の解析条件

- 密度フィールド: `rhoM_water`
- 密度しきい値: `500`
- 0判定許容値: `0`
- 連続ゼロ数: `3`
- 蒸発完了時刻: 連続ゼロ区間の最初の時刻

セル体積はOpenFOAM ASCIIの `constant/polyMesh/points`, `faces`, `owner`, `neighbour` から計算します。計算できない場合はGUIでセル体積または `dx, dy, dz` を入力してください。

## リポジトリメモ

このアプリの前提や実装上の注意は `AGENTS.md` にまとめています。
