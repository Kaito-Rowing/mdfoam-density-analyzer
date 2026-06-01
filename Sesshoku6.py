import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

def fit_sphere_algebraic(x, y, z):
    """
    点群 (x, y, z) に最小二乗法で球をフィッティングします。
    戻り値: 中心座標 (xc, yc, zc), 半径 R
    """
    try:
        if len(x) < 4: return None, None, None, None
        
        # 球の方程式: x^2 + y^2 + z^2 + Dx + Ey + Fz + G = 0
        # 行列 A * [D, E, F, G].T = b を解く
        
        A = np.c_[x, y, z, np.ones(len(x))]
        b = -(x**2 + y**2 + z**2)
        
        # 最小二乗法
        C, residues, rank, s = np.linalg.lstsq(A, b, rcond=None)
        D, E, F, G = C
        
        xc = -D / 2
        yc = -E / 2
        zc = -F / 2
        
        # R = sqrt(xc^2 + yc^2 + zc^2 - G)
        term = xc**2 + yc**2 + zc**2 - G
        R = np.sqrt(max(0, term))
        
        return xc, yc, zc, R
    except Exception as e:
        print(f"Fitting Error: {e}")
        return None, None, None, None

def get_contact_angle(xc, zc, R, z_base):
    """
    球パラメータと基板高さから接触角を計算します。
    (球フィッティングの場合、左右の区別はなく全周で同じ角度とみなします)
    """
    if xc is None or zc is None or R is None or R < 1e-12:
        return np.nan, np.nan
    
    # 1. cos(theta) を計算
    # 接触点での法線ベクトルとZ軸のなす角、あるいは幾何学的な高さ関係から算出
    # z_contact = z_base
    # cos(theta) = (z_base - zc) / -R  (下向きが接触の場合) -> 一般的には (z_base - zc) / R の逆余弦など
    
    # 図形的に:
    # 接触角 theta は、接点における接線と基板のなす角。
    # 重心高さ zc, 半径 R, 基板 z_base
    # cos(180 - theta) = (z_base - zc) / R  (液滴が上に凸の場合)
    # -cos(theta) = (z_base - zc) / R
    # theta = arccos( -(z_base - zc) / R )
    
    # あるいはシンプルに既存コードのロジックを踏襲
    cos_arg = (z_base - zc) / R
    
    # 2. 範囲外の値をクリップ
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
        
    # 3. 角度計算
    theta_rad = np.arccos(cos_arg)
    theta_deg = np.degrees(theta_rad)
    
    # 接触半径 (基板上での接触円の半径)
    r_contact = R * np.sin(theta_rad)
    
    return theta_deg, r_contact

def select_valid_columns_3d(df):
    """
    データフレームから数値列を抽出し、変動がある列をX, Y, Zとして返します。
    """
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 列数が3未満の場合は処理不能
    if df_numeric.shape[1] < 3:
        print("エラー: 3次元データ(x, y, z)が見つかりません。")
        return None, None, None
    
    # 標準偏差を計算
    stds = df_numeric.std()
    
    # 標準偏差が極端に小さい（定数列）を除外
    # 実データによってはノイズ程度しか動かない列があるかもしれないので、
    # stdが大きい順にトップ3を取るのが安全
    valid_cols = stds.sort_values(ascending=False).index[:3]
    
    # オリジナルの並び順に戻す (X, Y, Zの順序を保持するため)
    # データが [Time, X, Y, Z] などの場合に対応
    original_cols = df_numeric.columns.tolist()
    sorted_cols = sorted(valid_cols, key=lambda x: original_cols.index(x))
    
    df_final = df_numeric[sorted_cols]
    
    x = df_final.iloc[:, 0].values
    y = df_final.iloc[:, 1].values
    z = df_final.iloc[:, 2].values
    
    return x, y, z

def analyze_single_file(target_csv, output_dir='.', fit_range=(0.25, 1.0), 
                        xlim=None, ylim=None, save_fig=True):
    """
    1つのCSVファイルを解析し、結果を表示・保存するメイン関数です。
    """
    if not os.path.exists(target_csv):
        print(f"エラー: ファイルが見つかりません -> {target_csv}")
        return None, None

    filename = os.path.basename(target_csv)
    basename = os.path.splitext(filename)[0]
    
    try:
        df = pd.read_csv(target_csv)
        # === [変更] 3次元データ (x, y, z) を取得 ===
        x, y, z = select_valid_columns_3d(df)
        if x is None: return filename, None
        
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みまたは列抽出に失敗しました。\n{e}")
        return filename, None

    # 高さ方向のデータ範囲を取得
    z_min, z_max = np.min(z), np.max(z)
    z_height = z_max - z_min
    
    if z_height == 0:
        print("エラー: Z方向のデータ変化がありません。")
        return filename, None

    # --- 基板高さの決定 ---
    z_base_thresh = z_min + 0.05 * z_height
    base_points_mask = z <= z_base_thresh
    base_z = z[base_points_mask]
    z_base = np.mean(base_z) if len(base_z) > 0 else z_min
        
    # --- フィッティング用データの抽出 ---
    z_fit_lower = z_min + fit_range[0] * z_height
    z_fit_upper = z_min + fit_range[1] * z_height
    mask_fit = (z >= z_fit_lower) & (z <= z_fit_upper)
    
    x_fit, y_fit, z_fit = x[mask_fit], y[mask_fit], z[mask_fit]
    
    if len(x_fit) < 4:
        print(f"エラー: フィッティング対象のデータが不足しています。")
        return filename, None

    # --- 球フィッティング ---
    xc, yc, zc, R = fit_sphere_algebraic(x_fit, y_fit, z_fit)
    
    res_theta = np.nan
    
    # 座標変換（可視化用）
    # Zは基板を0にする
    z_offset = z_base
    
    # --- 結果計算 ---
    if xc is not None:
        theta_deg, r_contact = get_contact_angle(xc, zc, R, z_base)
        res_theta = theta_deg
    
    # === プロット作成 (2次元断面に投影) ===
    # 球の中心軸 (xc, yc) からの距離 r を計算して、r-z 平面にプロットします
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 中心軸からの距離を計算
    # フィッティング前の全データ
    r_all = np.sqrt((x - xc)**2 + (y - yc)**2) if xc is not None else np.zeros_like(x)
    # X座標の正負の分布を見るために、便宜的に元のX座標の符号をrに付与して左右に展開することも可能だが、
    # 球フィッティングなので「中心からの距離(r)」vs「高さ(z)」で右半分にプロットするのが一般的。
    # ここでは、視覚的にわかりやすくするため、rを左右対称にプロット(-r と +r)して球の断面に見せます。
    
    # データ点 (間引いて表示)
    step = max(1, len(x) // 2000) # 点が多すぎると重いので間引く
    ax.scatter(r_all[::step], z[::step] - z_offset, s=10, c='gray', alpha=0.3, label='Points (Radial)')
    # 左側にもミラー表示（形状確認用）
    ax.scatter(-r_all[::step], z[::step] - z_offset, s=10, c='gray', alpha=0.3)

    # フィッティングに使った点
    r_fit = np.sqrt((x_fit - xc)**2 + (y_fit - yc)**2)
    ax.scatter(r_fit, z_fit - z_offset, s=15, c='blue', alpha=0.5, label='Fit Range')
    ax.scatter(-r_fit, z_fit - z_offset, s=15, c='blue', alpha=0.5)

    # 基板ライン
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    
    # フィッティング円（球の断面）の描画
    if not np.isnan(res_theta):
        # 描画用の角度範囲
        sin_phi = (z_base - zc) / R
        sin_phi = np.clip(sin_phi, -1.0, 1.0)
        phi_start = np.arcsin(sin_phi) # 基板との交点
        phi_end = np.pi - phi_start    # 反対側
        
        theta_range = np.linspace(phi_start, phi_end, 100)
        
        # 断面円の座標 (中心軸からの距離 r_circle vs 高さ z_circle)
        # 球の断面なので、半径Rの円を描けば良い
        # r = R * cos(theta), z = R * sin(theta) + zc
        
        r_arc = R * np.cos(theta_range) # これはX方向の変位に対応
        z_arc = zc + R * np.sin(theta_range)
        
        # r_arc は -R to R の範囲になるので、横軸 X (中心からのズレ) としてそのままプロット
        ax.plot(r_arc, z_arc - z_offset, 'r-', linewidth=3, label=f'Sphere Fit: {res_theta:.1f}°')
        
        # 接触点 (r = +/- r_contact)
        ax.plot(r_contact, 0, 'rx', markersize=12, markeredgewidth=2)
        ax.plot(-r_contact, 0, 'rx', markersize=12, markeredgewidth=2)

    ax.set_aspect('equal', adjustable='box')
    
    # タイトル・ラベル設定
    title_text = f"{basename}\n{res_theta:.1f}°" if not np.isnan(res_theta) else f"{basename}\nFit Failed"
    ax.set_title(title_text, fontsize=24, fontweight='bold')
    ax.set_xlabel("Radial Distance from Center", fontsize=18)
    ax.set_ylabel("Z (Base=0)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right')

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    plt.tight_layout()
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"result_{basename}.png")
        plt.savefig(save_path, dpi=150)

    plt.close(fig)
    print(f"[{filename}] 解析完了 -> G: {res_theta:.2f}°")
    
    return filename, res_theta

def main():
    # 画像の固定スケール設定
    # 球フィッティングの断面表示用にX軸範囲を調整
    fixed_xlim = (-3e-9, 3e-9) 
    fixed_ylim = (-0.5e-9, 3.5e-9)
    
    # 解析対象ファイルのリストを作成
    target_files = []
    
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.isdir(arg_path):
            target_files = glob.glob(os.path.join(arg_path, "*.csv"))
        elif os.path.isfile(arg_path):
            target_files = [arg_path]
        else:
            print(f"エラー: 指定されたパスが見つかりません: {arg_path}")
            return
    else:
        target_files = glob.glob("*.csv")
        summary_name = "summary_contact_angles_sphere.csv"
        if summary_name in target_files:
            target_files.remove(summary_name)

    if not target_files:
        print("解析対象のCSVファイルが見つかりませんでした。")
        return

    print(f"--- 球フィッティング解析開始: {len(target_files)} 件 ---")
    
    summary_results = []
    output_dir = "results_sphere"
    
    for csv_file in sorted(target_files):
        fname, angle = analyze_single_file(
            csv_file, 
            output_dir=output_dir,
            fit_range=(0.5, 1.0), # トップ部分を重点的に使う場合
            xlim=fixed_xlim, 
            ylim=fixed_ylim
        )
        if fname is not None:
            summary_results.append({"Filename": fname, "ContactAngle": angle})

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = "summary_contact_angles_sphere.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n全処理が完了しました。")
        print(f"グラフ画像保存先: ./{output_dir}/")
        print(f"集計結果CSV: {summary_path}")

if __name__ == "__main__":
    main()