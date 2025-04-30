import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from skimage.color import rgb2gray 
from sklearn.preprocessing import StandardScaler
from kernels import kernel 
from kpca import KPCA 
import warnings
import time

# --- THÊM IMPORT CHO ĐÁNH GIÁ ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# ---------------------------------

# === HÀM TỰ CODE PCA BẰNG SVD ===
def manual_pca(X_scaled, n_components):
    """Thực hiện PCA bằng SVD"""
    print(f"    Đang thực hiện SVD cho PCA (d={n_components})...")
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    print(f"    Đang tính scores cho PCA...")
    X_pca = U[:, :n_components] * s[:n_components]
    return X_pca
# ================================

# === HÀM VẼ GALLERY ẢNH ===
def plot_gallery(title, images, image_shape, n_col=10, n_row=4, cmap=plt.cm.gray, labels=None, target_names=None):
    """Hàm tiện ích để vẽ gallery ảnh"""
    print(f"\n--- {title} ---")
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(title, size=16)
    max_images_to_show = n_row * n_col
    for i in range(min(max_images_to_show, images.shape[0])):
        plt.subplot(n_row, n_col, i + 1)
        img = images[i]
        if len(img.shape) == 1:
            img = img.reshape(image_shape)
        plt.imshow(img, cmap=cmap, interpolation='nearest')
        if labels is not None and target_names is not None:
             plt.title(target_names[labels[i]].split()[-1], size=9) # Lấy họ cho LFW
        elif labels is not None:
             display_label = labels[i] + 1 # Nhãn Olivetti 0-39 -> 1-40
             plt.title(f"Người {display_label}", size=9)
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
# ===========================

# === HÀM ĐÁNH GIÁ K-NN CV ===
def evaluate_knn_cv(X_features, y_labels, k=5, cv_folds=5):
    """Tính accuracy trung bình của k-NN bằng Cross-Validation"""
    if X_features is None or np.isnan(X_features).any() or np.isinf(X_features).any():
        return np.nan
    try:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Sử dụng n_jobs=-1 để tăng tốc nếu có thể
        cv_scores = cross_val_score(knn, X_features, y_labels, cv=cv_folds, n_jobs=-1)
        return np.mean(cv_scores)
    except Exception as e:
        print(f"      Lỗi khi tính CV: {e}")
        return np.nan
# ===========================

if __name__ == "__main__":
    start_time_total = time.time()

    # ==========================================================
    # PHẦN 1: XỬ LÝ VÀ SO SÁNH TRÊN TOÀN BỘ OLIVETTI
    # ==========================================================
    print("\n\n===== BẮT ĐẦU XỬ LÝ BỘ DỮ LIỆU OLIVETTI =====")
    olivetti = fetch_olivetti_faces()
    X_ol = olivetti.data
    y_ol = olivetti.target # Nhãn từ 0 đến 39
    images_ol = olivetti.images
    n_samples_ol, n_features_ol = X_ol.shape
    h_ol, w_ol = images_ol[0].shape
    n_classes_ol = len(np.unique(y_ol))

    print(f"\nOlivetti - Số lượng mẫu: {n_samples_ol}")
    print(f"Olivetti - Số lượng đặc trưng: {n_features_ol}")
    print(f"Olivetti - Số lượng lớp (người): {n_classes_ol}")
    print(f"Olivetti - Kích thước ảnh: ({h_ol}, {w_ol})")

    # Hiển thị ảnh mẫu Olivetti
    plot_gallery("Ảnh mẫu Olivetti (40 người - mỗi người 1 ảnh)", images_ol[::10], (h_ol, w_ol), n_col=10, n_row=4, labels=y_ol[::10])

    # Chuẩn hóa Olivetti
    print("\n--- Chuẩn hóa dữ liệu Olivetti ---")
    scaler_ol = StandardScaler()
    X_ol_scaled = scaler_ol.fit_transform(X_ol)

    # --- Định nghĩa tham số thử nghiệm Olivetti ---
    d_list_ol = [50, 100, 150, 200, 250] # Thử nhiều d cho Olivetti
    gamma_kpca_ol_fixed = 3e-7 
    print(f"\nOlivetti: Thử nghiệm giảm chiều với d = {d_list_ol}")
    print(f"Olivetti: Sử dụng gamma = {gamma_kpca_ol_fixed:.1E} cố định cho KPCA")
    print("Olivetti: Đánh giá bằng k-NN Cross-Validation (k=5, cv=5)")

    pca_ol_results = {}
    kpca_ol_results = {}

    # --- Vòng lặp qua d cho Olivetti ---
    for n_components_ol in d_list_ol:
        print(f"\n----- Đang xử lý Olivetti với d = {n_components_ol} -----")

        # PCA Tự Code
        print(f"  --- Thực hiện PCA Tự Code (d={n_components_ol}) ---")
        pca_knn_acc_ol = np.nan
        pca_error_ol = None
        try:
            X_pca_ol = manual_pca(X_ol_scaled, n_components_ol)
            print(f"    Đang tính k-NN CV cho PCA...")
            pca_knn_acc_ol = evaluate_knn_cv(X_pca_ol, y_ol)
            print(f"      Acc k-NN CV (PCA): {pca_knn_acc_ol:.4f}")
        except Exception as e:
            print(f"    LỖI khi chạy PCA với d={n_components_ol}: {e}")
            pca_error_ol = str(e)
        pca_ol_results[n_components_ol] = {'knn_accuracy': pca_knn_acc_ol, 'error': pca_error_ol}

        # KPCA
        print(f"\n  --- Thực hiện KPCA (gamma={gamma_kpca_ol_fixed:.1E}, d={n_components_ol}) ---")
        kpca_knn_acc_ol = np.nan
        kpca_error_ol = None
        try:
            ker_ol = kernel(gamma=gamma_kpca_ol_fixed)
            kernel_func_ol = ker_ol.rbf
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=UserWarning)
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 kpca_eval_ol = KPCA(X_ol_scaled.T, kernel_func_ol, n_components_ol)
                 X_ol_kpca = kpca_eval_ol.project().T

            if np.isnan(X_ol_kpca).any() or np.isinf(X_ol_kpca).any():
                print("    LỖI: KPCA tạo ra NaN/Inf.")
                kpca_error_ol = 'NaN/Inf in KPCA result'
            else:
                print(f"    Đang tính k-NN CV cho KPCA...")
                kpca_knn_acc_ol = evaluate_knn_cv(X_ol_kpca, y_ol)
                print(f"      Acc k-NN CV (KPCA): {kpca_knn_acc_ol:.4f}")
        except Exception as e:
            print(f"    LỖI GẶP PHẢI KPCA: {e}")
            kpca_error_ol = str(e)
        kpca_ol_results[n_components_ol] = {'knn_accuracy': kpca_knn_acc_ol, 'error': kpca_error_ol}

    # ==========================================================
    # Phần 2: Xử lý và So sánh trên LFW Subset (Không Resize)
    # ==========================================================
    print("\n\n===== BẮT ĐẦU XỬ LÝ BỘ DỮ LIỆU LFW =====")
    print("--- Đang tải dữ liệu LFW (Kích thước gốc) ---")
    try:
        lfw_people = fetch_lfw_people(min_faces_per_person=70, color=False)
        X_lfw = lfw_people.data
        y_lfw = lfw_people.target
        images_lfw = lfw_people.images
        h_lfw, w_lfw = images_lfw.shape[1:]
        n_samples_lfw, n_features_lfw = X_lfw.shape
        target_names_lfw = lfw_people.target_names
        n_classes_lfw = target_names_lfw.shape[0]
        print(f"\nThông tin dataset LFW:")
        print("Số lượng mẫu được chọn:", n_samples_lfw)
        print("Số lượng đặc trưng (pixel) gốc:", n_features_lfw)
        print("Số lượng lớp (người) được chọn:", n_classes_lfw)
        print("Kích thước ảnh:", (h_lfw, w_lfw))
    except Exception as e:
             print(f"Lỗi khi tải LFW: {e}. Bỏ qua phần LFW.")
             n_samples_lfw = 0 # Đặt cờ lỗi

    pca_lfw_results = {}
    kpca_lfw_results = {}

    if n_samples_lfw > 0:
        # Hiển thị ảnh mẫu LFW
        plot_gallery("Ảnh mẫu LFW", images_lfw, (h_lfw, w_lfw), n_col=8, n_row=4, labels=y_lfw, target_names=target_names_lfw)

        # Chuẩn hóa LFW
        print("\n--- Chuẩn hóa dữ liệu LFW ---")
        scaler_lfw = StandardScaler()
        X_lfw_scaled = scaler_lfw.fit_transform(X_lfw)

        # --- Định nghĩa tham số thử nghiệm LFW ---
        d_list_lfw = [50, 100, 150, 200, 250]
        gamma_kpca_lfw_fixed = 3e-7 # Gamma cho LFW
        print(f"\nThử nghiệm giảm chiều LFW với d = {d_list_lfw}")
        print(f"Sử dụng gamma = {gamma_kpca_lfw_fixed:.1E} cố định cho KPCA")
        print("Đánh giá bằng k-NN Cross-Validation (k=5, cv=5)")
        print("\nCẢNH BÁO: Quá trình này sẽ mất RẤT nhiều thời gian!")

        # === VÒNG LẶP QUA CÁC SỐ CHIỀU d CHO LFW ===
        for n_components_compare_lfw in d_list_lfw:
            print(f"\n----- Đang xử lý LFW với d = {n_components_compare_lfw} -----")

            # --- PCA TỰ CODE TRÊN LFW ---
            print(f"  --- Thực hiện PCA Tự Code (d={n_components_compare_lfw}) ---")
            pca_knn_acc = np.nan
            pca_error = None
            try:
                X_pca_lfw = manual_pca(X_lfw_scaled, n_components_compare_lfw)
                print(f"    Kích thước dữ liệu LFW sau PCA: {X_pca_lfw.shape}")
                print(f"    Đang tính k-NN CV cho PCA (d={n_components_compare_lfw})...")
                pca_knn_acc = evaluate_knn_cv(X_pca_lfw, y_lfw)
                print(f"      Acc k-NN CV (PCA): {pca_knn_acc:.4f}")
            except Exception as e:
                print(f"    LỖI khi chạy PCA với d={n_components_compare_lfw}: {e}")
                pca_error = str(e)
            pca_lfw_results[n_components_compare_lfw] = {'knn_accuracy': pca_knn_acc, 'error': pca_error}

            # --- KPCA TRÊN LFW ---
            print(f"\n  --- Thực hiện KPCA (gamma={gamma_kpca_lfw_fixed:.1E}, d={n_components_compare_lfw}) ---")
            kpca_knn_acc = np.nan
            kpca_error = None
            try:
                ker_lfw = kernel(gamma=gamma_kpca_lfw_fixed)
                kernel_func_lfw = ker_lfw.rbf
                # print(f"    Đang tính KPCA...")
                with warnings.catch_warnings():
                     warnings.simplefilter("ignore", category=UserWarning)
                     warnings.simplefilter("ignore", category=RuntimeWarning)
                     kpca_eval_lfw = KPCA(X_lfw_scaled.T, kernel_func_lfw, n_components_compare_lfw)
                     X_kpca_lfw = kpca_eval_lfw.project().T

                if np.isnan(X_kpca_lfw).any() or np.isinf(X_kpca_lfw).any():
                    print("    LỖI: KPCA tạo ra NaN/Inf.")
                    kpca_error = 'NaN/Inf in KPCA result'
                else:
                    # print(f"    KPCA xong. Đang đánh giá...")
                    print(f"    Đang tính k-NN CV cho KPCA (d={n_components_compare_lfw})...")
                    kpca_knn_acc = evaluate_knn_cv(X_kpca_lfw, y_lfw)
                    print(f"      Acc k-NN CV (KPCA): {kpca_knn_acc:.4f}")

            except Exception as e:
                print(f"    LỖI GẶP PHẢI (gamma={gamma_kpca_lfw_fixed}, d={n_components_compare_lfw}): {e}")
                kpca_error = str(e)

            kpca_lfw_results[n_components_compare_lfw] = {'knn_accuracy': kpca_knn_acc, 'error': kpca_error}

    # === IN KẾT QUẢ TÓM TẮT CUỐI CÙNG ===
    print("\n\n==========================================================")
    print(" Tóm tắt Kết quả So sánh Cuối cùng (k-NN CV Accuracy)")
    print("==========================================================")

    # --- Olivetti ---
    print(f"\n--- Olivetti (40 người, KPCA g={gamma_kpca_ol_fixed:.1E}) ---")
    print(f"{'Số chiều (d)':<15} | {'PCA Acc k-NN':<15} | {'KPCA Acc k-NN':<15}")
    print("----------------------------------------------------------")

    best_d_pca_knn_ol = -1
    best_acc_pca_knn_ol = -1
    best_d_kpca_knn_ol = -1
    best_acc_kpca_knn_ol = -1

    for d_val in d_list_ol:
        pca_metrics = pca_ol_results.get(d_val, {'knn_accuracy': np.nan, 'error': 'N/A'})
        kpca_metrics = kpca_ol_results.get(d_val, {'knn_accuracy': np.nan, 'error': 'N/A'})

        pca_knn_str = f"{pca_metrics['knn_accuracy']:.4f}" if not np.isnan(pca_metrics['knn_accuracy']) else "Lỗi" if pca_metrics['error'] else "N/A"
        kpca_knn_str = f"{kpca_metrics['knn_accuracy']:.4f}" if not np.isnan(kpca_metrics['knn_accuracy']) else "Lỗi" if kpca_metrics['error'] else "N/A"

        print(f"{d_val:<15} | {pca_knn_str:<15} | {kpca_knn_str:<15}")

        if not np.isnan(pca_metrics['knn_accuracy']) and pca_metrics['knn_accuracy'] > best_acc_pca_knn_ol:
            best_acc_pca_knn_ol = pca_metrics['knn_accuracy']
            best_d_pca_knn_ol = d_val
        if not np.isnan(kpca_metrics['knn_accuracy']) and kpca_metrics['knn_accuracy'] > best_acc_kpca_knn_ol:
            best_acc_kpca_knn_ol = kpca_metrics['knn_accuracy']
            best_d_kpca_knn_ol = d_val

    print("----------------------------------------------------------")
    print("Olivetti - Số chiều tốt nhất (k-NN Accuracy):")
    if best_d_pca_knn_ol != -1: print(f"  - PCA (Tự code SVD): d={best_d_pca_knn_ol} (Acc: {best_acc_pca_knn_ol:.4f})")
    if best_d_kpca_knn_ol != -1: print(f"  - KPCA (RBF):         d={best_d_kpca_knn_ol} (Acc: {best_acc_kpca_knn_ol:.4f})")
    print("==========================================================")

    # --- LFW ---
    if n_samples_lfw > 0: # Chỉ in nếu LFW được xử lý
        print(f"\n--- LFW (7 người, KPCA g={gamma_kpca_lfw_fixed:.1E}) ---")
        print(f"{'Số chiều (d)':<15} | {'PCA Acc k-NN':<15} | {'KPCA Acc k-NN':<15}")
        print("----------------------------------------------------------")

        best_d_pca_knn_lfw = -1
        best_acc_pca_knn_lfw = -1
        best_d_kpca_knn_lfw = -1
        best_acc_kpca_knn_lfw = -1

        for d_val in d_list_lfw:
            pca_metrics = pca_lfw_results.get(d_val, {'knn_accuracy': np.nan, 'error': 'N/A'})
            kpca_metrics = kpca_lfw_results.get(d_val, {'knn_accuracy': np.nan, 'error': 'N/A'})

            pca_knn_str = f"{pca_metrics['knn_accuracy']:.4f}" if not np.isnan(pca_metrics['knn_accuracy']) else "Lỗi" if pca_metrics['error'] else "N/A"
            kpca_knn_str = f"{kpca_metrics['knn_accuracy']:.4f}" if not np.isnan(kpca_metrics['knn_accuracy']) else "Lỗi" if kpca_metrics['error'] else "N/A"

            print(f"{d_val:<15} | {pca_knn_str:<15} | {kpca_knn_str:<15}")

            if not np.isnan(pca_metrics['knn_accuracy']) and pca_metrics['knn_accuracy'] > best_acc_pca_knn_lfw:
                best_acc_pca_knn_lfw = pca_metrics['knn_accuracy']
                best_d_pca_knn_lfw = d_val
            if not np.isnan(kpca_metrics['knn_accuracy']) and kpca_metrics['knn_accuracy'] > best_acc_kpca_knn_lfw:
                best_acc_kpca_knn_lfw = kpca_metrics['knn_accuracy']
                best_d_kpca_knn_lfw = d_val

        print("----------------------------------------------------------")
        print("LFW - Số chiều tốt nhất (k-NN Accuracy):")
        if best_d_pca_knn_lfw != -1: print(f"  - PCA (Tự code SVD): d={best_d_pca_knn_lfw} (Acc: {best_acc_pca_knn_lfw:.4f})")
        if best_d_kpca_knn_lfw != -1: print(f"  - KPCA (RBF):         d={best_d_kpca_knn_lfw} (Acc: {best_acc_kpca_knn_lfw:.4f})")
        print("==========================================================")
    else:
        print("\n--- LFW: Bỏ qua do lỗi tải dữ liệu ---")


    end_time_total = time.time()
    print(f"\nTổng thời gian chạy ước tính: {(end_time_total - start_time_total)/60:.2f} phút")