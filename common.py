# import cv2, time, csv, threading, queue
# import numpy as np
# import torch, pyttsx3
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# K_CALIB = 1000.0      # masofani kalibrlash koeffitsienti (o'zingiz sozlang)
# COOLDOWN = 3
# WARMUP = 10
#
# # ---- MiDaS ----
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
# midas_tf = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# def depth_estimate(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     batch = midas_tf(img).to(DEVICE)
#     with torch.no_grad():
#         pred = midas(batch)
#         pred = torch.nn.functional.interpolate(
#             pred.unsqueeze(1), size=img.shape[:2],
#             mode="bicubic", align_corners=False).squeeze()
#     return pred.cpu().numpy()
#
# # ---- Ovozli ogohlantirish (bitta thread) ----
# _q = queue.Queue()
# def _tts_worker():
#     eng = pyttsx3.init(); eng.setProperty("rate", 150)
#     while True:
#         eng.say(_q.get()); eng.runAndWait()
# threading.Thread(target=_tts_worker, daemon=True).start()
# _last = {}
# def speak(label, text, now):
#     if now - _last.get(label, 0) > COOLDOWN:
#         _q.put(text); _last[label] = now
#
# def _sync():
#     if DEVICE == "cuda": torch.cuda.synchronize()
#
# # ---- Ketma-ket / parallel ----
# def run_sequential(detect, frame):
#     t0 = time.perf_counter()
#     boxes = detect(frame); depth = depth_estimate(frame); _sync()
#     return boxes, depth, time.perf_counter() - t0
#
# def run_parallel(detect, frame):
#     out = {}
#     t0 = time.perf_counter()
#     a = threading.Thread(target=lambda: out.__setitem__("b", detect(frame)))
#     b = threading.Thread(target=lambda: out.__setitem__("d", depth_estimate(frame)))
#     a.start(); b.start(); a.join(); b.join(); _sync()
#     return out["b"], out["d"], time.perf_counter() - t0
#
# def distance_of(depth, x1, y1, x2, y2):
#     h, w = depth.shape
#     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#     dx, dy = max((x2 - x1) // 6, 1), max((y2 - y1) // 6, 1)
#     patch = depth[max(cy-dy,0):min(cy+dy,h), max(cx-dx,0):min(cx+dx,w)]
#     return K_CALIB / (float(np.median(patch)) + 1e-6)
#
# def main_loop(model_name, detect, max_frames=300):
#     cap = cv2.VideoCapture(0)
#     rows, n = [], 0
#     while True:
#         ok, frame = cap.read()
#         if not ok: break
#         _, _, t_seq = run_sequential(detect, frame)
#         boxes, depth, t_par = run_parallel(detect, frame)
#         n += 1
#         if n > WARMUP:
#             rows.append((n, t_seq, t_par, t_seq / t_par, 1/t_seq, 1/t_par))
#         now = time.time()
#         for (x1, y1, x2, y2, label) in boxes:
#             d = distance_of(depth, x1, y1, x2, y2)
#             if d < 2:   color, st = (0,0,255), "Juda yaqin"
#             elif d < 5: color, st = (0,255,255), "O'rtacha"
#             else:       color, st = (0,255,0), "Uzoq"
#             if d < 2: speak(label, f"Diqqat! {label}, {d:.1f} metr", now)
#             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#             cv2.putText(frame, f"{label} {d:.1f}m {st}", (x1, max(y1-8,15)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.putText(frame, f"{model_name} Seq {1/t_seq:.1f} FPS | Par {1/t_par:.1f} FPS | x{t_seq/t_par:.2f}",
#                     (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
#         cv2.imshow(model_name, frame)
#         if cv2.waitKey(1) == ord("q") or n >= max_frames + WARMUP: break
#     cap.release(); cv2.destroyAllWindows()
#     if rows:
#         a = np.array(rows)
#         with open(f"results_{model_name}.csv", "w", newline="") as f:
#             w = csv.writer(f)
#             w.writerow(["frame","t_seq_s","t_par_s","speedup","fps_seq","fps_par"])
#             w.writerows(rows)
#         print(f"\n{model_name}: seq={a[:,1].mean()*1000:.1f} ms, par={a[:,2].mean()*1000:.1f} ms, "
#               f"speedup={a[:,3].mean():.2f}x, FPS seq={a[:,4].mean():.1f}, FPS par={a[:,5].mean():.1f}")




import cv2
import time
import csv
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
import pyttsx3


# =====================================================
# 1. SOZLAMALAR
# =====================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CAMERA_INDEX = 0
WARMUP = 10
MAX_FRAMES = 300
COOLDOWN = 3.0

NEAR_DISTANCE = 2.0
MID_DISTANCE = 5.0

# MiDaS qiymati va haqiqiy masofa (metr) juftliklari.
# Quyidagi ro'yxatga faqat o'zingiz o'lchagan qiymatlarni kiriting.
#
# Format: (MiDaS_depth_value, haqiqiy_masofa_metrda)
#
# Misol tariqasida tasodifiy qiymatlar kiritmang.
CALIBRATION_POINTS = [
    # (haqiqiy_depth_qiymati_1, 1.0),
    # (haqiqiy_depth_qiymati_2, 2.0),
    # (haqiqiy_depth_qiymati_3, 3.0),
    # (haqiqiy_depth_qiymati_4, 4.0),
    # (haqiqiy_depth_qiymati_5, 5.0),
]


# =====================================================
# 2. MiDaS MODEL
# =====================================================

print(f"Hisoblash qurilmasi: {DEVICE}")
print("MiDaS modeli yuklanmoqda...")

midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
).to(DEVICE).eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)
midas_tf = midas_transforms.small_transform

print("MiDaS tayyor.")


def sync_device():
    """GPU hisoblashlari tugashini kutadi."""
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def depth_estimate(frame):
    """
    MiDaS nisbiy depth xaritasini qaytaradi.
    Kalibrlashsiz bu qiymatlar metr hisoblanmaydi.
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = midas_tf(image).to(DEVICE)

    with torch.inference_mode():
        prediction = midas(batch)

        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    return prediction.detach().cpu().numpy()


# =====================================================
# 3. MASOFANI METRDA KALIBRLASH
# =====================================================

def fit_distance_calibration(points):
    """
    Model:
        distance_m = a / depth_value + b

    points ichida kamida 3 ta haqiqiy o'lchov bo'lishi kerak.
    """
    if len(points) < 3:
        return None

    values = np.asarray([p[0] for p in points], dtype=float)
    distances = np.asarray([p[1] for p in points], dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(distances)
        & (values > 0)
        & (distances > 0)
    )

    values = values[valid]
    distances = distances[valid]

    if len(values) < 3:
        return None

    # Bir-biridan farqli depth qiymatlari bo'lishi kerak.
    x = 1.0 / values

    if np.unique(x).size < 3:
        return None

    a, b = np.polyfit(x, distances, 1)

    predicted = a * x + b
    rmse = float(np.sqrt(np.mean((predicted - distances) ** 2)))

    return float(a), float(b), rmse


DISTANCE_CALIBRATION = fit_distance_calibration(CALIBRATION_POINTS)

if DISTANCE_CALIBRATION is None:
    print(
        "OGOHLANTIRISH: masofa kalibrlanmagan. "
        "CALIBRATION_POINTS ga haqiqiy o'lchovlarni kiriting."
    )
else:
    print(
        "Kalibrlash tayyor. Kalibrlash nuqtalaridagi RMSE: "
        f"{DISTANCE_CALIBRATION[2]:.3f} m"
    )


def object_depth_value(depth, box):
    """Bounding box ichidagi median MiDaS qiymatini hisoblaydi."""
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = depth.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return float("nan")

    patch = depth[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch)]

    if valid.size == 0:
        return float("nan")

    return float(np.median(valid))


def distance_in_meters(depth_value):
    """Kalibrlangan model yordamida taxminiy masofani qaytaradi."""
    if DISTANCE_CALIBRATION is None:
        return None

    if not np.isfinite(depth_value) or depth_value <= 0:
        return None

    a, b, _ = DISTANCE_CALIBRATION
    distance = a / depth_value + b

    if not np.isfinite(distance) or distance <= 0:
        return None

    return float(distance)


# =====================================================
# 4. OVOZLI OGOHLANTIRISH
# =====================================================

tts_queue = queue.Queue()
last_spoken = {}
tts_lock = threading.Lock()


def tts_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
    except Exception as exc:
        print("Ovozli tizimni ishga tushirishda xato:", exc)
        return

    while True:
        message = tts_queue.get()

        try:
            if message is None:
                break

            engine.say(message)
            engine.runAndWait()

        except Exception as exc:
            print("TTS xatosi:", exc)

        finally:
            tts_queue.task_done()


threading.Thread(
    target=tts_worker,
    daemon=True
).start()


def speak(label, message, now):
    with tts_lock:
        if now - last_spoken.get(label, 0) >= COOLDOWN:
            last_spoken[label] = now
            tts_queue.put(message)


# =====================================================
# 5. SEQUENTIAL ISHLASH
# =====================================================

def run_sequential(detect, frame):
    sync_device()
    start = time.perf_counter()

    boxes = detect(frame)
    depth = depth_estimate(frame)

    sync_device()
    elapsed = time.perf_counter() - start

    return boxes, depth, elapsed


# =====================================================
# 6. PARALLEL ISHLASH
# =====================================================

executor = ThreadPoolExecutor(max_workers=2)


def run_parallel(detect, frame):
    sync_device()
    start = time.perf_counter()

    # Obyekt aniqlash va depth baholash alohida oqimlarda.
    future_boxes = executor.submit(detect, frame)
    future_depth = executor.submit(depth_estimate, frame)

    boxes = future_boxes.result()
    depth = future_depth.result()

    sync_device()
    elapsed = time.perf_counter() - start

    return boxes, depth, elapsed


# =====================================================
# 7. QORA STATISTIKA PANELI
# =====================================================

def draw_stats(frame, model_name, t_seq, t_par, frame_number):
    panel_w = min(540, frame.shape[1])
    panel_h = min(155, frame.shape[0])

    cv2.rectangle(
        frame,
        (0, 0),
        (panel_w, panel_h),
        (0, 0, 0),
        thickness=-1
    )

    fps_seq = 1.0 / t_seq if t_seq > 0 else 0.0
    fps_par = 1.0 / t_par if t_par > 0 else 0.0
    speedup = t_seq / t_par if t_par > 0 else 0.0

    lines = [
        f"Model: {model_name}",
        f"Sequential: {t_seq * 1000:.2f} ms | {fps_seq:.2f} FPS",
        f"Parallel:   {t_par * 1000:.2f} ms | {fps_par:.2f} FPS",
        f"Speedup: {speedup:.2f}x | Frame: {frame_number}"
    ]

    y = 27

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        y += 32


# =====================================================
# 8. ASOSIY SIKL
# =====================================================

def main_loop(model_name, detect, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Kamera ochilmadi. CAMERA_INDEX={CAMERA_INDEX} ni tekshiring."
        )

    rows = []
    frame_number = 0

    try:
        while frame_number < max_frames + WARMUP:
            ok, frame = cap.read()

            if not ok:
                print("Kameradan kadr olinmadi.")
                break

            frame_number += 1

            # Ikkala rejim bir xil kadrda bajariladi.
            # Navbatni almashtirish tartib ta'sirini biroz kamaytiradi.
            if frame_number % 2 == 0:
                _, _, t_seq = run_sequential(detect, frame)
                boxes, depth, t_par = run_parallel(detect, frame)
            else:
                boxes, depth, t_par = run_parallel(detect, frame)
                _, _, t_seq = run_sequential(detect, frame)

            if frame_number > WARMUP:
                rows.append({
                    "frame": frame_number,
                    "t_seq_s": t_seq,
                    "t_par_s": t_par,
                    "speedup": t_seq / t_par if t_par > 0 else 0.0,
                    "fps_seq": 1.0 / t_seq if t_seq > 0 else 0.0,
                    "fps_par": 1.0 / t_par if t_par > 0 else 0.0
                })

            now = time.time()

            # Obyektlarni parallel sinovdan olingan natijalar bilan chizamiz.
            for x1, y1, x2, y2, label in boxes:
                depth_value = object_depth_value(
                    depth,
                    (x1, y1, x2, y2)
                )

                distance_m = distance_in_meters(depth_value)

                if distance_m is None:
                    color = (0, 255, 255)
                    text = f"{label} | Masofa kalibrlanmagan"

                elif distance_m < NEAR_DISTANCE:
                    color = (0, 0, 255)
                    text = f"{label} | {distance_m:.2f} m | JUDA YAQIN"

                    speak(
                        str(label),
                        f"Diqqat! {label}, taxminan {distance_m:.1f} metr",
                        now
                    )

                elif distance_m < MID_DISTANCE:
                    color = (0, 255, 255)
                    text = f"{label} | {distance_m:.2f} m | YAQIN"

                else:
                    color = (0, 255, 0)
                    text = f"{label} | {distance_m:.2f} m | UZOQ"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 8, 165)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA
                )

            draw_stats(
                frame,
                model_name,
                t_seq,
                t_par,
                frame_number
            )

            cv2.imshow(model_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # =================================================
    # 9. CSV VA YAKUNIY NATIJALAR
    # =================================================

    if rows:
        filename = f"results_{model_name.replace('-', '_')}.csv"

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "frame",
                    "t_seq_s",
                    "t_par_s",
                    "speedup",
                    "fps_seq",
                    "fps_par"
                ]
            )

            writer.writeheader()
            writer.writerows(rows)

        seq_times = np.array([r["t_seq_s"] for r in rows])
        par_times = np.array([r["t_par_s"] for r in rows])

        mean_seq = float(np.mean(seq_times))
        mean_par = float(np.mean(par_times))

        # O'rtacha vaqtlar nisbati — umumiy speedup.
        mean_speedup = mean_seq / mean_par if mean_par > 0 else 0.0

        # FPS o'rtacha vaqtning teskarisi sifatida hisoblanadi.
        fps_seq = 1.0 / mean_seq if mean_seq > 0 else 0.0
        fps_par = 1.0 / mean_par if mean_par > 0 else 0.0

        print(f"\n========== {model_name} NATIJALARI ==========")
        print(f"O'lchangan kadrlar: {len(rows)}")
        print(f"Sequential vaqt: {mean_seq * 1000:.2f} ms")
        print(f"Parallel vaqt:   {mean_par * 1000:.2f} ms")
        print(f"Sequential FPS:  {fps_seq:.2f}")
        print(f"Parallel FPS:    {fps_par:.2f}")
        print(f"Speedup:         {mean_speedup:.3f}x")
        print(f"CSV fayl:        {filename}")

    else:
        print("Statistikani hisoblash uchun yetarli kadr yo'q.")
