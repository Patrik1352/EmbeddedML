#!/usr/bin/env python3
"""Замер задержки и пропускной способности ONNX-моделей на процессоре.

Запускается там, где модель будет жить: на Raspberry Pi 5. На ноутбуке скрипт
тоже работает, но числа там другие, и сравнивать их с платой нельзя.

    python bench_onnx.py models/mobilenet_v2_fp32.onnx models/mobilenet_v2_int8.onnx --threads 1,4
    python bench_onnx.py models/*.onnx --image cat.jpg   # заодно классифицирует картинку
"""
import argparse
import pathlib
import platform
import time

import numpy as np
import onnxruntime as ort


def make_session(path, threads):
    """Сессия ONNX Runtime с заданным числом потоков внутри одного оператора."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def measure(session, sample, warmup, iterations):
    """Возвращает список задержек в миллисекундах."""
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: sample})
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, {name: sample})
        finished = time.perf_counter()
        timings.append((finished - started) * 1000)
    return timings


def load_image(path):
    """Картинка в тензор 1×3×224×224 с нормализацией ImageNet."""
    from PIL import Image

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    with Image.open(path) as picture:
        picture = picture.convert("RGB").resize((224, 224))
        array = np.asarray(picture, dtype=np.float32) / 255.0
    array = (array - mean) / std
    array = array.transpose(2, 0, 1)
    return np.ascontiguousarray(array[np.newaxis, :])


def top5(session, sample):
    """Пять самых вероятных классов ImageNet по номерам."""
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: sample})[0][0]
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    order = np.argsort(probabilities)[::-1][:5]
    return [(int(index), float(probabilities[index])) for index in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    thread_counts = [int(value) for value in args.threads.split(",")]
    print(f"машина: {platform.machine()}, onnxruntime {ort.__version__}")
    print(f"провайдеры: {ort.get_available_providers()}")
    print()

    if args.image is not None:
        sample = load_image(args.image)
    else:
        generator = np.random.default_rng(0)
        sample = generator.standard_normal((1, 3, 224, 224)).astype(np.float32)

    header = f"{'модель':<34} {'потоков':>7} {'p50, мс':>9} {'p95, мс':>9} {'кадров/с':>9}"
    print(header)
    for model in args.models:
        path = pathlib.Path(model)
        size_mb = path.stat().st_size / 1_000_000
        for threads in thread_counts:
            session = make_session(path, threads)
            timings = measure(session, sample, args.warmup, args.iterations)
            p50 = float(np.percentile(timings, 50))
            p95 = float(np.percentile(timings, 95))
            per_second = 1000.0 / p50
            label = f"{path.name} ({size_mb:.1f} МБ)"
            print(f"{label:<34} {threads:>7} {p50:>9.1f} {p95:>9.1f} {per_second:>9.1f}")

    if args.image is not None:
        print()
        print("топ-5 классов ImageNet по номерам, чтобы сверить fp32 и int8:")
        for model in args.models:
            session = make_session(model, thread_counts[0])
            answer = top5(session, sample)
            pretty = ", ".join(f"{index}: {probability:.2f}" for index, probability in answer)
            print(f"  {pathlib.Path(model).name}: {pretty}")


if __name__ == "__main__":
    main()
