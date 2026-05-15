import argparse
import os
import sys

import cv2
import dlib
import numpy as np


def shape_to_np(shape):
    """Преобразует landmarks dlib в numpy-массив размера (68, 2)."""
    points = np.zeros((68, 2), dtype=np.int32)

    for i in range(68):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


def fill_convex_part(mask, points, value=255):
    """Закрашивает выпуклую область по точкам."""
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, value)


def remove_part(mask, points, dilation=5):
    """Удаляет область из маски, например глаза, рот или брови."""
    temp = np.zeros_like(mask)
    fill_convex_part(temp, points, 255)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation + 1, 2 * dilation + 1)
        )
        temp = cv2.dilate(temp, kernel)

    mask[temp > 0] = 0


def create_face_skin_mask(image_shape, landmarks):
    """
    Создаёт маску кожи лица на основе landmarks.

    Используются:
    0-16  — контур нижней части лица,
    17-26 — брови, по ним приблизительно восстанавливается область лба,
    36-47 — глаза,
    48-67 — рот.
    """

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Контур нижней части лица: подбородок и челюсть
    jaw = landmarks[0:17]

    # Верхнюю границу лица приближённо строим по линии бровей.
    brows = landmarks[17:27].copy()

    brow_center_y = np.mean(brows[:, 1])
    chin_y = landmarks[8, 1]
    face_height = max(1, chin_y - brow_center_y)

    forehead_offset = int(0.35 * face_height)

    upper_face = brows[::-1].copy()
    upper_face[:, 1] -= forehead_offset

    upper_face[:, 0] = np.clip(upper_face[:, 0], 0, width - 1)
    upper_face[:, 1] = np.clip(upper_face[:, 1], 0, height - 1)

    # Основной контур лица: челюсть + примерная линия лба
    face_contour = np.vstack([jaw, upper_face])

    cv2.fillPoly(mask, [face_contour.astype(np.int32)], 255)

    # Удаляем глаза
    remove_part(mask, landmarks[36:42], dilation=6)  # левый глаз
    remove_part(mask, landmarks[42:48], dilation=6)  # правый глаз

    # Удаляем брови
    remove_part(mask, landmarks[17:22], dilation=5)
    remove_part(mask, landmarks[22:27], dilation=5)

    # Удаляем рот и губы
    remove_part(mask, landmarks[48:68], dilation=5)

    # Немного сглаживаем края маски
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


def apply_mask(image, mask):
    """Оставляет только пиксели, попавшие в маску."""
    alpha = mask.astype(np.float32) / 255.0

    if image.ndim == 2:
        result = (image.astype(np.float32) * alpha).astype(np.uint8)
        return result

    if image.shape[2] == 4:
        result = image.copy()
        for c in range(3):
            result[:, :, c] = (image[:, :, c].astype(np.float32) * alpha).astype(np.uint8)

        result[:, :, 3] = (image[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
        return result

    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = (image[:, :, c].astype(np.float32) * alpha).astype(np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Создание маски кожных покровов лица по landmarks dlib"
    )

    parser.add_argument("input", help="Путь к входному изображению png/jpg/jpeg")
    parser.add_argument("output", help="Путь к выходному изображению")
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Путь к файлу shape_predictor_68_face_landmarks.dat"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: входной файл не найден: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.predictor):
        print(f"Ошибка: файл predictor не найден: {args.predictor}")
        sys.exit(1)

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Ошибка: не удалось прочитать изображение")
        sys.exit(1)

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    faces = detector(gray, 1)

    if len(faces) == 0:
        print("Лицо не найдено")
        sys.exit(1)

    full_mask = np.zeros(gray.shape, dtype=np.uint8)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = shape_to_np(shape)

        face_mask = create_face_skin_mask(image.shape, landmarks)
        full_mask = cv2.bitwise_or(full_mask, face_mask)

    result = apply_mask(image, full_mask)

    ok = cv2.imwrite(args.output, result)

    if not ok:
        print("Ошибка: не удалось сохранить результат")
        sys.exit(1)

    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()