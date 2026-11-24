import csv
import re
from pathlib import Path

import SimpleITK as sitk
import matplotlib.pyplot as plt

# New Hoffman dataset
root_folder = Path(
    "/Users/varakmesropian/PET-Brain-Motion-Compensation-main/"
    "Hoffman reconstruction from 0, 45, 90 degrees-2"
)

# Mask creation ONLY: the actual images registered below remain unblurred.
MASK_SMOOTHING_SIGMA_MM = 2.0
MASK_DILATION_MM = 2.0

OUTPUT_CSV = Path(__file__).with_name("masking_all_frames_results.csv")


def read_ground_truth(txt_path):
    text = txt_path.read_text()
    label = "# Detector to image space transformation matrix"
    if label not in text:
        raise ValueError(f"Matrix section not found in {txt_path}")
    matrix_text = text.split(label, 1)[1]
    rows = re.findall(r"\[([^\[\]]+)\]", matrix_text)[:4]
    if len(rows) != 4:
        raise ValueError(f"Expected four matrix rows in {txt_path}")
    matrix = [[float(value) for value in row.split()] for row in rows]
    if any(len(row) != 4 for row in matrix):
        raise ValueError(f"Expected a 4x4 matrix in {txt_path}")
    # Translation components of the matrix, in metres.
    return [matrix[i][3] for i in range(3)]


def load_dicom(folder):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {folder}")

    dicom_files = sorted(folder.glob("*.dcm"))
    if len(dicom_files) != 1:
        raise ValueError(f"Expected one DICOM file in {folder}, found {len(dicom_files)}")

    reader.SetFileNames([str(dicom_files[0])])
    image = reader.Execute()

    if image.GetDimension() == 4 and image.GetSize()[3] == 1:
        size = list(image.GetSize())
        size[3] = 0
        image = sitk.Extract(image, size, [0, 0, 0, 0])

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image in {folder}, got {image.GetDimension()}D")
    return sitk.Cast(image, sitk.sitkFloat32)


def register_with_mask(fixed_image, moving_image, fixed_mask):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricFixedMask(fixed_mask)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.2,
        minStep=0.0001,
        numberOfIterations=200,
        relaxationFactor=0.9,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # No Gaussian pre-blurring applied to either registration input.
    return registration_method.Execute(fixed_image, moving_image)


results = []

for angle in [0, 45, 90]:
    angle_folder = root_folder / f"{angle}_degrees"
    fixed_folder = angle_folder / "frame_1_no_motion"

    # Load the fixed image and create its mask once per angle.
    fixed_image = load_dicom(fixed_folder)
    mask_source = sitk.SmoothingRecursiveGaussian(
        fixed_image, MASK_SMOOTHING_SIGMA_MM
    )
    fixed_mask = sitk.OtsuThreshold(mask_source, 0, 1)

    # Expand the mask by approximately 3 mm in every direction
    radius = [
        max(1, round(MASK_DILATION_MM / spacing))
        for spacing in fixed_image.GetSpacing()
    ]

    fixed_mask = sitk.BinaryDilate(
        fixed_mask,
        radius,
        sitk.sitkBall,
        1
    )
    print(f"\nCreated fixed mask for {angle} degrees")

    # Display the middle slice of the original image and its mask.
    image_array = sitk.GetArrayFromImage(fixed_image)
    mask_array = sitk.GetArrayFromImage(fixed_mask)
    middle_slice = image_array.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image_array[middle_slice], cmap="gray")
    axes[0].set_title(f"{angle}° — Original PET Image")
    axes[0].axis("off")

    axes[1].imshow(mask_array[middle_slice], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"{angle}° — Smoothed Otsu Mask")
    axes[1].axis("off")
    fig.tight_layout()
    figure_path = Path(__file__).with_name(f"mask_{angle}_degrees.png")
    fig.savefig(figure_path, dpi=150)
    print(f"Saved mask image: {figure_path}")

    reference_txt = angle_folder / f"transform_no_motion_{angle}_deg.txt"
    reference_translation = read_ground_truth(reference_txt)

    for frame in [1, 2, 3]:
        moving_folder = angle_folder / f"frame_{frame}_motion"
        moving_image = load_dicom(moving_folder)
        motion_txt = moving_folder / f"transform_motion_frame_{frame}.txt"
        motion_translation = read_ground_truth(motion_txt)

        # Same reported ground-truth calculation as in your original code.
        ground_truth = [
            (motion_translation[i] - reference_translation[i]) * 1000
            for i in range(3)
        ]

        final_transform = register_with_mask(fixed_image, moving_image, fixed_mask)
        p = final_transform.GetParameters()
        estimated = [p[3], p[4], p[5]]
        differences = [estimated[i] - ground_truth[i] for i in range(3)]

        print(f"\n--- Angle {angle} degrees | Frame {frame} ---")
        print("Ground truth: " + ", ".join(f"{v:.3f}" for v in ground_truth) + " mm")
        print("Estimated:    " + ", ".join(f"{v:.3f}" for v in estimated) + " mm")
        print("Difference:   " + ", ".join(f"{v:+.3f}" for v in differences) + " mm")

        results.append({
            "angle_deg": angle,
            "frame": frame,
            "method": "smoothed_Otsu_fixed_mask_no_registration_blur",
            "mask_sigma_mm": MASK_SMOOTHING_SIGMA_MM,
            "gt_x_mm": ground_truth[0],
            "gt_y_mm": ground_truth[1],
            "gt_z_mm": ground_truth[2],
            "estimated_x_mm": estimated[0],
            "estimated_y_mm": estimated[1],
            "estimated_z_mm": estimated[2],
            "difference_x_mm": differences[0],
            "difference_y_mm": differences[1],
            "difference_z_mm": differences[2],
            "rotation_x_rad": p[0],
            "rotation_y_rad": p[1],
            "rotation_z_rad": p[2],
        })

with OUTPUT_CSV.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nCompleted {len(results)} registrations. Saved results to: {OUTPUT_CSV}")
print("Note: differences are numerical comparisons, not validated errors until coordinate conventions are confirmed.")

# Show all three angle-specific mask figures after registration finishes.
plt.show()