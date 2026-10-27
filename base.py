import os
import re
import SimpleITK as sitk
import numpy as np
import pandas as pd

ROOT = r"C:\Users\PCSHOP\OneDrive\Desktop\university\Courses\CPS40A\Hoffman reconstruction from 0, 45, 90 degrees\Hoffman reconstruction from 0, 45, 90 degrees"
ANGLES = [0, 45, 90]
FRAMES = [1, 2, 3]

# Test 1: blur once before registration
SIGMA_MM = 0

# Test 2: shrink + blur while registering
SHRINK_FACTORS = [4, 2, 1]
PYRAMID_SIGMAS = [4.0, 2.0, 0.0]   # mm, last value = blur in the final round

METHODS = ["pre_blur"]


# Load the DICOM image inside one folder
# Each .dcm file is already a full 3D volume, so read one file directly
def load_dicom(folder, file_index=0):
    dicom_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith("rotated.dcm")
    )

    if not dicom_files:
        raise ValueError(f"No .dcm files found in: {folder}")

    path = os.path.join(folder, dicom_files[file_index])
    print("    loading:", path)
    image = sitk.ReadImage(path)
    return sitk.Cast(image, sitk.sitkFloat32)

def gaussian_blur(image, sigma_mm):
    if sigma_mm <= 0:
        return image
    return sitk.SmoothingRecursiveGaussian(image, sigma_mm)

# Read the translation from the "Detector to image space" matrix, in mm
def read_translation(path):
    with open(path) as f:
        text = f.read()
 
    part = text.split("Detector to image space")[1]
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", part)]
    M = np.array(nums[:16]).reshape(4, 4)
    return M[:3, 3] * 1000   # meters -> mm

    
# sitk.WriteImage(fixed_image, fixed_path[:-4]+"mha")
# sitk.WriteImage(moving_image, moving_path[:-4]+"mha")

# Rigid registration of one moving image to the fixed image
def register(fixed_image, moving_image, use_pyramid):

    # Initialize a rigid transform (3D)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric: Mean Squared Error
    registration_method.SetMetricAsMeanSquares()

    # Interpolation used when resampling the moving image
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(
       learningRate=2.0,
       minStep=1e-4,
       numberOfIterations=1000,
       gradientMagnitudeTolerance=1e-8
   )
    # Shrink + blur while registering (Test 2 only)
    if use_pyramid:
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=SHRINK_FACTORS)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=PYRAMID_SIGMAS)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() 

    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetMetricSamplingStrategy(
        registration_method.NONE
    )

    # Set the initial transform
    registration_method.SetInitialTransform(
        initial_transform,
        inPlace=False
    )

    # Execute registration
    final_transform = registration_method.Execute(
        fixed_image,
        moving_image
    )

    print("    stop:", registration_method.GetOptimizerStopConditionDescription())
    print("    iterations:", registration_method.GetOptimizerIteration())

    p = final_transform.GetParameters()

    # Translation part only (mm)
    return np.array([p[3], p[4], p[5]])

# ---------------- Translation test: compare both methods ----------------
 
rows = []
 
for angle in ANGLES:
    print(f"\n===== {angle} degrees =====")
    angle_dir = os.path.join(ROOT, f"{angle}_degrees")
    base_file = os.path.join(angle_dir, f"transform_no_motion_{angle}_deg.txt")
 
    # Fixed image: frame 1 without motion (loaded once, used by both methods)
    fixed_raw = load_dicom(os.path.join(angle_dir, "frame_1_no_motion"))
 
    for frame in FRAMES:
        motion_dir = os.path.join(angle_dir, f"frame_{frame}_motion")
        motion_file = os.path.join(motion_dir, f"transform_motion_frame_{frame}.txt")
 
        moving_raw = load_dicom(motion_dir)
        true_t = read_translation(motion_file) - read_translation(base_file)
 
        for method in METHODS:
 
            if method == "pre_blur":
                # Test 1: blur once before, then register with no other blur
                fixed_image = gaussian_blur(fixed_raw, SIGMA_MM)
                moving_image = gaussian_blur(moving_raw, SIGMA_MM)
                est_t = register(fixed_image, moving_image, use_pyramid=False)
            else:
                # Test 2: no blur before, shrink + blur while registering
                est_t = register(fixed_raw, moving_raw, use_pyramid=True)
 
            diff = est_t - true_t
            error = np.linalg.norm(diff)
 
            print(f"  frame {frame}, {method}:")
            print(f"    true shift   {np.round(true_t, 3)} mm")
            print(f"    est. shift   {np.round(est_t, 3)} mm")
            print(f"    error        {error:.3f} mm")
 
            rows.append({
                "method": method,
                "angle": angle,
                "frame": frame,
                "true_tx_mm": true_t[0],
                "true_ty_mm": true_t[1],
                "true_tz_mm": true_t[2],
                "est_tx_mm": est_t[0],
                "est_ty_mm": est_t[1],
                "est_tz_mm": est_t[2],
                "error_x_mm": diff[0],
                "error_y_mm": diff[1],
                "error_z_mm": diff[2],
                "error_mm": error,
            })
 
df = pd.DataFrame(rows)
df.to_csv("translation_results_compare.csv", index=False)
 
print("\n===== Error per method and angle (mm) =====")
print(df.groupby(["method", "angle"])["error_mm"].agg(["mean", "max"]))
 
print("\n===== Overall error per method (mm) =====")
print(df.groupby("method")["error_mm"].agg(["mean", "max"]))
 
# Side by side, frame by frame
print("\n===== Frame by frame (mm) =====")
print(df.pivot_table(index=["angle", "frame"], columns="method", values="error_mm").round(3))