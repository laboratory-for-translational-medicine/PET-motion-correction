import SimpleITK as sitk
import numpy as np
import nrrd

fixed_path = r"C:\Users\elodi\Downloads\Hoffman reconstruction from 0 degrees\0_degrees\frame_1_no_motion\hoffman_20260807_160231_GATE_0_degrees.nrrd"
moving2_path = r"C:\Users\elodi\Downloads\Hoffman reconstruction from 0 degrees\0_degrees\frame_2_motion\hoffman_20260807_160231_GATE_0_degrees.nrrd"
moving3_path = r"C:\Users\elodi\Downloads\Hoffman reconstruction from 0 degrees\0_degrees\frame_3_motion\hoffman_20260807_160231_GATE_0_degrees.nrrd"
moving_path = moving2_path

# Load NRRD files
fixed_array, fixed_header = nrrd.read(fixed_path)
moving_array, moving_header = nrrd.read(moving_path)

# Convert arrays to SimpleITK images
fixed_image = sitk.GetImageFromArray(
    np.transpose(fixed_array, (2, 1, 0)).astype(np.float32)
)
moving_image = sitk.GetImageFromArray(
    np.transpose(moving_array, (2, 1, 0)).astype(np.float32)
)

# Restore spacing from the NRRD headers
fixed_image.SetSpacing(tuple(fixed_header["spacings"]))
moving_image.SetSpacing(tuple(moving_header["spacings"]))

# Restore origin, if available
if "space origin" in fixed_header:
    fixed_image.SetOrigin(tuple(fixed_header["space origin"]))

if "space origin" in moving_header:
    moving_image.SetOrigin(tuple(moving_header["space origin"]))
    
# sitk.WriteImage(fixed_image, fixed_path[:-4]+"mha")
# sitk.WriteImage(moving_image, moving_path[:-4]+"mha")

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
registration_method.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=300,
    convergenceMinimumValue=1e-7,
    convergenceWindowSize=10
)

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

p = final_transform.GetParameters()

print(
    f"angleX {p[0]:.3f} rad, "
    f"angleY {p[1]:.3f} rad, "
    f"angleZ {p[2]:.3f} rad, "
    f"translationX {p[3]:.3f} mm, "
    f"translationY {p[4]:.3f} mm, "
    f"translationZ {p[5]:.3f} mm"
)