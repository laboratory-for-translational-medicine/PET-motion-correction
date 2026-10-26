import os
import SimpleITK as sitk
from pathlib import Path

# Folder containing the new scans
root_folder = Path("/Users/varakmesropian/PET-Brain-Motion-Compensation-main/Hoffman reconstruction from 0, 45, 90 degrees")


# Start with the 0-degree scans
fixed_folder = root_folder / "0_degrees" / "frame_3_no_motion"
moving_folder = root_folder / "0_degrees" / "frame_3_motion"

def load_rotated_dicom(folder):
    reader = sitk.ImageSeriesReader()

    # Find DICOM series, following the professor's approach
    series_ids = reader.GetGDCMSeriesIDs(str(folder))

    if not series_ids:
        raise ValueError(f"No DICOM series found in {folder}")

    print(f"Found {len(series_ids)} DICOM series in {folder}")

    # Researcher requested ONLY the _rotated file
    rotated_files = sorted(folder.glob("*_rotated.dcm"))

    if len(rotated_files) != 1:
        raise ValueError(
            f"Expected one rotated DICOM in {folder}, found {len(rotated_files)}"
        )

    reader.SetFileNames([str(rotated_files[0])])
    image = reader.Execute()

    print("Original image size:", image.GetSize())

    # Remove the extra fourth dimension if its size is 1
    if image.GetDimension() == 4 and image.GetSize()[3] == 1:
        size = list(image.GetSize())
        size[3] = 0
        image = sitk.Extract(image, size, [0, 0, 0, 0])

    if image.GetDimension() != 3:
        raise ValueError("The loaded image is not 3D")

    return sitk.Cast(image, sitk.sitkFloat32)


fixed_image = load_rotated_dicom(fixed_folder)
moving_image = load_rotated_dicom(moving_folder)

print("Fixed image size:", fixed_image.GetSize())
print("Moving image size:", moving_image.GetSize())
print("Fixed dimensions:", fixed_image.GetDimension())




#-----------------------------------------
# Gaussian blur
#sigma = 0.5

#fixed_image = sitk.SmoothingRecursiveGaussian(fixed_image, sigma)
#moving_image = sitk.SmoothingRecursiveGaussian(moving_image, sigma)

#-----------------------------------------

# Initialize a rigid transform (3D)

#The code creates a rigid 3D transform.
#Rigid means the image can:
#rotate X
#rotate Y
#rotate Z

#move X
#move Y
#move Z
#but it cannot stretch or bend.
#So it has 6 controls total.


initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

#The GEOMETRY part means it initially lines up the images based on their geometric centers.
#So before the optimizer starts, it tries to put the two brains roughly on top of each other.


#-----------------------------------------
# Set up the registration method
#This basically creates the “registration machine.”
registration_method = sitk.ImageRegistrationMethod()

#-----------------------------------------
# Similarity metric: Mean Squared Error
'''''''''
This tells the program how to judge whether the images match.
Mean Squares is basically:
“Look at corresponding pixels in both images and measure how different their brightness values are.”

If the images line up well, the difference should get smaller.
So the optimizer’s goal is roughly:
make Mean Squared Error as small as possible
'''''''''
registration_method.SetMetricAsMeanSquares()   #changed this first, not work!



#-----------------------------------------
# Interpolation used when resampling the moving image

'''''''''
When the moving image gets shifted or rotated, some new positions fall between pixels.
Linear interpolation estimates what the image value should be at those in-between locations.
You can think of it as:
“If I move the brain halfway between two pixels, estimate the new value smoothly.”
'''''''''
registration_method.SetInterpolator(sitk.sitkLinear)

#-----------------------------------------
'''''''''
This is the part actually searching for the best alignment.

Think of it as a robot doing:
move a little
↓
check if the images match better
↓
move again
↓
check again
↓
repeat

learningRate=1.0 controls how big the optimizer’s steps are.
numberOfIterations=300 means it can try up to 300 updates.
convergenceMinimumValue and convergenceWindowSize help it decide:
“The result isn’t improving much anymore, I can stop.”
'''''''''
# Optimizer
registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.2,
    minStep=0.0001,
    numberOfIterations=200,
    relaxationFactor=0.9
)

# THIS CHANGE WORKS!
#-----------------------------------------
'''''''''
This is one of the important differences from Krutarth’s code.
Instead of manually saying something like:
rotation scale = 0.1
translation scale = 1

SimpleITK automatically figures out how to balance rotation and translation based on how much they physically move the image.

In kid terms:
The optimizer has rotation knobs and movement knobs, and this line helps make sure one type of knob does not overpower the other.


registration_method.SetMetricSamplingStrategy(
    registration_method.NONE
)

This means it uses all the image points when calculating the matching score.
It is not randomly sampling only part of the image.
So for this simple baseline, your professor is making it very straightforward.
'''''''''

registration_method.SetOptimizerScalesFromPhysicalShift()
registration_method.SetMetricSamplingStrategy(
    registration_method.NONE
)
#-----------------------------------------

'''''''''
This tells the registration:
“Start from the initial rigid transform we created earlier.”
'''''''''
# Set the initial transform
registration_method.SetInitialTransform(
    initial_transform,
    inPlace=False
)
#-----------------------------------------

'''''''''
This is where the optimizer actually runs and finds the best transform it can.
  reg_Homa_Varak
'''''''''

# Execute registration
final_transform = registration_method.Execute(
    fixed_image,
    moving_image
)
#-----------------------------------------


'''''''''
This extracts the 6 final values:
p[0] = rotation X
p[1] = rotation Y
p[2] = rotation Z

p[3] = translation X
p[4] = translation Y
p[5] = translation Z

Finally it prints them:
angleX
angleY
angleZ
translationX
translationY
translationZ

The angles are printed in radians, and the translations are printed in millimeters.
'''''''''
p = final_transform.GetParameters()

print(
    f"angleX {p[0]:.3f} rad, "
    f"angleY {p[1]:.3f} rad, "
    f"angleZ {p[2]:.3f} rad, "
    f"translationX {p[3]:.3f} mm, "
    f"translationY {p[4]:.3f} mm, "
    f"translationZ {p[5]:.3f} mm"
)
inverse_transform = final_transform.GetInverse()

print("Original transform:")
print(final_transform)

print("Inverse transform:")
print(inverse_transform)